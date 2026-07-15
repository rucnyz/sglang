"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a request to its token locations.
TokenToKVPoolAllocator manages the indices to kv cache data.
KVCache actually holds the physical kv cache.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import os
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

# Path-A actuator (XPoolActuator) requires both pools to be allocated
# via MultiTensorArena on a shared SharedHandlePool. Promote
if os.environ.get("SGLANG_HIMA") == "1":
    os.environ.setdefault("SGLANG_ARENA_SHARED", "1")


def kv_live_migration_enabled() -> bool:
    """Single source of truth for whether live KV-slot migration
    (cross_migrate src=kv, Stage-3) is enabled. Read by BOTH boot-time KV-pool
    construction (to set ``enable_kv_cache_copy`` so ``move_kv_cache`` works)
    and the Budgeter's Stage-3 walk gate (``SchedulerOwnerProvider``), so the
    two can never disagree. Fail-closed: default OFF until per-backend
    captured-graph replay coverage lands (proved on flashinfer; other backends
    not yet validated)."""
    return os.environ.get("SGLANG_XPOOL_KV_MIGRATE", "0") == "1"

import numpy as np
import torch
import triton

from sglang.jit_kernel.kvcache import can_use_store_cache, store_cache
from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import index_buf_accessor
from sglang.srt.layers.attention.dsa.quant_k_cache import (
    quantize_k_cache,
    quantize_k_cache_separate,
)
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.triton_ops.cache_move import (
    copy_all_layer_kv_cache_tiled,
    set_kv_buffer_prefix_valid_tiled,
)
from sglang.srt.mem_cache.utils import (
    get_mla_kv_buffer_triton,
    maybe_init_custom_mem_pool,
    set_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton_fp8_quant,
    set_mla_kv_scale_buffer_triton,
)
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.managers.schedule_batch import Req


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_cpu_has_amx_support = cpu_has_amx_support()
_is_hip = is_hip()
_is_fp8_fnuz = is_fp8_fnuz()
# `SGLANG_AITER_KV_CACHE_LAYOUT` is only meaningful on the ROCm AITER backend
# (HIP + --enable-aiter / SGLANG_USE_AITER=1). On any other platform / backend
# the SHUFFLE 5D pool layout has no consumer kernels, so the env var is
# silently ignored and the legacy NHD layout is used.
_use_aiter = bool(envs.SGLANG_USE_AITER.get()) and _is_hip


def get_tensor_size_bytes(t: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(t, list):
        return sum(get_tensor_size_bytes(x) for x in t)
    return np.prod(t.shape) * t.dtype.itemsize


def _set_kv_buffer_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    row_dim: int,  # head_num * head_dim
    store_dtype: torch.dtype,
    device_module: Any,
    size_limit: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    same_kv_dim: bool = True,
) -> None:
    row_bytes = row_dim * store_dtype.itemsize
    if (_is_cuda or _is_hip) and same_kv_dim and can_use_store_cache(row_bytes):
        return store_cache(
            k.view(-1, row_dim),
            v.view(-1, row_dim),
            k_cache.view(-1, row_dim),
            v_cache.view(-1, row_dim),
            indices,
            row_bytes=row_bytes,
            size_limit=size_limit,
        )

    if _is_cpu and _cpu_has_amx_support:
        return torch.ops.sgl_kernel.store_cache_cpu(
            k,
            v,
            k_cache,
            v_cache,
            indices,
            row_dim,
        )

    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode() and alt_stream is not None:
        current_stream = device_module.current_stream()
        alt_stream.wait_stream(current_stream)
        k_cache[indices] = k
        with device_module.stream(alt_stream):
            v_cache[indices] = v
        current_stream.wait_stream(alt_stream)
    else:  # fallback to naive implementation
        k_cache[indices] = k
        v_cache[indices] = v


def _set_kv_buffer_prefix_valid_impl(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    loc_2d: torch.Tensor,
    commit_lens: torch.Tensor,
    row_dim: int,
    store_dtype: torch.dtype,
) -> None:
    if k.numel() == 0 or loc_2d.numel() == 0 or commit_lens.numel() == 0:
        return

    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not loc_2d.is_contiguous():
        loc_2d = loc_2d.contiguous()
    if not commit_lens.is_contiguous():
        commit_lens = commit_lens.contiguous()

    row_bytes = row_dim * store_dtype.itemsize
    if row_bytes <= 0:
        return

    if row_bytes >= 8192:
        bytes_per_tile = 512
        num_warps = 8
    elif row_bytes >= 4096:
        bytes_per_tile = 256
        num_warps = 4
    else:
        bytes_per_tile = 128
        num_warps = 4

    grid = (
        int(loc_2d.shape[0]),
        int(loc_2d.shape[1]),
        triton.cdiv(row_bytes, bytes_per_tile),
    )

    set_kv_buffer_prefix_valid_tiled[grid](
        k,
        v,
        k_cache,
        v_cache,
        loc_2d,
        commit_lens,
        int(k.stride(0) * k.element_size()),
        int(v.stride(0) * v.element_size()),
        int(k_cache.stride(0) * k_cache.element_size()),
        int(v_cache.stride(0) * v_cache.element_size()),
        int(loc_2d.shape[1]),
        ROW_BYTES=row_bytes,
        BYTES_PER_TILE=bytes_per_tile,
        num_warps=num_warps,
        num_stages=2,
    )


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations.

    `size` is the LIVE admission cap (number of slot ids `alloc()` may
    hand out). When `max_size > size` (the dynamic-cap mode), the
    underlying `req_to_token` tensor is backed by a VA-stable
    `ReqTokenVAArena` whose `data_ptr()` is stable across `grow()` /
    `shrink()`. CUDA-captured kernels that bake in the pointer keep
    working post-resize.

    When `max_size == size` (the default, back-compat mode), the
    tensor is a standard `torch.zeros` allocation — byte-identical to
    the non-arena path. No arena involved.

    Dynamic-cap mode is opt-in: callers must pass `max_size > size`.
    """

    enable_mamba_extra_buffer_lazy: bool = False

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        *,
        max_size: Optional[int] = None,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        if max_size is None:
            max_size = size
        if max_size < size:
            raise ValueError(
                f"max_size={max_size} must be >= size={size}"
            )

        self.size = size
        # +1 padding row at index 0: cuda-graph padded batches default
        # req_pool_indices to 0, so dummy reads/writes land on the pad row
        # harmlessly. Valid slot ids are 1..size; admission never hands out
        # id 0. `_alloc_size` is the current row count (pad + live), tracked
        # so clear() can rebuild free_slots and grow/shrink can offset by +1.
        self._alloc_size = size + 1
        self.max_size = max_size
        self.max_context_len = max_context_len
        self.device = device

        # Row size in bytes for the (rows, max_context_len) int32 table.
        # All grows are in row units; arena handles the chunk rounding.
        self._row_bytes = max_context_len * 4  # int32

        if max_size > size:
            # Dynamic-cap mode: VA-stable backing.
            from sglang.srt.arena.req_token_arena import ReqTokenVAArena

            # Reserve max_size+1 rows (pad row 0 + max_size usable rows);
            # grow() maps more usable rows up to this ceiling.
            self._va_arena = ReqTokenVAArena(
                max_bytes=(max_size + 1) * self._row_bytes,
                device_id=torch.cuda.current_device(),
                pool_name="req_to_token",
            )
            # Map the pad row + `size` usable rows at boot.
            self._va_arena.set_mapped_bytes(self._alloc_size * self._row_bytes)
            # Tensor aliases the FULL VA range (shape = (max_size+1, ...)).
            # Slot ids in [1, size] are valid; admission gate (free_slots)
            # never hands out ids outside that range, so unmapped rows are
            # never accessed.
            self.req_to_token = self._va_arena.as_tensor(
                dtype=torch.int32,
                shape=(max_size + 1, max_context_len),
            )
            # Zero-init the mapped portion (pad row + live rows); matches the
            # torch.zeros baseline.
            self.req_to_token[: self._alloc_size].zero_()
            torch.cuda.synchronize()
        else:
            self._va_arena = None
            with memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                self.req_to_token = torch.zeros(
                    (self._alloc_size, max_context_len),
                    dtype=torch.int32,
                    device=device,
                )

        self.free_slots = list(range(1, self._alloc_size))

    def grow(self, new_size: int) -> int:
        """Extend the live admission cap to `new_size` (no-op if smaller).

        Maps additional physical pages, extends `free_slots` to expose
        slot ids `[size, new_size)` for allocation. Tensor `data_ptr()`
        unchanged. Returns the new `self.size`.

        Only valid in dynamic-cap mode (`max_size > size`).
        Raises if `new_size > max_size`.
        """
        if self._va_arena is None:
            raise RuntimeError(
                "ReqToTokenPool.grow requires dynamic-cap mode "
                "(construct with max_size > size)"
            )
        if new_size <= self.size:
            return self.size
        if new_size > self.max_size:
            raise ValueError(
                f"grow: new_size={new_size} > max_size={self.max_size}"
            )
        # Map physical pages for the new rows (pad row + new_size usable rows).
        self._va_arena.set_mapped_bytes((new_size + 1) * self._row_bytes)
        # Zero-init the newly-exposed rows so admission sees clean state.
        # Slot id s lives at row s; the new ids are size+1..new_size.
        self.req_to_token[self.size + 1 : new_size + 1].zero_()
        torch.cuda.synchronize()
        # Expose the new slot ids.
        self.free_slots.extend(range(self.size + 1, new_size + 1))
        self.size = new_size
        self._alloc_size = new_size + 1
        return self.size

    def shrink(self, new_size: int) -> int:
        """Reduce live admission cap to `new_size` (no-op if larger).

        Unmaps physical pages for rows `[new_size, size)`. Caller must
        ensure those slot ids are currently in `free_slots` (not held
        by a running req) — enforced by assertion.

        Returns new `self.size`.
        """
        if self._va_arena is None:
            raise RuntimeError(
                "ReqToTokenPool.shrink requires dynamic-cap mode"
            )
        if new_size >= self.size:
            return self.size
        if new_size < 0:
            raise ValueError(f"shrink: new_size={new_size} must be >= 0")
        # All slot ids in (new_size, size] must be currently free.
        free_set = set(self.free_slots)
        held = [s for s in range(new_size + 1, self.size + 1) if s not in free_set]
        if held:
            raise RuntimeError(
                f"ReqToTokenPool.shrink: cannot shrink to {new_size}; "
                f"slot ids still held: {held[:8]}{'…' if len(held)>8 else ''}"
            )
        # Drop the shrunk slot ids from free_slots (keep 1..new_size).
        self.free_slots = [s for s in self.free_slots if s <= new_size]
        # Unmap the physical pages (pad row + new_size usable rows).
        self._va_arena.set_mapped_bytes((new_size + 1) * self._row_bytes)
        self.size = new_size
        self._alloc_size = new_size + 1
        return self.size

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        # Indices of reqs that already have a req_pool_idx and will reuse
        # their existing slot (e.g. chunked prefill continuing across chunks).
        reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
        # NOTE: this check is relaxed temporarily
        # https://github.com/sgl-project/sglang/pull/20476
        # if not any(r.is_dllm() for r in reqs):
        #     assert (
        #         sum(1 for i in reusing if reqs[i].inflight_middle_chunks > 0) <= 1
        #     ), "only one chunked request may reuse req_pool_idx in a batch"
        assert all(
            reqs[i].inflight_middle_chunks > 0 or reqs[i].kv_committed_len > 0
            for i in reusing
        ), "reusing request must be chunked or have committed KV"

        need_size = len(reqs) - len(reusing)
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        offset = 0
        for r in reqs:
            if r.req_pool_idx is None:
                r.req_pool_idx = select_index[offset]
                offset += 1
        return [r.req_pool_idx for r in reqs]

    def free(self, req: Req):
        assert req.req_pool_idx is not None, "request must have req_pool_idx"
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None

    def clear(self):
        self.free_slots = list(range(1, self._alloc_size))


def _arena_tokens_per_chunk(chunk_bytes: int, per_token_bytes: int) -> int:
    """Per-sequence SSM slots that fit in one VMM chunk.

    The `MultiTensorArena` packs WHOLE per-token states per chunk, so it
    requires `chunk_bytes` to be an exact multiple of `per_token_bytes`
    (`MultiTensorArena.__init__` raises on `chunk_bytes % per_token != 0`).
    This validates the same constraint at pool-config time and returns
    `chunk_bytes // per_token_bytes`.

    A bare `chunk_bytes // per_token_bytes` floors a non-dividing state (and
    yields 0 for a per-token state larger than one chunk), then sizes the pool
    with that wrong value; the mismatch only surfaces later as a
    `MultiTensorArena` ValueError mid-boot. Failing here, at config, with an
    actionable message is clearer.
    """
    if per_token_bytes <= 0:
        raise ValueError(
            f"per-token SSM state must be positive, got {per_token_bytes} bytes"
        )
    if chunk_bytes % per_token_bytes != 0:
        suggested = (
            (chunk_bytes + per_token_bytes - 1) // per_token_bytes
        ) * per_token_bytes
        raise ValueError(
            f"SGLANG_ARENA_CHUNK_BYTES ({chunk_bytes}) must be a multiple of "
            f"the per-token SSM state size ({per_token_bytes} bytes); the mamba "
            f"VMM arena packs whole per-token states per chunk and cannot split "
            f"one across chunks. Set SGLANG_ARENA_CHUNK_BYTES to a multiple of "
            f"{per_token_bytes} (e.g. {suggested})."
        )
    return chunk_bytes // per_token_bytes


class MambaPool:
    @dataclass(frozen=True, kw_only=True)
    class State:
        conv: List[torch.Tensor]
        temporal: torch.Tensor

        def at_layer_idx(self, layer: int):
            kwargs = {}
            # Use fields instead of vars to avoid torch.compile graph break
            for f in fields(self):
                name = f.name
                v = getattr(self, name)
                if name in ("conv", "intermediate_conv_window"):
                    kwargs[name] = [conv[layer] for conv in v]
                else:
                    kwargs[name] = v[layer]

            return type(self)(**kwargs)

        def mem_usage_bytes(self):
            return sum(
                get_tensor_size_bytes(getattr(self, f.name))
                for f in dataclasses.fields(self)
            )

        def bytes_per_slot(self) -> int:
            """Physical bytes per single slot, independent of how many slots
            the backing tensors have (works for both arena and non-arena)."""
            total = 0
            for f in dataclasses.fields(self):
                v = getattr(self, f.name)
                if isinstance(v, list):
                    for t in v:
                        total += int(np.prod(t.shape[1:])) * t.dtype.itemsize
                else:
                    total += int(np.prod(v.shape[1:])) * v.dtype.itemsize
            return total

    @dataclass(frozen=True, kw_only=True)
    class SpeculativeState(State):
        intermediate_ssm: torch.Tensor
        intermediate_conv_window: List[torch.Tensor]

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        conv_state_shape = cache_params.shape.conv
        temporal_state_shape = cache_params.shape.temporal
        conv_dtype = cache_params.dtype.conv
        ssm_dtype = cache_params.dtype.temporal
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        num_mamba_layers = len(mamba_layer_ids)

        # Dynamic-cap mode (Phase 7): conv_state + free_slots sized for
        # max_size; live admission cap starts at `size`. The actuator
        # calls set_capacity_slots(N) up to max_size to lift the cap
        # (the existing grow path pulls slot ids from `_capped_slots`).
        #
        # Semantic of `self.size`: the LIVE capacity cap, updated by
        # `set_capacity_slots` on BOTH grow and shrink (Phase-7 dynamic).
        # This DIFFERS from `BaseTokenToKVPoolAllocator.size` on the KV
        # side, which stays fixed at init (high-water) and tracks the live
        # cap separately in `_cap`/`_capped_pages` — mamba folds the cap
        # into `self.size` directly. Two distinct shrink paths act on this
        # pool: (1) the admission-cap path `set_capacity_slots(n)` lowers
        # `self.size` to `n`; (2) the cross-pool cap-barrier path
        # (`_MambaCapAllocator.mark_pages_capped`) does NOT touch
        # `self.size` — it moves slots into `_capped_slots`, so allocatable
        # capacity drops below `self.size`. Hence `live_size = self.size −
        # (_capped_slots ≤ self.size).count()` — only capped slots WITHIN
        # the live range subtract, since `_capped_slots` may also carry
        # boot-deferred IDs in `(size, max_size]` whose chunks are unmapped.
        # See the `live_size` property for the exact masked-sum form.
        if max_size is None:
            max_size = size
        if max_size < size:
            raise ValueError(f"max_size={max_size} must be >= size={size}")
        self.size = size            # live cap (set_capacity_slots-updated)
        self.max_size = max_size    # pre-allocated upper bound
        self.device = device
        # Symmetric to `BaseTokenToKVPoolAllocator._alloc_lock`: the
        # cross-pool actuator's worker thread mutates `free_slots` /
        # `_capped_slots` / `self.size` via `set_capacity_slots` and
        # `unmark_slots` while the scheduler thread is in `alloc` /
        # `free` / `migrate_slot`. Without this lock the read-then-rebind
        # sequence in either side can race the other (mask shape
        # mismatch → IndexError, or stale cat → same slot handed twice).
        self._alloc_lock = threading.Lock()
        # Per-slot tensors are pre-allocated at `max_size` so they can
        # be addressed past init without reallocation (data_ptr stable
        # for CUDA-graph safety).
        alloc_size = max_size

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            conv_state = [
                torch.zeros(
                    size=(num_mamba_layers, alloc_size + 1) + conv_shape,
                    dtype=conv_dtype,
                    device=device,
                )
                for conv_shape in conv_state_shape
            ]

            if _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    _init_npu_conv_state,
                )

                conv_state = _init_npu_conv_state(
                    conv_state[0], conv_state_shape, speculative_num_draft_tokens
                )

            if _is_cpu and _cpu_has_amx_support:
                from sglang.srt.layers.amx_utils import _init_amx_conv_state

                # CPU uses a different layout of conv_state for kernel optimization
                conv_state = _init_amx_conv_state(conv_state)

            # When SGLANG_MAMBA_PERLAYER=1, temporal_state is a List[Tensor]
            # of length num_mamba_layers, each shape
            # (size+1, *temporal_state_shape). Mirrors the per-layer KV pool
            # layout, makes the pool VMM-arena-friendly. Default off.
            #
            # When SGLANG_MAMBA_ARENA=1 (implies SGLANG_MAMBA_PERLAYER=1),
            # temporal_state's per-layer tensors come from a MultiTensorArena,
            # sharing the chunk-bitmap actuator with the KV pool so cross-pool
            # transfer can move physical bytes.
            # Unconditionally declare the arena attribute so callers
            # can always do a direct None check instead of
            # `getattr(..., None)`. Stays None unless the arena branch
            # below actually constructs the MultiTensorArena.
            self._mamba_temporal_arena = None
            shared_arena = os.environ.get("SGLANG_ARENA_SHARED") == "1"
            self._mamba_arena = (
                os.environ.get("SGLANG_MAMBA_ARENA") == "1"
                or shared_arena
            )
            self._mamba_perlayer = (
                self._mamba_arena
                or os.environ.get("SGLANG_MAMBA_PERLAYER") == "1"
            )
            logger.info(
                "MambaPool: temporal layout=%s, arena=%s, shared=%s, "
                "num_layers=%d, size=%d, temporal_shape=%s, conv_shapes=%s",
                "per-layer-list" if self._mamba_perlayer else "stacked",
                self._mamba_arena,
                shared_arena,
                num_mamba_layers, size, tuple(temporal_state_shape),
                [tuple(s) for s in conv_state_shape],
            )
            if self._mamba_arena:
                from sglang.srt.arena.multi_tensor_arena import MultiTensorArena
                # Compute chunk-aligned slot count.
                # Phase A5: ablation honors SGLANG_ARENA_CHUNK_BYTES.
                # T1 (paper §3.2.1): default to CUDA VMM's native 2 MiB page
                # granularity on H200. Chunk-grain (e.g., 64 MiB) is selectable
                # via SGLANG_ARENA_CHUNK_BYTES for legacy A/B comparison.
                chunk_bytes = int(os.environ.get(
                    "SGLANG_ARENA_CHUNK_BYTES", str(2 * 1024 * 1024)
                ))
                per_token_bytes = (
                    int(np.prod(temporal_state_shape))
                    * torch.tensor([], dtype=ssm_dtype).element_size()
                )
                tokens_per_chunk = _arena_tokens_per_chunk(
                    chunk_bytes, per_token_bytes
                )
                tot = size + 1
                tot_aligned = (
                    (tot + tokens_per_chunk - 1) // tokens_per_chunk
                ) * tokens_per_chunk

                shared_pool = None
                # max_tokens > init_tokens reserves VA past the initial
                # physical mapping so cross-pool transfer can actually grow
                # this arena: when peer releases a chunk, the freed handle
                # gets cuMemMap'd into [init_chunks, max_chunks) of THIS
                # arena's VA range. Without this headroom, max_chunks =
                # init_chunks and the actuator's grow() returns 0 (B3 4-cell
                # v1: 10 fires, all unmapped=0 granted=0 — pure overhead).
                # The headroom is VA-only — no physical handles are
                # allocated for [init, max), they only get mapped at
                # transfer time. So this does NOT cost any KV/mamba budget.
                #
                # T5 (paper §3.2.1): SGLANG_ARENA_MAMBA_HEADROOM_BYTES, if
                # set, takes precedence over SGLANG_ARENA_MAMBA_HEADROOM_CHUNKS.
                # Default 80 GiB ensures the actuator can actually pull a
                # peer-released ~25 GiB recurrent budget into a peer pool
                # whose chunks are 2 MiB (T1).
                mamba_headroom_bytes_env = os.environ.get(
                    "SGLANG_ARENA_MAMBA_HEADROOM_BYTES",
                )
                if shared_arena and mamba_headroom_bytes_env is not None:
                    mamba_growth_chunks = (
                        int(mamba_headroom_bytes_env) // chunk_bytes
                    )
                elif shared_arena:
                    # Default to 80 GiB headroom; legacy CHUNKS env still
                    # honored for explicit overrides like "=4".
                    legacy_chunks_env = os.environ.get(
                        "SGLANG_ARENA_MAMBA_HEADROOM_CHUNKS"
                    )
                    if legacy_chunks_env is not None:
                        mamba_growth_chunks = int(legacy_chunks_env)
                    else:
                        mamba_growth_chunks = (
                            (80 * 1024 * 1024 * 1024) // chunk_bytes
                        )
                else:
                    mamba_growth_chunks = 0
                mamba_max_tokens = (
                    tot_aligned + mamba_growth_chunks * tokens_per_chunk
                )
                if shared_arena:
                    from sglang.srt.arena.shared_pool import (
                        get_or_create_shared_handle_pool,
                    )
                    shared_pool = get_or_create_shared_handle_pool(
                        device_id=torch.cuda.current_device(),
                        chunk_bytes=chunk_bytes,
                    )

                # Static-min/soft split (paper §sec:design-l2-actuator). All
                # init_chunks worth of physical handles are cuMemMap'd at
                # boot — the pool boots at full baseline capacity, no
                # donation to a shared queue. The static_min defines the
                # FLOOR below which the cross-pool actuator never shrinks
                # ("the bytes the pool needs to admit any traffic"). When
                # shared_arena is enabled we set static_min
                # to 1 chunk per sub-pool, leaving (init - 1) chunks per
                # sub-pool transferable via drain protocol on fire. When
                # shared_arena is off, static_min = init = no shrink
                # possible (matches non-L2 baseline behavior identically).
                init_chunks = tot_aligned // tokens_per_chunk
                mamba_static_min_chunks = 1 if shared_arena else init_chunks
                mamba_static_min_tokens = mamba_static_min_chunks * tokens_per_chunk
                self._mamba_temporal_arena = MultiTensorArena(
                    device_id=torch.cuda.current_device(),
                    n_layers=num_mamba_layers,
                    n_kinds=1,
                    per_token_shape=tuple(temporal_state_shape),
                    dtype=ssm_dtype,
                    max_tokens=mamba_max_tokens,
                    init_tokens=tot_aligned,
                    static_min_tokens=mamba_static_min_tokens,
                    chunk_bytes=chunk_bytes,
                    external_handle_pool=shared_pool,
                )
                temporal_state = [
                    self._mamba_temporal_arena.tensor(i, 0)
                    for i in range(num_mamba_layers)
                ]
                # Match torch.zeros initial state for slot 0 (pad slot).
                for buf in temporal_state:
                    buf[:1].zero_()
                logger.info(
                    "MambaPool arena: tot=%d (aligned=%d), tokens_per_chunk=%d, "
                    "chunk_bytes=%d, per_token_bytes=%d, shared=%s, "
                    "subpool_offset=%d, n_subpools=%d",
                    tot, tot_aligned, tokens_per_chunk, chunk_bytes,
                    per_token_bytes, shared_arena,
                    self._mamba_temporal_arena._subpool_offset,
                    num_mamba_layers,
                )
            elif self._mamba_perlayer:
                temporal_state = [
                    torch.zeros(
                        size=(alloc_size + 1,) + temporal_state_shape,
                        dtype=ssm_dtype,
                        device=device,
                    )
                    for _ in range(num_mamba_layers)
                ]
            else:
                temporal_state = torch.zeros(
                    size=(num_mamba_layers, alloc_size + 1) + temporal_state_shape,
                    dtype=ssm_dtype,
                    device=device,
                )
            if speculative_num_draft_tokens is not None:
                if _is_npu:
                    temporal_state = temporal_state.transpose(-1, -2)
                    temporal_state_shape = (
                        *temporal_state_shape[:-2],
                        temporal_state_shape[-1],
                        temporal_state_shape[-2],
                    )
                # Cache intermediate SSM states per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, HV, K, V]
                intermediate_ssm_state_cache = torch.zeros(
                    size=(
                        num_mamba_layers,
                        spec_state_size + 1,
                        speculative_num_draft_tokens,
                        temporal_state_shape[0],
                        temporal_state_shape[1],
                        temporal_state_shape[2],
                    ),
                    dtype=ssm_dtype,
                    device="cuda",
                )
                # Cache intermediate conv windows (last K-1 inputs) per draft token during target verify
                # Shape: [num_layers, size + 1, speculative_num_draft_tokens, dim, K-1]
                intermediate_conv_window_cache = [
                    torch.zeros(
                        size=(
                            num_mamba_layers,
                            spec_state_size + 1,
                            speculative_num_draft_tokens,
                            conv_shape[0],
                            conv_shape[1],
                        ),
                        dtype=conv_dtype,
                        device="cuda",
                    )
                    for conv_shape in conv_state_shape
                ]
                self.mamba_cache = self.SpeculativeState(
                    conv=conv_state,
                    temporal=temporal_state,
                    intermediate_ssm=intermediate_ssm_state_cache,
                    intermediate_conv_window=intermediate_conv_window_cache,
                )
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                    f"intermediate_ssm_state_cache size: {get_tensor_size_bytes(intermediate_ssm_state_cache) / GB:.2f}GB "
                    f"intermediate_conv_window_cache size: {get_tensor_size_bytes(intermediate_conv_window_cache) / GB:.2f}GB "
                )
            else:
                self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
                logger.info(
                    f"Mamba Cache is allocated. "
                    f"max_mamba_cache_size: {size}, "
                    f"conv_state size: {get_tensor_size_bytes(conv_state) / GB:.2f}GB, "
                    f"ssm_state size: {get_tensor_size_bytes(temporal_state) / GB:.2f}GB "
                )
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # In dynamic-cap mode (max_size > size), free_slots covers
            # ONLY the live cap; the deferred ids [size+1, max_size+1)
            # go into _capped_slots so the existing set_capacity_slots
            # grow path can pull them in when the actuator lifts the cap.
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            if self.size < self.max_size:
                self._capped_slots = torch.arange(
                    self.size + 1, self.max_size + 1,
                    dtype=torch.int64, device=self.device,
                )
            else:
                self._capped_slots = torch.empty(
                    0, dtype=torch.int64, device=self.device,
                )
            # Paper §sec:design-l2: at boot, pool maps init_chunks (= live cap
            # slots usable). Allocator hands out the live range — engine
            # behaves identically to non-L2 baseline. The actuator only
            # shrinks via drain protocol (cap allocator → wait for in-flight
            # tail-slot reqs to drain → cuMemUnmap), which dynamically
            # re-caps the allocator at fire time. No boot-time cap needed.
            self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
            self.num_mamba_layers = num_mamba_layers

    def get_speculative_mamba2_params_all_layers(self) -> SpeculativeState:
        assert isinstance(self.mamba_cache, self.SpeculativeState)
        return self.mamba_cache

    def mamba2_layer_cache(self, layer_id: int):
        return self.mamba_cache.at_layer_idx(layer_id)

    def available_size(self):
        if hasattr(self, "_allocator") and self._allocator is not None:
            return self._allocator.available_size()
        return len(self.free_slots)

    def clear_slots(self, indices: torch.Tensor):
        """Zero out mamba state at the given pool indices. Must run on the
        forward stream. Driven by `req.mamba_needs_clear`: a freshly allocated
        slot is handed out dirty and zeroed here, not at alloc time, so alloc
        launches no extra kernel and the clear overlaps the forward pass.

        Expands a scalar GPU zero into each slot (no CPU-GPU sync). The scalar
        is allocated ONCE per dtype and broadcast-expanded per tensor; a fresh
        `torch.zeros(1)` per layer would launch one extra tiny allocation
        kernel per layer for an identical result. The arena backs `temporal`
        as a per-layer list (slot on dim 0); the non-arena pool stacks it into
        one tensor (slot on dim 1), so both layouts are handled.
        """
        need_size = len(indices)
        conv = self.mamba_cache.conv
        if conv:
            conv_zero = torch.zeros(1, dtype=conv[0].dtype, device=conv[0].device)
            for t in conv:
                t[:, indices] = conv_zero.expand(
                    t.shape[0], need_size, *t.shape[2:]
                )
        if isinstance(self.mamba_cache.temporal, list):
            temporal = self.mamba_cache.temporal
            temporal_zero = torch.zeros(
                1, dtype=temporal[0].dtype, device=temporal[0].device
            )
            for t in temporal:
                t[indices] = temporal_zero.expand(need_size, *t.shape[1:])
        else:
            t = self.mamba_cache.temporal
            z = torch.zeros(1, dtype=t.dtype, device=t.device).expand(
                t.shape[0], need_size, *t.shape[2:]
            )
            t[:, indices] = z

    @property
    def _no_cross_fire(self) -> bool:
        """True when `_capped_slots` is empty AND the live cap still equals the
        pre-allocated upper bound, so every live slot id lies in
        `[1, self.size] == [1, self.max_size]`: no freed id can exceed the cap
        and no id can be a capped (unmapped) slot.

        `free` uses this to take the plain free-list path (matching the
        non-budgeter baseline), skipping the `torch.isin` membership test and
        the `free_index > self.size` mask whose `.any()` forces a
        device-to-host sync on every free.

        SAFETY rests on one invariant: `_capped_slots` holds the id of EVERY
        slot whose VMM chunk is unmapped. Every path that unmaps a chunk
        populates `_capped_slots` or lowers `self.size` under `_alloc_lock`
        before the unmap (`migrate_slot`, `set_capacity_slots` SHRINK, the
        above-cap `free` branch, `_MambaCapAllocator.mark_pages_capped`), so
        this predicate reads False the instant any live slot could be unmapped;
        `unmark_slots` / `set_capacity_slots` GROW only restore ids the actuator
        just re-mapped. The ONE known violator is `clear()`: it rebuilds
        `_capped_slots` from `self.size`, dropping below-cap actuator marks
        (flush-boundary crash), so the fast path is only fully sound once
        `clear()` preserves those marks. The crash manifests identically on the slow
        path (post-`clear()` `_capped_slots` is empty, so its filter is a no-op
        too), so it is orthogonal to this fast path.
        """
        return self._capped_slots.numel() == 0 and self.size == self.max_size

    def migrate_slot(self, src: int, dst: int) -> bool:
        """T4 (paper §3.2.3): atomic per-slot migration. Copies the
        recurrent-state contents of slot `src` to slot `dst`, then
        updates allocator side state: `src` joins _capped_slots
        (held out, about to be unmapped), `dst` removed from free_slots
        (now live with src's data).

        Caller's responsibility: update any in-flight request whose
        `mamba_pool_idx == src` to `dst`. The slot's tensor bytes are
        moved here; the *owning-request pointer* is the caller's job
        (it has the scheduler reference; this pool does not).

        Caller must also wrap the call with `torch.cuda.synchronize`
        before (so all in-flight kernels using src finish reading) and
        after (so the copy is visible before the next kernel launch).
        Strictly between scheduler steps this is automatic.

        Returns True if migration succeeded; False if `dst` is not
        currently free or `src` and `dst` are equal.
        """
        if src == dst:
            return False
        with self._alloc_lock:
            # Verify dst is free.
            is_dst_free = bool((self.free_slots == dst).any().item())
            if not is_dst_free:
                return False

            # Copy state across all conv tensors and the temporal tensor(s).
            for t in self.mamba_cache.conv:
                t[:, dst, ...].copy_(t[:, src, ...])
            if isinstance(self.mamba_cache.temporal, list):
                for t in self.mamba_cache.temporal:
                    t[dst, ...].copy_(t[src, ...])
            else:
                t = self.mamba_cache.temporal
                t[:, dst, ...].copy_(t[:, src, ...])
            # Speculative-decoding: SpeculativeState adds intermediate_ssm
            # (Tensor) + intermediate_conv_window (List[Tensor]). Without
            # this branch, migration silently strips speculative state and
            # the next decode reads stale data.
            if isinstance(self.mamba_cache, MambaPool.SpeculativeState):
                t = self.mamba_cache.intermediate_ssm
                t[:, dst, ...].copy_(t[:, src, ...])
                for t in self.mamba_cache.intermediate_conv_window:
                    t[:, dst, ...].copy_(t[:, src, ...])

            # Allocator-side state: dst removed from free_slots; src into
            # _capped_slots (NOT free_slots — its chunk is about to be
            # unmapped by the actuator).
            self.free_slots = self.free_slots[self.free_slots != dst]
            existing = self._capped_slots
            src_t = torch.tensor([src], dtype=self.free_slots.dtype, device=self.device)
            if existing.numel() == 0:
                self._capped_slots = src_t
            elif not bool(torch.isin(src_t, existing).any().item()):
                # Dedupe against existing _capped_slots: a slot can be
                # recycled and migrated again before its prior cap is
                # released; without this, `_capped_slots` double-counts.
                self._capped_slots = torch.cat([existing, src_t])
            self._assert_capped_slots_invariant()
            return True

    def clear(self):
        # flush_cache resets the pool to "every MAPPED slot is free", but a slot
        # whose VMM chunk is currently unmapped must STAY capped — otherwise the
        # next alloc hands out unmapped VA. `_capped_slots` already
        # holds exactly the unmapped set (every mutator maintains the invariant;
        # `unmark_slots` drops an id only when its chunk is re-mapped), of two
        # kinds: below-cap actuator marks (id ≤ size, recorded by
        # `_MambaCapAllocator.mark_pages_capped` without lowering self.size) and
        # the boot-deferred / shrunk tail (size, max_size]. So we PRESERVE
        # `_capped_slots` as-is and rebuild `free_slots` as the live range minus
        # the capped ids. Reconstructing `_capped_slots` from self.size alone
        # would drop the below-cap marks and re-enter unmapped slots.
        # Mirrors KV's `CappedFreeList.reset`, which keeps its `marks` set.
        with self._alloc_lock:
            live = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
            capped = self._capped_slots
            if capped.numel() > 0:
                self.free_slots = live[~torch.isin(live, capped)]
            else:
                self.free_slots = live
            self._assert_capped_slots_invariant()

    @property
    def live_size(self) -> int:
        if hasattr(self, "_allocator") and self._allocator is not None:
            return self._allocator.live_size
        capped = self._capped_slots
        if capped.numel() == 0:
            return self.size
        n_below_cap = int((capped <= self.size).sum().item())
        return self.size - n_below_cap

    def unmark_slots(self, ids: torch.Tensor) -> int:
        """Restore specific slot IDs from ``_capped_slots`` back into
        ``free_slots``. Mirrors KV's
        ``BaseTokenToKVPoolAllocator.unmark_pages_capped(ids)``.

        Used by the cross-pool actuator with the IDs returned by
        ``chunk_arena.grow`` — i.e. the slot positions whose chunks
        were just freshly mapped. ``self.size`` extends to cover any
        restored ID that lies above the current cap.

        Contract:
          - Restore = ``_capped_slots ∩ ids``. IDs in ``ids`` not in
            ``_capped_slots`` are silently ignored (already free or
            never owned), so callers can pass raw arena output.
          - Returns the actual restore count.
        """
        if ids.numel() == 0:
            return 0
        with self._alloc_lock:
            capped = self._capped_slots
            if capped.numel() == 0:
                return 0
            # Restore = capped ∩ ids. Drop those from _capped_slots,
            # append to free_slots.
            in_restore = torch.isin(capped, ids)
            restore = capped[in_restore]
            if restore.numel() == 0:
                return 0
            # `self.size` is the upper bound of the live slot range.
            # `_capped_slots` may carry IDs both ≤ size (marked off by
            # `_MambaCapAllocator.mark_pages_capped`) and > size (boot-
            # deferred slots from `[init+1..max_size]` pre-allocation).
            # Restoring IDs ≤ size does not move the boundary; restoring
            # IDs > size extends it to cover the new maximum.
            self._capped_slots = capped[~in_restore]
            self.free_slots = torch.cat([self.free_slots, restore])
            self.size = max(self.size, int(restore.max().item()))
            return int(restore.numel())

    def live_capacity_tokens(self) -> int:
        return self.live_size

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        for i in range(len(self.mamba_cache.conv)):
            self.mamba_cache.conv[i][:, dst_indices] = self.mamba_cache.conv[i][
                :, src_indices
            ]
        if isinstance(self.mamba_cache.temporal, list):
            for t in self.mamba_cache.temporal:
                t[dst_indices] = t[src_indices]
        else:
            self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[
                :, src_indices
            ]

    def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:
        dst_index = self._allocator.alloc(1)
        if dst_index is None:
            return None
        self.copy_from(src_index, dst_index)
        return dst_index

    def get_cpu_copy(self, indices):
        current_platform.synchronize()
        conv_cpu = [
            conv[:, indices].to("cpu", non_blocking=True)
            for conv in self.mamba_cache.conv
        ]
        if isinstance(self.mamba_cache.temporal, list):
            temporal_cpu = [
                t[indices].to("cpu", non_blocking=True)
                for t in self.mamba_cache.temporal
            ]
        else:
            temporal_cpu = self.mamba_cache.temporal[:, indices].to(
                "cpu", non_blocking=True
            )
        current_platform.synchronize()
        return conv_cpu, temporal_cpu

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        conv_cpu, temporal_cpu = mamba_cache_cpu
        current_platform.synchronize()
        for i, conv in enumerate(self.mamba_cache.conv):
            conv[:, indices] = conv_cpu[i].to(conv.device, non_blocking=True)
        if isinstance(self.mamba_cache.temporal, list):
            for i, t in enumerate(self.mamba_cache.temporal):
                t[indices] = temporal_cpu[i].to(t.device, non_blocking=True)
        else:
            self.mamba_cache.temporal[:, indices] = temporal_cpu.to(
                self.mamba_cache.temporal.device, non_blocking=True
            )
        current_platform.synchronize()

    def get_contiguous_buf_infos(self):
        """
        Get buffer info for RDMA registration.
        Only returns conv and temporal state buffers, excluding intermediate buffers
        used for speculative decoding (intermediate_ssm, intermediate_conv_window).

        When temporal is a per-layer-split List[Tensor] (len ==
        num_mamba_layers, entries don't carry a layer axis), the entries
        are treated directly as per-layer buffers (no extra layer-indexing).
        """
        # Per-logical-state list of "layer-indexable views"; each entry is
        # something where `entry[layer_id]` returns the per-layer buffer.
        # For stacked tensors and conv-shape lists this is the entry itself;
        # for per-layer-split lists the wrapping list IS already layer-indexed.
        state_views = []
        for fname in vars(self.mamba_cache):
            if fname in ("intermediate_ssm", "intermediate_conv_window"):
                continue
            value = getattr(self.mamba_cache, fname)
            if isinstance(value, list):
                if (
                    len(value) == self.num_mamba_layers
                    and value[0].shape[0] != self.num_mamba_layers
                ):
                    # Per-layer split: the list itself is the layer-indexed view.
                    state_views.append(value)
                else:
                    # List of per-conv-shape stacked tensors.
                    for v in value:
                        state_views.append(v)
            else:
                state_views.append(value)

        data_ptrs, data_lens, item_lens = [], [], []
        for view in state_views:
            data_ptrs += [
                view[i].data_ptr() for i in range(self.num_mamba_layers)
            ]
            data_lens += [view[i].nbytes for i in range(self.num_mamba_layers)]
            item_lens += [
                view[i][0].nbytes for i in range(self.num_mamba_layers)
            ]
        return data_ptrs, data_lens, item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each (state-tensor, layer).

        For mamba state, the layout is:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        Each logical state-tensor contributes num_mamba_layers entries.

        When SGLANG_MAMBA_PERLAYER=1 and temporal is a List[Tensor] of
        length num_mamba_layers (each entry shape
        (size+1, sliceable_dim, ...)), it counts as ONE logical
        state-tensor (sliceable_dim = entry[0].shape[1], repeated
        num_mamba_layers times).
        """
        sliceable_dims = []
        for fname in vars(self.mamba_cache):
            value = getattr(self.mamba_cache, fname)
            if isinstance(value, list):
                # Distinguish "list of conv-shape tensors, each layer-stacked"
                # from "per-layer-split temporal (one tensor per layer)".
                if (
                    len(value) == self.num_mamba_layers
                    and value[0].shape[0] != self.num_mamba_layers
                ):
                    sliceable_dims.append(value[0].shape[1])
                else:
                    for v in value:
                        sliceable_dims.append(v.shape[2])
            else:
                # Stacked: [num_layers, size+1, sliceable_dim, ...]
                sliceable_dims.append(value.shape[2])

        dim_per_tensor = []
        for d in sliceable_dims:
            dim_per_tensor += [d] * self.num_mamba_layers
        return dim_per_tensor


# Mamba active-slot slots a single hybrid request needs, by prefix-cache mode.
# The extra slots over 1 reserve ping-pong / lazy-copy buffers so a cached
# prefix's recurrent state can be reused without clobbering the live slot.
# Single source of truth: both alloc_req_slots (supply-side eviction target) and
# HybridReqToTokenPool.available_size (admission gate) price a request off these.
MAMBA_STATE_PER_REQ_PREFIX_CACHE = 3
MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY = 2
MAMBA_STATE_PER_REQ_NO_CACHE = 1


class HybridReqToTokenPool(ReqToTokenPool):
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        *,
        size: int,
        mamba_size: int,
        mamba_spec_state_size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        enable_mamba_extra_buffer: bool,
        enable_mamba_extra_buffer_lazy: bool = False,
        speculative_num_draft_tokens: int = None,
        enable_overlap_schedule: bool = True,
        start_layer: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        super().__init__(
            size=size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=enable_memory_saver,
            max_size=max_size,
        )

        self.mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1
        self.enable_mamba_extra_buffer = enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = enable_mamba_extra_buffer_lazy
        self.enable_memory_saver = enable_memory_saver
        # On-demand mamba (active-slot) grow: optional synchronous k2m grow
        # callback the BudgetAgent installs. The active-slot alloc below calls it
        # when mamba_pool.alloc fails, so a mamba-phase burst grows mamba from
        # idle KV instead of crashing "Not enough space for mamba cache" — the
        # symmetric counterpart of the KV allocator's _kv_grow_hook. This lets
        # the budgeter's on-demand mamba grow floor stay at the current working
        # set instead of statically reserving max_running for a future burst.
        # None on stock sglang / Budgeter off.
        self._mamba_active_grow_hook = None
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        # Mamba growable headroom for cross-pool fires. The mamba
        # pool must be in dynamic-cap mode (max_size > size) for the
        # cross-pool actuator's k2m grants to `unmark_slots` chunks into
        # the live cap; with max_size == size, `_capped_slots` is empty
        # and every grant is a no-op (the pool stays pinned at boot size,
        # cache_hit never recovers). The old gate keyed this on the REQ
        # pool's dynamic-cap mode (`self.max_size > self.size`), which is
        # the wrong condition — an arena-backed (cross-fire) mamba pool
        # needs the headroom regardless. We therefore also enable it when
        # the mamba pool is arena-backed. The cap is a bounded multiple of
        # the boot mamba size (NOT the full arena VA headroom): conv_state
        # is physically allocated at max_size (only the SSM/temporal state
        # is arena-backed VA-on-demand), so an unbounded max_size would
        # balloon conv. SGLANG_XPOOL_MAMBA_MAX_FACTOR tunes it (default 4×,
        # enough to reach the workload-optimal mamba size on the starve
        # bench where S* ≈ 4× the 64-slot boot).
        _mamba_arena = (
            os.environ.get("SGLANG_ARENA_SHARED") == "1"
            or os.environ.get("SGLANG_MAMBA_ARENA") == "1"
        )
        if self.max_size > self.size:
            mamba_max_size = self.max_size * 3
        elif _mamba_arena:
            _factor = max(2, int(
                os.environ.get("SGLANG_XPOOL_MAMBA_MAX_FACTOR", "4")
            ))
            mamba_max_size = mamba_size * _factor
        else:
            mamba_max_size = None
        self._init_mamba_pool(
            mamba_size=mamba_size,
            mamba_spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_mamba_extra_buffer=enable_mamba_extra_buffer,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            mamba_max_size=mamba_max_size,
        )

    def _init_mamba_pool(
        self,
        mamba_size: int,
        mamba_spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_mamba_extra_buffer: bool,
        speculative_num_draft_tokens: int = None,
        mamba_max_size: Optional[int] = None,
    ):
        self.mamba_pool = MambaPool(
            size=mamba_size,
            spec_state_size=mamba_spec_state_size,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            device=device,
            enable_memory_saver=self.enable_memory_saver,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            max_size=mamba_max_size,
        )
        self.mamba_allocator = MambaSlotAllocator(
            size=mamba_size,
            device=device,
            max_size=mamba_max_size,
        )
        self.mamba_pool._allocator = self.mamba_allocator
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}

        self.device = device
        # Size by the full req_to_token row count (max_size + 1, including the
        # pad row 0). The tensor aliases the whole VA range, so its shape[0] is
        # stable across grow/shrink: data_ptr() stays put, required for
        # CUDA-graph capture safety. Cost: ~4 bytes per row, negligible vs the
        # multi-GiB req_to_token table that needs the full VA-arena treatment.
        req_pool_size = self.req_to_token.shape[0]
        self.req_index_to_mamba_index_mapping: torch.Tensor = torch.zeros(
            req_pool_size, dtype=torch.int32, device=self.device
        )
        if enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping: torch.Tensor = (
                torch.zeros(
                    (req_pool_size, self.mamba_ping_pong_track_buffer_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            )

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def mamba_slots_per_req(self, supports_mamba: bool = True) -> int:
        """Mamba active slots one fresh request consumes. Single source of truth
        shared by ``alloc_req_slots`` (how many mamba slots to free before an
        alloc) and ``available_size`` (how many requests the mamba pool can back)
        so the two can never drift."""
        if not supports_mamba:
            return MAMBA_STATE_PER_REQ_NO_CACHE
        return (
            MAMBA_STATE_PER_REQ_PREFIX_CACHE_LAZY
            if self.enable_mamba_extra_buffer_lazy
            else MAMBA_STATE_PER_REQ_PREFIX_CACHE
        )

    def mamba_admittable_reqs(
        self, mamba_evictable: int, supports_mamba: bool = True
    ) -> int:
        """How many fresh requests the mamba pool can back right now =
        (free active slots + evictable cached snapshots) / slots-per-req.

        A hybrid request needs a mamba active slot in addition to a req slot, but
        the base ``available_size`` counts only req slots — so admitting on req
        slots alone over-commits mamba and turns a should-be DEFER into an
        ``alloc_req_slots`` crash. The admission gate (``get_num_allocatable_reqs``)
        bounds the batch by this so it falls to 0 when mamba is exhausted and the
        request stays queued (the design's defer action). ``mamba_evictable`` is
        supplied by the caller from the tree cache (the pool does not own it),
        mirroring the KV available+evictable gate in ``PrefillAdder``; counting
        evictable is what keeps a lightly-loaded pool from throttling on the free
        slots alone."""
        return (
            self.mamba_allocator.available_size() + mamba_evictable
        ) // self.mamba_slots_per_req(supports_mamba)

    # For chunk prefill req, we do not need to allocate mamba cache,
    # We could use allocated mamba cache instead.
    def alloc(self, reqs: List["Req"]) -> Optional[List[int]]:
        fresh_reqs = [r for r in reqs if r.req_pool_idx is None]
        select_index = super().alloc(reqs)
        if select_index is None:
            return None

        # Reqs that THIS call gave a fresh mamba slot / ping-pong buffer to (not
        # reused from a radix-cache COW or a prior chunk). Only these are
        # cleared and freed on a rollback.
        fresh_mamba_reqs: list["Req"] = []
        fresh_track_reqs: list["Req"] = []

        mamba_indices: list[torch.Tensor] = []
        mamba_ping_pong_track_buffers: list[torch.Tensor] = []
        for req in reqs:
            if req.mamba_pool_idx is not None:  # for radix cache / continuing chunked
                pass
            else:
                mid = self._alloc_active_mamba_slot()
                if mid is None:
                    self._rollback_active_alloc(
                        fresh_reqs, fresh_mamba_reqs, fresh_track_reqs
                    )
                    return None
                req.mamba_pool_idx = mid[0]
                # Fresh slot is handed out dirty; the forward stream zeroes it
                # via clear_slots, driven by this flag (deferred-clear model).
                req.mamba_needs_clear = True
                fresh_mamba_reqs.append(req)
            mamba_indices.append(req.mamba_pool_idx)
            if self.enable_mamba_extra_buffer:
                if req.mamba_ping_pong_track_buffer is None:
                    buf = self.mamba_allocator.alloc(self.mamba_ping_pong_track_buffer_size)
                    if buf is None:
                        self._rollback_active_alloc(
                            fresh_reqs, fresh_mamba_reqs, fresh_track_reqs
                        )
                        return None
                    req.mamba_ping_pong_track_buffer = buf
                    req.mamba_next_track_idx = 0
                    fresh_track_reqs.append(req)
                mamba_ping_pong_track_buffers.append(req.mamba_ping_pong_track_buffer)
        assert len(select_index) == len(
            mamba_indices
        ), "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio or --max-mamba-cache-size."
        if self.enable_mamba_extra_buffer:
            assert len(select_index) == len(
                mamba_ping_pong_track_buffers
            ), "Not enough space for mamba ping pong idx, try to increase --mamba-full-memory-ratio."
        mamba_index_tensor = torch.stack(mamba_indices).to(dtype=torch.int32)
        self.req_index_to_mamba_index_mapping[select_index] = mamba_index_tensor
        if self.enable_mamba_extra_buffer:
            ping_pong_tensor = torch.stack(mamba_ping_pong_track_buffers)
            self.req_index_to_mamba_ping_pong_track_buffer_mapping[select_index] = (
                ping_pong_tensor
            )
        return select_index

    def _alloc_active_mamba_slot(self) -> Optional[torch.Tensor]:
        """Allocate one active mamba slot, or None. On a full pool, fire the
        on-demand k2m grow hook (idle KV -> mamba) and retry once. Lets the
        on-demand mamba grow floor stay at the current working set instead of
        statically reserving max_running. Never asserts; the caller rolls back
        and returns None so the scheduler back-pressures."""
        mid = self.mamba_allocator.alloc(1)
        if mid is None and self._mamba_active_grow_hook is not None:
            if self._mamba_active_grow_hook(1):
                mid = self.mamba_allocator.alloc(1)
        return mid

    def _rollback_active_alloc(
        self,
        fresh_reqs: List["Req"],
        fresh_mamba_reqs: List["Req"],
        fresh_track_reqs: List["Req"],
    ) -> None:
        """Undo a partially-completed `alloc` batch, leaving the pool and every
        req exactly as before the call so the caller can defer cleanly. Frees
        only what THIS call acquired: the fresh mamba slots, the fresh ping-pong
        buffers, and the fresh req_pool slots. Reused (radix-cache COW / chunked)
        slots are left untouched."""
        for req in fresh_mamba_reqs:
            self.mamba_allocator.free(req.mamba_pool_idx.unsqueeze(0))
            req.mamba_pool_idx = None
            # Clear the deferred-clear flag set alongside the fresh slot, so a
            # rolled-back req never carries `mamba_needs_clear=True` with
            # `mamba_pool_idx=None` (mirrors Req.reset). The forward-stream
            # collector would otherwise do `None.unsqueeze(0)`.
            req.mamba_needs_clear = False
        for req in fresh_track_reqs:
            self.mamba_allocator.free(req.mamba_ping_pong_track_buffer)
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None
        for req in fresh_reqs:
            self.free_slots.append(req.req_pool_idx)
            req.req_pool_idx = None

    def get_mamba_indices(self, req_indices: torch.Tensor) -> torch.Tensor:
        return self.req_index_to_mamba_index_mapping[req_indices]

    def mamba2_layer_cache(self, layer_id: int):
        assert layer_id in self.mamba_map
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.mamba_pool.mamba2_layer_cache(self.mamba_map[layer_id])

    def get_speculative_mamba2_params_all_layers(self) -> MambaPool.SpeculativeState:
        return self.mamba_pool.get_speculative_mamba2_params_all_layers()

    def get_state_buf_infos(self):
        return self.mamba_pool.get_contiguous_buf_infos()

    def get_state_dim_per_tensor(self):
        return self.mamba_pool.get_state_dim_per_tensor()

    def get_mamba_ping_pong_other_idx(self, mamba_next_track_idx: int) -> int:
        if self.mamba_ping_pong_track_buffer_size == 2:
            return 1 - mamba_next_track_idx
        else:
            return mamba_next_track_idx

    def get_mamba_ping_pong_keep_idx(self, req: Req) -> int:
        """Return the ping-pong index holding the most recent tracked state.

        In lazy mode the valid state stays at next_track_idx (no eager swap).
        In normal mode it is at the "other" index (swapped after each track).
        """
        if self.enable_mamba_extra_buffer_lazy:
            return req.mamba_next_track_idx
        return self.get_mamba_ping_pong_other_idx(req.mamba_next_track_idx)

    def _alloc_ping_pong_buffer(self, req: Req):
        """Allocate the ping-pong track buffer for a new request.

        Lazy mode allocates 1 slot with the second set to -1 (allocated
        on demand at track boundaries). Normal mode allocates all slots upfront.
        """
        n = (
            1
            if self.enable_mamba_extra_buffer_lazy
            else self.mamba_ping_pong_track_buffer_size
        )
        slots = self.mamba_allocator.alloc(n)
        assert slots is not None, (
            "Not enough space for mamba ping pong idx, "
            "try to increase --mamba-full-memory-ratio."
        )
        buf = torch.full(
            (self.mamba_ping_pong_track_buffer_size,),
            -1,
            dtype=slots.dtype,
            device=slots.device,
        )
        buf[:n] = slots
        req.mamba_ping_pong_track_buffer = buf
        req.mamba_next_track_idx = 0

    def set_mamba_ping_pong_slot(self, req: Req, idx: int, value):
        """Update a ping-pong slot value and sync the device-side mapping.

        The req holds the authoritative buffer; this keeps the
        req_index_to_mamba_ping_pong_track_buffer_mapping in sync so that
        set_mamba_track_indices_from_reqs reads correct slot indices.
        """
        req.mamba_ping_pong_track_buffer[idx] = value
        self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx] = (
            req.mamba_ping_pong_track_buffer
        )

    def donate_mamba_ping_pong_slot(
        self, req: Req, new_slot: torch.Tensor
    ) -> torch.Tensor:
        """Donate the tracked-state ping-pong slot to the radix cache.

        Returns the old slot index (shape [1]) for cache insertion and
        replaces it with new_slot so the request can continue tracking.
        In lazy mode the valid state is at next_track_idx; in normal mode
        it is at the "other" index.
        """
        donate_idx = self.get_mamba_ping_pong_keep_idx(req)
        mamba_value_donated = (
            req.mamba_ping_pong_track_buffer[donate_idx].unsqueeze(-1).clone()
        )
        assert mamba_value_donated.item() != -1, (
            f"Donated mamba slot is -1: donate_idx={donate_idx}, "
            f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
            f"next_track_idx={req.mamba_next_track_idx}, "
            f"rid={req.rid}"
        )
        self.set_mamba_ping_pong_slot(req, donate_idx, new_slot[0])
        return mamba_value_donated

    def free_mamba_cache(
        self, req: Req, mamba_ping_pong_track_buffer_to_keep: Optional[int] = None
    ):
        mamba_index = req.mamba_pool_idx
        assert mamba_index is not None, "double free? mamba_index is None"
        self.mamba_allocator.free(mamba_index.unsqueeze(0))
        req.mamba_pool_idx = None

        if self.enable_mamba_extra_buffer:
            mamba_ping_pong_track_buffer_to_free = (
                self.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx]
            )
            if mamba_ping_pong_track_buffer_to_keep is not None:
                assert mamba_ping_pong_track_buffer_to_keep in [
                    0,
                    1,
                ], f"mamba_ping_pong_track_buffer_to_keep must be 0 or 1, {mamba_ping_pong_track_buffer_to_keep=}"
                # Avoid Python-list advanced indexing on a device tensor.
                # The ping-pong buffer size is either 2 (normal) or 1 (spec decode).
                if self.mamba_ping_pong_track_buffer_size == 2:
                    idx_to_free = 1 - mamba_ping_pong_track_buffer_to_keep
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[
                            idx_to_free : idx_to_free + 1
                        ]
                    )
                else:
                    assert self.mamba_ping_pong_track_buffer_size == 1, (
                        f"Unexpected mamba_ping_pong_track_buffer_size="
                        f"{self.mamba_ping_pong_track_buffer_size}"
                    )
                    assert mamba_ping_pong_track_buffer_to_keep == 0, (
                        "mamba_ping_pong_track_buffer_to_keep must be 0 when "
                        "mamba_ping_pong_track_buffer_size is 1"
                    )
                    # Keep the only slot, so free nothing.
                    mamba_ping_pong_track_buffer_to_free = (
                        mamba_ping_pong_track_buffer_to_free[0:0]
                    )
            if self.enable_mamba_extra_buffer_lazy:
                mamba_ping_pong_track_buffer_to_free = (
                    mamba_ping_pong_track_buffer_to_free[
                        mamba_ping_pong_track_buffer_to_free != -1
                    ]
                )
            self.mamba_allocator.free(mamba_ping_pong_track_buffer_to_free)
            # Match the req.mamba_pool_idx=None clear above so the next
            # alloc() doesn't see a stale ping-pong reference on the req
            # and skip allocation (which would silently reuse a freed
            # tensor on the req side while the new pool slot leaks).
            req.mamba_ping_pong_track_buffer = None
            req.mamba_next_track_idx = None

    def clear(self):
        logger.info("Reset HybridReqToTokenPool")
        super().clear()
        self.mamba_allocator.clear()
        self.req_index_to_mamba_index_mapping.zero_()
        if self.enable_mamba_extra_buffer:
            self.req_index_to_mamba_ping_pong_track_buffer_mapping.zero_()


@dataclass
class KVWriteLoc:
    """Write target(s) for ``KVCache.set_kv_buffer``.

    ``loc`` is the full-pool write location; ``swa_loc`` is the pre-translated
    full->SWA location for hybrid SWA pools (``None`` otherwise). Bundling them
    lets a backend issue one ``set_kv_buffer`` call regardless of pool type.
    """

    loc: torch.Tensor
    swa_loc: Optional[torch.Tensor] = None


def unwrap_write_loc(loc_info):
    """Return ``(loc, swa_loc)`` from a ``KVWriteLoc`` or a bare loc tensor."""
    if isinstance(loc_info, KVWriteLoc):
        return loc_info.loc, loc_info.swa_loc
    return loc_info, None


class KVCache(abc.ABC):
    # Optional ChunkArena backing for cross-pool growth. Declared at
    # the base so EVERY KVCache subclass exposes it and callers can do a
    # direct `pool._kv_arena is None` check instead of `getattr(.., None)`.
    # MHATokenToKVPool overrides it with a live MultiTensorArena when
    # SGLANG_KV_ARENA=1; non-arena backends (MLA, torch.zeros MHA) leave it
    # None. HybridLinearKVPool forwards it to its inner full-attention pool.
    _kv_arena = None

    def can_move_kv_cache(self) -> bool:
        """Whether `move_kv_cache` can actually relocate a slot's bytes on this
        pool. Declared at the base so callers (the Stage-3 migration walk +
        allocator.can_migrate_slot) can ask directly instead
        of probing for the method — a generic KVCache CANNOT (default False).
        MHATokenToKVPool overrides it (True once warmed via enable_kv_cache_copy
        or under the native-copy path); MLA pools have no move_kv_cache and
        keep the False default, so an MLA(-hybrid) pool refuses migration with
        ZERO mutation rather than asserting/AttributeError-ing mid-fire."""
        return False

    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.layer_num = layer_num
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.mem_usage = 0

        # used for chunked cpu-offloading
        self.cpu_offloading_chunk_size = 8192

        # default state for optional layer-wise transfer control
        self.layer_transfer_counter = None

        # for disagg with nvlink
        self.enable_custom_mem_pool, self.custom_mem_pool, _ = (
            maybe_init_custom_mem_pool(device=self.device)
        )

    def _finalize_allocation_log(self, num_tokens: int):
        """Common logging and mem_usage computation for KV cache allocation.
        Supports both tuple (K, V) size returns and single KV size returns.
        """
        kv_size_bytes = self.get_kv_size_bytes()
        if isinstance(kv_size_bytes, tuple):
            k_size, v_size = kv_size_bytes
            k_size_GB = k_size / GB
            v_size_GB = v_size / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, K size: {k_size_GB:.2f} GB, V size: {v_size_GB:.2f} GB"
            )
            self.mem_usage = k_size_GB + v_size_GB
        else:
            kv_size_GB = kv_size_bytes / GB
            logger.info(
                f"KV Cache is allocated. dtype: {self.dtype}, #tokens: {num_tokens}, KV size: {kv_size_GB:.2f} GB"
            )
            self.mem_usage = kv_size_GB

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError()

    def maybe_get_custom_mem_pool(self):
        return self.custom_mem_pool


class MHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim if v_head_dim is not None else head_dim
        )

        # Optional SHUFFLE 5D ("vectorized") physical layout for K/V.
        # Selected by `SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d` on the ROCm
        # AITER backend (HIP + SGLANG_USE_AITER=1). When active:
        #   K shape: (num_blocks, H, D_k // X, page, X)
        #   V shape: (num_blocks, H, page // X, D_v, X)   where X = 16 / dtype_bytes
        # aiter `mha_batch_prefill_func` consumes these 5D shapes natively and
        # aiter `pa_decode_gluon` reads SHUFFLE blocks directly during decode.
        # An explicit `kv_cache_layout=` argument always wins (e.g. SWAKVPool
        # passes "nhd" to keep its SWA sub-pool on the legacy layout); on
        # non-AITER platforms the env var is ignored and NHD is forced since
        # no consumer kernel exists for SHUFFLE 5D outside the AITER backend.
        self.kv_cache_layout = "nhd"
        if _use_aiter:
            layout = envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower()
            if layout not in ("nhd", "vectorized_5d"):
                raise ValueError(
                    f"Unsupported SGLANG_AITER_KV_CACHE_LAYOUT={layout!r}; "
                    "expected 'nhd' or 'vectorized_5d'."
                )
            self.kv_cache_layout = layout
            if layout == "vectorized_5d":
                # X is the inner vectorization width in the SHUFFLE layout,
                # determined by the STORAGE dtype (not the compute dtype) since
                # it controls how many elements fit in 16 bytes of the on-pool
                # tensor. For fp8 storage X=16, for bf16/fp16 X=8.
                self._kv_vector_x = 16 // self.store_dtype.itemsize
                assert (self.size + self.page_size) % self.page_size == 0
                assert self.page_size % self._kv_vector_x == 0, (
                    f"page_size={self.page_size} must be divisible by "
                    f"X={self._kv_vector_x} for vectorized_5d layout"
                )
                assert self.head_dim % self._kv_vector_x == 0
                assert self.v_head_dim % self._kv_vector_x == 0

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)

        _use_alt_stream = _is_cuda or current_platform.is_cuda_alike()
        self.alt_stream = (
            self.device_module.Stream()
            if _use_alt_stream and enable_alt_stream
            else None
        )

        if enable_kv_cache_copy:
            self._init_kv_copy_and_warmup()
        else:
            self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # for store_cache JIT kernel
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = self.head_dim == self.v_head_dim

    def _init_kv_copy_and_warmup(self):
        # Zero-layer pool (e.g. all-SWA model's full sub-pool) has no buffers.
        if self.layer_num == 0:
            self._kv_copy_config = None
            return

        # Heuristics for KV copy tiling
        _KV_COPY_STRIDE_THRESHOLD_LARGE = 8192
        _KV_COPY_STRIDE_THRESHOLD_MEDIUM = 4096
        _KV_COPY_TILE_SIZE_LARGE = 512
        _KV_COPY_TILE_SIZE_MEDIUM = 256
        _KV_COPY_TILE_SIZE_SMALL = 128
        _KV_COPY_NUM_WARPS_LARGE_TILE = 8
        _KV_COPY_NUM_WARPS_SMALL_TILE = 4

        stride_bytes = int(self.data_strides[0].item())
        if stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_LARGE:
            bytes_per_tile = _KV_COPY_TILE_SIZE_LARGE
        elif stride_bytes >= _KV_COPY_STRIDE_THRESHOLD_MEDIUM:
            bytes_per_tile = _KV_COPY_TILE_SIZE_MEDIUM
        else:
            bytes_per_tile = _KV_COPY_TILE_SIZE_SMALL

        # Calculate num_locs_upper to avoid large Triton specialization (e.g. 8192)
        chunk_upper = 128 if bytes_per_tile >= _KV_COPY_TILE_SIZE_LARGE else 256

        self._kv_copy_config = {
            "bytes_per_tile": bytes_per_tile,
            "byte_tiles": (stride_bytes + bytes_per_tile - 1) // bytes_per_tile,
            "num_warps": (
                _KV_COPY_NUM_WARPS_SMALL_TILE
                if bytes_per_tile <= _KV_COPY_TILE_SIZE_MEDIUM
                else _KV_COPY_NUM_WARPS_LARGE_TILE
            ),
            "num_locs_upper": chunk_upper,
        }

        dummy_loc = torch.zeros(chunk_upper, dtype=torch.int64, device=self.device)
        grid = (self.data_ptrs.numel(), self._kv_copy_config["byte_tiles"])

        copy_all_layer_kv_cache_tiled[grid](
            self.data_ptrs,
            self.data_strides,
            dummy_loc,
            dummy_loc,
            1,
            chunk_upper,
            BYTES_PER_TILE=self._kv_copy_config["bytes_per_tile"],
            num_warps=self._kv_copy_config["num_warps"],
            num_stages=2,
        )

    def _create_buffers(self):
        # Unconditionally declare the arena attribute so callers can
        # always do a direct None check instead of `getattr(..., None)`.
        # Stays None for the non-arena (torch.zeros) backend.
        self._kv_arena = None
        # Optional ChunkArena-backed allocation. Gated by
        # SGLANG_KV_ARENA=1. Restricted to head_dim == v_head_dim for now;
        # falls through to default for the asymmetric case.
        # SGLANG_ARENA_SHARED=1 implies KV_ARENA=1 and routes this arena's
        # MultiTensorArena onto the process-singleton SharedHandlePool so
        # cross-pool (KV ↔ mamba) transfer can move physical handles
        # between the two pools.
        shared_arena = os.environ.get("SGLANG_ARENA_SHARED") == "1"
        use_arena = (
            (os.environ.get("SGLANG_KV_ARENA") == "1" or shared_arena)
            and self.head_dim == self.v_head_dim
            and not self.enable_custom_mem_pool
        )
        logger.info(
            "MHATokenToKVPool buffers: backend=%s (SGLANG_KV_ARENA=%s, "
            "SGLANG_ARENA_SHARED=%s, head_dim==v_head_dim=%s, "
            "custom_mem_pool=%s), size=%d, page_size=%d, layer_num=%d, "
            "head_num=%d, head_dim=%d",
            "arena" if use_arena else "torch.zeros",
            os.environ.get("SGLANG_KV_ARENA", "<unset>"),
            os.environ.get("SGLANG_ARENA_SHARED", "<unset>"),
            self.head_dim == self.v_head_dim,
            self.enable_custom_mem_pool,
            self.size, self.page_size, self.layer_num,
            self.head_num, self.head_dim,
        )
        if use_arena:
            from sglang.srt.arena.multi_tensor_arena import MultiTensorArena
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                # For the first cut, init = max (no soft-cap headroom);
                # set_capacity_tokens(n) will be added to enable runtime
                # resize once the budgeter is wired in 2e.4.d.
                tot = self.size + self.page_size
                # Phase A5: ablation honors SGLANG_ARENA_CHUNK_BYTES.
                # T1 (paper §3.2.1): default to CUDA VMM's native 2 MiB page
                # granularity on H200. Chunk-grain (e.g., 64 MiB) is selectable
                # via SGLANG_ARENA_CHUNK_BYTES for legacy A/B comparison.
                chunk_bytes = int(os.environ.get(
                    "SGLANG_ARENA_CHUNK_BYTES", str(2 * 1024 * 1024)
                ))
                per_token_bytes = (
                    self.head_num * self.head_dim
                    * torch.tensor([], dtype=self.store_dtype).element_size()
                )
                tokens_per_chunk = _arena_tokens_per_chunk(
                    chunk_bytes, per_token_bytes
                )
                # Round up to chunk-aligned token count.
                tot_aligned = (
                    (tot + tokens_per_chunk - 1) // tokens_per_chunk
                ) * tokens_per_chunk

                shared_pool = None
                # See MambaPool note above: VA-only headroom for the actuator.
                # T5 (paper §3.2.1): SGLANG_ARENA_KV_HEADROOM_BYTES takes
                # precedence over the legacy SGLANG_ARENA_KV_HEADROOM_CHUNKS.
                # Default 80 GiB ensures the KV pool can grow into a
                # peer-released ~80 GiB budget at 2 MiB grain (T1).
                kv_headroom_bytes_env = os.environ.get(
                    "SGLANG_ARENA_KV_HEADROOM_BYTES"
                )
                if shared_arena and kv_headroom_bytes_env is not None:
                    kv_growth_chunks = (
                        int(kv_headroom_bytes_env) // chunk_bytes
                    )
                elif shared_arena:
                    legacy_chunks_env = os.environ.get(
                        "SGLANG_ARENA_KV_HEADROOM_CHUNKS"
                    )
                    if legacy_chunks_env is not None:
                        kv_growth_chunks = int(legacy_chunks_env)
                    else:
                        kv_growth_chunks = (
                            (80 * 1024 * 1024 * 1024) // chunk_bytes
                        )
                else:
                    kv_growth_chunks = 0
                kv_max_tokens = tot_aligned + kv_growth_chunks * tokens_per_chunk
                if shared_arena:
                    from sglang.srt.arena.shared_pool import (
                        get_or_create_shared_handle_pool,
                    )
                    shared_pool = get_or_create_shared_handle_pool(
                        device_id=torch.cuda.current_device(),
                        chunk_bytes=chunk_bytes,
                    )

                # Static-min/soft split — see MambaPool note above.
                # Boot maps init_chunks fully; static_min is the floor for
                # actuator shrink (1 chunk per sub-pool when shared_arena=on,
                # else == init for non-L2 baseline behavior).
                init_chunks = tot_aligned // tokens_per_chunk
                kv_static_min_chunks = 1 if shared_arena else init_chunks
                kv_static_min_tokens = kv_static_min_chunks * tokens_per_chunk
                self._kv_arena = MultiTensorArena(
                    device_id=torch.cuda.current_device(),
                    n_layers=self.layer_num,
                    n_kinds=2,
                    per_token_shape=(self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    max_tokens=kv_max_tokens,
                    init_tokens=tot_aligned,
                    static_min_tokens=kv_static_min_tokens,
                    chunk_bytes=chunk_bytes,
                    external_handle_pool=shared_pool,
                )
                # Paper §sec:design-l2: pool boots at full init capacity. The
                # actuator dynamically caps the allocator during drain when
                # firing a shrink; no boot-time cap needed.
                logger.info(
                    "MHATokenToKVPool arena: tot_tokens=%d (tot_aligned=%d), "
                    "tokens_per_chunk=%d, chunk_bytes=%d, per_token_bytes=%d, "
                    "shared=%s, subpool_offset=%d, n_subpools=%d",
                    tot, tot_aligned, tokens_per_chunk, chunk_bytes,
                    per_token_bytes, shared_arena,
                    self._kv_arena._subpool_offset,
                    self.layer_num * 2,
                )
            self.k_buffer = [self._kv_arena.tensor(i, 0) for i in range(self.layer_num)]
            self.v_buffer = [self._kv_arena.tensor(i, 1) for i in range(self.layer_num)]
            # Match torch.zeros semantics for the padded slot at index 0.
            for buf in self.k_buffer + self.v_buffer:
                buf[: self.page_size].zero_()
        else:
            with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
                with (
                    torch.cuda.use_mem_pool(self.custom_mem_pool)
                    if self.enable_custom_mem_pool
                    else nullcontext()
                ):
                    # [size, head_num, head_dim] for each layer
                    # The padded slot 0 is used for writing dummy outputs from padded tokens.
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (
                                self.size + self.page_size,
                                self.head_num,
                                self.v_head_dim,
                            ),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def set_capacity_tokens(self, n_tokens: int) -> int:
        """Resize the KV pool to back exactly `n_tokens` of capacity (plus padding).

        Only valid when the pool was created with SGLANG_KV_ARENA=1. Returns
        the actual capacity (rounded up to chunk granularity by the arena).
        Caller is responsible for ensuring no live allocation references
        token indices >= the new capacity before calling shrink.
        """
        if self._kv_arena is None:
            raise RuntimeError(
                "set_capacity_tokens requires SGLANG_KV_ARENA=1 at pool creation"
            )
        target = n_tokens + self.page_size
        # The arena rounds to its chunk granularity; clamp to its max.
        chunk = self._kv_arena._arena.chunk_size
        per_token = self._kv_arena.per_token_bytes
        tokens_per_chunk = chunk // per_token
        target_aligned = (
            (target + tokens_per_chunk - 1) // tokens_per_chunk
        ) * tokens_per_chunk
        target_aligned = min(target_aligned, self._kv_arena.max_tokens)
        prev = self._kv_arena.current_capacity_tokens()
        self._kv_arena.set_capacity_tokens(target_aligned)
        new_advertised = target_aligned - self.page_size
        logger.info(
            "MHATokenToKVPool.set_capacity_tokens: req=%d -> aligned=%d "
            "(prev=%d, advertised=%d, page_size=%d)",
            n_tokens, target_aligned, prev, new_advertised, self.page_size,
        )
        return new_advertised

    def live_capacity_tokens(self) -> int:
        """Currently-backed token capacity (excludes padding)."""
        if self._kv_arena is not None:
            return self._kv_arena.current_capacity_tokens() - self.page_size
        return self.size  # static path: capacity == size

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += get_tensor_size_bytes(k_cache)
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += get_tensor_size_bytes(v_cache)
        return k_size_bytes, v_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes * self.page_size
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = self.k_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = self.v_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(self.k_buffer[0].device, non_blocking=True)
                v_chunk = v_cpu.to(self.v_buffer[0].device, non_blocking=True)
                self.k_buffer[layer_id][chunk_indices] = k_chunk
                self.v_buffer[layer_id][chunk_indices] = v_chunk
        current_platform.synchronize()

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        # note: get_key_buffer is hooked with synchronization for layer-wise KV cache loading
        # it is supposed to be used only by attention backend not for information purpose
        # same applies to get_value_buffer and get_kv_buffer
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc, _ = unwrap_write_loc(loc_info)
        # Catch stale slot ids here instead of as illegal-addr / silent KV
        # corruption in the store_kvcache write (gated on SGLANG_ENABLE_ASYNC_ASSERT).
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.kv_cache_layout == "vectorized_5d":
            # Late-import to keep the NHD path import-clean.
            from sglang.srt.layers.attention.utils import (
                launch_reshape_and_cache_shuffle_5d,
            )

            # The writer kernel uses key.stride(0) directly as the source
            # token stride; head/dim are assumed contiguous within each
            # token (stride(1)=head_size, stride(2)=1). Both hold for K/V
            # produced by QKV split + RoPE in upstream attention even when
            # the outer per-token stride is non-canonical, so we skip the
            # protective .contiguous() copies that would otherwise fire
            # large per-layer elementwise kernels.
            launch_reshape_and_cache_shuffle_5d(
                cache_k,
                cache_v,
                self.k_buffer[layer_id - self.start_layer],
                self.v_buffer[layer_id - self.start_layer],
                loc,
            )
            return

        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
            device_module=self.device_module,
            # size + page_size = real slots + the reserved padding slot (padded /
            # dummy tokens write there); valid index range is [0, size + page_size).
            size_limit=self.k_buffer[0].shape[0] if self._kv_arena is not None else self.size + self.page_size,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
            raise ValueError(
                "commit_lens must match loc_2d batch size: "
                f"{tuple(commit_lens.shape)=} {tuple(loc_2d.shape)=}."
            )

        num_rows = int(loc_2d.numel())
        if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
            raise ValueError(
                "dense KV rows must match loc_2d size: "
                f"{tuple(cache_k.shape)=} {tuple(cache_v.shape)=} {tuple(loc_2d.shape)=}."
            )

        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.contiguous().view(self.store_dtype)
            cache_v = cache_v.contiguous().view(self.store_dtype)
        else:
            cache_k = cache_k.contiguous()
            cache_v = cache_v.contiguous()

        if loc_2d.device != self.k_buffer[0].device:
            loc_2d = loc_2d.to(device=self.k_buffer[0].device, non_blocking=True)
        if commit_lens.device != self.k_buffer[0].device:
            commit_lens = commit_lens.to(
                device=self.k_buffer[0].device, non_blocking=True
            )
        if loc_2d.dtype != torch.int64:
            loc_2d = loc_2d.to(torch.int64)
        if commit_lens.dtype != torch.int32:
            commit_lens = commit_lens.to(torch.int32)

        if not (_is_cuda or _is_hip):
            row_offsets = torch.arange(loc_2d.shape[1], device=loc_2d.device)
            valid_mask = row_offsets[None, :] < commit_lens.to(torch.int64)[:, None]
            valid_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                return
            self.set_kv_buffer(
                layer,
                loc_2d.reshape(-1).index_select(0, valid_idx),
                cache_k.index_select(0, valid_idx),
                cache_v.index_select(0, valid_idx),
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
            return

        _set_kv_buffer_prefix_valid_impl(
            cache_k,
            cache_v,
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc_2d,
            commit_lens,
            row_dim=self.row_dim,
            store_dtype=self.store_dtype,
        )

    def can_move_kv_cache(self) -> bool:
        # Capable under the native-copy path (no warmup needed) OR once
        # enable_kv_cache_copy has initialized _kv_copy_config. Mirrors the two
        # branches of move_kv_cache below so the capability check can't drift
        # from what move_kv_cache actually requires.
        return (
            envs.SGLANG_NATIVE_MOVE_KV_CACHE.get()
            or self._kv_copy_config is not None
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # Zero-layer pool (e.g. all-SWA model's full sub-pool) has no buffers.
        if self.layer_num == 0:
            return

        # Catch stale indices here instead of as illegal-addr or silent KV corruption.
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if envs.SGLANG_NATIVE_MOVE_KV_CACHE.get():
            move_kv_cache_native(self.k_buffer, self.v_buffer, tgt_loc, src_loc)
            return

        N = tgt_loc.numel()
        if N == 0:
            return

        assert (
            self._kv_copy_config is not None
        ), "KV copy not initialized. Set enable_kv_cache_copy=True in __init__"

        cfg = self._kv_copy_config
        cap = int(cfg.get("num_locs_upper", 256))
        grid = (self.data_ptrs.numel(), cfg["byte_tiles"])

        if N <= cap:
            upper = next_power_of_2(N)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc,
                src_loc,
                N,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )
            return

        # Huge N: chunk, but each chunk's upper is still pow2(<= cap)
        for start in range(0, N, cap):
            end = min(start + cap, N)
            chunk_len = end - start
            upper = next_power_of_2(chunk_len)
            copy_all_layer_kv_cache_tiled[grid](
                self.data_ptrs,
                self.data_strides,
                tgt_loc[start:end],
                src_loc[start:end],
                chunk_len,
                upper,
                BYTES_PER_TILE=cfg["bytes_per_tile"],
                num_warps=cfg["num_warps"],
                num_stages=2,
            )


class NoOpMHATokenToKVPool(MHATokenToKVPool):
    """KV cache pool that skips physical K/V buffer allocation.

    Used in embedding-mode prefill-only workloads with the FA
    fa_skip_kv_cache path, where no layer reads or writes KV cache because
    attention uses raw K/V via flash_attn_varlen_func. Other prefill-only paths
    such as scoring/MIS may benefit from the same idea later, but some still
    stage K/V through paged cache today.

    This class keeps the scheduler's view of pool capacity (self.size is
    honored for admission) but allocates only (page_size, head_num, head_dim)
    placeholder tensors per layer to satisfy any code paths that dereference
    the buffers.

    Callers MUST ensure no real set_kv_buffer/get_*_buffer calls happen against
    this pool; those paths raise loudly so misuse is visible.
    """

    def _create_buffers(self):
        # Allocate minimal placeholder buffers. They exist purely so that code
        # paths holding `k_buffer` / `v_buffer` references (pointer tables,
        # layer-transfer counters, stride arithmetic) keep working without
        # None-guards scattered across the codebase. Shape is
        # [page_size, head_num, head_dim] per layer so that the unconditional
        # `key_cache.view(-1, page_size, head_num, head_dim)` in the FA backend
        # at the top of forward_extend succeeds regardless of --page-size.
        # Total footprint is still on the order of KB vs GBs for a real pool.
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (self.page_size, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs], dim=0)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def _finalize_allocation_log(self, num_tokens: int):
        self.mem_usage = 0.0
        placeholder_bytes = (
            2
            * self.layer_num
            * self.page_size
            * self.head_num
            * max(self.head_dim, self.v_head_dim)
            * self.store_dtype.itemsize
        )
        logger.info(
            f"KV Cache skipped (no-op pool). Logical #tokens: {num_tokens}, "
            f"physical K/V size: ~{placeholder_bytes / 1024:.1f} KB placeholder"
        )

    def get_kv_size_bytes(self):
        # Report zero so downstream memory accounting matches reality.
        return (0, 0)

    def set_kv_buffer(self, *args, **kwargs):
        raise RuntimeError(
            "NoOpMHATokenToKVPool.set_kv_buffer was called. This pool is only "
            "valid in prefill-only modes (e.g. --is-embedding, scoring) with "
            "the FA backend's fa_skip_kv_cache path active; the attention "
            "backend must never write to it. Check that the workload truly "
            "performs no decode and that the FA backend's fa_skip_kv_cache "
            "preconditions are met."
        )

    def get_key_buffer(self, layer_id: int):
        # Return the placeholder. The FA backend reads this before taking the
        # fa_skip_kv_cache branch (which does not use it); the placeholder shape
        # is (page_size, head_num, head_dim) so downstream .view() calls succeed.
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        # no-op; embedding mode has no KV cache to move
        return


class MHATokenToKVPoolFP4(MHATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                # [size, head_num, head_dim] for each layer
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = self.head_num
                k = self.head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer
        del self.k_scale_buffer
        del self.v_scale_buffer

    def _get_key_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.k_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.k_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        # for internal use of referencing
        if self.store_dtype != self.dtype:
            cache_v_nope_fp4 = self.v_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_v_nope_fp4_sf = self.v_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_v_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_v_nope_fp4, cache_v_nope_fp4_sf
            )
            return cache_v_nope_fp4_dequant
        return self.v_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        loc, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA-FP4)")
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_k)
            cache_v, cache_v_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(cache_v)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

            cache_k_fp4_sf = cache_k_fp4_sf.view(self.store_dtype)
            cache_v_fp4_sf = cache_v_fp4_sf.view(self.store_dtype)

        if get_is_capture_mode() and self.alt_stream is not None:
            # Overlap the copy of K and V cache for small batch size
            current_stream = self.device_module.current_stream()
            self.alt_stream.wait_stream(current_stream)
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            with self.device_module.stream(self.alt_stream):
                self.v_buffer[layer_id - self.start_layer][loc] = cache_v

                self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf
            current_stream.wait_stream(self.alt_stream)
        else:
            self.k_buffer[layer_id - self.start_layer][loc] = cache_k
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v

            self.k_scale_buffer[layer_id - self.start_layer][loc] = cache_k_fp4_sf
            self.v_scale_buffer[layer_id - self.start_layer][loc] = cache_v_fp4_sf


class HybridLinearKVPool(KVCache):
    """KV cache with separate pools for full and linear attention layers."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        page_size: int,
        head_num: int,
        head_dim: int,
        full_attention_layer_ids: List[int],
        enable_kvcache_transpose: bool,
        device: str,
        mamba_pool: MambaPool,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        # TODO: refactor mla related args
        use_mla: bool = False,
        kv_lora_rank: int = None,
        qk_rope_head_dim: int = None,
        start_layer: Optional[int] = None,
    ):
        self.size = size
        self.dtype = dtype
        self.device = device
        self.full_layer_nums = len(full_attention_layer_ids)
        self.page_size = page_size
        self.start_layer = start_layer if start_layer is not None else 0
        self.layer_transfer_counter = None
        self.head_num = head_num
        self.head_dim = head_dim
        self.mamba_pool = mamba_pool
        # TODO MHATransposedTokenToKVPool if enable_kvcache_transpose is True
        assert not enable_kvcache_transpose
        self.use_mla = use_mla
        if not use_mla:
            TokenToKVPoolClass = MHATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mha_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMHATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=self.full_layer_nums,
                device=device,
                enable_memory_saver=enable_memory_saver,
                # Live KV-slot migration relocates bytes via move_kv_cache,
                # which needs _kv_copy_config. The MLA branch
                # below has no move_kv_cache (migrate_slot fails loud there),
                # so the flag is forwarded only on the MHA path.
                enable_kv_cache_copy=enable_kv_cache_copy,
            )
        else:
            TokenToKVPoolClass = MLATokenToKVPool

            if current_platform.is_out_of_tree():
                TokenToKVPoolClass = current_platform.get_mla_kv_pool_cls()
            elif _is_npu:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                TokenToKVPoolClass = NPUMLATokenToKVPool

            self.full_kv_pool = TokenToKVPoolClass(
                size=size,
                page_size=self.page_size,
                dtype=dtype,
                layer_num=self.full_layer_nums,
                device=device,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                enable_memory_saver=enable_memory_saver,
            )
        self.full_attention_layer_id_mapping = {
            id: i for i, id in enumerate(full_attention_layer_ids)
        }
        if use_mla:
            self.mem_usage = self.get_kv_size_bytes() / GB
        else:
            k_size, v_size = self.get_kv_size_bytes()
            self.mem_usage = (k_size + v_size) / GB

    @property
    def _kv_arena(self):
        # The arena (if any) lives on the inner full-attention pool; forward
        # so callers see the same `_kv_arena` interface as a plain KVCache
        # (KV-growable wiring).
        return self.full_kv_pool._kv_arena

    def get_kv_size_bytes(self):
        return self.full_kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        mamba_data_ptrs, mamba_data_lens, mamba_item_lens = (
            self.mamba_pool.get_contiguous_buf_infos()
        )
        return mamba_data_ptrs, mamba_data_lens, mamba_item_lens

    def get_state_dim_per_tensor(self):
        """Get the sliceable dimension size for each mamba state tensor."""
        return self.mamba_pool.get_state_dim_per_tensor()

    def maybe_get_custom_mem_pool(self):
        return self.full_kv_pool.maybe_get_custom_mem_pool()

    def _transfer_full_attention_id(self, layer_id: int):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(
                f"{layer_id=} not in full attention layers: {self.full_attention_layer_id_mapping.keys()}"
            )
        return self.full_attention_layer_id_mapping[layer_id]

    def register_layer_transfer_counter(self, layer_transfer_counter: LayerDoneCounter):
        self.layer_transfer_counter = layer_transfer_counter
        # The layer-wise wait logic is executed at the Hybrid LinearPool level;
        # no additional wait is needed in the full_kv_pool
        self.full_kv_pool.register_layer_transfer_counter(None)

    def _wait_for_layer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    def get_key_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        self._wait_for_layer(layer_id)
        layer_id = self._transfer_full_attention_id(layer_id)
        return self.full_kv_pool.get_kv_buffer(layer_id)

    @contextmanager
    def _transfer_id_context(self, layer: RadixAttention):
        @contextmanager
        def _patch_layer_id(layer):
            original_layer_id = layer.layer_id
            layer.layer_id = self._transfer_full_attention_id(layer.layer_id)
            try:
                yield
            finally:
                layer.layer_id = original_layer_id

        with _patch_layer_id(layer):
            yield

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ):
        layer_id = self._transfer_full_attention_id(layer.layer_id)
        if not self.use_mla:
            self.full_kv_pool.set_kv_buffer(
                None,
                loc,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=layer_id,
            )
        else:
            with self._transfer_id_context(layer):
                self.full_kv_pool.set_kv_buffer(
                    layer,
                    loc,
                    cache_k,
                    cache_v,
                )

    def can_move_kv_cache(self) -> bool:
        # Forward to the inner full-attention pool: MHA inner -> its real
        # capability; MLA inner -> KVCache base default False (MLA has no
        # move_kv_cache), so an MLA hybrid refuses migration cleanly.
        return self.full_kv_pool.can_move_kv_cache()

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        self.full_kv_pool.move_kv_cache(tgt_loc, src_loc)

    def get_cpu_copy(self, indices, mamba_indices=None):
        kv_cpu = self.full_kv_pool.get_cpu_copy(indices)
        mamba_cpu = (
            self.mamba_pool.get_cpu_copy(mamba_indices)
            if mamba_indices is not None
            else None
        )
        return kv_cpu, mamba_cpu

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        kv_cpu, mamba_cpu = cache_cpu
        self.full_kv_pool.load_cpu_copy(kv_cpu, indices)
        if mamba_cpu is not None and mamba_indices is not None:
            self.mamba_pool.load_cpu_copy(mamba_cpu, mamba_indices)

    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        assert self.use_mla, "set_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            self.full_kv_pool.set_mla_kv_buffer(layer, loc, cache_k_nope, cache_k_rope)

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        assert self.use_mla, "get_mla_kv_buffer called when use_mla is False"
        with self._transfer_id_context(layer):
            return self.full_kv_pool.get_mla_kv_buffer(layer, loc, dst_dtype)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_dsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_dsa = use_dsa
        self.dsa_kv_cache_store_fp8 = (
            use_dsa
            and dtype == torch.float8_e4m3fn
            and override_kv_cache_dim is not None
        )
        # When override_kv_cache_dim is provided with dsa model, we assume the
        # override kv cache dim is correct and use it directly.
        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        if not use_dsa:
            # DSA will allocate indexer KV cache later and then log the total size
            self._finalize_allocation_log(size)

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                self.kv_buffer = [
                    torch.zeros(
                        (self.size + self.page_size, 1, self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    # for disagg
    def get_contiguous_buf_infos(self):
        # MLA has only one kv_buffer, so only the information of this buffer needs to be returned.
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in range(self.layer_num)]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in range(self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer].view(self.dtype)

        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id - self.start_layer][
                ..., : self.kv_lora_rank
            ].view(self.dtype)
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id

        if _is_hip and self.use_dsa and self.dtype == fp8_dtype:
            # HIP FP8 path uses raw MLA KV layout (nope + rope) without per-block scales.
            # Fuse BF16/FP16 -> FP8 cast with paged KV write.
            set_mla_kv_buffer_triton_fp8_quant(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
                fp8_dtype,
            )
        elif self.dsa_kv_cache_store_fp8:
            # OPTIMIZATION: Quantize k_nope and k_rope separately to avoid concat overhead
            # This also enables reuse of set_mla_kv_buffer_triton two-tensor write path
            # quantize_k_cache_separate returns (nope_part, rope_part) as uint8 bytes
            cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
                cache_k_nope, cache_k_rope
            )

            # Reuse existing two-tensor write kernel (works with FP8 byte layout)
            # cache_k_nope_fp8: (num_tokens, 1, 528) uint8 [nope_fp8(512) | scales(16)]
            # cache_k_rope_fp8: (num_tokens, 1, 128) uint8 [rope_bf16_bytes(128)]
            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp8,
                cache_k_rope_fp8,
            )
        else:
            if cache_k_nope.dtype != self.dtype:
                cache_k_nope = cache_k_nope.to(self.dtype)
                cache_k_rope = cache_k_rope.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope,
                cache_k_rope,
            )

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        # get k nope and k rope from the kv buffer, and optionally cast them to dst_dtype.
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        get_mla_kv_buffer_triton(kv_buffer, loc, cache_k_nope, cache_k_rope)
        return cache_k_nope, cache_k_rope

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Relocate accepted-token combined MLA KV (latent + rope) per layer."""
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")

        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for kv_cache in self.kv_buffer:
            kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[0].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()


class MLATokenToKVPoolFP4(MLATokenToKVPool):
    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # The padded slot 0 is used for writing dummy outputs from padded tokens.
                m = self.size + self.page_size
                n = 1  # head_num
                k = self.kv_cache_dim  # head_dim

                scale_block_size = 16
                self.store_dtype = torch.uint8

                self.kv_buffer = [
                    torch.zeros(
                        (m, n, k // 2),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

                self.kv_scale_buffer = [
                    torch.zeros(
                        (m, k // scale_block_size),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def _clear_buffers(self):
        del self.kv_buffer
        del self.kv_scale_buffer

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype != self.dtype:
            cache_k_nope_fp4 = self.kv_buffer[layer_id - self.start_layer].view(
                torch.uint8
            )
            cache_k_nope_fp4_sf = self.kv_scale_buffer[layer_id - self.start_layer]

            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_nope_fp4_dequant = BlockFP4KVQuantizeUtil.batched_dequantize(
                cache_k_nope_fp4, cache_k_nope_fp4_sf
            )
            return cache_k_nope_fp4_dequant

        return self.kv_buffer[layer_id - self.start_layer]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # loc_info may be a KVWriteLoc; MLA pools have no SWA target.
        loc, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA-FP4)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        if cache_k.dtype != self.dtype:
            from sglang.srt.layers.quantization.kvfp4_tensor import (
                BlockFP4KVQuantizeUtil,
            )

            cache_k_fp4, cache_k_fp4_sf = BlockFP4KVQuantizeUtil.batched_quantize(
                cache_k
            )

        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k_fp4.view(
                self.store_dtype
            )
            self.kv_scale_buffer[layer_id - self.start_layer][loc] = (
                cache_k_fp4_sf.view(self.store_dtype)
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA-FP4)"
        )
        layer_id = layer.layer_id

        if self.dsa_kv_cache_store_fp8:
            # original cache_k: (num_tokens, num_heads 1, hidden 576); we unsqueeze the page_size=1 dim here
            # TODO no need to cat
            cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
            cache_k = quantize_k_cache(cache_k.unsqueeze(1)).squeeze(1)
            cache_k = cache_k.view(self.store_dtype)
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k
        else:
            if cache_k_nope.dtype != self.dtype:
                from sglang.srt.layers.quantization.kvfp4_tensor import (
                    BlockFP4KVQuantizeUtil,
                )

                cache_k_nope_fp4, cache_k_nope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_nope)
                )
                cache_k_rope_fp4, cache_k_rope_fp4_sf = (
                    BlockFP4KVQuantizeUtil.batched_quantize(cache_k_rope)
                )

            if self.store_dtype != self.dtype:
                cache_k_nope = cache_k_nope.view(self.store_dtype)
                cache_k_rope = cache_k_rope.view(self.store_dtype)

            set_mla_kv_buffer_triton(
                self.kv_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4,
                cache_k_rope_fp4,
            )
            set_mla_kv_scale_buffer_triton(
                self.kv_scale_buffer[layer_id - self.start_layer],
                loc,
                cache_k_nope_fp4_sf,
                cache_k_rope_fp4_sf,
            )


class DSATokenToKVPool(MLATokenToKVPool):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8
    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16

    def __init__(
        self,
        size: int,
        page_size: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        index_head_dim: int,
        enable_memory_saver: bool,
        kv_cache_dim: int,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        index_buf_size: Optional[int] = None,
    ):
        override_dim = (
            kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
        )

        super().__init__(
            size,
            page_size,
            dtype,
            kv_lora_rank,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
            use_dsa=True,
            override_kv_cache_dim=override_dim,
        )
        # self.index_k_dtype = torch.float8_e4m3fn
        # self.index_k_scale_dtype = torch.float32
        self.index_head_dim = index_head_dim
        if index_buf_size is None:
            index_buf_size = size
        # num head == 1 and head dim == 128 for index_k in DSA
        assert index_head_dim == 128

        if _is_hip:
            if aiter_can_use_preshuffle_paged_mqa():
                assert (
                    self.page_size % 16 == 0
                ), f"HIP preshuffle requires page_size to be a multiple of 16, got {self.page_size}"
            else:
                assert (
                    self.page_size == 1
                ), f"HIP legacy DSA path requires page_size == 1, got {self.page_size}"
        else:
            assert self.page_size == 64
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            self.index_k_with_scale_buffer = [
                torch.zeros(
                    # Layout:
                    #     ref: test_attention.py :: kv_cache_cast_to_fp8
                    #     shape: (num_pages, page_size 64 * head_dim 128 + page_size 64 * fp32_nbytes 4)
                    #     data: for page i,
                    #         * buf[i, :page_size * head_dim] for fp8 data
                    #         * buf[i, page_size * head_dim:].view(float32) for scale
                    (
                        (index_buf_size + page_size + 1) // self.page_size,
                        self.page_size
                        * (
                            index_head_dim + index_head_dim // self.quant_block_size * 4
                        ),
                    ),
                    dtype=self.index_k_with_scale_buffer_dtype,
                    device=device,
                )
                for _ in range(layer_num)
            ]
        self._finalize_allocation_log(size)

    def _clear_buffers(self):
        del self.kv_buffer
        del self.index_k_with_scale_buffer

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        """Move latent KV and the DSA indexer cache (key + scale) in lockstep."""
        super().move_kv_cache(tgt_loc, src_loc)

        if tgt_loc.numel() == 0:
            return

        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for index_k in self.index_k_with_scale_buffer:
            index_k[tgt_loc_flat] = index_k[src_loc_flat]

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetK.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_continuous(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetS.execute(
            self, buf, seq_len=seq_len, page_indices=page_indices
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        """
        Fused method to get both index K and scale data in a single call using Triton.
        More efficient than calling get_index_k_continuous and get_index_k_scale_continuous separately.

        :param layer_id: Layer index
        :param seq_len: Sequence length
        :param page_indices: Page indices tensor
        :return: tuple of (k_fp8, k_scale) where
                 k_fp8: (seq_len, index_head_dim), uint8
                 k_scale: (seq_len, 4), uint8
        """
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        return index_buf_accessor.GetKAndS.execute(
            self,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
        index_buf_accessor.SetKAndS.execute(
            pool=self, buf=buf, loc=loc, index_k=index_k, index_k_scale=index_k_scale
        )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # DSA keeps a page-indexed index_k_with_scale_buffer alongside kv_buffer.
        # Retract frees the slots/pages and they get reused by other reqs'
        # set_index_k_scale_buffer, so we must offload it here too -- otherwise
        # resume restores kv_buffer but leaves foreign index/scale in place and
        # DSA attention reads garbage at those token positions.
        kv_cache_cpu = super().get_cpu_copy(indices, mamba_indices=mamba_indices)

        page_indices = indices[:: self.page_size] // self.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            index_k_cpu.append([])
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.index_k_with_scale_buffer[layer_id][
                    chunk_page_indices
                ].to("cpu", non_blocking=True)
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()

        return {"kv": kv_cache_cpu, "index_k": index_k_cpu}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        super().load_cpu_copy(
            kv_cache_cpu_dict["kv"], indices, mamba_indices=mamba_indices
        )

        page_indices = indices[:: self.page_size] // self.page_size
        index_k_cpu = kv_cache_cpu_dict["index_k"]
        torch.cuda.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.page_size)
        for layer_id in range(self.layer_num):
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(
                    self.index_k_with_scale_buffer[0].device, non_blocking=True
                )
                self.index_k_with_scale_buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()

    def get_state_buf_infos(self):
        data_ptrs = [
            self.index_k_with_scale_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        data_lens = [
            self.index_k_with_scale_buffer[i].nbytes for i in range(self.layer_num)
        ]
        item_lens = [
            self.index_k_with_scale_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        return data_ptrs, data_lens, item_lens

    def get_kv_size_bytes(self):
        kv_size_bytes = super().get_kv_size_bytes()
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes


def move_kv_cache_native(
    k_buffer: List[torch.Tensor],
    v_buffer: List[torch.Tensor],
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
):
    if tgt_loc.numel() == 0:
        return

    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for k_cache, v_cache in zip(k_buffer, v_buffer):
        k_cache[tgt_loc_flat] = k_cache[src_loc_flat]
        v_cache[tgt_loc_flat] = v_cache[src_loc_flat]
