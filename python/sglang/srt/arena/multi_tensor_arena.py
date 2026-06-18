"""
MultiTensorArena: a ChunkArena pool group for the KV-style layout.

SGLang's KV pool exposes per-layer per-kind (k, v) tensors of shape
`(N_tokens, head_num, head_dim)`. Token rows are contiguous within each
tensor but token i across layers/k+v lives in different tensors. To
"add 1 chunk" of KV capacity at the planner level we have to grow each
sub-tensor's tail by the same number of bytes.

This class:
  - reserves one VA arena that contains N_LAYERS * N_KINDS sub-ranges,
    each sized for full max-token capacity;
  - pre-allocates the per-tensor `torch.Tensor` at shape (N_MAX, ...) so
    the engine sees the full shape and indexing works for any
    layer/kind/token, even though only the first `current_tokens` rows
    are physically backed;
  - exposes `set_capacity_tokens(n)` that fans out to all sub-pools,
    keeping all sub-tensor capacities synchronized.

Invariants:
  - Each sub-pool has the same chunk_size and the same currently-mapped
    chunk count. The token capacity is always
    `min_chunks_across_subpools * tokens_per_chunk`.
  - tensor(layer, kind).data_ptr() is the same VA forever; only the
    first `current_tokens` rows are accessible.
"""

from __future__ import annotations
import ctypes
import logging
import os
from typing import List, Optional, Tuple

import torch

from sglang.srt.arena.chunk_arena import (
    ChunkArena,
    SharedHandlePool,
    _DPTR,  # noqa: F401
)


logger = logging.getLogger(__name__)
_SO_NAME = "arena_multi64.so"
_C_NAME = "arena_multi64.c"
_MAX_SUBPOOLS = 64


def _ensure_arena_so_built() -> str:
    """Build arena_multi64.so from arena_multi64.c if missing.

    The .so is a build artifact (gitignored), so a fresh checkout has
    only the .c source. Auto-compile on first use so callers don't
    have to wire a separate build step. One-time cost (~50 ms).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, _SO_NAME)
    c_path = os.path.join(here, _C_NAME)
    if os.path.exists(so_path):
        if not os.path.exists(c_path) or \
                os.path.getmtime(so_path) >= os.path.getmtime(c_path):
            return so_path
    import subprocess
    logger.info("Building %s from %s (one-time)", _SO_NAME, _C_NAME)
    rc = subprocess.run(
        ["gcc", "-O2", "-fPIC", "-shared", c_path, "-o", so_path],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        raise RuntimeError(
            f"Failed to build {_SO_NAME}: {rc.stderr}\n"
            f"(manual: gcc -O2 -fPIC -shared {c_path} -o {so_path})"
        )
    return so_path


def _per_token_bytes(per_token_shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for d in per_token_shape:
        n *= d
    return n * torch.tensor([], dtype=dtype).element_size()


class MultiTensorArena:
    """A KV-style multi-tensor pool over ChunkArena.

    One arena, `n_layers * n_kinds` sub-pools (one per (layer, kind)
    tensor). All sub-pools have the same byte capacity and grow/shrink
    in lock-step. A token-capacity API on top translates to per-sub-pool
    chunk operations.
    """

    def __init__(
        self,
        device_id: int,
        n_layers: int,
        n_kinds: int,            # 2 for KV (k, v); 1 for single-buffer pools
        per_token_shape: Tuple[int, ...],
        dtype: torch.dtype,
        max_tokens: int,
        init_tokens: int,
        chunk_bytes: int = 32 * 1024 * 1024,
        external_handle_pool: Optional[SharedHandlePool] = None,
        subpool_offset: Optional[int] = None,
        # static_min_tokens is the actuator runtime SHRINK FLOOR, not a
        # boot-time reserve. At boot every sub-pool is cuMemMap'd to its
        # full `init_chunks_per_pool` (see the boot loop below); nothing is
        # left soft in the shared queue. The cross-pool actuator's drain
        # protocol may later shrink a pool below boot at fire time, but
        # never past static_min_tokens. Default: static_min == init (the
        # pool can grow but is never shrunk below its boot mapping).
        static_min_tokens: Optional[int] = None,
    ) -> None:
        n_subpools = n_layers * n_kinds
        # Resolve subpool_offset: explicit value wins; else auto-assign from
        # the shared pool (so multiple MTAs in one process don't collide on
        # arena_multi64.so's 64 fixed pool slots); else 0 (single-arena mode).
        if subpool_offset is None:
            if external_handle_pool is not None:
                subpool_offset = external_handle_pool.allocate_subpool_range(n_subpools)
            else:
                subpool_offset = 0
        if subpool_offset + n_subpools > _MAX_SUBPOOLS:
            raise ValueError(
                f"need C indices [{subpool_offset}, {subpool_offset + n_subpools}) "
                f"but arena_multi64.so only has {_MAX_SUBPOOLS}"
            )

        self.n_layers = n_layers
        self.n_kinds = n_kinds
        self.per_token_shape = per_token_shape
        self.dtype = dtype
        self.max_tokens = max_tokens
        # Shift C-side pool indices when sharing arena_multi64.so between
        # two MultiTensorArenas (e.g., KV at offset 0, mamba at offset
        # n_kv_subpools). Logical sub-pool index `i` maps to C index
        # `subpool_offset + i`.
        self._subpool_offset = subpool_offset

        per_token = _per_token_bytes(per_token_shape, dtype)
        if chunk_bytes % per_token != 0:
            raise ValueError(
                f"chunk_bytes {chunk_bytes} not a multiple of per-token bytes {per_token}"
            )
        self.per_token_bytes = per_token
        self.tokens_per_chunk = chunk_bytes // per_token

        if max_tokens % self.tokens_per_chunk != 0:
            raise ValueError(
                f"max_tokens {max_tokens} not a multiple of tokens_per_chunk {self.tokens_per_chunk}"
            )
        self.max_chunks_per_pool = max_tokens // self.tokens_per_chunk
        if init_tokens > max_tokens:
            raise ValueError("init_tokens > max_tokens")
        if init_tokens % self.tokens_per_chunk != 0:
            raise ValueError(
                f"init_tokens {init_tokens} not a multiple of tokens_per_chunk {self.tokens_per_chunk}"
            )
        self.init_chunks_per_pool = init_tokens // self.tokens_per_chunk

        # static_min defaults to init (= boot maps everything, no soft).
        if static_min_tokens is None:
            static_min_tokens = init_tokens
        if static_min_tokens > init_tokens:
            raise ValueError(
                f"static_min_tokens {static_min_tokens} > init_tokens {init_tokens}"
            )
        if static_min_tokens % self.tokens_per_chunk != 0:
            raise ValueError(
                f"static_min_tokens {static_min_tokens} not a multiple of "
                f"tokens_per_chunk {self.tokens_per_chunk}"
            )
        self.static_min_chunks_per_pool = static_min_tokens // self.tokens_per_chunk

        # Self-owned: provision physical handles for the full max-chunks
        # range so any planner-requested grow within [init, max] succeeds.
        # Shared mode: each arena pays for its INITIAL handle quota —
        # these are cuMemCreate'd at boot, and the boot loop below
        # cuMemMaps all `init_chunks_per_pool` of them into each sub-pool.
        if external_handle_pool is None:
            n_handles = n_subpools * self.max_chunks_per_pool
        else:
            n_handles = n_subpools * self.init_chunks_per_pool

        self._arena = ChunkArena(
            device_id=device_id,
            chunk_size=chunk_bytes,
            n_handles=n_handles,
            pool_capacities=[
                (self._pool_name(i), self.max_chunks_per_pool)
                for i in range(n_subpools)
            ],
            external_handle_pool=external_handle_pool,
        )

        # Boot maps the full init_chunks_per_pool worth of physical
        # handles into each sub-pool, so the pool boots at baseline
        # capacity with no unmapped reserve. The actuator's drain protocol
        # shrinks down to static_min only at fire time, never at boot.
        for i in range(n_subpools):
            # grow() returns list[int] of mapped slot IDs; this boot path
            # only checks the count.
            granted = len(self._arena.grow(
                self._pool_name(i), self.init_chunks_per_pool))
            if granted != self.init_chunks_per_pool:
                raise RuntimeError(
                    f"sub-pool {i} only got {granted} of "
                    f"{self.init_chunks_per_pool} init chunks"
                )

        # Hand each sub-pool to the C-side allocator at full init capacity.
        so_path = _ensure_arena_so_built()
        self._lib = ctypes.CDLL(so_path)
        self._lib.multi_init.argtypes = [
            ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t]
        self._lib.multi_set_capacity.argtypes = [ctypes.c_int, ctypes.c_size_t]
        for i in range(n_subpools):
            base = self._arena.pool_va_base(self._pool_name(i))
            self._lib.multi_init(
                self._c_index(i), base, chunk_bytes,
                self.init_chunks_per_pool)

        # Tensor construction via at::from_blob over cuMemMap-backed VA
        # (vAttention's pattern). Tensor's storage has a no-op deleter;
        # arena owns the VA lifetime. Bypasses PyTorch's caching
        # allocator + MemPool entirely, which avoids pytorch issue
        # 165419 (use_mem_pool disables expandable_segments process-
        # wide, costing ~3% TTFT under live attention + CUDA graph
        # capture). Verified by
        # dev/interlayer/0_page_state_machine/alloc_lock/bisect_arena_path.sh.
        self._tensors: List[torch.Tensor] = []
        self._so_path = so_path

        from sglang.srt.arena.from_blob_ext import tensor_from_va
        for i in range(n_subpools):
            # Each sub-pool's VA range starts at pool_va_base; the
            # tensor's data_ptr is the first byte of that range, and
            # its shape is (max_tokens, *).
            va = self._arena.pool_va_base(self._pool_name(i))
            t = tensor_from_va(
                va=va,
                sizes=(max_tokens, *per_token_shape),
                dtype=dtype,
                device_index=device_id,
            )
            if os.environ.get("SGLANG_ARENA_ZERO_INIT_LIVE") == "1":
                live_tokens = self.static_min_chunks_per_pool * self.tokens_per_chunk
                if live_tokens > 0:
                    t[:live_tokens].zero_()
            self._tensors.append(t)

        torch.cuda.synchronize()
        logger.info(
            "MultiTensorArena initialized: n_layers=%d, n_kinds=%d, "
            "n_subpools=%d, chunk_bytes=%d, max_tokens=%d, init_tokens=%d, "
            "static_min_tokens=%d (actuator floor), tokens_per_chunk=%d, "
            "va_base=0x%x, boot_mapped=%d (= init_chunks), "
            "transferable_per_pool=%d (= init - static_min)",
            self.n_layers, self.n_kinds, n_subpools, chunk_bytes,
            max_tokens, init_tokens, static_min_tokens, self.tokens_per_chunk,
            self._arena.va_base,
            self.init_chunks_per_pool,
            self.init_chunks_per_pool - self.static_min_chunks_per_pool,
        )

    # ------------------------------------------------------------------

    def _pool_name(self, i: int) -> str:
        # ChunkArena pool names must be unique across all arenas sharing the
        # same SharedHandlePool. We include the C-index so two MTAs at
        # different subpool_offsets get disjoint name spaces.
        return f"sub{self._c_index(i)}"

    def _c_index(self, i: int) -> int:
        return self._subpool_offset + i

    def _subpool_index(self, layer: int, kind: int) -> int:
        return layer * self.n_kinds + kind

    def tensor(self, layer: int, kind: int) -> torch.Tensor:
        """Return the (sub-pool's) currently-allocated tensor."""
        return self._tensors[self._subpool_index(layer, kind)]

    def current_capacity_tokens(self) -> int:
        """Min mapped chunks across all sub-pools, in token units."""
        n_subpools = self.n_layers * self.n_kinds
        min_chunks = min(
            self._arena.pool_mapped_chunks(self._pool_name(i))
            for i in range(n_subpools)
        )
        return min_chunks * self.tokens_per_chunk

    def set_capacity_tokens(self, n_tokens: int) -> None:
        """Grow or shrink ALL sub-pools to back exactly n_tokens of capacity.

        The per-tensor torch.Tensors are pre-allocated at max_tokens over a
        fixed VA in __init__ (`tensor_from_va`), so this never re-allocates:
        it only maps/unmaps physical chunks per sub-pool (`grow`/`shrink`)
        and updates the C-side capacity register (`multi_set_capacity`).
        Each tensor's data_ptr stays the same VA forever.
        """
        if n_tokens > self.max_tokens:
            raise ValueError(f"n_tokens {n_tokens} > max_tokens {self.max_tokens}")
        if n_tokens % self.tokens_per_chunk != 0:
            raise ValueError(
                f"n_tokens {n_tokens} not a multiple of tokens_per_chunk {self.tokens_per_chunk}"
            )
        new_chunks = n_tokens // self.tokens_per_chunk
        n_subpools = self.n_layers * self.n_kinds
        prev_tokens = self.current_capacity_tokens()
        logger.info(
            "MultiTensorArena.set_capacity_tokens: %d -> %d tokens "
            "(chunks %d -> %d, n_subpools=%d)",
            prev_tokens, n_tokens,
            prev_tokens // self.tokens_per_chunk, new_chunks, n_subpools,
        )

        # Walk each sub-pool and grow/shrink to new_chunks.
        for i in range(n_subpools):
            cur = self._arena.pool_mapped_chunks(self._pool_name(i))
            if new_chunks > cur:
                self._arena.grow(self._pool_name(i), new_chunks - cur)
            elif new_chunks < cur:
                self._arena.shrink(self._pool_name(i), cur - new_chunks)
            self._lib.multi_set_capacity(self._c_index(i), new_chunks)

    def cleanup(self) -> None:
        # from_blob tensors have no-op deleters so dropping them is
        # cheap; arena.cleanup() unmaps the underlying VA.
        del self._tensors[:]
        import gc
        gc.collect()
        torch.cuda.synchronize()
        self._arena.cleanup()
