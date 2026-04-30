"""
Phase 2e.4 — MultiTensorArena: a ChunkArena pool group for the KV-style layout.

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
import os
from typing import List, Tuple

import torch

from chunk_arena import ChunkArena, _DPTR  # noqa: F401


_SO_NAME = "arena_multi64.so"
_MAX_SUBPOOLS = 64


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
    ) -> None:
        n_subpools = n_layers * n_kinds
        if n_subpools > _MAX_SUBPOOLS:
            raise ValueError(
                f"need {n_subpools} sub-pools but arena_multi64.so only has {_MAX_SUBPOOLS}"
            )

        self.n_layers = n_layers
        self.n_kinds = n_kinds
        self.per_token_shape = per_token_shape
        self.dtype = dtype
        self.max_tokens = max_tokens

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

        # Total physical handles = n_subpools * init_chunks_per_pool, plus
        # headroom for the planner to grow at runtime. The arena reserves
        # max_chunks_per_pool of VA per pool, but only init_chunks worth
        # of physical pages are backed; growth requires more handles.
        # For the smoke test we provision room for the full max (no soft cap).
        n_handles = n_subpools * self.max_chunks_per_pool

        self._arena = ChunkArena(
            device_id=device_id,
            chunk_size=chunk_bytes,
            n_handles=n_handles,
            pool_capacities=[
                (self._pool_name(i), self.max_chunks_per_pool)
                for i in range(n_subpools)
            ],
        )

        # Initial mapping: init_chunks_per_pool to each sub-pool.
        for i in range(n_subpools):
            granted = self._arena.grow(self._pool_name(i), self.init_chunks_per_pool)
            if granted != self.init_chunks_per_pool:
                raise RuntimeError(
                    f"sub-pool {i} only got {granted} of {self.init_chunks_per_pool} init chunks"
                )

        # Hand each sub-pool to the C-side allocator.
        so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _SO_NAME)
        self._lib = ctypes.CDLL(so_path)
        self._lib.multi_init.argtypes = [
            ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t]
        self._lib.multi_set_capacity.argtypes = [ctypes.c_int, ctypes.c_size_t]
        for i in range(n_subpools):
            base = self._arena.pool_va_base(self._pool_name(i))
            self._lib.multi_init(i, base, chunk_bytes, self.init_chunks_per_pool)

        # Per-sub-pool MemPool + tensor allocation. SOFT-CAP DESIGN:
        # the C-side allocator's `n_chunks` controls what bump-alloc
        # *would return*; we set it temporarily to max_chunks_per_pool
        # so PyTorch can grab a max-shape segment, then restore it to
        # init_chunks_per_pool. The tensor's data_ptr is at chunk 0,
        # the tensor's shape is (max_tokens, *), but only the first
        # init_tokens rows are physically backed. Engine code respects
        # `current_capacity_tokens()` as the soft ceiling.
        #
        # For this to be safe, the tensor must be allocated with
        # `torch.empty` (no zero-init), since zero-init would touch
        # unbacked VA past init_tokens and fault.
        from torch.cuda.memory import CUDAPluggableAllocator
        self._tensors: List[torch.Tensor] = []
        self._mempools: List[torch.cuda.MemPool] = []
        self._so_path = so_path

        for i in range(n_subpools):
            plug = CUDAPluggableAllocator(
                so_path, f"pool{i}_malloc", f"pool{i}_free")
            mp = torch.cuda.MemPool(allocator=plug.allocator())
            self._mempools.append(mp)
            # Temporarily expose max-chunks so PyTorch's segment grab
            # for the (max_tokens, *) tensor succeeds. This is OK
            # because the over-promised VA past init_chunks is reserved
            # but unmapped; torch.empty doesn't probe.
            self._lib.multi_set_capacity(i, self.max_chunks_per_pool)
            with torch.cuda.use_mem_pool(mp):
                t = torch.empty(
                    (max_tokens, *per_token_shape), dtype=dtype, device="cuda")
            # Restore so subsequent torch.empty inside this MemPool
            # would respect the live capacity. (Not used in practice;
            # only one tensor per sub-pool.)
            self._lib.multi_set_capacity(i, self.init_chunks_per_pool)
            self._tensors.append(t)

        torch.cuda.synchronize()

    # ------------------------------------------------------------------

    def _pool_name(self, i: int) -> str:
        return f"sub{i}"

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

        For this smoke version, growing means we re-allocate each tensor
        at the new shape. (A production version would pre-allocate at
        N_MAX and only need to update the C-side capacity register; that
        requires PyTorch UntypedStorage-from-pointer plumbing we defer.)
        """
        if n_tokens > self.max_tokens:
            raise ValueError(f"n_tokens {n_tokens} > max_tokens {self.max_tokens}")
        if n_tokens % self.tokens_per_chunk != 0:
            raise ValueError(
                f"n_tokens {n_tokens} not a multiple of tokens_per_chunk {self.tokens_per_chunk}"
            )
        new_chunks = n_tokens // self.tokens_per_chunk
        n_subpools = self.n_layers * self.n_kinds

        # Walk each sub-pool and grow/shrink to new_chunks.
        for i in range(n_subpools):
            cur = self._arena.pool_mapped_chunks(self._pool_name(i))
            if new_chunks > cur:
                self._arena.grow(self._pool_name(i), new_chunks - cur)
            elif new_chunks < cur:
                self._arena.shrink(self._pool_name(i), cur - new_chunks)
            self._lib.multi_set_capacity(i, new_chunks)

    def cleanup(self) -> None:
        # PyTorch's caching allocator caches segments backed by our chunks.
        # We must release those segments before the arena unmaps physical
        # handles, or the next caching-allocator touch faults.
        del self._tensors[:]
        del self._mempools[:]
        import gc
        gc.collect()
        torch.cuda.empty_cache()  # force caching allocator to drop our segments
        torch.cuda.synchronize()
        self._arena.cleanup()
