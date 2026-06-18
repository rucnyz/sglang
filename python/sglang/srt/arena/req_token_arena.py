"""VA-stable backing for per-request bookkeeping tensors.

Wraps SharedHandlePool + ChunkArena + tensor_from_va to give a torch
tensor whose `data_ptr()` is stable across grow/shrink. Used by
ReqToTokenPool (and friends) to support dynamic admission cap without
invalidating CUDA-captured Triton kernels that bake in the tensor
pointer (see `dev/interlayer/dyn_admission_cap/audit_cuda_graphs.md`).

Design: caller specifies `max_bytes` at construction → arena reserves
that much contiguous VA (free per D5). Caller then calls
`set_mapped_bytes(n)` to physically back any prefix of that range.
The tensor returned by `as_tensor` aliases the full VA via
`at::from_blob`; rows beyond `mapped_bytes` are unmapped and faulting
on access — caller is responsible for bounds (e.g. ReqToTokenPool's
`free_slots` only hands out ids in `[0, mapped_rows)`).
"""
import logging
from typing import Optional, Sequence

import torch

from sglang.srt.arena.chunk_arena import ChunkArena, SharedHandlePool
from sglang.srt.arena.from_blob_ext import tensor_from_va

logger = logging.getLogger(__name__)


class ReqTokenVAArena:
    """VA-stable backing arena for a per-request bookkeeping tensor."""

    def __init__(
        self,
        max_bytes: int,
        device_id: int,
        *,
        chunk_bytes: int = 2 * 1024 * 1024,
        shared_pool: Optional[SharedHandlePool] = None,
        pool_name: str = "req_token",
    ):
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
        if chunk_bytes <= 0 or chunk_bytes & (chunk_bytes - 1) != 0:
            raise ValueError(
                f"chunk_bytes must be positive power of 2, got {chunk_bytes}"
            )
        self.chunk_bytes = chunk_bytes
        self.device_id = device_id
        # Round max_bytes up to chunk boundary.
        n_chunks_max = (max_bytes + chunk_bytes - 1) // chunk_bytes
        self.n_chunks_max = n_chunks_max
        self.max_bytes = n_chunks_max * chunk_bytes
        self._pool_name = pool_name

        # ChunkArena reserves VA + manages handles. We use a single pool.
        # `n_handles` is ignored when an `external_handle_pool` is provided
        # (cross-pool mode); otherwise create our own pool of n_chunks_max
        # handles.
        self._arena = ChunkArena(
            device_id=device_id,
            chunk_size=chunk_bytes,
            n_handles=n_chunks_max if shared_pool is None else 0,
            pool_capacities=[(pool_name, n_chunks_max)],
            external_handle_pool=shared_pool,
        )
        self._va_base = self._arena.pool_va_base(pool_name)
        self._mapped_chunks = 0
        logger.info(
            "ReqTokenVAArena[%s]: device=%d va_base=0x%x chunk_bytes=%d "
            "n_chunks_max=%d max_bytes=%d MiB",
            pool_name, device_id, self._va_base, chunk_bytes,
            n_chunks_max, self.max_bytes >> 20,
        )

    # -- public API ---------------------------------------------------------

    @property
    def mapped_bytes(self) -> int:
        return self._mapped_chunks * self.chunk_bytes

    @property
    def mapped_chunks(self) -> int:
        return self._mapped_chunks

    def set_mapped_bytes(self, n_bytes: int) -> None:
        """Grow / shrink physical mapping so AT LEAST n_bytes are backed.

        Rounded up to chunk boundary. Shrink unmaps from the tail
        (highest VA first). VA reservation unchanged. Tensor data_ptr
        unchanged.
        """
        if n_bytes < 0 or n_bytes > self.max_bytes:
            raise ValueError(
                f"set_mapped_bytes: n_bytes={n_bytes} out of range "
                f"[0, {self.max_bytes}]"
            )
        needed_chunks = (n_bytes + self.chunk_bytes - 1) // self.chunk_bytes
        if needed_chunks > self._mapped_chunks:
            n_grow = needed_chunks - self._mapped_chunks
            # grow() returns list[int] of mapped slot IDs; this path
            # only needs the count.
            grew = len(self._arena.grow(self._pool_name, n_grow))
            self._mapped_chunks += grew
            if grew != n_grow:
                raise RuntimeError(
                    f"ReqTokenVAArena: partial grow ({grew}/{n_grow}); "
                    f"handle pool exhausted?"
                )
        elif needed_chunks < self._mapped_chunks:
            n_shrink = self._mapped_chunks - needed_chunks
            shrunk = self._arena.shrink(self._pool_name, n_shrink)
            self._mapped_chunks -= shrunk

    def as_tensor(
        self, dtype: torch.dtype, shape: Sequence[int]
    ) -> torch.Tensor:
        """Return a torch tensor aliasing the arena's full VA range.

        `data_ptr()` is the arena's VA base — stable for the arena's
        lifetime. The tensor sees all `max_bytes` worth of rows, but
        rows past `mapped_bytes` are unmapped. Caller is responsible
        for not reading/writing them (e.g. via a free-slots gate).
        """
        elem_bytes = torch.empty((), dtype=dtype).element_size()
        n_elems = 1
        for d in shape:
            n_elems *= d
        bytes_needed = n_elems * elem_bytes
        if bytes_needed > self.max_bytes:
            raise ValueError(
                f"as_tensor: shape {tuple(shape)} dtype {dtype} needs "
                f"{bytes_needed} bytes > max_bytes {self.max_bytes}"
            )
        return tensor_from_va(
            va=self._va_base,
            sizes=shape,
            dtype=dtype,
            device_index=self.device_id,
        )

    def data_ptr(self) -> int:
        return self._va_base

    def cleanup(self) -> None:
        """Release all mapped chunks + the underlying VA reservation."""
        self._arena.cleanup()
