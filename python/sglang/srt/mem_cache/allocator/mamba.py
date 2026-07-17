"""CappedFreeList-based slot allocator for the Mamba state pool.

Mirrors ``TokenToKVPoolAllocator`` on the KV side: the CappedFreeList is the
SINGLE source of truth for which slots are free, capped (cross-pool reserved),
or live. ``MambaPool`` is storage-only (tensors); all allocation + cap
management lives here.

Slot 0 is reserved (pad row); live slots are [1, live_cap].
"""

from __future__ import annotations

import threading
from typing import Iterator, Optional

import torch

from sglang.srt.mem_cache.capped_free_list import _NO_TAIL, CappedFreeList


class MambaSlotAllocator:
    """CappedFreeList-based mamba slot allocator.

    Architecturally symmetric with ``TokenToKVPoolAllocator``: one object
    owns alloc / free / cap / live_size, and the pool (MambaPool) is
    storage-only.
    """

    def __init__(
        self,
        size: int,
        device: str,
        max_size: Optional[int] = None,
    ):
        ceiling = size if max_size is None else int(max_size)
        if ceiling < size:
            raise ValueError(f"max_size={ceiling} must be >= size={size}")
        self.max_size = ceiling
        self.size = ceiling
        self.device = device
        self._fl = CappedFreeList(ceiling, device, need_sort=False, boot_cap=size)
        self._alloc_lock = threading.Lock()
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    # ---- capacity ----

    @property
    def live_size(self) -> int:
        return self._fl.live()

    def set_capacity(self, n: int) -> None:
        n = max(1, min(n, self.max_size))
        with self._alloc_lock:
            self._fl.set_cap(n)

    # ---- alloc / free ----

    def available_size(self) -> int:
        return self._fl.available()

    def alloc_group_begin(self, num_reqs: int):
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        with self._alloc_lock:
            return self._fl.alloc(need_size)

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        with self._alloc_lock:
            self._fl.free(free_index)

    def clear(self):
        with self._alloc_lock:
            self._fl.reset()
            self._alloc_iter = None

    # ---- cross-pool cap (mark / unmark) ----

    def mark(self, ids: torch.Tensor) -> int:
        with self._alloc_lock:
            return self._fl.mark(ids)

    def unmark(self, ids: torch.Tensor) -> int:
        with self._alloc_lock:
            return self._fl.unmark(ids)

    # ---- compatibility read accessors ----

    @property
    def free_slots(self) -> torch.Tensor:
        return self._fl.free_ids

    @property
    def _capped_slots(self) -> torch.Tensor:
        return self._fl.capped_ids()

    # ---- invariant check (test oracle, not hot path) ----

    def _assert_invariant(self) -> None:
        fl = self._fl
        for name, ids in (("free", fl.free_ids), ("pending", fl.pending)):
            if ids.numel():
                assert int(ids.max()) < fl.tail_lo, (
                    f"{name} list reaches the capped tail "
                    f"(max={int(ids.max())} >= tail_lo={fl.tail_lo})")
        free_all = fl._free_union()
        assert int(torch.unique(free_all).numel()) == int(free_all.numel()), (
            "duplicate id in the free list")
        if fl.marks.numel():
            assert bool(torch.isin(fl.marks, free_all).all()), (
                "a drained mark is not in the free list")

    def count_reachable_capped(self, cap_t: torch.Tensor) -> int:
        with self._alloc_lock:
            return self._fl.count_reachable(cap_t)
