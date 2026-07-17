"""CappedFreeList-based KV allocator.

HiMA's TokenToKVPoolAllocator: wraps a CappedFreeList that handles the
cross-pool capped-page state (the implicit tail + drained marks) so
alloc/free never filter — O(1) allocatable-count, no 560k-tensor isin.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.capped_free_list import _NO_TAIL, CappedFreeList

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

_CAPPED_LO_EMPTY = 1 << 62


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
        max_size: Optional[int] = None,
    ):
        # NOTE: does not call super().__init__. All free-list + capped state
        # lives in `self._fl` (a CappedFreeList); the base ctor would set plain
        # `free_pages`/`_capped_pages` attributes that collide with the
        # read-only compatibility properties below. The common fields the base
        # ctor sets are reproduced here.
        ceiling = size if max_size is None else int(max_size)
        if ceiling < size:
            raise ValueError(f"max_size={ceiling} must be >= size={size}")
        self.max_size = ceiling
        self.size = ceiling                  # page-id ceiling (= page space)
        self.page_size = 1
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort
        # Optional synchronous m2k grow callback the BudgetAgent installs (grow
        # KV from idle mamba); consulted by common.py's alloc-fail path before
        # declaring OOM. None on stock sglang / Budgeter off.
        self._kv_grow_hook = None
        self.is_not_in_free_group = True
        self.free_group = []
        # Protects `_fl` against concurrent mutation: the cross-pool actuator's
        # worker thread runs mark/unmark/set_cap on this allocator while the
        # scheduler thread runs alloc/free. ~100 ns acquire, negligible against
        # alloc's per-call cost.
        self._alloc_lock = threading.Lock()
        # The free list, and the WHOLE capped-page state. `boot_cap = size` is
        # the live cap at boot; the reserved grow headroom `(size, ceiling]`
        # starts capped as an IMPLICIT tail (an int boundary, never a ~1.5 M-id
        # tensor — that materialization was the per-token isin tax this design
        # removes). A cross-fire grow raises the boundary; a drain adds a tiny
        # mark. The free list never contains a capped id, so alloc/free never
        # filter — see CappedFreeList.
        self._fl = CappedFreeList(ceiling, device, need_sort, boot_cap=size)
        self.clear()

    # ------------------------------------------------------------------ #
    # Compatibility read accessors. External readers (the pool-leak checker,
    # OwnerProvider, Budgeter telemetry, debug logs) and tests reach into the
    # allocator's free/capped tensors by these names; they now project the
    # CappedFreeList state. Read-only: nothing external writes them (the base
    # ctor that did is bypassed; `restore_state` writes `_fl` directly).
    # ------------------------------------------------------------------ #
    @property
    def free_pages(self) -> torch.Tensor:
        return self._fl.free_ids

    @property
    def release_pages(self) -> torch.Tensor:
        return self._fl.pending

    @property
    def _capped_pages(self) -> torch.Tensor:
        """The full capped set (tail ids + marks), materialized on demand.
        Cold path only — the per-token path never reads this."""
        return self._fl.capped_ids()

    @property
    def _cap(self) -> int:
        """Current live admission cap = the highest backed id. It IS the tail
        boundary, so it tracks cross-fire grows and can never go stale."""
        return self.size if self._fl.tail_lo == _NO_TAIL else self._fl.tail_lo - 1

    @property
    def _capped_lo(self) -> int:
        """Lowest capped id (tail start, or a lower mark). Debug telemetry."""
        lo = self._fl.tail_lo
        if self._fl.marks.numel():
            lo = min(lo, int(self._fl.marks.min()))
        return lo

    @property
    def _n_allocatable(self) -> int:
        """Allocatable id count (= available_size: free ids minus the drained
        marks, plus the release buffer). Debug telemetry only."""
        return self._fl.available()

    @property
    def live_size(self) -> int:
        """Backed page capacity = ceiling − capped. The pool-leak checker reads
        this as the pool `total`."""
        return self._fl.live()

    def clear(self):
        """Reset allocation state to empty-cache while PRESERVING capacity. A
        /flush_cache clears cached entries but does NOT unmap arena chunks, so
        the backed region (boot headroom + any cross-fire grow/shrink/marks)
        must survive it — the free list is rebuilt as `[1, live cap)` minus the
        cross-fire marks. Delegated to CappedFreeList.reset(). Holds
        `_alloc_lock` (the cross-fire worker may be mutating `_fl` concurrently)."""
        with self._alloc_lock:
            self._fl.reset()
            self.is_not_in_free_group = True
            self.free_group = []

    def available_size(self):
        """Allocatable slot count. O(1): the free list already excludes every
        capped id, so there is nothing to subtract (page_size is 1 here)."""
        return self._fl.available()

    def merge_and_sort_free(self):
        with self._alloc_lock:
            self._fl.merge()

    def _merge_and_sort_free_unlocked(self):
        # The base method rebinds `free_pages` (a read-only property here). It is
        # unreachable on this allocator (merge_and_sort_free is overridden), but
        # override it to name the right path if a future base-path call appears.
        raise AssertionError(
            "_merge_and_sort_free_unlocked is a base-allocator method; the arena "
            "allocator merges through self._fl.merge() (see merge_and_sort_free)."
        )

    def backup_state(self):
        return (self._fl.free_ids, self._fl.pending)

    def restore_state(self, state):
        with self._alloc_lock:
            self._fl.free_ids, self._fl.pending = state

    def alloc(self, need_size: int):
        """Hand out `need_size` slots, or None if short. The free list holds
        only allocatable ids, so this is a head-pop with no capped filtering
        and no per-token GPU sync (the merge of released slots is internal)."""
        with self._alloc_lock:
            return self._fl.alloc(need_size)

    def free(self, free_index: torch.Tensor):
        """Return slots to the free list. No capped/ceiling filtering: a capped
        page is never live (the cross-fire cap_barrier caps only genuinely-free
        pages), so it can never reach `free`. Inside a free-group, batch until
        `free_group_end`."""
        if free_index.numel() == 0:
            return
        with self._alloc_lock:
            if self.is_not_in_free_group:
                self._fl.free(free_index)
            else:
                self.free_group.append(free_index)

    # --------------------------- cross-fire ---------------------------- #
    def mark_pages_capped(self, page_indices: torch.Tensor) -> int:
        """Hold page ids out of the free list after the actuator unmapped their
        chunks (k2m drain). Returns the count newly capped."""
        if page_indices.numel() == 0:
            return 0
        with self._alloc_lock:
            return self._fl.mark(page_indices)

    def unmark_pages_capped(self, page_indices: torch.Tensor) -> int:
        """Reverse of `mark_pages_capped`: return ids to the free list after an
        m2k grow re-mapped their chunks. Returns the count un-capped."""
        if page_indices.numel() == 0:
            return 0
        with self._alloc_lock:
            return self._fl.unmark(page_indices)

    def set_capacity_pages(self, n_pages: int) -> None:
        """Restrict the allocator to hand out only ids <= `n_pages` (the
        Budgeter's contiguous tick-path resize). The caller guarantees no
        in-flight alloc references id > n_pages."""
        with self._alloc_lock:
            self._fl.set_cap(n_pages)

    def count_referenced(self, cap_t: torch.Tensor) -> int:
        """How many of `cap_t` are NOT free (still backing a live/evictable
        slot)? The cap_barrier aborts the fire if non-zero."""
        return self._fl.count_referenced(cap_t)

    def count_reachable_capped(self, cap_t: torch.Tensor) -> int:
        """How many of `cap_t` leaked back into the free list after the
        cap-barrier? The fire worker aborts before cuMemUnmap if non-zero."""
        return self._fl.count_reachable(cap_t)

    def can_migrate_slot(self) -> bool:
        return self.page_size == 1 and self._kvcache.can_move_kv_cache()

    def migrate_slot(self, src: int, dst: int) -> bool:
        """Relocate a live KV slot `src` -> `dst` so a fire can free
        `src`'s page. Moves the bytes (kvcache), then swaps free state: `dst`
        leaves the free list (now live, holding src's data) and `src` re-enters
        it (now free). Returns False if `dst` is not allocatable. The caller
        marks `src`'s page in the same scheduler-thread critical section."""
        if self.page_size != 1:
            raise RuntimeError(
                f"migrate_slot is page_size==1-only; this allocator has "
                f"page_size={self.page_size} (page-id space ≠ token-slot space)."
            )
        if src == 0 or dst == 0 or src == dst:
            return False
        if not self._kvcache.can_move_kv_cache():
            raise RuntimeError(
                f"migrate_slot: kvcache {type(self._kvcache).__name__} cannot "
                f"move_kv_cache (MLA pool, or MHA without enable_kv_cache_copy) "
                f"— KV-slot migration is unsupported here."
            )
        with self._alloc_lock:
            if not self._fl.is_allocatable(dst):
                return False
            tgt = torch.tensor([dst], dtype=torch.int64, device=self.device)
            srcv = torch.tensor([src], dtype=torch.int64, device=self.device)
            self._kvcache.move_kv_cache(tgt, srcv)
            return self._fl.relocate(freed=src, taken=dst)

    def _assert_capped_invariant(self) -> None:
        """Test-time structural oracle for the free list. The per-mutation
        fail-fast lives inside CappedFreeList (`mark`/`unmark`/`set_capacity`/
        `relocate` raise at the bad mutation); this is called by the growable-
        allocator tests after each mutation, NOT on the prod hot path — the
        checks below materialize the free union (`torch.unique` + `isin`), too
        heavy to run per free/alloc. Asserts the invariants the free list
        guarantees: (a) the free list never reaches the contiguous capped tail,
        (b) every drained `mark` is itself a free id (a drained page is free,
        just skipped by alloc, so `marks ⊆ free`), and (c) the free list has no
        duplicate id. (a)+(b) keep the allocatable set (free minus marks) disjoint
        from the capped set; (c) keeps `available()` (which subtracts each mark
        once) consistent with the alloc slow path (which masks a marked id by
        value)."""
        fl = self._fl
        for name, ids in (("free", fl.free_ids), ("pending", fl.pending)):
            if ids.numel():
                assert int(ids.max()) < fl.tail_lo, (
                    f"{name} list reaches the capped tail "
                    f"(max={int(ids.max())} >= tail_lo={fl.tail_lo})")
        free_all = fl._free_union()
        assert int(torch.unique(free_all).numel()) == int(free_all.numel()), (
            "duplicate id in the free list (a double-free or bad relocate "
            "injected one; the alloc slow path would short an allocation)")
        if fl.marks.numel():
            assert bool(torch.isin(fl.marks, free_all).all()), (
                "a drained mark is not in the free list (marks must be free ids)")

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
