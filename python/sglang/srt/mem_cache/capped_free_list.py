"""CappedFreeList — page-id free list for the dynamic-cap KV arena.

The arena reserves `[1, size]` of VA but only a prefix is physically backed; the
Budgeter grows/shrinks the backed region by trading 2 MiB handles with the mamba
pool. A page-id is therefore allocatable (backed+free), live (backed+in-use), or
capped (unbacked → never handed out). This class owns that bookkeeping; the
kv-cache bytes stay on the allocator.

State — four fields:
    free_ids  sorted* int64 — EVERY free id (allocatable + drained). *iff need_sort.
    pending   int64 — freed-but-not-yet-merged release buffer (need_sort only).
    tail_lo   int  — capped tail is the contiguous `[tail_lo, size]`; `size+1` = none.
              An int, NEVER a tensor: materializing this ~560k-id tail and filtering
              it per alloc/free was the old decode tax.
    marks     small int64 — mid-range ids a drain unmapped; FREE but unbacked, so
              they stay in `free_ids` and `alloc` skips them. Usually empty.
Capped ⟺ `id >= tail_lo or id in marks`; allocatable ⟺ free and not in `marks`.

Fast because the hot path never touches the tail: `free` is a plain append (a
capped page is never live → never freed); `alloc` pops the lowest n free ids,
checking nothing when no drain is in flight, else one small `isin` over `marks`.
Cross-fire `mark`/`unmark` are O(K) (edit `marks` only, no `free_ids` realloc);
only `set_cap`/boot-cap growth rebuild `free_ids`, and those are the gentle tick
path, not per-fire drains.

Concurrency: mutators assume the caller holds the allocator's `_alloc_lock`.
"""

from __future__ import annotations

from typing import Optional

import torch

# `tail_lo` sentinel: tail starts above the id space, i.e. nothing capped at the
# top. A plain int so the hot path compares without reading a tensor.
_NO_TAIL = 1 << 62


class CappedFreeList:
    def __init__(
        self,
        size: int,
        device,
        need_sort: bool,
        boot_cap: Optional[int] = None,
    ) -> None:
        """Build a free list over page-ids `[1, size]` (id 0 is the padded
        sentinel and is never allocatable). `boot_cap` is the highest id
        physically backed at boot; `(boot_cap, size]` starts capped (the grow
        headroom). `boot_cap=None` (or `== size`) means fully backed, no tail.
        """
        self.size = int(size)
        self.device = device
        self.need_sort = bool(need_sort)
        cap = self.size if boot_cap is None else int(boot_cap)
        self.tail_lo = cap + 1 if cap < self.size else _NO_TAIL
        self.marks = torch.empty(0, dtype=torch.int64, device=device)
        self.free_ids = torch.empty(0, dtype=torch.int64, device=device)
        self.pending = torch.empty(0, dtype=torch.int64, device=device)
        self.reset()

    # ------------------------------------------------------------------ #
    # capacity helpers (derived from tail_lo + marks; never scan the tail) #
    # ------------------------------------------------------------------ #
    @property
    def n_capped(self) -> int:
        """How many ids are currently capped = the contiguous tail size plus
        the mid-range marks. Pure integer arithmetic; never builds the tail."""
        tail = 0 if self.tail_lo == _NO_TAIL else (self.size - self.tail_lo + 1)
        return tail + int(self.marks.numel())

    def live(self) -> int:
        """Backed capacity = total id space minus the capped (unbacked) ids.
        This is what the pool-leak checker reads as the pool `total`."""
        return self.size - self.n_capped

    def available(self) -> int:
        """Allocatable count = every free id minus the drained (marked) ones.
        O(1): `marks` is a subset of `free_ids`/`pending`, so a single length
        subtraction; the contiguous tail is excluded by construction."""
        return (
            int(self.free_ids.numel())
            + int(self.pending.numel())
            - int(self.marks.numel())
        )

    # ----------------------------- hot path ---------------------------- #
    def alloc(self, n: int) -> Optional[torch.Tensor]:
        """Hand out `n` allocatable ids (the lowest free ids that are not
        drained), or None if fewer than `n` are available.

        The release buffer is folded into the sorted free list first when a
        request can't be served from `free_ids` alone — either because it is
        short of ids, OR because a drain is in flight (`marks` non-empty): with
        marks present the allocatable count is `free_ids ∪ pending` minus the
        drained ids, so the slice/skip below MUST run over the complete free
        list, not a partial `free_ids` (else a request that `available()` can
        satisfy from `pending` would return a SHORT tensor)."""
        if self.need_sort and (self.marks.numel() or n > self.free_ids.numel()):
            self._merge_pending()
        if n > self.available():
            return None
        head = self.free_ids[:n]
        if self.marks.numel() == 0 or not bool(torch.isin(head, self.marks).any()):
            # No drain in flight, or the lowest n free ids are all allocatable
            # (drained ids are high). Hand out the head directly.
            self.free_ids = self.free_ids[n:]
            return head
        # A drained id is among the lowest n free ids: hand out the lowest n
        # allocatable (non-marked) ids and drop exactly those positions. Relies
        # on `free_ids` having no duplicates (the invariant `_assert_capped_
        # invariant` enforces) — a duplicated marked id would mask both copies
        # and short the result.
        keep_pos = (~torch.isin(self.free_ids, self.marks)).nonzero(as_tuple=True)[0]
        take = keep_pos[:n]
        out = self.free_ids[take]
        drop = torch.zeros(self.free_ids.numel(), dtype=torch.bool, device=self.device)
        drop[take] = True
        self.free_ids = self.free_ids[~drop]
        return out

    def free(self, ids: torch.Tensor) -> None:
        """Return ids to the list. Sorted lists buffer into `pending` (merged
        lazily); unsorted lists append straight to `free_ids`. A capped id can
        never reach here (a marked/tail page is never live).

        The free list is a SET of free slots: returning a slot that is already
        free must be a no-op, not a second copy. The append below would create a
        duplicate if an upstream caller double-frees a slot, which pushes
        `available()` past `live()` and drives the scheduler's usage count
        negative (`#full token` < 0 -> CUDA illegal-access once the inflated
        list hands one physical slot to two requests). Detect that over-add in
        O(1) and restore the set invariant by deduping only on the (rare)
        violation, so the hot path stays append-only (no per-token isin tax)."""
        if ids.numel() == 0:
            return
        ids = _norm(ids, self.device)
        if self.need_sort:
            self.pending = torch.cat((self.pending, ids))
        else:
            self.free_ids = torch.cat((self.free_ids, ids))
        if self.available() > self.live():
            self._dedup_free_union()

    def _dedup_free_union(self) -> None:
        """Restore the free-list set invariant after a detected over-add: drop
        duplicate ids from `free_ids ∪ pending` (a slot returned more than
        once). Gated behind the O(1) `available() > live()` check in `free`, so
        the O(n) unique is never paid on the hot path."""
        merged = torch.cat((self.free_ids, self.pending))
        self.free_ids = torch.unique(merged)  # torch.unique: sorted + deduped
        self.pending = torch.empty(0, dtype=torch.int64, device=self.device)

    def _merge_pending(self) -> None:
        """Fold the release buffer into the sorted free list. Sorts only the
        working set (free + pending), never the 560k tail."""
        if self.pending.numel() == 0:
            return
        merged = torch.cat((self.free_ids, self.pending))
        self.free_ids, _ = torch.sort(merged)
        self.pending = torch.empty(0, dtype=torch.int64, device=self.device)

    def merge(self) -> None:
        """Public merge (used by the allocator's `merge_and_sort_free`)."""
        self._merge_pending()

    # --------------------------- cross-fire ---------------------------- #
    def mark(self, ids: torch.Tensor) -> int:
        """Cap specific free ids (a k2m drain unmapped their chunks). They stay
        in `free_ids` (no realloc); `alloc` skips them. Returns the count newly
        marked. O(K) in the number of marked ids — the per-fire scheduler-thread
        cost stays tiny regardless of how large the free list is."""
        ids = _norm(ids, self.device)
        if ids.numel() == 0:
            return 0
        max_id = int(ids.max())
        if max_id > self.size:
            raise AssertionError(
                f"mark: id {max_id} exceeds ceiling size={self.size}. A capped "
                f"page-id must lie in [1, size]; an out-of-range id corrupts the "
                f"capped accounting (live_size would go negative). Fail-fast at "
                f"the mutation site rather than leak a bad id silently."
            )
        # An id already capped (in the tail, or already marked) is a no-op; dedup
        # keeps `n_capped` honest (a double-mark would undercount `live`).
        fresh = ids[ids < self.tail_lo]
        if self.marks.numel():
            fresh = fresh[~torch.isin(fresh, self.marks)]
        fresh = torch.unique(fresh)
        if fresh.numel() == 0:
            return 0
        self.marks = torch.cat((self.marks, fresh))
        return int(fresh.numel())

    def unmark(self, ids: torch.Tensor) -> int:
        """Un-cap ids (an m2k grow re-mapped their chunks) and return how many
        were actually un-capped. Dispatches by where each id sits — a mid-range
        mark is cleared, a tail id grows the live region — both cheap (the
        cross-fire normally uncaps the lowest-capped ids, arena
        `first_free_slot`)."""
        ids = _norm(ids, self.device)
        if ids.numel() == 0:
            return 0
        max_id = int(ids.max())
        if max_id > self.size:
            raise AssertionError(
                f"unmark: id {max_id} exceeds ceiling size={self.size}. The KV "
                f"arena grew past the allocator page-id space — bound the "
                f"cross-fire grant to max_size. Silently dropping it would "
                f"orphan the mapped chunk (donated handle + HBM leak)."
            )
        ids = torch.unique(ids)
        return (
            self._clear_marks(ids[ids < self.tail_lo])
            + self._grow_into_tail(ids[ids >= self.tail_lo])
        )

    def _clear_marks(self, ids: torch.Tensor) -> int:
        """A previously-drained page is re-mapped: drop it from `marks` (it is
        already in `free_ids`, so it becomes allocatable again). O(K), no
        realloc. Returns the count cleared."""
        if ids.numel() == 0 or self.marks.numel() == 0:
            return 0
        in_marks = torch.isin(self.marks, ids)
        self.marks = self.marks[~in_marks]
        return int(in_marks.sum())

    def _grow_into_tail(self, ids: torch.Tensor) -> int:
        """Grow the live region into the contiguous tail up to `max(ids)+1`: the
        whole exposed range joins the free list, and any id in that range NOT
        being un-capped (a gap, only in the pathological non-prefix case) stays
        capped as a `mark` (kept in the free list → `marks ⊆ free`). Returns the
        count un-capped (= the ids that were in the tail)."""
        if ids.numel() == 0:
            return 0
        new_lo = int(ids.max()) + 1
        exposed = torch.arange(
            self.tail_lo, new_lo, dtype=torch.int64, device=self.device
        )
        still_capped = exposed[~torch.isin(exposed, ids)]
        self._add_to_free(exposed)
        if still_capped.numel():
            self.marks = torch.cat((self.marks, still_capped))
        self.tail_lo = new_lo if new_lo <= self.size else _NO_TAIL
        return int(ids.numel())

    def set_cap(self, n: int) -> None:
        """Set the contiguous backed region to `[1, n]` by moving the tail
        boundary to `n + 1` (the Budgeter's tick-path resize). `tail_lo` is the
        single source of truth for the boundary, so there is no separate cap
        that can go stale against a fire-path grow."""
        n = max(1, min(int(n), self.size))
        new_lo = n + 1 if n < self.size else _NO_TAIL
        old_lo = self.tail_lo
        if new_lo == old_lo:
            return
        old_b = self.size + 1 if old_lo == _NO_TAIL else old_lo
        new_b = self.size + 1 if new_lo == _NO_TAIL else new_lo
        if new_b > old_b:
            # Grow: ids [old_b, new_b) were the tail, now allocatable.
            self.tail_lo = new_lo
            self._add_to_free(
                torch.arange(old_b, new_b, dtype=torch.int64, device=self.device)
            )
        else:
            # Shrink: ids [new_b, old_b) become the tail. Drop them from the free
            # list and from `marks` (the enlarged tail now covers them).
            self.tail_lo = new_lo
            if self.marks.numel():
                self.marks = self.marks[self.marks < new_b]
            if self.free_ids.numel():
                self.free_ids = self.free_ids[self.free_ids < new_b]
            if self.pending.numel():
                self.pending = self.pending[self.pending < new_b]

    def relocate(self, freed: int, taken: int) -> bool:
        """Migration id-swap: `taken` was allocatable and now holds relocated
        bytes (-> live, leaves the free list); `freed` gave its bytes away
        (-> free, joins the list). Returns False if `taken` was not actually
        allocatable."""
        if not self.is_allocatable(taken):
            return False
        marked = self.marks.numel() and bool((self.marks == freed).any())
        if freed >= self.tail_lo or marked or self.is_allocatable(freed):
            raise AssertionError(
                f"relocate: freed id {freed} must be a LIVE backed slot — not "
                f"capped (>= tail_lo={self.tail_lo} or marked) and not already "
                f"free — else free_ids gets a duplicate or reaches the tail."
            )
        self.free_ids = self.free_ids[self.free_ids != taken]
        self.pending = self.pending[self.pending != taken]
        self._add_to_free(_scalar(freed, self.device))
        return True

    # ----------------------------- queries ----------------------------- #
    def is_allocatable(self, id_: int) -> bool:
        """Is this id free AND not drained (i.e. `alloc` could hand it out)?"""
        t = _scalar(id_, self.device)
        in_free = (
            self.free_ids.numel() and bool(torch.isin(t, self.free_ids).any())
        ) or (
            self.pending.numel() and bool(torch.isin(t, self.pending).any())
        )
        if not in_free:
            return False
        return not (
            self.marks.numel() and bool(torch.isin(t, self.marks).any())
        )

    def count_referenced(self, ids: torch.Tensor) -> int:
        """How many of `ids` are a BACKED page that is NOT free — i.e. still
        backing a live/evictable slot? A drained (marked) page counts as free
        (it is not live); a tail id (`>= tail_lo`) is unbacked, not live, so it
        is NOT referenced. The cross-fire `cap_barrier` aborts if this is
        non-zero."""
        ids = _norm(ids, self.device)
        if ids.numel() == 0:
            return 0
        not_free = ~torch.isin(ids, self._free_union())
        backed = ids < self.tail_lo
        return int((not_free & backed).sum())

    def count_reachable(self, ids: torch.Tensor) -> int:
        """How many of `ids` are still ALLOCATABLE — i.e. leaked back into the
        live free path after a cap-barrier (a concurrent alloc could grab one)?
        A properly-marked target is in `free_ids` but `alloc` skips it, so it is
        NOT reachable; an un-marked free target IS. The fire worker aborts
        before any cuMemUnmap if this is non-zero."""
        ids = _norm(ids, self.device)
        if ids.numel() == 0:
            return 0
        free_all = self._free_union()
        if free_all.numel() == 0:
            return 0
        reachable = torch.isin(ids, free_all)
        if self.marks.numel():
            reachable = reachable & (~torch.isin(ids, self.marks))
        return int(reachable.sum())

    def capped_ids(self) -> torch.Tensor:
        """Materialize the full capped set (tail ids + marks). Cold path only
        (owner-map build, leak check): the per-token path never calls this."""
        parts = []
        if self.tail_lo != _NO_TAIL:
            parts.append(
                torch.arange(
                    self.tail_lo, self.size + 1,
                    dtype=torch.int64, device=self.device,
                )
            )
        if self.marks.numel():
            parts.append(self.marks)
        if not parts:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def reset(self) -> None:
        """Reset ALLOCATION state to empty-cache while PRESERVING capacity
        (`tail_lo`, `marks`): a flush clears cached entries but does not unmap
        arena chunks, so the backed region must survive it. The free list
        becomes every backed id `[1, tail_lo)`; the drained `marks` stay marked
        (still skipped by `alloc`)."""
        hi = self.size if self.tail_lo == _NO_TAIL else self.tail_lo - 1
        self.free_ids = torch.arange(
            1, hi + 1, dtype=torch.int64, device=self.device
        )
        self.pending = torch.empty(0, dtype=torch.int64, device=self.device)
        # Drop any stale marks that the live cap no longer covers (a shrink that
        # happened while flushed); keep the rest — those chunks are still
        # unmapped and must stay skipped.
        if self.marks.numel() and self.tail_lo != _NO_TAIL:
            self.marks = self.marks[self.marks < self.tail_lo]

    # --------------------------- internals ----------------------------- #
    def _free_union(self) -> torch.Tensor:
        """The free + release ids as one tensor (cold-path queries)."""
        if self.pending.numel() == 0:
            return self.free_ids
        if self.free_ids.numel() == 0:
            return self.pending
        return torch.cat((self.free_ids, self.pending))

    def _add_to_free(self, ids: torch.Tensor) -> None:
        """Add `ids` to the free list (they became allocatable). Keeps the list
        sorted when `need_sort`, so alloc's head stays the lowest ids."""
        if ids.numel() == 0:
            return
        merged = torch.cat((self.free_ids, ids.to(self.free_ids.dtype)))
        if self.need_sort:
            merged, _ = torch.sort(merged)
        self.free_ids = merged


def _norm(ids: torch.Tensor, device) -> torch.Tensor:
    """Coerce an id tensor onto `device` as int64."""
    return ids.to(device).to(torch.int64)


def _scalar(id_: int, device) -> torch.Tensor:
    """A 1-element int64 tensor for a single id."""
    return torch.tensor([int(id_)], dtype=torch.int64, device=device)
