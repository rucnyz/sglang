"""Behavioral regression for the arena allocator's capped-page handling.

`TokenToKVPoolAllocator` delegates all free-list + capped-page bookkeeping to a
`CappedFreeList` (`self._fl`). The free list holds ONLY allocatable ids — a
capped id is never in it — so the per-decode-step `alloc`/`free` path does zero
`torch.isin` and zero GPU sync over the reserved headroom (the ~600µs/step tax
the old Convention-A design paid; see README §2026-06-08 / §2026-06-12). The
CappedFreeList contract is unit-tested in `test_capped_free_list.py`; THIS file
pins the same guarantees at the allocator-integration level (the real
`TokenToKVPoolAllocator`, the cross-fire API the actuator calls):

  (A) CORRECTNESS — alloc NEVER returns a capped page, under both capped
      layouts (high-tail headroom AND an arbitrary mark_pages_capped id) and
      across a full cross-fire sequence.
  (B) NO PER-TOKEN ISIN — alloc/free do not call torch.isin (the removed tax).
  (C) ACCOUNTING — available_size / live_size track every cap mutation
      (set_capacity grow/shrink, mark/unmark) exactly and never go negative.
  (D) #320 — the design eliminates the stale-`_cap` bug CLASS: `_cap` is the
      live tail boundary (never stale after an m2k grow), and a referenced page
      is caught by `count_referenced` before it can ever be capped.

Run: .venv/bin/python dev/interlayer/4_e2e/cc_zero_downside/test_capped_lo_fastpath.py
"""
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.capped_free_list import _NO_TAIL

DEV = "cpu"


def _mk(size=8, max_size=16, need_sort=False):
    # size = live cap, max_size = page-id ceiling. Headroom (size, max_size]
    # boots capped (the dynamic-cap arena pattern). The real arena KV allocator
    # runs need_sort=True (sorted free list).
    return TokenToKVPoolAllocator(
        size=size, dtype=torch.bfloat16, device=DEV, kvcache=None,
        need_sort=need_sort, max_size=max_size,
    )


def _capped_set(a):
    return set(a._capped_pages.tolist())


def _free_set(a):
    return set(a.free_pages.tolist()) | set(a.release_pages.tolist())


def _assert_no_capped(out, a, where):
    if out is None:
        return
    bad = set(out.tolist()) & _capped_set(a)
    assert not bad, f"{where}: alloc returned CAPPED pages {bad}"


def _allocatable_set(a):
    # The ids alloc may hand out = free ids that are NOT drained (capped). A
    # drained page stays in free_pages (Convention A) but is in _capped_pages
    # too, so it is excluded here.
    return _free_set(a) - _capped_set(a)


def _check_consistent(a, where):
    """The cross-cutting invariant after every mutation: the ALLOCATABLE set
    (free ids minus the drained/capped ones) is disjoint from the capped set;
    available counts exactly the allocatable set; live = ceiling − capped.
    (Replaces the old `_n_allocatable == searchsorted` counter, which no longer
    exists.)"""
    capped = _capped_set(a)
    allocatable = _allocatable_set(a)
    assert not (allocatable & capped), (
        f"{where}: allocatable ∩ capped = {allocatable & capped}")
    assert a.available_size() == len(allocatable), (
        f"{where}: available_size={a.available_size()} != |allocatable|={len(allocatable)}")
    assert a.live_size == a.size - len(capped), (
        f"{where}: live_size={a.live_size} != size−capped={a.size - len(capped)}")
    assert a.available_size() >= 0, f"{where}: available_size negative"


# --------------------------------------------------------------------------- #
# (A) correctness: alloc never returns a capped page                          #
# --------------------------------------------------------------------------- #
def test_alloc_never_returns_capped_tail():
    a = _mk(8, 16)
    seen = set()
    for _ in range(8):                       # exhaust the 8 live pages
        out = a.alloc(1)
        _assert_no_capped(out, a, "alloc-tail")
        if out is not None:
            seen |= set(out.tolist())
    assert seen == set(range(1, 9)), seen    # none from the capped [9..16]
    assert a.alloc(1) is None, "handed out a capped page at exhaustion"


def test_alloc_never_returns_capped_arbitrary_mark():
    a = _mk(16, 16)                          # no headroom; full live
    a.mark_pages_capped(torch.tensor([3, 4, 5], dtype=torch.int64, device=DEV))
    _check_consistent(a, "after-mark")
    handed = set()
    for _ in range(20):
        out = a.alloc(2)
        _assert_no_capped(out, a, "alloc-arbitrary-mark")
        if out is not None:
            handed |= set(out.tolist())
    assert not (handed & {3, 4, 5}), handed


def test_init_and_clear_preserve_capacity():
    a = _mk(8, 16)
    assert _capped_set(a) == set(range(9, 17)), _capped_set(a)  # boot headroom
    _check_consistent(a, "init")
    a.alloc(3)
    a.clear()                                # flush: capacity preserved
    assert _capped_set(a) == set(range(9, 17)), _capped_set(a)
    assert a.available_size() == 8           # free list rebuilt to full live cap
    _check_consistent(a, "clear")


def test_capacity_across_set_capacity_and_unmark():
    a = _mk(8, 16)                           # capped [9..16]
    a.set_capacity_pages(12)                 # grow: uncap (8,12] → capped [13..16]
    assert _capped_set(a) == set(range(13, 17))
    assert a.live_size == 12
    _check_consistent(a, "grow")
    a.set_capacity_pages(10)                 # shrink: cap (10,12] → capped [11..16]
    assert _capped_set(a) == set(range(11, 17))
    assert a.live_size == 10
    _check_consistent(a, "shrink")
    a.unmark_pages_capped(torch.tensor([11, 12], dtype=torch.int64, device=DEV))
    assert {11, 12} <= _free_set(a)
    _check_consistent(a, "unmark")
    # empty-capped path
    b = _mk(16, 16)
    assert _capped_set(b) == set()
    assert b._fl.tail_lo == _NO_TAIL
    _check_consistent(b, "empty")


# --------------------------------------------------------------------------- #
# (B) no per-token isin                                                       #
# --------------------------------------------------------------------------- #
def test_alloc_does_no_isin():
    # The whole point of the redesign: with a capped tail present, alloc still
    # does ZERO torch.isin (the free list has no capped id to filter).
    for need_sort in (False, True):
        a = _mk(8, 16, need_sort=need_sort)
        calls = {"n": 0}
        orig = torch.isin

        def counting(*args, **kw):
            calls["n"] += 1
            return orig(*args, **kw)

        torch.isin = counting
        try:
            out = a.alloc(4)                 # head [1..4], capped tail [9..16]
        finally:
            torch.isin = orig
        _assert_no_capped(out, a, f"no-isin ns={need_sort}")
        assert calls["n"] == 0, (
            f"need_sort={need_sort}: alloc called torch.isin {calls['n']}x — the "
            f"O(capped) scan is back (the per-step arena tax)")


def test_free_does_no_isin():
    # free() returns slots with no capped filtering — zero isin per slot-free
    # (this was the ~370µs/free tax that halved decode throughput).
    a = _mk(8, 16, need_sort=True)
    held = a.alloc(4)
    calls = {"n": 0}
    orig = torch.isin

    def counting(*args, **kw):
        calls["n"] += 1
        return orig(*args, **kw)

    torch.isin = counting
    try:
        a.free(held)
    finally:
        torch.isin = orig
    assert calls["n"] == 0, f"free called torch.isin {calls['n']}x — the removed tax"


# --------------------------------------------------------------------------- #
# (C) accounting                                                              #
# --------------------------------------------------------------------------- #
def test_available_tracks_alloc_free_reclaim():
    a = _mk(8, 16, need_sort=True)            # capped tail [9..16]
    assert a.available_size() == 8
    a.alloc(3)
    assert a.available_size() == 5
    _check_consistent(a, "after-alloc")
    a.free(torch.tensor([1, 2], dtype=torch.int64, device=DEV))  # → release
    assert a.available_size() == 7           # release counts as available
    a.set_capacity_pages(12)                 # grow
    assert a.available_size() == 11          # +4 uncapped, +2 released, 5 free
    a.clear()
    assert a.available_size() == 12          # free reset to full live cap
    _check_consistent(a, "after-clear")


def test_need_sort_reclaims_released_below_cap():
    # need_sort=True + capped tail: freed ids buffer in release_pages and MUST
    # be reclaimed via the internal merge when the free list is exhausted —
    # alloc must not starve while available_size > 0.
    a = _mk(4, 8, need_sort=True)            # live [1..4], capped tail [5..8]
    assert a.available_size() == 4
    b1 = a.alloc(2)
    b2 = a.alloc(2)                          # consume all 4 live ids
    assert b1 is not None and b2 is not None
    assert a.available_size() == 0
    assert a.alloc(1) is None, "handed out a capped page / phantom slot"
    a.free(torch.tensor([1, 2], dtype=torch.int64, device=DEV))  # → release
    assert a.available_size() == 2
    out = a.alloc(2)                         # MUST reclaim [1,2], not starve
    _assert_no_capped(out, a, "reclaim")
    assert out is not None, "alloc starved: released pages were not reclaimed"
    assert set(out.tolist()) == {1, 2}, out.tolist()
    _check_consistent(a, "after-reclaim")


def test_fragmented_capped_stays_allocatable():
    # A non-tail capped id (cross-fire mark) does NOT block the free ids around
    # it: every non-capped id stays allocatable (the old design under-counted
    # them behind the lowest cap; the free list now holds them directly).
    a = _mk(16, 16, need_sort=True)           # full live, no headroom
    a.mark_pages_capped(torch.tensor([3], dtype=torch.int64, device=DEV))
    assert a.available_size() == 15           # all but the marked id
    _check_consistent(a, "frag-mark")
    handed = set()
    for _ in range(16):
        out = a.alloc(1)
        _assert_no_capped(out, a, "frag-alloc")
        _check_consistent(a, "frag-after-alloc")
        if out is not None:
            handed |= set(out.tolist())
    assert 3 not in handed, handed
    assert handed == set(range(1, 17)) - {3}, handed  # every non-capped id reachable


def test_unmark_grow_tracks_available_no_orphan():
    # #319/#320 at the allocator unit level: an m2k cross-fire grow takes the
    # `unmark_pages_capped` path. After it uncaps N pages they MUST become
    # allocatable — available_size += N, live_size += N, alloc can reach them.
    # If they uncapped but didn't enter the free list they would orphan (#319
    # leak / #320 alloc-None-despite-room).
    a = _mk(4, 8, need_sort=True)            # live [1..4], capped tail [5..8]
    assert a.available_size() == 4 and a.live_size == 4
    a.unmark_pages_capped(torch.tensor([5, 6], dtype=torch.int64, device=DEV))
    assert a.live_size == 6, f"live_size did not track unmark grow: {a.live_size}"
    assert a.available_size() == 6, (
        f"unmark orphaned the grown pages: available={a.available_size()} != 6")
    _check_consistent(a, "after-unmark-grow")
    out = a.alloc(6)
    assert out is not None, "alloc starved despite available_size==6 after grow"
    _assert_no_capped(out, a, "unmark-grow-alloc")
    assert a.alloc(1) is None, "handed out beyond the uncapped live cap"


def test_available_size_correct_when_capping_free_and_guard_catches_inflight():
    """available_size() stays correct and non-negative under the cap contract:
    only GENUINELY-FREE pages may be drained. Capping the free pages drops
    available by exactly that count. What keeps a LIVE (in-flight) page from
    ever being capped — which WOULD corrupt the accounting (the #319 crash) — is
    the cap_barrier's `count_referenced` guard, which this test pins flags the
    in-flight pages so the fire aborts."""
    a = _mk(8, 8, need_sort=True)            # live [1..8], no capped tail
    held = a.alloc(6)                        # 6 live, 2 free
    assert held is not None
    # The guard: every in-flight page is reported referenced (the fire aborts
    # rather than cap it — capping a live page is the #319 contract violation).
    assert a.count_referenced(held) == 6, a.count_referenced(held)
    # The legal cap (the 2 free pages) drops available correctly, never negative.
    free_now = a.free_pages.clone()
    a.mark_pages_capped(free_now)
    assert a.available_size() == 0, a.available_size()
    assert a.available_size() >= 0
    _check_consistent(a, "free-cap")


# --------------------------------------------------------------------------- #
# (A) + (C): full cross-fire sequence                                         #
# --------------------------------------------------------------------------- #
def test_need_sort_cross_fire_sequence():
    # The arena runs need_sort=True, so a full fire sequence (set_capacity
    # grow/shrink, mark/unmark) interleaved with alloc/free must keep the free
    # list capped-free and the accounting exact at every step.
    a = _mk(8, 16, need_sort=True)            # live [1..8], capped tail [9..16]
    _check_consistent(a, "ns-fire-init")
    a.alloc(3)                                # [1,2,3]
    a.set_capacity_pages(12)                  # GROW → uncap (8,12]
    _check_consistent(a, "ns-fire-grow")
    h = a.alloc(4)                            # now allowed to reach into [9..12]
    _assert_no_capped(h, a, "ns-fire-grow-alloc")
    a.set_capacity_pages(10)                  # SHRINK → cap (10,12]
    _check_consistent(a, "ns-fire-shrink")
    a.free(torch.tensor([1, 2, 3], dtype=torch.int64, device=DEV))  # → release
    # Drain caps only GENUINELY-FREE pages (the cap_barrier contract): 4..7 are
    # live, so cap two free headroom ids 8,9.
    a.mark_pages_capped(torch.tensor([8, 9], dtype=torch.int64, device=DEV))
    _check_consistent(a, "ns-fire-mark")
    handed = set()
    for _ in range(12):                       # drain; forces merge (reclaim 1,2,3)
        out = a.alloc(2)
        _assert_no_capped(out, a, "ns-fire-drain")
        _check_consistent(a, "ns-fire-drain")
        if out is not None:
            handed |= set(out.tolist())
    assert not (handed & {8, 9}), handed      # drained ids never handed out
    a.unmark_pages_capped(torch.tensor([8, 9], dtype=torch.int64, device=DEV))
    _check_consistent(a, "ns-fire-unmark")


# --------------------------------------------------------------------------- #
# (D) #320: the stale-cap bug CLASS is eliminated                            #
# --------------------------------------------------------------------------- #
def test_base_init_does_not_read_size_property():
    """Regression: the BASE allocator's init must NOT read `self.size`.
    `SWATokenToKVPoolAllocator` exposes `size` as a @property whose backing it
    sets only AFTER `super().__init__()` returns, so a base init that read
    `self.size` would AttributeError. (The arena `TokenToKVPoolAllocator` does
    not subclass-init this way, but the base contract other allocators rely on
    must hold.)"""

    class _SizeAfterSuper(BaseTokenToKVPoolAllocator):
        def __init__(self, size):
            super().__init__(size, 1, torch.bfloat16, DEV, None, False)
            self._late_size = size

        @property
        def size(self):
            return self._late_size

        def clear(self):
            pass

        def alloc(self, need_size):
            return None

        def free(self, free_index):
            pass

    a = _SizeAfterSuper(16)  # must not raise AttributeError
    assert a.size == 16


def test_count_referenced_free_vs_allocated_320():
    """#320 free-only-cap guard: `count_referenced(cap_t)` reports any cap
    target still backing an allocated slot (absent from free + release) —
    capping such a page would strand its cache. Genuinely-free targets report
    0. This is the gate `cap_barrier` checks before the irreversible mark."""
    a = _mk(16, 16, need_sort=True)            # free = [1..16]
    alloced = a.alloc(4)
    assert alloced is not None and alloced.numel() == 4
    alloced_ids = sorted(alloced.tolist())
    free_ids = sorted(a.free_pages.tolist())
    assert a.count_referenced(alloced.clone()) == 4
    some_free = torch.tensor(free_ids[:4], dtype=torch.int64, device=DEV)
    assert a.count_referenced(some_free) == 0
    mixed = torch.tensor(
        alloced_ids[:2] + free_ids[:2], dtype=torch.int64, device=DEV
    )
    assert a.count_referenced(mixed) == 2
    assert a.count_referenced(torch.empty(0, dtype=torch.int64, device=DEV)) == 0


def test_referenced_page_guard_is_mandatory_320():
    """The free path trusts that a capped page is NEVER live (it does no
    per-free capped filter — that was the removed tax). That trust is upheld by
    `cap_barrier`, which calls `count_referenced` and aborts the fire if any
    cap target is still referenced. This test pins the guard: a live page is
    correctly flagged, so the fire never caps it. (Capping a live page would be
    a contract violation that corrupts the free list — which is exactly why the
    guard, not a defensive free-time filter, is the mechanism.)"""
    a = _mk(16, 16, need_sort=True)
    victim = a.alloc(4)[:1].clone()              # one referenced (live) page
    assert a.count_referenced(victim) == 1, (
        "count_referenced failed to flag a live page — cap_barrier would let a "
        "fire cap it and strand its cache")
    # A genuinely-free page (the contract-honoring case) reports 0.
    free_id = torch.tensor([sorted(a.free_pages.tolist())[0]],
                           dtype=torch.int64, device=DEV)
    assert a.count_referenced(free_id) == 0


def test_grown_page_cap_tracks_no_stale_cap_320():
    """#320 bug CLASS eliminated. An m2k cross-fire grows the live KV cap by
    un-capping headroom pages via `unmark_pages_capped`. In the old design this
    did NOT bump `self._cap` (only the set_capacity SHRINK path did), so `_cap`
    went STALE and `free()`'s `> _cap → re-cap` branch wrongly re-capped freed
    pages in the grown region (the #320 residual crash). The new design has NO
    separate cap field — `_cap` IS the live tail boundary, so it tracks the
    grow exactly and there is no re-cap branch at all: a freed grown page just
    rejoins the free list."""
    a = _mk(8, 16, need_sort=True)            # boot live cap 8, headroom [9,16]
    assert a._cap == 8
    a.unmark_pages_capped(torch.tensor([9], dtype=torch.int64, device=DEV))
    assert 9 not in _capped_set(a)
    assert a._cap == 9, (
        f"_cap did not track the m2k grow (got {a._cap}); the old stale-cap "
        f"trap would leave it at 8")
    # The engine allocates the grown slot (9 is now a live, mapped page) and
    # later evicts the cache on it → free([9]). It must rejoin the free list,
    # NOT get re-capped (the old `> stale _cap` branch did exactly that).
    held = a.alloc(9)                         # consume [1..9], 9 now live
    assert held is not None and 9 in set(held.tolist())
    a.free(torch.tensor([9], dtype=torch.int64, device=DEV))
    a.merge_and_sort_free()
    assert 9 not in _capped_set(a), (
        f"freed grown page 9 was RE-CAPPED: capped={sorted(_capped_set(a))}")
    assert 9 in _free_set(a), "freed grown page 9 did not rejoin the free list"
    _check_consistent(a, "grown-free")


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:
                fails += 1
                print("FAIL", name, "->", e)
                traceback.print_exc()
    print(f"\n{fails} FAILED" if fails else "\nALL PASS")
    sys.exit(1 if fails else 0)
