"""Contract tests for `CappedFreeList` — the isolated capped page-id free list.

Run (CPU, no GPU needed):
    .venv/bin/python dev/interlayer/4_e2e/cc_zero_downside/test_capped_free_list.py

These pin the invariants the arena KV allocator relies on, so the unit can be
verified in isolation. Two kinds of contract:

  CORRECTNESS — the hot path (`alloc`/`free`/`available`/`live`) never hands out
  a capped id and never materializes the tail; the cross-fire path
  (`mark`/`unmark`/`set_cap`) keeps `n_capped`/`live` exact and the allocatable
  set capped-free; the cold queries agree with the state.

  PERFORMANCE TARGET (the whole point of the design) — a cross-pool DRAIN
  (`mark`) and its later RESTORE (`unmark`) are O(K) in the number of drained
  ids and DO NOT reallocate `free_ids`, regardless of how large the free list
  is. This is what keeps the per-fire scheduler-thread cost from spiking decode
  (the old Convention-A design rebuilt the whole free array on every mark). The
  target is pinned by object-identity (a mark must not rebind `free_ids`) plus a
  scaling check (mark on a 500k-free pool costs ~the same as on a 50-free pool).
"""
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch

from sglang.srt.mem_cache.capped_free_list import CappedFreeList, _NO_TAIL

DEV = "cpu"


def _ids(*xs):
    return torch.tensor(list(xs), dtype=torch.int64, device=DEV)


def _set(t):
    return set(int(x) for x in t.tolist())


def _allocatable(fl):
    """The ids `alloc` may hand out = free ids that are not drained (marked)."""
    free = _set(fl.free_ids) | _set(fl.pending)
    return free - _set(fl.marks)


def _capped(fl):
    """Every capped id = the implicit tail plus the explicit marks."""
    tail = set(range(fl.tail_lo, fl.size + 1)) if fl.tail_lo != _NO_TAIL else set()
    return tail | _set(fl.marks)


def _assert_invariants(fl):
    """Global invariants that must hold after every operation."""
    free = _set(fl.free_ids)
    pend = _set(fl.pending)
    marks = _set(fl.marks)
    # the free list (incl. drained marks) never reaches the contiguous tail
    assert all(i < fl.tail_lo for i in free | pend), "free list reached the tail"
    # marks are a subset of the free list (a drained page is free, not live)
    assert marks <= (free | pend), f"marks not subset of free: {marks - (free|pend)}"
    # free_ids / pending disjoint
    assert not (free & pend), "free_ids overlaps pending"
    # allocatable and capped are disjoint (the headline safety property)
    assert not (_allocatable(fl) & _capped(fl)), "allocatable overlaps capped"
    # available counts exactly the allocatable set
    assert fl.available() == len(_allocatable(fl)), (
        f"available()={fl.available()} != |allocatable|={len(_allocatable(fl))}")
    # live = backed capacity = size - capped
    assert fl.live() == fl.size - fl.n_capped
    assert fl.live() == fl.size - len(_capped(fl))
    # sorted invariant
    if fl.need_sort and fl.free_ids.numel() > 1:
        assert bool((fl.free_ids[1:] >= fl.free_ids[:-1]).all()), "free not sorted"


# --------------------------------------------------------------------------- #
# construction                                                                #
# --------------------------------------------------------------------------- #
def test_boot_no_headroom():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True)
    assert fl.tail_lo == _NO_TAIL
    assert fl.n_capped == 0
    assert fl.live() == 8 and fl.available() == 8
    assert _allocatable(fl) == set(range(1, 9))
    _assert_invariants(fl)


def test_boot_with_headroom():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    assert fl.tail_lo == 5
    assert fl.n_capped == 4 and fl.live() == 4 and fl.available() == 4
    assert _allocatable(fl) == {1, 2, 3, 4}
    _assert_invariants(fl)


# --------------------------------------------------------------------------- #
# hot path                                                                    #
# --------------------------------------------------------------------------- #
def test_alloc_pops_head_never_capped():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    got = fl.alloc(3)
    assert _set(got) == {1, 2, 3}
    assert _set(fl.free_ids) == {4}
    assert fl.alloc(2) is None      # only 1 left
    _assert_invariants(fl)


def test_free_merge_roundtrip():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    a = fl.alloc(4)
    assert fl.available() == 0
    fl.free(a)
    assert fl.pending.numel() == 4 and fl.available() == 4
    fl.merge()
    assert _set(fl.free_ids) == {1, 2, 3, 4} and fl.pending.numel() == 0
    _assert_invariants(fl)


def test_alloc_triggers_lazy_merge():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    a = fl.alloc(4)
    fl.free(a)
    got = fl.alloc(4)               # forces merge
    assert _set(got) == {1, 2, 3, 4}
    _assert_invariants(fl)


def test_unsorted_no_pending():
    fl = CappedFreeList(size=8, device=DEV, need_sort=False, boot_cap=4)
    a = fl.alloc(2)
    fl.free(a)
    assert fl.pending.numel() == 0 and fl.available() == 4
    _assert_invariants(fl)


# --------------------------------------------------------------------------- #
# cross-fire: mark / unmark — CORRECTNESS                                     #
# --------------------------------------------------------------------------- #
def test_mark_caps_free_ids_alloc_skips_them():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    n = fl.mark(_ids(7, 8))          # drain the two highest free pages
    assert n == 2
    assert _set(fl.marks) == {7, 8}
    assert fl.live() == 6 and fl.available() == 6
    # alloc must never hand out a drained id, even draining the whole pool
    handed = set()
    while True:
        out = fl.alloc(1)
        if out is None:
            break
        handed |= _set(out)
    assert not (handed & {7, 8}) and handed == {1, 2, 3, 4, 5, 6}
    _assert_invariants(fl)


def test_mark_low_id_slow_path_still_skips():
    # Pathological: drain LOW ids (not the usual high ones). alloc's slow path
    # must still skip them.
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    fl.mark(_ids(1, 2, 3))
    assert fl.available() == 5
    got = fl.alloc(5)
    assert not (_set(got) & {1, 2, 3}) and _set(got) == {4, 5, 6, 7, 8}
    assert fl.alloc(1) is None
    _assert_invariants(fl)


def test_alloc_folds_pending_when_drain_in_flight():
    """When a drain is in flight (marks non-empty) and allocatable ids are
    parked in un-merged `pending`, alloc must fold pending in first — never
    return a SHORT non-None tensor (the engine treats that as a successful
    under-allocation → out-of-bounds KV writes)."""
    for free_ids, pending, marks, want in [
        ([7], [8, 9], [8], {7, 9}),       # allocatable {7,9}
        ([], [3, 4, 5], [3], {4, 5}),     # allocatable {4,5}, free_ids empty
    ]:
        fl = CappedFreeList(size=16, device=DEV, need_sort=True, boot_cap=16)
        fl.free_ids = _ids(*free_ids)
        fl.pending = _ids(*pending) if pending else torch.empty(0, dtype=torch.int64)
        fl.marks = _ids(*marks)
        assert fl.available() == 2, (free_ids, pending, marks, fl.available())
        got = fl.alloc(2)
        assert got is not None and got.numel() == 2, (
            f"alloc(2) returned a SHORT tensor "
            f"{None if got is None else got.tolist()} for free={free_ids} "
            f"pending={pending} marks={marks}")
        assert _set(got) == want, (got.tolist(), want)
        assert not (_set(got) & _set(fl.marks)), "alloc returned a drained id"
        _assert_invariants(fl)


def test_mark_dedup_keeps_live_honest():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    fl.mark(_ids(3))
    assert fl.mark(_ids(3)) == 0     # re-mark is a no-op
    assert fl.live() == 7
    _assert_invariants(fl)


def test_mark_out_of_ceiling_raises():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    try:
        fl.mark(_ids(9))
        assert False, "expected AssertionError for id past ceiling"
    except AssertionError as e:
        assert "ceiling" in str(e) or "size" in str(e)


def test_unmark_mark_roundtrip():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    fl.mark(_ids(3, 5))
    assert fl.unmark(_ids(3, 5)) == 2
    assert fl.marks.numel() == 0
    assert fl.live() == 8 and _allocatable(fl) == set(range(1, 9))
    _assert_invariants(fl)


def test_unmark_tail_prefix_grows_into_headroom():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    assert fl.tail_lo == 5
    n = fl.unmark(_ids(5, 6))        # grow into the two lowest headroom ids
    assert n == 2 and fl.tail_lo == 7
    assert {5, 6} <= _allocatable(fl) and fl.live() == 6
    _assert_invariants(fl)


def test_unmark_noncontiguous_tail_demotes_gap_to_marks():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    n = fl.unmark(_ids(7))           # uncap 7, leave 5,6 capped
    assert n == 1 and fl.tail_lo == 8
    assert _set(fl.marks) == {5, 6}  # the gap becomes mid-range marks
    assert fl.live() == 5 and 7 in _allocatable(fl)
    _assert_invariants(fl)


def test_unmark_past_ceiling_raises():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    try:
        fl.unmark(_ids(9))
        assert False, "expected AssertionError"
    except AssertionError as e:
        assert "ceiling" in str(e)


# --------------------------------------------------------------------------- #
# cross-fire: mark / unmark — PERFORMANCE TARGET (the optimization)           #
# --------------------------------------------------------------------------- #
def test_mark_does_not_realloc_free_ids():
    """THE optimization: a drain `mark` is O(K) and must NOT rebuild `free_ids`
    (the per-fire scheduler-thread cost must not scale with the free list).
    Pinned by object identity — `mark` only edits the tiny `marks` set."""
    fl = CappedFreeList(size=600_000, device=DEV, need_sort=True, boot_cap=600_000)
    snapshot = fl.free_ids
    fl.mark(_ids(599_990, 599_991, 599_992, 599_993))   # drain 4 high pages
    assert fl.free_ids is snapshot, (
        "mark reallocated free_ids — the per-fire cost now scales with the free "
        "list size (the Convention-A regression the design removes)")
    assert _set(fl.marks) == {599_990, 599_991, 599_992, 599_993}


def test_unmark_of_marked_does_not_realloc_free_ids():
    """The RESTORE half: un-capping a previously-drained page is O(K) and must
    not rebuild `free_ids` either (it just clears the mark)."""
    fl = CappedFreeList(size=600_000, device=DEV, need_sort=True, boot_cap=600_000)
    fl.mark(_ids(599_990, 599_991, 599_992, 599_993))
    snapshot = fl.free_ids
    n = fl.unmark(_ids(599_990, 599_991, 599_992, 599_993))
    assert n == 4
    assert fl.free_ids is snapshot, (
        "unmark of a drained page reallocated free_ids — the restore should "
        "only clear the mark")
    assert fl.marks.numel() == 0


def test_mark_unmark_cost_is_independent_of_free_size():
    """Scaling sanity check: drain/restore on a 500k-free pool must cost ~the
    same as on a 50-free pool (O(K), not O(free)). Generous bound (10x) — the
    point is "doesn't scale with the free list", not a tight number."""
    def churn_us(boot):
        fl = CappedFreeList(size=boot, device=DEV, need_sort=True, boot_cap=boot)
        ids = _ids(*range(boot - 8, boot))
        t0 = time.perf_counter_ns()
        for _ in range(200):
            fl.mark(ids)
            fl.unmark(ids)
        return (time.perf_counter_ns() - t0) / 1000 / 200

    small = churn_us(64)
    large = churn_us(500_000)
    assert large < small * 10 + 50, (
        f"mark/unmark scaled with the free list: small={small:.1f}us "
        f"large={large:.1f}us (>10x). The drain is O(free), not O(K).")


# --------------------------------------------------------------------------- #
# cross-fire: set_cap (contiguous tick-path resize)                           #
# --------------------------------------------------------------------------- #
def test_set_cap_grow_exposes_range():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    fl.set_cap(6)
    assert fl.tail_lo == 7 and {5, 6} <= _allocatable(fl) and fl.live() == 6
    _assert_invariants(fl)


def test_set_cap_shrink_hides_range_and_drops_marks():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    fl.mark(_ids(6))
    fl.set_cap(5)                    # tail [6,8] subsumes the mark at 6
    assert fl.tail_lo == 6 and fl.marks.numel() == 0 and fl.live() == 5
    assert all(i <= 5 for i in _allocatable(fl))
    _assert_invariants(fl)


def test_set_cap_full_clears_tail():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    fl.set_cap(8)
    assert fl.tail_lo == _NO_TAIL and fl.n_capped == 0 and fl.live() == 8
    _assert_invariants(fl)


# --------------------------------------------------------------------------- #
# queries                                                                     #
# --------------------------------------------------------------------------- #
def test_count_referenced_and_reachable():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    fl.alloc(3)                                  # 1,2,3 live
    assert fl.count_referenced(_ids(1, 2, 4)) == 2   # 1,2 live; 4 free
    assert fl.count_reachable(_ids(1, 4, 5)) == 2    # 4,5 allocatable; 1 live
    # a drained (marked) target is NOT reachable (alloc skips it)
    fl.mark(_ids(5))
    assert fl.count_reachable(_ids(4, 5)) == 1        # 4 yes, 5 marked → no
    assert fl.count_referenced(_ids(5)) == 0          # 5 is free (not live)
    _assert_invariants(fl)


def test_capped_ids_materializes_tail_plus_marks():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    fl.mark(_ids(2))
    assert _set(fl.capped_ids()) == {2, 5, 6, 7, 8}
    assert fl.capped_ids().numel() == fl.n_capped


# --------------------------------------------------------------------------- #
# reset / relocate                                                            #
# --------------------------------------------------------------------------- #
def test_reset_preserves_capacity_and_marks():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=4)
    fl.mark(_ids(2))
    fl.alloc(2)
    fl.reset()
    assert fl.tail_lo == 5 and _set(fl.marks) == {2}
    assert _allocatable(fl) == {1, 3, 4}        # [1,4] minus the drained 2
    assert fl.live() == 3
    _assert_invariants(fl)


def test_relocate_swaps_live_free():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    src = fl.alloc(1)
    src_id = int(src[0])
    dst_id = int(fl.free_ids[-1])
    assert fl.relocate(freed=src_id, taken=dst_id)
    assert dst_id not in _allocatable(fl)        # now live
    assert src_id in _allocatable(fl)            # now free
    _assert_invariants(fl)


def test_relocate_rejects_nonfree_dst():
    fl = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    fl.alloc(2)                      # 1,2 live
    assert fl.relocate(freed=1, taken=2) is False
    fl2 = CappedFreeList(size=8, device=DEV, need_sort=True, boot_cap=8)
    fl2.mark(_ids(7))                # 7 drained → not allocatable
    assert fl2.relocate(freed=1, taken=7) is False


# --------------------------------------------------------------------------- #
# the headline invariant under churn: alloc NEVER returns a capped id         #
# --------------------------------------------------------------------------- #
def test_alloc_never_returns_capped_under_cross_fire_churn():
    fl = CappedFreeList(size=64, device=DEV, need_sort=True, boot_cap=32)
    live = []
    ops = [
        ("alloc", 5), ("mark", (30, 31)), ("alloc", 4), ("set_cap", 40),
        ("free", None), ("unmark", (30, 31)), ("alloc", 8), ("set_cap", 36),
        ("mark", (3,)), ("alloc", 6), ("unmark", (33, 34, 35)), ("free", None),
        ("mark", (20, 21)), ("alloc", 3), ("unmark", (20, 21)), ("free", None),
    ]
    for op, arg in ops:
        if op == "alloc":
            got = fl.alloc(arg)
            if got is not None:
                assert not (_set(got) & _capped(fl)), f"alloc returned capped: {got}"
                live.extend(got.tolist())
        elif op == "free" and live:
            half = live[: len(live) // 2]
            live = live[len(live) // 2:]
            fl.free(_ids(*half))
        elif op == "mark":
            # cap_barrier contract: only GENUINELY-FREE, currently-allocatable
            # ids may be drained.
            tgt = [i for i in arg if i in _allocatable(fl)]
            if tgt:
                fl.mark(_ids(*tgt))
        elif op == "unmark":
            fl.unmark(_ids(*arg))
        elif op == "set_cap":
            # the budgeter shrinks only over free headroom, never below a live id
            floor = max(live) if live else 1
            fl.set_cap(max(arg, floor))
        _assert_invariants(fl)


def test_available_equals_alloc_capacity_under_churn():
    """The honest-accounting guarantee: at any point, alloc can hand out exactly
    available() ids, none capped."""
    fl = CappedFreeList(size=32, device=DEV, need_sort=True, boot_cap=24)
    fl.alloc(10)
    fl.mark(_ids(20, 21, 22))        # drain 3
    n = fl.available()
    got = fl.alloc(n)
    assert got is not None and got.numel() == n
    assert not (_set(got) & _capped(fl))
    assert fl.alloc(1) is None       # truly exhausted
    _assert_invariants(fl)


def test_double_free_is_idempotent_no_overadd():
    """A slot freed twice must be a no-op, not a second copy: the free list is a
    SET of free ids. An over-add inflates available() past live()/size, which
    drives the scheduler usage count negative (#full token < 0) and eventually
    hands one physical slot to two requests (CUDA illegal-access). Pre-fix the
    double-free left available()==11 > size=8; the set invariant pins it at 8."""
    fl = CappedFreeList(size=8, device=DEV, need_sort=True)
    out = fl.alloc(3)
    assert out is not None and out.numel() == 3
    fl.free(out); fl.merge()
    fl.free(out); fl.merge()                 # double-free of the SAME ids
    assert fl.available() <= fl.size, (fl.available(), fl.size)
    assert fl.available() <= fl.live()
    union = torch.cat((fl.free_ids, fl.pending))
    assert int(torch.unique(union).numel()) == int(union.numel())  # no dup id
    _assert_invariants(fl)


def test_double_free_unsorted_no_overadd():
    """Same set invariant on the unsorted (need_sort=False) free list."""
    fl = CappedFreeList(size=8, device=DEV, need_sort=False)
    out = fl.alloc(3)
    assert out is not None and out.numel() == 3
    fl.free(out)
    fl.free(out)                             # double-free
    assert fl.available() <= fl.size, (fl.available(), fl.size)
    union = torch.cat((fl.free_ids, fl.pending))
    assert int(torch.unique(union).numel()) == int(union.numel())
    _assert_invariants(fl)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"  PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
