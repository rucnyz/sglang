"""Crash-boundary coverage driven by the REAL cross-pool cap path.

The fast-path / perf tests run with a hand-set `_capped_slots`; this file drives
the PRODUCTION cap/restore path the cross-pool actuator uses on an m2k fire
(`_MambaCapAllocator.mark_pages_capped` / `unmark_pages_capped`), so the
#312/#329 unmapped-VA guard and the #327 flush-boundary preservation are pinned
against the code that actually fires, not a stand-in.

  1. mark -> `_no_cross_fire` flips False and marked slots leave `free_slots`:
     while any chunk is unmapped the fast path is OFF (so `free` filters).
  2. free of a still-capped slot is DROPPED, never returned to `free_slots`
     (the #312/#329 crash guard), exercised through real cap state.
  3. unmark (the fire fully reverses) -> marked slots return to `free_slots` and
     `_no_cross_fire` flips back True: the fast path RE-ENABLES only after every
     chunk is re-mapped (the boundary the perf/fast-path tests never cross).
  4. #327: clear() (flush_cache) MUST preserve below-cap actuator marks;
     rebuilding `_capped_slots` from `self.size` alone drops them and the next
     alloc hands out unmapped VA -> illegal memory at the next-rep start.

Builds the real `MambaPool` via the shared `_build_pool` (real geometry, CPU).
Run: .venv/bin/python dev/interlayer/3_budgeter/mamba_pool_perf/test_cap_cycle_real_actuator.py
"""
import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from test_mamba_pool_invariants import _build_pool  # noqa: E402  real MambaPool
from sglang.srt.arena.mamba_actuator import _MambaCapAllocator  # noqa: E402

DEVICE = "cpu"


def _ids(*xs):
    return torch.tensor(list(xs), dtype=torch.int64, device=DEVICE)


def test_1_mark_disables_fastpath():
    """An m2k donate caps live-range slots via the real actuator: the fast path
    must turn OFF and the marked (unmapped) slots must leave free_slots."""
    pool = _build_pool(per_layer=True, size=8, max_size=8)
    assert pool._no_cross_fire is True
    cap = _MambaCapAllocator(pool)
    cap.mark_pages_capped(_ids(5, 6, 7, 8))
    assert pool._no_cross_fire is False, "fast path must be OFF while chunks unmapped"
    assert set(pool.free_slots.tolist()).isdisjoint({5, 6, 7, 8}), \
        "marked (unmapped) slots still allocatable"
    assert set(pool._capped_slots.tolist()) == {5, 6, 7, 8}
    print("  PASS  1  real-actuator mark disables fast path; marked slots leave free")


def test_2_free_capped_slot_is_dropped():
    """A request on a capped slot finishes and frees it; its chunk is unmapped,
    so free must DROP it (the #312/#329 guard), via real cap state."""
    pool = _build_pool(per_layer=True, size=8, max_size=8)
    _MambaCapAllocator(pool).mark_pages_capped(_ids(7))
    pool.free(_ids(7))
    assert 7 not in pool.free_slots.tolist(), "capped slot 7 handed back -> #312/#329 crash"
    print("  PASS  2  free of a real-capped slot is dropped, not returned")


def test_3_unmark_reenables_fastpath():
    """The fire fully reverses (chunks re-mapped via unmark): the fast path must
    RE-ENABLE only once every capped slot is restored -- the boundary the
    fast-path/perf tests never cross."""
    pool = _build_pool(per_layer=True, size=8, max_size=8)
    cap = _MambaCapAllocator(pool)
    cap.mark_pages_capped(_ids(5, 6, 7, 8))
    assert pool._no_cross_fire is False
    cap.unmark_pages_capped(_ids(5, 6))            # partial restore: still capped
    assert pool._no_cross_fire is False, "fast path must stay OFF while any chunk unmapped"
    cap.unmark_pages_capped(_ids(7, 8))            # full restore
    assert pool._no_cross_fire is True, "fast path must RE-ENABLE after full restore"
    assert set(pool.free_slots.tolist()) == set(range(1, 9))
    print("  PASS  3  unmark (fire reverses) re-enables the fast path only when fully restored")


def test_4_clear_preserves_below_cap_marks_327():
    """#327: after an m2k donate caps below-cap slots, flush_cache (clear) must
    keep them capped. Rebuilding _capped_slots from self.size drops them and the
    next alloc hands out unmapped VA -> illegal memory at the next-rep start."""
    pool = _build_pool(per_layer=True, size=8, max_size=8)
    _MambaCapAllocator(pool).mark_pages_capped(_ids(5, 6, 7, 8))   # chunks unmapped, id <= size
    pool.clear()                                                   # flush_cache
    capped = set(pool._capped_slots.tolist())
    free = set(pool.free_slots.tolist())
    assert {5, 6, 7, 8} <= capped, f"clear() dropped actuator marks (#327): capped={sorted(capped)}"
    assert free.isdisjoint({5, 6, 7, 8}), \
        f"clear() returned unmapped slots to free (#327): {sorted(free & {5, 6, 7, 8})}"
    assert pool._no_cross_fire is False, "clear() wrongly re-enabled fast path with unmapped slots live"
    print("  PASS  4  clear() preserves below-cap actuator marks (#327)")


def test_5_clear_preserves_marks_with_boot_deferred_tail():
    """#327 with a dynamic-cap pool (size < max_size): clear() must preserve BOTH
    the below-cap marks AND the boot-deferred tail (size, max_size]."""
    pool = _build_pool(per_layer=True, size=6, max_size=8)   # tail {7,8} capped at boot
    assert set(pool._capped_slots.tolist()) == {7, 8}
    _MambaCapAllocator(pool).mark_pages_capped(_ids(5, 6))   # below-cap marks
    pool.clear()
    capped = set(pool._capped_slots.tolist())
    assert {5, 6, 7, 8} <= capped, f"clear() lost marks or tail (#327): capped={sorted(capped)}"
    assert set(pool.free_slots.tolist()).isdisjoint({5, 6, 7, 8})
    print("  PASS  5  clear() preserves below-cap marks + boot-deferred tail")


def main():
    tests = [
        test_1_mark_disables_fastpath,
        test_2_free_capped_slot_is_dropped,
        test_3_unmark_reenables_fastpath,
        test_4_clear_preserves_below_cap_marks_327,
        test_5_clear_preserves_marks_with_boot_deferred_tail,
    ]
    print(f"\nReal-actuator cap cycle + #327 (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\ncap cycle: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
