"""Correctness pin for the MambaPool `free` no-cross-fire fast path.

`free` skips the `torch.isin` capped-membership test and the
`free_index > self.size` mask (whose `.any()` forces a per-call device sync)
ONLY when `_no_cross_fire` holds: `_capped_slots` empty AND
`self.size == self.max_size`. This file proves:

  1. No cross-fire => the predicate is True and freed ids return to
     free_slots verbatim (baseline behavior).
  2. The instant a shrink lowers the cap (`set_capacity_slots(n < size)`),
     the predicate flips False and the capped-aware slow path runs: an
     above-cap free is routed to `_capped_slots`, never back to free_slots
     (the #312/#329 unmapped-VA crash guard).
  3. A populated `_capped_slots` (via `migrate_slot`) also flips the
     predicate False so a freed capped id is dropped on the floor, not
     handed back out.

Builds a real `MambaPool` via `MambaPool.__new__` with only the
allocator-side state the bookkeeping paths touch (the dtype/shape geometry is
irrelevant to free routing), mirroring
`dev/interlayer/0_page_state_machine/alloc_lock/test_mamba_alloc_lock.py`.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python test_mamba_free_fastpath.py
     (CPU also fine; no kernels needed for free routing)
"""
import sys
import threading

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch  # noqa: E402

from sglang.srt.mem_cache.memory_pool import MambaPool  # noqa: E402

DEVICE = "cpu"   # free routing is pure index bookkeeping; no GPU needed


def _build_pool(size: int, max_size: int) -> MambaPool:
    pool = MambaPool.__new__(MambaPool)
    pool.size = size
    pool.max_size = max_size
    pool.device = DEVICE
    pool._alloc_lock = threading.Lock()
    pool.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device=DEVICE)
    if size < max_size:
        pool._capped_slots = torch.arange(
            size + 1, max_size + 1, dtype=torch.int64, device=DEVICE
        )
    else:
        pool._capped_slots = torch.empty(0, dtype=torch.int64, device=DEVICE)
    # migrate_slot touches the cache tensors; not exercised here.
    return pool


def test_1_no_cross_fire_takes_fast_path():
    """size == max_size, capped empty => predicate True; free returns ids
    straight to free_slots (baseline)."""
    pool = _build_pool(size=8, max_size=8)
    assert pool._no_cross_fire is True
    freed = torch.tensor([3, 5], dtype=torch.int64, device=DEVICE)
    pool.free_slots = pool.free_slots[~torch.isin(pool.free_slots, freed)]
    pool.free(freed)
    fs = set(pool.free_slots.tolist())
    assert fs == set(range(1, 9)), f"fast path lost/mis-routed ids: {sorted(fs)}"
    assert pool._capped_slots.numel() == 0
    print("  PASS  1  no cross-fire: predicate True, free returns to free_slots")


def test_2_headroom_disables_fast_path():
    """max_size > size (boot-deferred headroom in _capped_slots) => predicate
    False even before any shrink, so the capped-aware path always runs while
    headroom slots exist."""
    pool = _build_pool(size=4, max_size=8)
    assert pool._no_cross_fire is False
    print("  PASS  2  boot headroom (capped non-empty): predicate False")


def test_3_shrink_flips_predicate_and_routes_above_cap():
    """After set_capacity_slots(n < size), self.size < max_size => predicate
    False; freeing a slot that is above the new cap AND was LIVE (not moved to
    _capped_slots by the shrink) must hit the `mask_above` routing branch and be
    held in _capped_slots, never returned to free_slots.

    The shrink only moves currently-FREE ids > n into _capped_slots. To exercise
    the `mask_above` branch (not the isin-drop branch test_4 covers), slot 6 is
    made live (pulled from free_slots) BEFORE the shrink, so it is NOT in
    _capped_slots when freed; the `free_index > size` mask is what routes it."""
    pool = _build_pool(size=8, max_size=8)
    assert pool._no_cross_fire is True
    # Slot 6 is allocated/live: pull it from free_slots before the shrink.
    pool.free_slots = pool.free_slots[pool.free_slots != 6]
    pool.set_capacity_slots(4)          # shrink: free ids > 4 (5,7,8) -> _capped_slots
    assert pool.size == 4
    assert pool._no_cross_fire is False, "shrink must disable the fast path"
    assert 6 not in pool._capped_slots.tolist(), \
        "slot 6 was live, the shrink must not have capped it (else isin-drop, not mask_above)"
    # Free the live above-cap slot: isin finds nothing (6 not capped), the
    # `free_index > size` mask routes it into _capped_slots.
    pool.free(torch.tensor([6], dtype=torch.int64, device=DEVICE))
    assert 6 not in pool.free_slots.tolist(), "above-cap id leaked to free_slots"
    assert 6 in pool._capped_slots.tolist(), "above-cap id not routed by mask_above branch"
    print("  PASS  3  shrink flips predicate; live above-cap free routed via mask_above")


def test_4_capped_member_dropped_not_handed_back():
    """A populated _capped_slots (via migrate_slot side state) flips the
    predicate False so freeing a capped id drops it on the floor, never
    returning it to free_slots (the unmapped-VA crash guard)."""
    pool = _build_pool(size=8, max_size=8)
    # Simulate the cross-pool cap-barrier marking slot 7 off: put it in
    # _capped_slots and pull it from free_slots (what mark_pages_capped does).
    pool.free_slots = pool.free_slots[pool.free_slots != 7]
    pool._capped_slots = torch.tensor([7], dtype=torch.int64, device=DEVICE)
    assert pool._no_cross_fire is False
    # An in-flight req on slot 7 finishes and frees it; it must be dropped.
    pool.free(torch.tensor([7], dtype=torch.int64, device=DEVICE))
    assert 7 not in pool.free_slots.tolist(), "capped id 7 handed back out (crash)"
    print("  PASS  4  capped member free dropped, not returned to free_slots")


def main():
    tests = [
        test_1_no_cross_fire_takes_fast_path,
        test_2_headroom_disables_fast_path,
        test_3_shrink_flips_predicate_and_routes_above_cap,
        test_4_capped_member_dropped_not_handed_back,
    ]
    print(f"\nMambaPool free fast-path correctness (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nfree fast-path: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
