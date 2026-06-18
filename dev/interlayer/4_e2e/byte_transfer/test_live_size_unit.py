"""Dverify — `live_size` includes `_capped_pages` (verify-gap-2).

Locks in the e5f6d34421 fix at
`python/sglang/srt/mem_cache/allocator.py:76-93`:

  Pre-fix:  live_size = _cap if _cap is not None else size
  Post-fix: live_size = size - _capped_pages.numel()

The allocator has TWO mechanisms that hold pages out:
  (a) `set_capacity_pages(n)` — soft cap, sets `_cap` AND adds
      pages > n to `_capped_pages`
  (b) `mark_pages_capped(ids)` — specific pages held out (used by
      cross-pool actuator after cuMemUnmap) — adds to `_capped_pages`
      WITHOUT touching `_cap`

Pre-fix `live_size` only consulted `_cap`, missing the
`mark_pages_capped` contribution. After every cross-pool fire,
`mark_pages_capped` removed slots from `free_pages` but `live_size`
stayed at `size`. Sglang's `_check_pool_invariant` reads `live_size`
as `total` → saw `total > available + evictable` → fired
`ValueError: pool memory leak detected!` and SIGQUIT'd the scheduler.

Test-first protocol:
  1. `git checkout e5f6d34421~1 -- python/sglang/srt/mem_cache/allocator.py`
  2. Run this test → MUST FAIL (live_size won't reflect
     mark_pages_capped contribution)
  3. `git checkout e5f6d34421 -- python/sglang/srt/mem_cache/allocator.py`
  4. Run this test → MUST PASS
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator


def _build_allocator(size=4096):
    """Pure-CPU instance; kvcache=None is OK because alloc/free/cap
    don't touch the kv buffer (only allocator-side bookkeeping)."""
    return TokenToKVPoolAllocator(
        size=size, dtype=torch.bfloat16, device="cpu",
        kvcache=None, need_sort=False,
    )


# ---------- sub-tests ----------

def test_1_fresh_allocator_live_size_equals_size():
    """Boot state: _capped_pages empty → live_size == size."""
    alloc = _build_allocator(size=4096)
    assert alloc.live_size == 4096, (
        f"fresh allocator: live_size={alloc.live_size}, expected 4096"
    )
    capped = getattr(alloc, "_capped_pages", None)
    n_capped = int(capped.numel()) if capped is not None else 0
    assert n_capped == 0, f"_capped_pages should be empty, got {n_capped}"


def test_2_mark_pages_capped_drops_live_size():
    """The cross-pool actuator path: mark_pages_capped(8 pages) →
    live_size drops by 8. PRE-FIX BUG: pre-fix live_size returns _cap
    (None/size) and ignores _capped_pages → still reads as size,
    triggers the leak checker downstream."""
    alloc = _build_allocator(size=4096)
    cap_target = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17],
                              dtype=torch.int64)
    moved = alloc.mark_pages_capped(cap_target)
    assert moved == 8, f"mark_pages_capped: expected 8 moved, got {moved}"
    assert alloc.live_size == 4096 - 8, (
        f"BUG: live_size={alloc.live_size}, expected {4096 - 8} after "
        f"mark_pages_capped(8). Pre-fix live_size only consulted _cap "
        f"and missed the mark_pages_capped contribution — sglang's "
        f"_check_pool_invariant would see total=4096 > available+evictable "
        f"and fire 'pool memory leak detected' SIGQUIT."
    )


def test_3_unmark_pages_capped_restores_live_size():
    """Symmetric: unmark restores capacity."""
    alloc = _build_allocator(size=4096)
    cap_target = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17],
                              dtype=torch.int64)
    alloc.mark_pages_capped(cap_target)
    assert alloc.live_size == 4088

    # Unmark 4 of them
    alloc.unmark_pages_capped(torch.tensor([10, 11, 12, 13], dtype=torch.int64))
    assert alloc.live_size == 4096 - 4, (
        f"unmark 4 of 8: live_size={alloc.live_size}, expected {4096 - 4}"
    )

    # Unmark remaining 4
    alloc.unmark_pages_capped(torch.tensor([14, 15, 16, 17], dtype=torch.int64))
    assert alloc.live_size == 4096, (
        f"all unmarked: live_size={alloc.live_size}, expected 4096"
    )


def test_4_set_capacity_pages_lowers_live_size():
    """The OTHER cap mechanism (set_capacity_pages) also drops
    live_size — this case worked pre-fix too (because pre-fix
    consulted _cap), but the new formula must keep it working."""
    alloc = _build_allocator(size=4096)
    alloc.set_capacity_pages(2048)
    assert alloc.live_size == 2048, (
        f"set_capacity_pages(2048): live_size={alloc.live_size}, "
        f"expected 2048"
    )

    # Grow back
    alloc.set_capacity_pages(4096)
    assert alloc.live_size == 4096


def test_5_both_mechanisms_together_compose():
    """Critical case the pre-fix formula gets WRONG: combine
    set_capacity_pages with mark_pages_capped (specific page ids
    < the cap). Live should drop by BOTH contributions.

    Pre-fix: live = _cap = 2048 — ignores mark_pages_capped
    Post-fix: live = size - _capped = 4096 - (2048 + 8) = 2040"""
    alloc = _build_allocator(size=4096)
    # First soft-cap to 2048 (moves pages [2049..4096] to _capped)
    alloc.set_capacity_pages(2048)
    pre_capped = int(alloc._capped_pages.numel())
    assert alloc.live_size == 2048
    assert pre_capped == 2048

    # Now mark_pages_capped(8 pages inside the cap)
    cap_target = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17],
                              dtype=torch.int64)
    alloc.mark_pages_capped(cap_target)
    post_capped = int(alloc._capped_pages.numel())
    assert post_capped == 2048 + 8

    assert alloc.live_size == 4096 - (2048 + 8), (
        f"BUG: combined cap mechanisms — live_size={alloc.live_size}, "
        f"expected {4096 - (2048 + 8)} = size − _capped_pages.numel(). "
        f"Pre-fix returns _cap=2048, missing the 8 mark_pages_capped "
        f"contributions."
    )


# ---------- runner ----------

def main():
    tests = [
        ("1 fresh allocator: live_size == size",
         test_1_fresh_allocator_live_size_equals_size),
        ("2 mark_pages_capped drops live_size (the leaked-pre-fix path)",
         test_2_mark_pages_capped_drops_live_size),
        ("3 unmark_pages_capped restores live_size",
         test_3_unmark_pages_capped_restores_live_size),
        ("4 set_capacity_pages still works (regression guard)",
         test_4_set_capacity_pages_lowers_live_size),
        ("5 set_capacity_pages + mark_pages_capped compose correctly",
         test_5_both_mechanisms_together_compose),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            import traceback
            traceback.print_exc()
    print(f"\nDverify live_size: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
