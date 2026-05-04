"""T2 direct test: with need_sort=True, alloc(n) returns the LOWEST n
free indices (not whatever happens to be at the front of free_pages).
Constructed so the invariant is testable in isolation, no statistical
workload.

Concrete steps:
  1. Allocator size 100, need_sort=True
  2. alloc(50) → handed out 1..50
  3. free a known set spread across the index space: {5, 30, 75, 90}
     (some low, some mid, some high)
  4. alloc(80) — request more than free_pages alone has (50 free
     already in free_pages = [51..100], plus 4 in release_pages)
     so merge_and_sort_free triggers; final free_pages should be
     sorted ascending: [5, 30, 51..74, 75, 76..89, 90, 91..100]
     and alloc(80) returns the smallest 80 of those.
  5. With need_sort=False (path B): same workload, alloc(80) returns
     whatever's at the front, which is the unsorted residual.

This isolates the placement-bias mechanism from any workload model:
the invariant is "alloc returns lowest free indices when need_sort=True".
"""

import sys
import torch
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator


class _DummyKVCache:
    pass


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pool_size = 100

    # Path A: need_sort=True
    a_on = TokenToKVPoolAllocator(
        size=pool_size, dtype=torch.bfloat16, device=device,
        kvcache=_DummyKVCache(), need_sort=True,
    )
    a_on.alloc(50)
    # Free 4 known live indices, spread low-to-high within the alloc'd range.
    a_on.free(torch.tensor([5, 12, 30, 47], dtype=torch.int64, device=device))
    # release_pages = [5, 12, 30, 47]; free_pages = [51..100]
    # alloc(80) needs 80, free_pages has 50 → merge_and_sort triggers.
    # After merge: free_pages = sorted([5, 30, 51..100, 75, 90])
    #            = [5, 30, 51, 52, ..., 74, 75, 76, ..., 89, 90, 91, ..., 100]
    # Length = 50 + 4 = 54. alloc(80) wants 80, only 54 available → returns None.
    # OK adjust: alloc(54) — returns ALL of them in sorted order.
    got = a_on.alloc(54)
    assert got is not None and got.numel() == 54
    got_sorted = sorted(got.tolist())
    # Expected set: 4 freed-back indices plus 50..100 still-never-alloc'd.
    expected = sorted(set([5, 12, 30, 47] + list(range(51, 101))))
    assert got_sorted == expected, \
        f"need_sort=True alloc returns wrong set: {got_sorted[:10]}..."

    # The CRITICAL invariant: the returned indices are sorted.
    assert got.tolist() == sorted(got.tolist()), \
        "need_sort=True alloc must return sorted (lowest-first) indices"
    print(f"[bias-on] alloc(54) returned indices in sorted order: "
          f"first 5 = {got.tolist()[:5]}, last 5 = {got.tolist()[-5:]}")

    # Path B: need_sort=False
    a_off = TokenToKVPoolAllocator(
        size=pool_size, dtype=torch.bfloat16, device=device,
        kvcache=_DummyKVCache(), need_sort=False,
    )
    a_off.alloc(50)
    # free goes directly to free_pages tail (no release_pages, no sort).
    a_off.free(torch.tensor([5, 12, 30, 47], dtype=torch.int64, device=device))
    # free_pages now = [51..100, 5, 30, 75, 90] (NOT sorted).
    got_off = a_off.alloc(54)
    assert got_off is not None
    # Without sort, alloc returns front-of-free_pages = [51..100, 5, 30, 75, 90].
    # Specifically, the first index returned should be 51, NOT 5.
    print(f"[bias-off] alloc(54) returned (NOT sorted): "
          f"first 5 = {got_off.tolist()[:5]}, last 5 = {got_off.tolist()[-5:]}")
    assert got_off.tolist()[0] == 51, \
        f"need_sort=False expected first=51 (FIFO), got {got_off.tolist()[0]}"
    assert got_off.tolist() != sorted(got_off.tolist()), \
        "need_sort=False should NOT return sorted (placement bias absent)"

    # Side-by-side comparison: where does the lowest freed-back index (=5) end up?
    pos_on = got.tolist().index(5)
    pos_off = got_off.tolist().index(5)
    print(f"\nPosition of freed-then-realloced index 5 in alloc result:")
    print(f"  bias-on:  position {pos_on} (handed out FIRST — low-index priority)")
    print(f"  bias-off: position {pos_off} (handed out late — went to FIFO tail)")
    assert pos_on < pos_off, \
        f"index 5 should be handed out earlier with bias-on: {pos_on} < {pos_off}? FAIL"

    print("\nT2 direct placement-bias observation test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
