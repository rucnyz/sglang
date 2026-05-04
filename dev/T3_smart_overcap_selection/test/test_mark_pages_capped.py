"""T3 correctness test: mark_pages_capped / unmark_pages_capped maintain
the invariant that any page id removed from free_pages cannot be
returned by alloc(). The earlier T3 wiring bypassed cap_allocator_only,
which would let the allocator hand out a page id whose VA was
cuMemUnmapped — the next kernel touch would crash.
"""

import sys
import torch
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator


class _DummyKVCache:
    pass


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    size = 100
    a = TokenToKVPoolAllocator(
        size=size,
        dtype=torch.bfloat16,
        device=device,
        kvcache=_DummyKVCache(),
        need_sort=True,
    )

    # Allocate 30 pages (1..30), mark pages 91..100 capped (highest 10).
    a.alloc(30)
    capped_ids = torch.arange(91, 101, dtype=torch.int64, device=device)
    moved = a.mark_pages_capped(capped_ids)
    assert moved == 10, f"expected 10 pages moved to capped, got {moved}"
    print(f"[mark capped 91..100] moved={moved}, "
          f"free_pages.numel={a.free_pages.numel()}, "
          f"_capped_pages.numel={a._capped_pages.numel()}")

    # Now alloc(60) — must NOT return any page in 91..100, even though
    # only 30 pages are live and 70 nominally free.
    got = a.alloc(60)
    assert got is not None and got.numel() == 60
    assert got.max().item() <= 90, \
        f"alloc returned a capped page! got max={got.max().item()}"
    print(f"[alloc 60] max={got.max().item()} (must be <= 90)")

    # Try to alloc more — only 0 free pages left (since 30 + 60 = 90 used,
    # and 10 are capped, total non-capped = 90).
    leftover = a.alloc(5)
    assert leftover is None, \
        f"alloc should have returned None (no free pages); got {leftover}"
    print("[alloc 5] returned None as expected (free exhausted, capped untouched)")

    # Unmark capped: pages 91..100 return to free.
    unmoved = a.unmark_pages_capped(capped_ids)
    assert unmoved == 10, f"expected 10 unmark, got {unmoved}"
    print(f"[unmark 91..100] unmoved={unmoved}")

    # Now alloc(10) should succeed (and return pages 91..100).
    final = a.alloc(10)
    assert final is not None and final.numel() == 10
    assert sorted(final.tolist()) == list(range(91, 101)), \
        f"expected 91..100 returned after unmark, got {sorted(final.tolist())}"
    print(f"[alloc 10] got {sorted(final.tolist())}")

    # Symmetric: mark pages already capped is a no-op.
    moved_again = a.mark_pages_capped(torch.tensor([91, 92], device=device))
    # 91, 92 are no longer in free (just allocated above), so 0 moved.
    assert moved_again == 0, \
        f"already-allocated pages should not be marked: moved={moved_again}"
    print(f"[mark already-allocated] moved={moved_again} (correctly 0)")

    print("\nT3 mark_pages_capped invariant test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
