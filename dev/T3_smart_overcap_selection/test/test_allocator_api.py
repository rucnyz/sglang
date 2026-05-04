"""T3 unit test: free_page_mask() and select_drain_pages() on
BaseTokenToKVPoolAllocator return correct results across alloc/free cycles.

No SGLang server boot needed — exercises the allocator in isolation.
"""

import sys
import torch
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator


class _DummyKVCache:
    """Minimal KVCache stand-in that exposes only what the allocator touches."""
    def __init__(self):
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

    # Initial: pages 1..size free; index 0 always False (sentinel).
    mask = a.free_page_mask()
    assert mask.shape == (size + 1,), f"mask shape: {mask.shape}"
    assert mask.sum().item() == size, f"all free: {mask.sum().item()}"
    assert not mask[0].item(), "sentinel slot 0 must not be in free mask"
    print(f"[init] free_pages count={a.free_pages.numel()}, mask sum={mask.sum().item()}")

    # Allocate 30 pages; expect pages 1..30 (lowest-first, since need_sort=True).
    got = a.alloc(30)
    assert got is not None and got.numel() == 30
    print(f"[alloc 30] got pages min={got.min().item()} max={got.max().item()}")
    mask = a.free_page_mask()
    assert mask.sum().item() == 70, f"70 free after alloc 30: {mask.sum().item()}"

    # select_drain_pages(10, prefer="high") should give us indices 91..100.
    drain = a.select_drain_pages(10, prefer="high")
    assert drain.numel() == 10
    assert drain.min().item() == 91 and drain.max().item() == 100, \
        f"high-prefer expected 91..100, got {drain.tolist()}"
    print(f"[drain top 10] got {sorted(drain.tolist())}")

    # Free some pages back.
    a.free(torch.tensor([5, 7, 12], device=device))
    mask = a.free_page_mask()
    # We have 70 free + 3 released-not-merged = 73 in the union.
    assert mask.sum().item() == 73, f"73 free after free 3: {mask.sum().item()}"
    assert mask[5].item() and mask[7].item() and mask[12].item(), \
        "freed indices should show as free"

    # select_drain_pages(15, prefer="low") should pick from low free indices
    # (the freshly-freed 5,7,12 plus the still-untouched 31..).
    drain_low = a.select_drain_pages(15, prefer="low")
    assert drain_low.min().item() == 5, \
        f"low-prefer should start at 5: {drain_low.tolist()[:5]}"
    print(f"[drain low 15] got {sorted(drain_low.tolist())}")

    # Edge: ask for more than available.
    drain_too_many = a.select_drain_pages(200, prefer="high")
    assert drain_too_many.numel() == 73, \
        f"capped at 73 free: {drain_too_many.numel()}"

    print("\nT3 allocator API unit test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
