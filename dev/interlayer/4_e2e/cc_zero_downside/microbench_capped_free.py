"""Micro-bench: the per-slot free() cost, arena (capped tail) vs plain.

The old Convention-A design kept the ~560k reserved-headroom page-ids INSIDE
`free_pages` and ran `torch.isin(freed, _capped_pages)` on EVERY slot-free —
~370µs/free, which halved decode throughput. The CappedFreeList design keeps the
tail OUT of the free list (an implicit `tail_lo` int), so free() is a plain
append: arena free() should now match plain free() (~tens of µs), not 11× it.

Run: CUDA_VISIBLE_DEVICES=7 .venv/bin/python \
       dev/interlayer/4_e2e/cc_zero_downside/microbench_capped_free.py
"""
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

DEV = "cuda:0"
BOOT = 560_000
CEIL = 1_120_000
BS = 64
ITERS = 300


def _mk(arena: bool):
    return TokenToKVPoolAllocator(
        size=BOOT, dtype=torch.bfloat16, device=DEV, kvcache=None,
        need_sort=True, max_size=(CEIL if arena else None),
    )


def _bench(arena: bool):
    a = _mk(arena)
    # Fill near-full so the working free list is small (the decode regime).
    held = []
    while a.available_size() > 4 * BS:
        out = a.alloc(BS)
        if out is None:
            break
        held.append(out)
    torch.cuda.synchronize()

    # Steady churn: free a block, alloc it back. Time free() and alloc()
    # separately.
    free_us = []
    alloc_us = []
    blk = held.pop()
    for i in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        a.free(blk)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        out = a.alloc(BS)
        torch.cuda.synchronize()
        t2 = time.perf_counter_ns()
        free_us.append((t1 - t0) / 1000)
        alloc_us.append((t2 - t1) / 1000)
        blk = out if out is not None else held.pop()

    free_us.sort()
    alloc_us.sort()
    med_free = free_us[len(free_us) // 2]
    med_alloc = alloc_us[len(alloc_us) // 2]
    return med_free, med_alloc, a.available_size()


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    # Warm CUDA.
    _ = torch.zeros(1, device=DEV)
    torch.cuda.synchronize()

    pf, pa, _ = _bench(arena=False)
    af, aa, _ = _bench(arena=True)
    print(f"plain : free={pf:7.1f}us  alloc={pa:7.1f}us")
    print(f"arena : free={af:7.1f}us  alloc={aa:7.1f}us")
    ratio = af / pf if pf else float("inf")
    print(f"arena free / plain free = {ratio:.2f}x   "
          f"(old Convention-A was ~11x; target ~1x)")
    # The tax is gone if arena free is within ~3x of plain (the old design was
    # 11x). Generous bound — both are now O(bs) appends; any residual gap is
    # sort-buffer noise, not a 560k isin.
    ok = ratio < 3.0
    print("\nPASS: free() tax removed" if ok else "\nFAIL: arena free still taxed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
