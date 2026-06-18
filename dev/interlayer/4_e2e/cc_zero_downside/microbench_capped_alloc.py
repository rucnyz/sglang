"""Op-level microbench for the arena per-decode-step alloc overhead.

Root cause (evidence: dev/interlayer/4_e2e/cc_zero_downside/README §2026-06-08):
the arena KV allocator boots with ~1.5M reserved-headroom pages parked in
`_capped_pages` (capacity a cross-fire can grow into). `TokenToKVPoolAllocator.
alloc(bs)` then runs `torch.isin(head, capped)` against that ~1.5M-element set
EVERY decode step — measured 7µs (static, capped empty) → 604µs (arena). The
isin almost always misses (capped ids are the high tail; alloc takes the low
head), so it is pure waste in steady state.

This microbench reproduces the alloc decision ops at the real sizes and times
three variants, to confirm at the OP level that the fix restores static parity
BEFORE touching production code:

  STATIC  : capped empty            → head = free_pages[:bs]; advance      (off path)
  CURRENT : isin(head, capped_1.5M) → the shipped capped-aware filter      (regressed)
  FIXED   : cheap lower-bound pre-check (head.max() < capped_min → skip isin)

Pass criterion (the goal): FIXED median ≈ STATIC (≈0 increment vs off), i.e.
the +600µs is eliminated. CURRENT reproduces the ~600µs regression.

Run: .venv/bin/python dev/interlayer/4_e2e/cc_zero_downside/microbench_capped_alloc.py
"""
import statistics
import sys

import torch

# Sizes from the cc bench arena boot (README §2026-06-08): free_pages spans the
# full reserved id space (~3.05M), capped = the upper headroom half (~1.5M).
SIZE = 3_050_862
CAP = 1_525_391          # live cap; capped = arange(CAP+1, SIZE+1)
BS = 8                   # decode batch (1 token/req); cc steady state bs≈5-10
N_ALLOC = 300            # allocs timed per rep
N_REPS = 5
DEVICE = "cuda"


def _make(device):
    free_pages = torch.arange(1, SIZE + 1, dtype=torch.int64, device=device)
    capped = torch.arange(CAP + 1, SIZE + 1, dtype=torch.int64, device=device)
    return free_pages, capped


def _time_variant(name, fn):
    """fn(free_pages, capped) does ONE alloc-decision; we time N_ALLOC of them
    on a fixed free_pages snapshot (we measure the per-step DECISION cost, not
    the shrinking, so each call sees the same representative state)."""
    reps = []
    for _ in range(N_REPS):
        free_pages, capped = _make(DEVICE)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(N_ALLOC):
            fn(free_pages, capped)
        end.record()
        torch.cuda.synchronize()
        reps.append(start.elapsed_time(end) / N_ALLOC * 1000.0)  # µs/alloc
    return name, statistics.mean(reps), statistics.pstdev(reps)


def static_decision(free_pages, capped):
    # capped empty fast path: just take the head.
    head = free_pages[:BS]
    return head


def current_decision(free_pages, capped):
    # Shipped capped-aware path (allocator.py alloc): isin(head, capped) + sync.
    head = free_pages[:BS]
    head_in_capped = torch.isin(head, capped)
    if not bool(head_in_capped.any().item()):
        return head
    # (slow full-scan branch omitted; cc never reaches it — head is always
    # below the capped tail, so the cost is the isin+any+item above.)
    return head


# Production form: `_capped_lo` is maintained as a PLAIN PYTHON INT (the min
# capped id) at every _capped_pages mutation, so the hot path reads no GPU
# scalar. The only per-alloc sync is the single `.any().item()` on a BS-sized
# boolean — O(BS), independent of capped size. Sound for ANY capped layout:
# every capped id >= _capped_lo, so "no head id >= _capped_lo" ⟹ none capped.
_CAPPED_LO = CAP + 1  # == int(capped.min()); cached, no GPU read


def fixed_decision(free_pages, capped):
    head = free_pages[:BS]
    if not bool((head >= _CAPPED_LO).any().item()):
        return head  # no capped in head — skip the O(capped) isin
    head_in_capped = torch.isin(head, capped)
    if not bool(head_in_capped.any().item()):
        return head
    return head


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); return 1
    print(f"device={torch.cuda.get_device_name()}  SIZE={SIZE} CAP={CAP} "
          f"capped={SIZE-CAP} BS={BS} N_ALLOC={N_ALLOC} N_REPS={N_REPS}\n")
    results = []
    for name, fn in [("STATIC (off)", static_decision),
                     ("CURRENT (arena)", current_decision),
                     ("FIXED (pre-check)", fixed_decision)]:
        results.append(_time_variant(name, fn))
    base = results[0][1]
    print(f"{'variant':20s} {'µs/alloc':>12s} {'±std':>8s} {'vs STATIC':>12s}")
    for name, mean, std in results:
        print(f"{name:20s} {mean:12.2f} {std:8.2f} {mean-base:+11.2f}µs")
    static_us, current_us, fixed_us = (r[1] for r in results)
    print()
    print(f"CURRENT reproduces the regression: +{current_us-static_us:.1f}µs vs static")
    # Goal: eliminate the O(capped) algorithmic cost. FIXED is O(BS)+one sync,
    # independent of capped size — so it must remove the bulk of the regression
    # (>=95%) and leave only a single-sync residual (a few tens of µs), which is
    # <1% of a ~12.6 ms decode step (verified for real by the e2e re-run).
    removed = (current_us - fixed_us) / (current_us - static_us)
    residual = fixed_us - static_us
    ok = removed >= 0.95 and residual < 60.0
    print(f"FIXED vs STATIC: +{residual:.1f}µs residual (one sync, O(BS) not O(capped))")
    print(f"regression removed: {removed*100:.1f}%  ->  "
          f"{'PASS (algorithmic cost eliminated)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
