"""Perf-target pins for the MambaPool hot-path fix (per-layer minus stacked).

`bench_mamba_pool_ops.py` MEASURES the per-op overhead but asserts nothing.
This file pins the FIX's design targets so a future change that reintroduces
the overhead fails CI, not just a silent slowdown (perf tests assert the
theoretical target, not merely "no regression"):

  free       strictly free: the no-cross-fire fast path is the baseline
             `torch.cat`, no isin / no `>size` mask / no `.item()` device
             sync. Target delta ~0 (pre-fix was +11us/call).
  alloc      hoist realized: the scalar `torch.zeros(1)` is allocated once
             per dtype, not once per layer, removing 23 tiny alloc kernels.
             Target well below the pre-fix +168us; the residual is only the
             inherent 24-vs-1 per-layer indexed-write launches (~+55us).
  live_size  early-returns `self.size` when capped is empty: target ~0.
  copy_from  FLAGGED inherent: the per-layer temporal copy is 24-vs-1 launches
             and can only be removed by an arena-layout change (single stacked
             temporal backing), which trades off per-sub-pool independence.
             NOT pinned to 0; pinned as a REGRESSION GUARD (<= the known
             ~+92us) so it cannot silently get worse.

Runs N=3 outer reps at B=1 (where kernel-launch overhead dominates; at large
B byte bandwidth, layout-independent, masks it) and asserts on the mean.
Reports mean +- std (run-to-run timing noise is real).

GPU required, ~23 GB/layout at production geometry. Pick an IDLE GPU
(contention invalidates timing): CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  dev/interlayer/3_budgeter/mamba_pool_perf/test_mamba_pool_perf_targets.py
"""
import os
import statistics as st
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

from bench_mamba_pool_ops import (  # noqa: E402  reuse the real microbench
    _bench_alloc,
    _bench_copy_from,
    _bench_free,
    _bench_live_size,
    _build_pool,
)

N_REPS = 3
B = 1  # launch overhead dominates; the fix is about launches, not bytes.

# Targets (us). free/live_size pin the strictly-free wins; alloc pins the
# hoist; copy_from is the flagged-inherent regression guard.
TARGETS = {
    "free": 6.0,        # ~0 expected; pre-fix +11
    "alloc": 100.0,     # ~+55 expected (inherent); pre-fix +168
    "live_size": 3.0,   # ~0
    "copy_from": 130.0,  # ~+92 inherent (arena-layout tradeoff); guard, not 0
}


def _measure_deltas():
    """One rep: build each layout, time the ops, return per-op (per-layer -
    stacked) deltas in us."""
    timings = {}
    for label, per_layer in (("stacked", False), ("per-layer-list", True)):
        pool = _build_pool(per_layer)
        assert pool._capped_slots.numel() == 0, "perf pin requires no-fire pool"
        timings[label] = {
            "copy_from": _bench_copy_from(pool, B),
            "alloc": _bench_alloc(pool, B),
            "free": _bench_free(pool, B),
            "live_size": _bench_live_size(pool),
        }
        del pool
        torch.cuda.empty_cache()
    st_, pl = timings["stacked"], timings["per-layer-list"]
    return {op: pl[op] - st_[op] for op in st_}


def main():
    assert torch.cuda.is_available(), "needs an IDLE GPU (CUDA_VISIBLE_DEVICES=<idle>)"
    free_mb = torch.cuda.mem_get_info()[0] / 1024**2
    assert free_mb > 60_000, (
        f"only {free_mb:.0f} MiB free on this GPU; production geometry needs "
        f"~23 GB/layout and a contended GPU invalidates timing. Pick an idle GPU."
    )

    reps = []
    for r in range(N_REPS):
        d = _measure_deltas()
        reps.append(d)
        print(f"  rep {r + 1}/{N_REPS}: " + "  ".join(f"{k}={v:+.1f}us" for k, v in d.items()))

    print(f"\nMambaPool perf targets (per-layer minus stacked, B={B}, N={N_REPS}):")
    print(f"  {'op':<12}{'mean us':>10}{'std':>8}{'target us':>12}{'verdict':>10}")
    print("  " + "-" * 50)
    ok = True
    for op, target in TARGETS.items():
        vals = [r[op] for r in reps]
        mean, std = st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)
        passed = mean <= target
        ok = ok and passed
        print(f"  {op:<12}{mean:>10.1f}{std:>8.1f}{target:>12.1f}"
              f"{'PASS' if passed else 'FAIL':>10}")

    print()
    if ok:
        print("perf targets: ALL PASS (strictly-free wins realized; copy_from "
              "within the flagged inherent bound)")
        return 0
    print("perf targets: FAIL — a hot-path op regressed past its design target")
    return 1


if __name__ == "__main__":
    sys.exit(main())
