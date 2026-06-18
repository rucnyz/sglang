"""Read gap.jsonl produced by SGLANG_GAP_TRACE_LOG, print distribution."""
from __future__ import annotations

import json
import statistics
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_gap_trace.py <path>")
        sys.exit(1)
    path = sys.argv[1]
    gap_gpu = []
    gap_cpu = []
    prev_bs = []
    this_bs = []
    n = 0
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
        except Exception:
            continue
        gap_gpu.append(e["gap_us_gpu"])
        gap_cpu.append(e["gap_us_cpu"])
        prev_bs.append(e["prev_bs"])
        this_bs.append(e["this_bs"])
        n += 1

    if n == 0:
        print(f"no entries in {path}")
        sys.exit(2)

    def _stats(name, vals):
        srt = sorted(vals)
        p50 = srt[len(srt) // 2]
        p90 = srt[int(0.90 * len(srt))]
        p99 = srt[int(0.99 * len(srt))]
        mn = min(vals)
        mx = max(vals)
        mean = statistics.mean(vals)
        return (
            f"  {name:18s}  n={len(vals):6d}  min={mn:7d}  "
            f"p50={p50:7d}  p90={p90:7d}  p99={p99:7d}  "
            f"max={mx:7d}  mean={int(mean):7d}  (us)"
        )

    print(f"=== gap distribution from {path} (n={n} batch transitions) ===")
    print(_stats("gap_us_gpu", gap_gpu))
    print(_stats("gap_us_cpu", gap_cpu))
    print(_stats("prev_bs", prev_bs))
    print(_stats("this_bs", this_bs))

    # Coverage histogram for gap_us_gpu against the ~82 ms fire cost.
    fire_cost_us = 82_000  # bench_cumem_costs.py result
    n_ge_full = sum(1 for v in gap_gpu if v >= fire_cost_us)
    n_ge_half = sum(1 for v in gap_gpu if v >= fire_cost_us // 2)
    n_ge_pos = sum(1 for v in gap_gpu if v > 0)
    n_neg = sum(1 for v in gap_gpu if v < 0)
    print()
    print(f"=== A2 / A3 decision data ===")
    print(f"  gap > 0       : {n_ge_pos}/{n} ({100*n_ge_pos/n:.1f}%) "
          f"— GPU was actually idle between these batches")
    print(f"  gap >= 41 ms  : {n_ge_half}/{n} ({100*n_ge_half/n:.1f}%) "
          f"— half a fire fits naturally")
    print(f"  gap >= 82 ms  : {n_ge_full}/{n} ({100*n_ge_full/n:.1f}%) "
          f"— full fire fits naturally (G-natural viable here)")
    print(f"  gap < 0       : {n_neg}/{n} ({100*n_neg/n:.1f}%) "
          f"— CUDA never drained before next launch (no boundary)")

    print()
    print(f"=== verdict ===")
    if n_neg > 0.5 * n:
        print("  A2 REFUTED: >50% of transitions show no clean GPU boundary.")
        print("  G is dead. Revisit C or accept current state.")
    elif n_ge_full > 0.5 * n:
        print("  A3 CONFIRMED: >50% of natural gaps fit a full fire.")
        print("  G-natural viable. Advance to step 3.")
    elif n_ge_pos > 0.5 * n:
        print("  A2 holds, A3 partial: gaps are positive but small.")
        print("  G-forced is the path; ~82 ms bubble per fire. Advance to step 3.")
    else:
        print("  Mixed signals. Inspect raw distribution.")


if __name__ == "__main__":
    main()
