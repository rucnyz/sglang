"""burst_recovery validator — burst-recovery queue p99 comparison.

design.md §burst_recovery PASS criterion:
  queue_p99_phase_B_inter ≤ queue_p99_phase_B_off × 1.10

We don't have direct queue-depth telemetry, so we approximate using
the bench-server's per-request `wait_time` (time from POST to first
output token; this is dominated by queue + scheduling latency, not
generation). For a burst that exceeds steady-state capacity, an
`inter` run with Admitter firing should produce shorter waits than
the baseline because the Admitter expands the dst pool synchronously
as bursts arrive.

This validator parses Phase B's `bench.json` for both `off` and
`inter` and compares wait-time p99.

Falsification: queue_p99_phase_B_inter / queue_p99_phase_B_off ≥ 1.50
means the Admitter isn't catching the burst fast enough (or made it
worse via overhead).
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _wait_times(path: str) -> list[float]:
    """Extract per-request wait time from a bench_serving JSON."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    # bench_serving stores per-request stats in various shapes
    # depending on version; try a few keys.
    waits = []
    if "ttft" in data and isinstance(data["ttft"], list):
        # Per-req TTFT directly
        waits = [float(x) for x in data["ttft"]]
    elif "p99_ttft" in data:
        waits = [float(data.get("p99_ttft", 0))]
    elif "wait_time" in data and isinstance(data["wait_time"], list):
        waits = [float(x) for x in data["wait_time"]]
    return waits


def _summary(path: str) -> dict:
    """Extract aggregate stats from bench_serving JSON."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        k: data.get(k)
        for k in (
            "duration", "completed", "total_input_tokens",
            "total_output_tokens",
            "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
            "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
            "mean_e2e_latency_ms", "p99_e2e_latency_ms",
        )
    }


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * 0.99)
    return s[min(idx, len(s) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ratio-threshold", type=float, default=1.10,
                    help="Max ratio of inter/off Phase-B p99 TTFT (default 1.10)")
    ap.add_argument("--falsify-threshold", type=float, default=1.50,
                    help="Above this ratio, falsify (Admitter making it worse)")
    args = ap.parse_args()

    off_phaseB = os.path.join(args.out_dir, "off.phaseB.bench.json")
    inter_phaseB = os.path.join(args.out_dir, "inter.phaseB.bench.json")

    off_stats = _summary(off_phaseB)
    inter_stats = _summary(inter_phaseB)

    if not off_stats:
        print(f"FAIL: missing {off_phaseB}")
        return 1
    if not inter_stats:
        print(f"FAIL: missing {inter_phaseB}")
        return 1

    print("burst_recovery Phase B summary:")
    print(f"  off:   completed={off_stats.get('completed')}, "
          f"p99_ttft={off_stats.get('p99_ttft_ms', 0):.1f}ms, "
          f"mean_ttft={off_stats.get('mean_ttft_ms', 0):.1f}ms")
    print(f"  inter: completed={inter_stats.get('completed')}, "
          f"p99_ttft={inter_stats.get('p99_ttft_ms', 0):.1f}ms, "
          f"mean_ttft={inter_stats.get('mean_ttft_ms', 0):.1f}ms")

    off_p99 = float(off_stats.get("p99_ttft_ms") or 0)
    inter_p99 = float(inter_stats.get("p99_ttft_ms") or 0)
    if off_p99 <= 0:
        print(f"FAIL: off Phase B p99_ttft is {off_p99} — bench results malformed")
        return 1

    ratio = inter_p99 / off_p99
    print(f"\nD11: queue_p99_phase_B inter / off = {ratio:.3f}")
    print(f"     (inter={inter_p99:.1f}ms, off={off_p99:.1f}ms)")

    if ratio <= args.ratio_threshold:
        print(f"\nD11: PASS — inter ≤ off × {args.ratio_threshold:.2f} "
              f"(actual ratio {ratio:.2f}). Admitter caught the burst.")
        return 0
    if ratio >= args.falsify_threshold:
        print(f"\nD11: FAIL (FALSIFIED) — inter ≥ off × {args.falsify_threshold:.2f} "
              f"(actual ratio {ratio:.2f}). Admitter isn't catching the burst:\n"
              "  - c^xfer EWMA may be too high (suppressing fires)\n"
              "  - cost model may be mis-ranking candidates\n"
              "  - actuator wall may be higher than fire_wall_curve expected")
        return 1
    print(f"\nD11: SOFT FAIL — ratio {ratio:.2f} is between {args.ratio_threshold} "
          f"and {args.falsify_threshold}; Admitter showed marginal effect "
          "but didn't clear the 10% threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
