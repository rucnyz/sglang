"""disc_top2_discrimination (§disc_top2_discrimination) — top-2 cost discrimination.

design.md §calibration_sanity disc_top2_discrimination: for decisions where own-free was NOT chosen
(i.e., the cost model actually had to compare), the **median ratio
between the cheapest and second-cheapest finite costs is ≤ 10×**. If
this is much larger, the cost model isn't discriminating — one
candidate trivially dominates the rest.

Test: filter Admitter log to non-own_free decisions; for each, sort
finite costs ascending; compute `costs[1] / costs[0]`. Assert median
ratio ≤ threshold (default 10).

Falsification: median ratio >> 10 means in production our "5-candidate
cost comparison" is effectively "always pick X" — the architecture is
overhead without semantic benefit. Either re-calibrate or simplify
to a 2-candidate selector.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admitter-log", required=True)
    ap.add_argument("--max-ratio", type=float, default=10.0,
                    help="Max median top-2 cost ratio (default 10x)")
    ap.add_argument("--settle", type=int, default=30,
                    help="Skip first N decisions (EWMA warm-up)")
    args = ap.parse_args()

    try:
        with open(args.admitter_log) as f:
            raw = [json.loads(ln) for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"FAIL: log not found: {args.admitter_log}")
        return 1

    post = raw[args.settle:]
    ratios = []
    for e in post:
        if e.get("action") == "own_free":
            continue
        costs = e.get("candidate_costs_us") or {}
        finite = sorted(v for v in costs.values() if v is not None)
        if len(finite) < 2:
            continue
        if finite[0] <= 0:
            # Skip zero-or-negative cheapest (would inflate ratio to inf).
            continue
        ratios.append(finite[1] / finite[0])

    print(f"disc_top2_discrimination: non-own_free decisions with ≥2 finite costs = "
          f"{len(ratios)}")
    if not ratios:
        print(
            "disc_top2_discrimination: FAIL — no non-own_free decisions with ≥2 finite "
            "costs to compute discrimination on. Workload was either too "
            "easy (always own_free) or too contentious (only 1 finite "
            "candidate); disc_top2_discrimination undefined here.\n"
            "  diagnostic: run a workload with a mix of regimes."
        )
        return 1

    ratios.sort()
    median = ratios[len(ratios) // 2]
    p10 = ratios[len(ratios) // 10] if len(ratios) >= 10 else ratios[0]
    p90 = ratios[len(ratios) * 9 // 10] if len(ratios) >= 10 else ratios[-1]

    print(f"disc_top2_discrimination: top-2 cost ratio (cheapest_2 / cheapest_1):")
    print(f"  min    = {ratios[0]:.2f}")
    print(f"  p10    = {p10:.2f}")
    print(f"  median = {median:.2f}")
    print(f"  p90    = {p90:.2f}")
    print(f"  max    = {ratios[-1]:.2f}")

    if median > args.max_ratio:
        print(
            f"\nD6m-disc: FAIL — median ratio {median:.2f}× > "
            f"{args.max_ratio:.0f}×. One candidate trivially dominates the "
            f"second-cheapest by a huge margin; the 5-candidate framework "
            f"isn't discriminating on this workload."
        )
        print(
            "  diagnostic: either (1) the cost model magnitudes are\n"
            "  miscalibrated (e.g. defer cost dwarfs everything because\n"
            "  SGLANG_XPOOL_QUEUE_WAIT_US is too high), or (2) the workload\n"
            "  consistently puts one candidate cheaply ahead and the others\n"
            "  are only theoretically reachable."
        )
        return 1

    print(f"\nD6m-disc: PASS — median top-2 ratio {median:.2f}× ≤ "
          f"{args.max_ratio:.0f}×; cost model discriminates meaningfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
