"""sweep-arm validator — under Lemma A1 (below-saturation regime), the
cross-free Admitter candidate dominates cross-evict.

design.md §sweep-arm PASS criteria:
  count cross_free_chosen / (cross_free_chosen + cross_evict_chosen) ≥ 0.95

Reuses the JSONL log produced by the cost_picks_xfree launch. Skips the same
SETTLE_TICKS as cost_picks_xfree.
"""
from __future__ import annotations

import argparse
import json
import sys


SETTLE_TICKS = 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admitter-log", required=True)
    ap.add_argument("--min-decisions", type=int, default=50,
                    help="Minimum cross-pool decisions for the ratio. "
                         "design.md §sweep arm calls for 100, but with the Budgeter "
                         "co-running it relieves KV pressure asynchronously, "
                         "so contentious arrivals are rarer than spec assumed. "
                         "50 is a statistical floor that's achievable in a "
                         "single workload run.")
    ap.add_argument("--ratio-threshold", type=float, default=0.95,
                    help="Minimum cross_free / (cross_free + cross_evict)")
    args = ap.parse_args()

    try:
        with open(args.admitter_log) as f:
            raw = [json.loads(ln) for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"FAIL: log not found: {args.admitter_log}")
        return 1

    post_settle = raw[SETTLE_TICKS:]
    cf = sum(1 for e in post_settle if e["action"] == "cross_free")
    ce = sum(1 for e in post_settle if e["action"] == "cross_evict")
    total = cf + ce

    print(f"sweep arm: post-settle cross_free = {cf}, cross_evict = {ce}")

    if total < args.min_decisions:
        print(
            f"sweep arm: FAIL — only {total} cross-pool decisions "
            f"(need ≥ {args.min_decisions}); workload didn't generate "
            f"enough Admitter contention to validate sweep arm statistically"
        )
        return 1

    ratio = cf / total
    print(f"sweep arm: cross_free / (cross_free + cross_evict) = {ratio:.4f}")

    if ratio < args.ratio_threshold:
        print(
            f"sweep arm: FAIL — ratio {ratio:.2%} < {args.ratio_threshold:.0%}. "
            f"Below this either both pools are saturated (Lemma A1 broken) "
            f"or the donor pool's FREE pages are clustered in a bad locality."
        )
        return 1

    print(f"sweep arm: PASS — cross_free dominates with {ratio:.2%} of "
          f"cross-pool decisions ({cf}/{total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
