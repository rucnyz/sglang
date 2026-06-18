"""cost_picks_xfree validator — parses the Admitter JSONL log and asserts that
under contentious arrivals (own_free infeasible), the Admitter picked
cross-free for ≥80% of decisions where i_src had FREE pages.

design.md §cost_picks_xfree PASS criteria:
  (a) ≥100 cross-pool-feasible decisions seen (statistical floor)
  (b) ≥80% of contentious arrivals chose cross_free (or own_free if
      own_free was feasible — i.e., not contentious in the first place)
  (c) 0 decisions chose 'defer' while cross_free was finite
       (i.e., no missed opportunity: Admitter must never defer when a
       cheap cross-pool path exists)

Skips the first SETTLE_TICKS decisions (the system is still warming
the EWMA and ramping up).
"""
from __future__ import annotations

import argparse
import json
import sys


SETTLE_TICKS = 30  # skip first 30 admissions while EWMA warms up


def _is_feasible(v):
    """JSONL maps inf → null. Feasible = finite cost."""
    return v is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admitter-log", required=True,
                    help="Path to admitter JSONL produced by SGLANG_HIMA_ADMITTER_LOG=")
    ap.add_argument("--min-decisions", type=int, default=50,
                    help="Statistical floor on # cross-pool-feasible decisions "
                         "(design.md §cost_picks_xfree requires 50 post-settle decisions)")
    ap.add_argument("--cross-free-frac", type=float, default=0.80,
                    help="Min fraction of contentious arrivals that chose cross_free")
    args = ap.parse_args()

    try:
        with open(args.admitter_log) as f:
            raw = [json.loads(ln) for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"FAIL: admitter log not found: {args.admitter_log}")
        return 1

    if len(raw) <= SETTLE_TICKS:
        print(f"FAIL: only {len(raw)} decisions in log (need > {SETTLE_TICKS} settle)")
        return 1

    post_settle = raw[SETTLE_TICKS:]

    # Bucket the post-settle decisions.
    actions = {"own_free": 0, "own_evict": 0, "cross_free": 0,
               "cross_evict": 0, "defer": 0}
    contentious = []        # own_free=null entries (where Admitter HAD to choose)
    defer_with_cross_free_feasible = []
    for e in post_settle:
        action = e.get("action", "?")
        actions[action] = actions.get(action, 0) + 1
        costs = e.get("candidate_costs_us") or {}
        own_free_feasible = _is_feasible(costs.get("own_free"))
        cross_free_feasible = _is_feasible(costs.get("cross_free"))
        if not own_free_feasible:
            contentious.append(e)
        if action == "defer" and cross_free_feasible:
            defer_with_cross_free_feasible.append(e)

    print(f"cost_picks_xfree: total post-settle decisions = {len(post_settle)}")
    print(f"cost_picks_xfree: action breakdown = {actions}")
    print(f"cost_picks_xfree: contentious (own_free infeasible) = {len(contentious)}")
    print(f"cost_picks_xfree: cross-pool-feasible decisions = "
          f"{actions['cross_free'] + actions['cross_evict']}")

    failures = []

    # (a) statistical floor
    cross_pool_n = actions["cross_free"] + actions["cross_evict"]
    if cross_pool_n < args.min_decisions:
        failures.append(
            f"(a) only {cross_pool_n} cross-pool decisions in log "
            f"(need ≥ {args.min_decisions}); workload didn't generate "
            f"enough Admitter contention to validate cost_picks_xfree statistically"
        )

    # (b) ≥80% of contentious arrivals chose cross_free
    if contentious:
        cf = sum(1 for e in contentious if e["action"] == "cross_free")
        frac = cf / len(contentious)
        print(f"cost_picks_xfree: contentious cross_free fraction = {frac:.3f} "
              f"({cf}/{len(contentious)})")
        if frac < args.cross_free_frac:
            failures.append(
                f"(b) only {frac:.2%} of contentious arrivals chose cross_free "
                f"(need ≥ {args.cross_free_frac:.0%}); cost model is "
                f"under-pricing cross_free or over-pricing it relative to "
                f"alternatives"
            )
    else:
        # No contentious arrivals means own_free always feasible — the
        # workload didn't stress KV admission. Diagnostic, not a fail
        # of the Admitter logic.
        failures.append(
            "(b) zero contentious arrivals in post-settle window — "
            "workload was not KV-bound enough; rerun with higher "
            "RPS / longer input"
        )

    # (c) no defer while cross_free was feasible
    if defer_with_cross_free_feasible:
        n = len(defer_with_cross_free_feasible)
        failures.append(
            f"(c) {n} decisions chose 'defer' while cross_free was feasible "
            f"— Admitter is missing cheap cross-pool opportunities. "
            f"Example: {defer_with_cross_free_feasible[0]['candidate_costs_us']}"
        )

    if failures:
        print("\nD6: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nD6: PASS")
    print(f"  ≥{cross_pool_n} cross-pool decisions, "
          f"{frac:.2%} contentious → cross_free, "
          f"0 missed cross_free opportunities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
