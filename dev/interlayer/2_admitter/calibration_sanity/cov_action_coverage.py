"""cov_action_coverage (§cov_action_coverage) — Admitter action coverage.

design.md §calibration_sanity cov_action_coverage: under a representative workload, **each of the
5 Admitter actions is chosen at least 1% of the time**. If any action's
selection rate is 0, that action is dead code under this workload and
the 5-candidate cost model is theatre.

Test: parse the Admitter JSONL log produced by a cost_picks_xfree run. Compute
selection rates of {own_free, own_evict, cross_free, cross_evict,
defer}. Assert all 5 ≥ 1%.

Falsification: any action at 0 (or below 1%) means either:
  - the cost model mis-prices it (re-calibrate),
  - the action is genuinely unreachable in this workload (delete the
    candidate, or document as saturation-only),
  - or this workload's regime is too narrow for cov_action_coverage (try a
    different workload).

Per spec §calibration_sanity, own-free is allowed to dominate (60-90%); we just check
the others aren't 0.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admitter-log", required=True,
                    help="Path to admitter JSONL produced by SGLANG_HIMA_ADMITTER_LOG=")
    ap.add_argument("--min-rate", type=float, default=0.01,
                    help="Minimum selection rate per action (default 1%%)")
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
    if not post:
        print(f"FAIL: log has only {len(raw)} entries, none post-settle")
        return 1

    actions = {"own_free": 0, "own_evict": 0, "cross_free": 0,
               "cross_evict": 0, "defer": 0}
    for e in post:
        a = e.get("action", "?")
        actions[a] = actions.get(a, 0) + 1
    total = len(post)
    rates = {a: c / total for a, c in actions.items()}

    print(f"cov_action_coverage: post-settle decisions = {total}")
    print(f"cov_action_coverage: action rates:")
    for a in ("own_free", "own_evict", "cross_free", "cross_evict", "defer"):
        marker = "OK " if rates[a] >= args.min_rate else "LOW"
        print(f"  [{marker}] {a:12s} {rates[a]*100:6.2f}% ({actions[a]}/{total})")

    dead = [a for a in actions if rates[a] < args.min_rate]

    if dead:
        print(f"\nD6m-cov: FAIL — actions below {args.min_rate*100:g}%: {dead}")
        print(
            "  diagnostic: the 5-candidate cost model is effectively\n"
            f"  {5 - len(dead)}-candidate under this workload. Either:\n"
            "    (1) the cost model mis-prices these actions → recalibrate,\n"
            "    (2) the actions are genuinely unreachable here → document\n"
            "        as workload-specific or remove from the framework,\n"
            "    (3) the workload regime is too narrow → run additional\n"
            "        workloads (mamba-saturated, mixed, etc.) and combine logs."
        )
        return 1

    print(f"\nD6m-cov: PASS — all 5 actions ≥ {args.min_rate*100:g}% in this workload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
