"""Zero-downside (no-regression) validator for the request-bounded cc A/B.

WIDE is designed to be zero-downside: the Admitter only ADDS admit options
(it never removes the baseline's), and a cross-fire harvests free / low-value
capacity, so the inter cell should never be worse than off. This validator
asserts that property on a REQUEST-BOUNDED replay (run_cc.sh with
MAX_SESSIONS=N) over N paired reps.

Two confounds this guards against:

1. Time-bounded replay (MAX_SESSIONS=0): off and inter complete DIFFERENT
   session sets within the time cap, so cache_hit = Σcached/Σprompt is taken
   over different populations. Use --max-sessions N to fix the session set.

2. Deadline truncation / host contention: even with --max-sessions N, if the
   sessions do not all COMPLETE before --max-time-min (or the node is
   contended), the two cells finish different *fractions* of the same N
   sessions, so their request counts diverge and the comparison is invalid
   (observed: a rep where off completed 2327 requests but inter only 1020).
   The COMPLETION-PARITY guard below excludes any rep whose two cells differ
   in num_requests_valid by more than --min-parity, so only reps where both
   cells did comparable work feed the no-regression verdict. Size N small
   enough that both cells fully complete all N sessions.

PASS iff, over the BALANCED reps, the median paired delta does not regress
beyond a noise band for any metric (mean/p99 TTFT, output tps, cache_hit).
Reuses cc_traces_headline/validate_cc.py plumbing. Companion to that
validator, which asserts a WIN; this one asserts no LOSS.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cc_traces_headline")
)
from validate_cc import _WIN_METRICS, _load_cell, _paired_delta  # noqa: E402


def _valid_count(out_dir: str, cell: str):
    """num_requests_valid for one cell, used for the completion-parity guard."""
    path = os.path.join(out_dir, cell, "bench.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    return d.get("num_requests_valid")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dirs", nargs="+", required=True,
                    help="N request-bounded off/inter run dirs (paired)")
    ap.add_argument("--eps-frac", type=float, default=0.03,
                    help="noise band for ttft/tps metrics (fractional)")
    ap.add_argument("--eps-pp", type=float, default=0.01,
                    help="noise band for cache_hit (percentage-point)")
    ap.add_argument("--min-parity", type=float, default=0.85,
                    help="exclude a rep if min(off,inter)/max(off,inter) of "
                         "num_requests_valid is below this (truncation guard)")
    args = ap.parse_args()

    runs = [_load_cell(d) for d in args.out_dirs]
    print(f"=== zero-downside (no-regression), request-bounded: "
          f"{', '.join(args.out_dirs)} ===")

    balanced = []
    for d, (off, inter, fires) in zip(args.out_dirs, runs):
        ov, iv = _valid_count(d, "off"), _valid_count(d, "inter_admitter")
        if not ov or not iv:
            print(f"  {d}: EXCLUDED — missing bench (off={ov} inter={iv})")
            continue
        parity = min(ov, iv) / max(ov, iv)
        parts = []
        for metric, direction, _ in _WIN_METRICS:
            dv = _paired_delta(off, inter, metric, direction)
            unit = "pp" if direction == "pp" else "%"
            parts.append(f"{metric}={dv * 100:+.2f}{unit}"
                         if dv is not None else f"{metric}=NA")
        tag = "ok" if parity >= args.min_parity else "EXCLUDED (truncated)"
        print(f"  {d}: off_req={ov} inter_req={iv} parity={parity:.2f} [{tag}]  "
              f"fires={fires}  " + "  ".join(parts))
        if parity >= args.min_parity:
            balanced.append((off, inter, fires))

    if not balanced:
        print("\nZERO-DOWNSIDE: INCONCLUSIVE — no rep had balanced completion "
              "(shrink N_SESSIONS / raise MAX_TIME_MIN / quieter node)")
        return 2

    regressions = []
    print(f"\nZero-downside per-metric over {len(balanced)} balanced rep(s) "
          f"(median per-run delta; negative = regression):")
    for metric, direction, _ in _WIN_METRICS:
        deltas = [_paired_delta(off, inter, metric, direction)
                  for off, inter, _ in balanced]
        deltas = [d for d in deltas if d is not None]
        if not deltas:
            print(f"  [ NA ] {metric}: no data")
            continue
        med = statistics.median(deltas)
        eps = args.eps_pp if direction == "pp" else args.eps_frac
        ok = med >= -eps
        unit = "pp" if direction == "pp" else "%"
        marker = "ok  " if ok else "REGR"
        print(f"  [{marker}] {metric}: median Δ={med * 100:+.2f}{unit} "
              f"(per-run {[round(d * 100, 2) for d in deltas]}) "
              f"floor=-{eps * 100:g}{unit}")
        if not ok:
            regressions.append(metric)

    if regressions:
        print(f"\nZERO-DOWNSIDE: FAIL — regressed beyond the noise band on "
              f"{regressions}")
        return 1
    print("\nZERO-DOWNSIDE: PASS — no metric regressed beyond the noise band")
    return 0


if __name__ == "__main__":
    sys.exit(main())
