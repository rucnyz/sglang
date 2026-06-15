#!/usr/bin/env python3
"""Do-no-harm A-vs-B: per-task agent-execution time, ours vs baseline.

`agent_execution` is the LLM-serving phase of each harbor task (prefill +
decode + tool round-trips) — the only phase the KV-scheduling daemon can
affect. We deliberately exclude docker env-setup and the verifier's test.sh
(daemon-irrelevant, and the source of the broken-task hangs). Errored tasks
(non-null exception_info) are excluded.

Usage: python3 parse.py [results_root]
"""
import json, glob, statistics, sys
from datetime import datetime

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
TAG = "tp4fix228"


def _parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def durations(arm):
    out = []
    for c in (1, 2, 3):
        pat = f"{ROOT}/run_K_{arm}_{TAG}_{arm}_cycle{c}/harbor_jobs/*/instance_*/result.json"
        for pf in glob.glob(pat):
            try:
                d = json.load(open(pf))
            except Exception:
                continue
            if d.get("exception_info") is not None:
                continue
            ae = d.get("agent_execution") or {}
            s, f = _parse(ae.get("started_at")), _parse(ae.get("finished_at"))
            if s and f:
                sec = (f - s).total_seconds()
                if sec > 0:
                    out.append(sec)
    return out


def main():
    rows = {}
    for arm in ("a3", "a3_kvoff"):
        ds = sorted(durations(arm))
        if not ds:
            print(f"{arm}: no data")
            continue
        rows[arm] = ds
        p90 = ds[int(0.9 * len(ds)) - 1]
        print(
            f"{arm:9s}  n={len(ds):3d}  agent_exec  "
            f"mean={statistics.mean(ds):6.1f}s  median={statistics.median(ds):6.1f}s  "
            f"std={statistics.pstdev(ds):6.1f}s  p90={p90:6.1f}s  max={max(ds):6.1f}s"
        )
    if {"a3", "a3_kvoff"} <= rows.keys():
        a, b = rows["a3"], rows["a3_kvoff"]
        ma, mb = statistics.mean(a), statistics.mean(b)
        sa, sb = statistics.pstdev(a), statistics.pstdev(b)
        delta = (ma - mb) / mb * 100
        # PLAN §5 significance: non-overlapping mean±std bands
        sig = (ma + sa) < (mb - sb) or (mb + sb) < (ma - sa)
        print(f"\nΔmean = {delta:+.1f}%  (ours vs baseline)")
        print(f"significant (mean±std bands disjoint)? {sig}  "
              f"→ do-no-harm {'VIOLATED' if sig and ma > mb else 'HOLDS'}")


if __name__ == "__main__":
    main()
