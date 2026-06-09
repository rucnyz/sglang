#!/usr/bin/env python3
"""#208 — aggregate the A3 (HBM-pressure) 5-arm + const-V_u campaign.

Multi-metric: per-trial wall time is high-variance (runaway tail), so the
load-bearing signals are the lower-variance, scheduler-sensitive ones —
cache hit rate, re-prefill volume, prefill/queue latency, and daemon
activity.  All computed per cycle, then mean ± std across the N cycles.

Metrics per arm:
  wall_s        per-trial wall mean (s)            — high variance, context only
  cache_hit%    n_cache_tokens / n_input_tokens    — PRIMARY (ratio, low var)
  reprefill_Mt  (n_input − n_cache)/1e6 tokens      — prefill work not avoided
  output_kt     n_output_tokens/1e3                 — work-done sanity (confound)
  errs          n_errored_trials                    — failure-rate confound
  e2e_p50/p99   per-request end-to-end latency (s)   — from sglang request.finished
  queue_p50     per-request queue wait (s)          — admission/pressure latency
  migr/paus     daemon migrate_applied / pauses     — what the daemon DID (daemon arms)

Usage:
    python parse_a3_campaign.py <STAMP>      # e.g. 20260605_085244
    python parse_a3_campaign.py              # reads /tmp/a3_campaign_stamp.txt
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
sys.path.insert(0, str(_SHARED))
from parse_4arm import RESULTS, cycle_stats  # noqa: E402

try:
    from parse_ttft import parse_sglang_json  # noqa: E402
except Exception:  # pragma: no cover
    parse_sglang_json = None
try:
    from parse_daemon_events import summarize_cycle as _daemon_summary  # noqa: E402
except Exception:  # pragma: no cover
    _daemon_summary = None

ARM_PREFIX = {
    "LRU": "run_LRU_now", "TA": "run_TA_now", "OURS_inline": "run_H_prime_now",
    "OURS_full": "run_K_a3", "const_V_u": "run_K_a3",
}
ARM_TAG = {
    "LRU": "lru", "TA": "ta", "OURS_inline": "ours_inline",
    "OURS_full": "ours_full", "const_V_u": "const_vu",
}
DAEMON_ARMS = {"OURS_full", "const_V_u"}


def arm_dirs(arm: str, stamp: str):
    pat = f"{ARM_PREFIX[arm]}_a3camp_{stamp}_{ARM_TAG[arm]}_cycle*"
    return sorted(d for d in RESULTS.glob(pat) if d.is_dir())


def _cycle_aggregate_stats(d: Path):
    """The cycle-level aggregate result.json carries n_input/cache/output."""
    best = None
    for p in glob.glob(str(d / "harbor_jobs/**/result.json"), recursive=True):
        try:
            s = json.load(open(p)).get("stats", {})
        except Exception:
            continue
        if s.get("n_input_tokens"):
            best = s
    return best


def _ttft_metrics(d: Path):
    """Per-request e2e + queue latency from the cycle's sglang log."""
    if parse_sglang_json is None:
        return None
    log = d / "sglang_v4flash.log"
    if not log.exists():
        return None
    e2e, queue = [], []
    for r in parse_sglang_json(log):
        if r["e2e_latency"] > 0:
            e2e.append(r["e2e_latency"])
        queue.append(r["queue_time"])
    if not e2e:
        return None
    e2e.sort(); queue.sort()
    return {
        "e2e_p50": e2e[len(e2e) // 2], "e2e_p99": e2e[int(0.99 * len(e2e))],
        "queue_p50": queue[len(queue) // 2] if queue else 0.0,
    }


def _daemon_metrics(d: Path):
    if _daemon_summary is None:
        return None
    log = d / "daemon.log"
    if not log.exists():
        return None
    try:
        s = _daemon_summary(log)
    except Exception:
        return None
    occ = [o for _, o in s.get("occ_series", [])] or [0.0]
    return {
        "migr": s.get("migrate_applied", 0),
        "paus": len(s.get("pauses", [])),
        "occ_peak": max(occ), "occ_mean": statistics.mean(occ),
    }


def _ms(xs):
    if not xs:
        return "—"
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:.2f}±{sd:.2f}"


def main() -> int:
    stamp = sys.argv[1] if len(sys.argv) > 1 else \
        Path("/tmp/a3_campaign_stamp.txt").read_text().strip().split("=", 1)[1]
    print(f"# A3 (HBM-pressure) 5-arm + const-V_u campaign — stamp {stamp}\n")
    summ = {}
    for arm in ARM_PREFIX:
        dirs = arm_dirs(arm, stamp)
        acc = {k: [] for k in ("wall", "hit", "repf", "out", "err",
                               "e2e50", "e2e99", "q50", "migr", "paus",
                               "occp", "occm")}
        for d in dirs:
            cs = cycle_stats(d)
            agg = _cycle_aggregate_stats(d)
            if cs is None or agg is None:
                continue
            acc["wall"].append(cs["mean"])
            ni, nc, no = agg["n_input_tokens"], agg["n_cache_tokens"], agg["n_output_tokens"]
            acc["hit"].append(100 * nc / ni)
            acc["repf"].append((ni - nc) / 1e6)
            acc["out"].append(no / 1e3)
            acc["err"].append(agg.get("n_errored_trials", 0))
            tt = _ttft_metrics(d)
            if tt:
                acc["e2e50"].append(tt["e2e_p50"]); acc["e2e99"].append(tt["e2e_p99"])
                acc["q50"].append(tt["queue_p50"])
            if arm in DAEMON_ARMS:
                dm = _daemon_metrics(d)
                if dm:
                    acc["migr"].append(dm["migr"]); acc["paus"].append(dm["paus"])
                    acc["occp"].append(dm["occ_peak"]); acc["occm"].append(dm["occ_mean"])
        if acc["hit"]:
            summ[arm] = acc
            print(f"## {arm} (N={len(acc['hit'])})\n")
            print(f"* wall_s       {_ms(acc['wall'])}")
            print(f"* cache_hit%   {_ms(acc['hit'])}    ← PRIMARY")
            print(f"* reprefill_Mt {_ms(acc['repf'])}")
            print(f"* output_kt    {_ms(acc['out'])}   errs {_ms(acc['err'])}")
            if acc["e2e50"]:
                print(f"* e2e_p50 {_ms(acc['e2e50'])}  e2e_p99 {_ms(acc['e2e99'])}  "
                      f"queue_p50 {_ms(acc['q50'])}")
            if acc["migr"]:
                print(f"* daemon: migr_applied {_ms(acc['migr'])}  pauses {_ms(acc['paus'])}  "
                      f"occ_peak {_ms(acc['occp'])}  occ_mean {_ms(acc['occm'])}")
            print()

    # Welch-z on the low-variance primary metrics (not wall).
    def zrow(a, b, key, label):
        if a not in summ or b not in summ:
            return None
        xa, xb = summ[a][key], summ[b][key]
        if len(xa) < 1 or len(xb) < 1:
            return None
        ma, mb = statistics.mean(xa), statistics.mean(xb)
        sa = statistics.stdev(xa) if len(xa) > 1 else 0.0
        sb = statistics.stdev(xb) if len(xb) > 1 else 0.0
        se = (sa ** 2 / len(xa) + sb ** 2 / len(xb)) ** 0.5
        z = (ma - mb) / se if se > 0 else 0.0
        return f"| {label} | {ma:.2f} vs {mb:.2f} | {ma-mb:+.2f} | {se:.2f} | {z:+.2f} |"

    print("## Decomposition — cache_hit% (Welch z; |z|>1.96 = sig)\n")
    print("| comparison | A vs B | Δ | SE | z |")
    print("|---|---|---|---|---|")
    for a, b, lbl in [("OURS_full", "const_V_u", "V_u RANKING (full−const)"),
                      ("const_V_u", "LRU", "MACHINERY (const−LRU)"),
                      ("OURS_full", "OURS_inline", "DAEMON (full−inline)"),
                      ("OURS_full", "TA", "vs TA"),
                      ("OURS_full", "LRU", "vs LRU")]:
        r = zrow(a, b, "hit", lbl)
        if r:
            print(r)

    print("\n## Ranking by cache_hit% (higher = better eviction)\n")
    for arm in sorted(summ, key=lambda x: -statistics.mean(summ[x]["hit"])):
        print(f"* {arm}: {statistics.mean(summ[arm]['hit']):.2f}%  "
              f"(wall {statistics.mean(summ[arm]['wall']):.0f}s, "
              f"reprefill {statistics.mean(summ[arm]['repf']):.2f}M)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
