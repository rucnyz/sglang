"""Parse the Tier-2 e2e characterization sweep (#230) into a comparison table.

Reads results/run_K_a3_char_<arm>_p<pressure>_c<cycle>/ dirs, aggregates per
arm across cycles (mean ± std), and prints markdown.  The headline metric is
radix prefix cache-hit rate (the inverse of re-prefill waste — a unit evicted
then reused = a miss = recompute); plus TTFT, throughput, and the
inline-evict-vs-imperative split.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[3] / "results"


def _daemon_metrics(d: Path) -> dict:
    dl = d / "daemon.log"
    out = {"migrate": 0, "rejects": None, "hint_delay_ms": 0}
    if not dl.exists():
        return out
    txt = dl.read_text(errors="ignore")
    out["migrate"] = txt.count("event=migrate_enqueued")
    m = re.findall(r'n_failures_total=(\d+)', txt)
    if m:
        out["rejects"] = int(m[-1])
    dd = re.findall(r'delay_ms=(\d+)', txt)
    if dd:
        out["hint_delay_ms"] = int(dd[-1])
    return out


def _cache_hit(d: Path) -> float | None:
    """sglang cache-report hit rate from the sglang log (--enable-cache-report).
    Falls back to None if absent."""
    sl = d / "sglang.log"
    if not sl.exists():
        return None
    hits = totals = 0
    for line in sl.read_text(errors="ignore").splitlines():
        m = re.search(r'cached?[_-]?token[s]?[^0-9]*(\d+).*?(?:prompt|input)[^0-9]*(\d+)',
                      line, re.I)
        if m:
            hits += int(m.group(1)); totals += int(m.group(2))
    return hits / totals if totals else None


def _harbor_resolved(d: Path) -> tuple[int, int]:
    jobs = list((d / "harbor_jobs").glob("**/result.json"))
    ok = 0
    for j in jobs:
        try:
            r = json.loads(j.read_text())
            if r.get("resolved") or r.get("passed") or r.get("success"):
                ok += 1
        except Exception:
            pass
    return ok, len(jobs)


def main() -> int:
    dirs = sorted(glob.glob(str(RESULTS / "run_K_a3_char_*")))
    if not dirs:
        print("no char_* result dirs yet — run e2e/run_sweep.sh first")
        return 0
    by_arm: dict[str, list[Path]] = defaultdict(list)
    for d in dirs:
        name = os.path.basename(d)
        m = re.match(r"run_K_a3_char_(.+)_p\d+_c\d+", name)
        if m:
            by_arm[m.group(1)].append(Path(d))

    print("# Tier-2 e2e characterization results (#230)\n")
    print("| arm | cycles | cache-hit (mean) | rejects | migrate | "
          "resolved | hint-delay |")
    print("|---|---|---|---|---|---|---|")
    for arm in sorted(by_arm):
        ds = by_arm[arm]
        hits = [h for h in (_cache_hit(d) for d in ds) if h is not None]
        dm = [_daemon_metrics(d) for d in ds]
        rej = [m["rejects"] for m in dm if m["rejects"] is not None]
        mig = [m["migrate"] for m in dm]
        delay = next((m["hint_delay_ms"] for m in dm if m["hint_delay_ms"]), 0)
        res = [_harbor_resolved(d) for d in ds]
        rsum = sum(o for o, _ in res); tsum = sum(t for _, t in res)
        hit_s = f"{statistics.mean(hits):.3f}" if hits else "n/a"
        rej_s = f"{statistics.mean(rej):.0f}" if rej else "n/a"
        mig_s = f"{statistics.mean(mig):.0f}" if mig else "n/a"
        print(f"| {arm} | {len(ds)} | {hit_s} | {rej_s} | {mig_s} | "
              f"{rsum}/{tsum} | {delay}ms |")
    print("\n_higher cache-hit = less re-prefill waste = better eviction. "
          "The ours_d* gradient is the on-hardware hint-latency budget._")
    return 0


if __name__ == "__main__":
    sys.exit(main())
