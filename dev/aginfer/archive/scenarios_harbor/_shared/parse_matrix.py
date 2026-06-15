#!/usr/bin/env python3
"""Aggregate T9 N=3 matrix results.

Usage:
    python verify/t9/parse_matrix.py <matrix_root>

For each `cycleN_<config>/` (symlink or dir) under matrix_root:
  * Parse harbor_jobs/<run_id>/instance_*/result.json → per-trial duration_s
  * Parse sglang.log Prefill-batch lines → cumulative new/cached tokens

Aggregate per config:
  * across-cycle mean ± std of (per-cycle mean trial duration)
  * across-cycle mean ± std of cache hit ratio (cached / (cached + new))

Emit SUMMARY.md in matrix_root.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PREFILL_RE = re.compile(
    r"Prefill batch.*"
    r"#new-token:\s*(\d+).*"
    r"#cached-token:\s*(\d+)"
)


def parse_harbor(cycle_dir: Path):
    """Return list of per-trial durations (s)."""
    durations = []
    harbor_jobs = cycle_dir / "harbor_jobs"
    if not harbor_jobs.exists():
        return durations
    for run_dir in sorted(harbor_jobs.iterdir()):
        if not run_dir.is_dir():
            continue
        for inst_dir in sorted(run_dir.iterdir()):
            res = inst_dir / "result.json"
            if not res.exists():
                continue
            try:
                r = json.loads(res.read_text())
            except Exception:
                continue
            if r.get("started_at") and r.get("finished_at"):
                s = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
                f = datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00"))
                durations.append((f - s).total_seconds())
    return durations


def parse_sglang(cycle_dir: Path):
    """Return (total_new, total_cached) tokens summed over Prefill batches."""
    # run_k.sh copies the real launch-time log here.
    log_path = cycle_dir / "sglang_v4flash.log"
    if not log_path.exists():
        # Fallback: the wrapper-stdout log (mostly empty).
        log_path = cycle_dir / "sglang.log"
    if not log_path.exists():
        return (0, 0)
    total_new = 0
    total_cached = 0
    try:
        with log_path.open(errors="ignore") as f:
            for line in f:
                m = PREFILL_RE.search(line)
                if m:
                    total_new += int(m.group(1))
                    total_cached += int(m.group(2))
    except Exception as e:
        print(f"warn: failed to parse {log_path}: {e}", file=sys.stderr)
    return (total_new, total_cached)


def stats(xs):
    if not xs:
        return dict(n=0)
    return dict(
        n=len(xs),
        mean=statistics.mean(xs),
        stdev=statistics.stdev(xs) if len(xs) > 1 else 0.0,
        p50=statistics.median(xs),
        p99=sorted(xs)[int(0.99 * len(xs))] if len(xs) > 2 else max(xs),
        min=min(xs),
        max=max(xs),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix_root", type=Path)
    args = ap.parse_args()

    root: Path = args.matrix_root
    cycles_by_config = defaultdict(list)  # config → [(cycle_idx, dir)]
    for child in sorted(root.iterdir()):
        m = re.match(r"cycle(\d+)_(\w+)", child.name)
        if not m:
            continue
        idx, cfg = int(m.group(1)), m.group(2)
        target = child.resolve()
        cycles_by_config[cfg].append((idx, target))

    print(f"# T9 N=3 matrix aggregation: {root.name}\n")
    print(f"Generated: {datetime.now().isoformat()}\n")

    config_summaries = {}
    for cfg in sorted(cycles_by_config):
        print(f"## Config: {cfg}\n")
        cycle_means = []
        cycle_hits = []
        per_cycle_rows = []
        for idx, cyc_dir in cycles_by_config[cfg]:
            durations = parse_harbor(cyc_dir)
            new_tok, cached_tok = parse_sglang(cyc_dir)
            hit_rate = cached_tok / (cached_tok + new_tok) if (cached_tok + new_tok) else 0.0
            d_stats = stats(durations)
            per_cycle_rows.append({
                "cycle": idx,
                "dir": str(cyc_dir),
                "n_trials": d_stats.get("n", 0),
                "mean_s": d_stats.get("mean"),
                "p50_s": d_stats.get("p50"),
                "p99_s": d_stats.get("p99"),
                "stdev_s": d_stats.get("stdev"),
                "new_tok": new_tok,
                "cached_tok": cached_tok,
                "hit_rate": hit_rate,
            })
            if d_stats.get("mean") is not None:
                cycle_means.append(d_stats["mean"])
            cycle_hits.append(hit_rate)

        # Per-cycle table
        print("| cycle | n | mean_s | p50_s | p99_s | stdev_s | new_tok | cached_tok | hit_rate |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in per_cycle_rows:
            mean_s = r['mean_s']
            p50_s = r['p50_s']
            p99_s = r['p99_s']
            stdev_s = r['stdev_s']
            print(
                f"| {r['cycle']} | {r['n_trials']} | "
                f"{mean_s:.1f} | "
                f"{p50_s:.1f} | "
                f"{p99_s:.1f} | "
                f"{stdev_s:.1f} | "
                f"{r['new_tok']} | {r['cached_tok']} | {r['hit_rate']:.4f} |"
                if mean_s is not None else
                f"| {r['cycle']} | {r['n_trials']} | n/a | n/a | n/a | n/a | "
                f"{r['new_tok']} | {r['cached_tok']} | {r['hit_rate']:.4f} |"
            )
        print()

        # Across-cycle aggregate
        if cycle_means:
            m_mean = statistics.mean(cycle_means)
            m_std = statistics.stdev(cycle_means) if len(cycle_means) > 1 else 0.0
            h_mean = statistics.mean(cycle_hits) if cycle_hits else 0.0
            h_std = statistics.stdev(cycle_hits) if len(cycle_hits) > 1 else 0.0
            print(f"**Across-cycle (N={len(cycle_means)}):**")
            print(f"* per-trial mean: **{m_mean:.1f} ± {m_std:.1f} s**")
            print(f"* cache hit rate: **{h_mean:.4f} ± {h_std:.4f}**")
            config_summaries[cfg] = (m_mean, m_std, h_mean, h_std, len(cycle_means))
        print()

    # Final comparison
    if "baseline" in config_summaries and "ours" in config_summaries:
        b_mean, b_std, b_hit, b_hit_std, b_n = config_summaries["baseline"]
        o_mean, o_std, o_hit, o_hit_std, o_n = config_summaries["ours"]
        delta = o_mean - b_mean
        # Welch-ish: SE of difference
        se = (b_std**2 / b_n + o_std**2 / o_n) ** 0.5 if b_n and o_n else 0.0
        z = delta / se if se > 0 else 0.0
        print("## Final comparison")
        print()
        print(f"| metric | baseline | ours | Δ |")
        print(f"|---|---|---|---|")
        print(f"| per-trial mean s | {b_mean:.1f} ± {b_std:.1f} | {o_mean:.1f} ± {o_std:.1f} | {delta:+.1f} |")
        print(f"| cache hit rate | {b_hit:.4f} ± {b_hit_std:.4f} | {o_hit:.4f} ± {o_hit_std:.4f} | {o_hit - b_hit:+.4f} |")
        print()
        print(f"**Δ mean = {delta:+.1f} s, SE ≈ {se:.1f} s, z ≈ {z:+.2f}**")
        if z < -2.0:
            print("✓ ours significantly faster than baseline (z < -2σ)")
        elif z > 2.0:
            print("✗ ours significantly SLOWER than baseline (z > 2σ)")
        else:
            print("✗ no significant difference within noise (|z| < 2σ)")


if __name__ == "__main__":
    main()
