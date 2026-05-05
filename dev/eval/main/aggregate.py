#!/usr/bin/env python3
"""Aggregate paper-main run results into one CSV.

Walks dev/eval/runs/<run-name>/<model>/<regime>/<cell-or-trial>/ and emits
one row per (model, regime, cell) — averaged across trials for the 4-cell
ablation, single row for vLLM and each static-best ratio.

Output columns match paper tab:main-cross-model:
    model regime cell n_trials out_tps mean_ttft_ms p99_ttft_ms median_e2e_ms reqs xfers_total xfers_granted

Usage:
    python3 dev/eval/main/aggregate.py <run-root>
    → writes <run-root>/main_table.csv to stdout
"""
import json
import os
import statistics
import sys
from pathlib import Path


def read_bench(p: Path) -> dict | None:
    """Read bench.json — handles both formats:
    - multiturn_summary.json: pretty-printed dict
    - sglang.bench_serving output: one JSON object per line (last is summary)
    """
    if not p.exists():
        return None
    try:
        with open(p) as f:
            content = f.read().strip()
        if not content:
            return None
        # Try parse as a single pretty JSON first (multi-turn case)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        # Fall back to last-line (bench_serving multi-row case)
        last = content.splitlines()[-1]
        return json.loads(last)
    except Exception:
        return None


def metric(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def summarize_cell(cell_dirs: list[Path]) -> dict | None:
    """Average bench numbers across trial directories for one cell."""
    rows = []
    for d in cell_dirs:
        bj = read_bench(d / "bench.json")
        if bj is None:
            continue
        rows.append(bj)
        # Also peek xpool summary if present
    if not rows:
        return None

    def col(name, *aliases):
        vals = [metric(r, name, *aliases) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return statistics.mean(vals) if vals else None

    out = {
        "n_trials": len(rows),
        "out_tps":      col("output_throughput", "out_throughput", "output_tps"),
        "mean_ttft_ms": col("mean_ttft_ms", "ttft_mean_ms"),
        "p99_ttft_ms":  col("p99_ttft_ms", "ttft_p99_ms"),
        "median_e2e_ms": col("median_e2e_latency_ms", "e2e_median_ms",
                             "median_e2e_ms", "p50_e2e_ms"),
        "mean_e2e_ms":  col("mean_e2e_latency_ms", "mean_e2e_ms", "e2e_mean_ms"),
        "reqs":         col("completed", "n_requests", "total_requests",
                            "num_requests_valid"),
    }
    # xpool stats from the first cell with budgeter.jsonl summary present
    fires_total = 0; granted_total = 0
    for d in cell_dirs:
        xs = d / "xpool_summary.json"
        if xs.exists():
            try:
                with open(xs) as f:
                    xj = json.load(f)
                fires_total += xj.get("fires_total", 0)
                granted_total += xj.get("granted_total", 0)
            except Exception:
                pass
    out["xfers_total"] = fires_total // max(1, len(rows))
    out["xfers_granted"] = granted_total // max(1, len(rows))
    return out


def collect(root: Path) -> list[dict]:
    rows = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for regime_dir in sorted(model_dir.iterdir()):
            if not regime_dir.is_dir():
                continue
            regime = regime_dir.name

            # 4-cell ablation: gather trials per cell label
            by_cell = {}
            for entry in regime_dir.iterdir():
                if not entry.is_dir():
                    continue
                # trial<N>_intra<I>_inter<J>
                name = entry.name
                if name.startswith("trial"):
                    parts = name.split("_", 1)
                    cell = parts[1] if len(parts) == 2 else name
                    by_cell.setdefault(cell, []).append(entry)
                elif name.startswith("static_best") or name == "vllm":
                    # static-best subdirs are nested one level
                    if name == "vllm":
                        by_cell["vllm"] = [entry]
                    else:
                        # static_best/ contains static_best_r0.3/, _r0.5/, ...
                        for sub in entry.iterdir():
                            if sub.is_dir() and sub.name.startswith("static_best_r"):
                                by_cell[sub.name] = [sub]

            for cell, dirs in sorted(by_cell.items()):
                summary = summarize_cell(dirs)
                if summary is None:
                    continue
                rows.append({
                    "model": model, "regime": regime, "cell": cell,
                    **summary,
                })
    return rows


def main():
    if len(sys.argv) != 2:
        print("usage: aggregate.py <run-root>", file=sys.stderr); sys.exit(1)
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr); sys.exit(1)
    rows = collect(root)
    if not rows:
        print("no rows produced", file=sys.stderr); sys.exit(1)

    cols = ["model", "regime", "cell", "n_trials",
            "out_tps", "mean_ttft_ms", "p99_ttft_ms",
            "median_e2e_ms", "mean_e2e_ms",
            "reqs", "xfers_total", "xfers_granted"]
    print(",".join(cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print(",".join(vals))


if __name__ == "__main__":
    main()
