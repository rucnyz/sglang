#!/usr/bin/env python3
"""Post-variance diagnostic: explain why L2 won (or didn't) on the
multi-turn long-horizon agent variance run.

Inputs: a variance root dir like
  dev/eval/runs/multiturn-variance-YYYYMMDD-HHMMSS

Reads each cell's budgeter.jsonl + multiturn_summary.json and produces:
  - Pool saturation: did baseline KV usage actually hit ABOVE_HIGH /
    truly saturate during the bench, and for how long?
  - Fire effectiveness: did (1,1) successful fires materially change
    pool capacities? KV cap before vs after fire? mamba cap?
  - Per-trial separation: where exactly is (1,1) better/worse than
    (0,0) — mean TTFT, P99 TTFT, P99 E2E, output TPS?
  - Queueing depth proxy: num_running_reqs / num_queue_reqs trajectory
    in baseline cell — is the engine actually queueing?

Output: a markdown table on stdout.
"""

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple


def load_summary(out_dir: str) -> Optional[dict]:
    fp = os.path.join(out_dir, "multiturn_summary.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        return json.load(f)


def load_budgeter(out_dir: str, cell: str) -> List[dict]:
    fp = os.path.join(out_dir, f"{cell}_budgeter.jsonl")
    if not os.path.exists(fp):
        return []
    rows = []
    with open(fp) as f:
        for L in f:
            try:
                rows.append(json.loads(L))
            except Exception:
                pass
    return rows


def load_metrics(out_dir: str) -> List[dict]:
    fp = os.path.join(out_dir, "multiturn_metrics.jsonl")
    if not os.path.exists(fp):
        return []
    out = []
    with open(fp) as f:
        for L in f:
            try:
                out.append(json.loads(L))
            except Exception:
                pass
    return out


def diagnose_cell(root: str, trial: int, cell: Tuple[int, int]) -> dict:
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    out_dir = os.path.join(root, f"trial{trial}_{cstr}")
    summary = load_summary(out_dir)
    budgeter = load_budgeter(out_dir, cstr)
    metrics = load_metrics(out_dir)

    # ---- pool saturation analysis ----
    kv_above_high_ticks = 0
    mamba_above_high_ticks = 0
    peak_kv_inst = 0.0
    peak_mamba_inst = 0.0
    high_threshold_kv = 0.5  # default L2 threshold; just for "above high" counting
    high_threshold_m = 0.5
    n_running_max = 0
    n_queue_max = 0
    evict_total = 0
    fires = []
    for r in budgeter:
        kv = r.get("xpool_plan_usage_kv_inst", 0.0) or 0.0
        m = r.get("xpool_plan_usage_mamba_inst", 0.0) or 0.0
        if kv > peak_kv_inst:
            peak_kv_inst = kv
        if m > peak_mamba_inst:
            peak_mamba_inst = m
        if kv >= high_threshold_kv:
            kv_above_high_ticks += 1
        if m >= high_threshold_m:
            mamba_above_high_ticks += 1
        nr = r.get("num_running_reqs", 0) or 0
        nq = r.get("num_queue_reqs", 0) or 0
        if isinstance(nr, dict):
            nr = nr.get("total", 0)
        if isinstance(nq, dict):
            nq = nq.get("total", 0)
        if nr > n_running_max:
            n_running_max = nr
        if nq > n_queue_max:
            n_queue_max = nq
        evict_total += int(r.get("num_evicted_tokens_recent", 0) or 0)
        if r.get("xpool_direction") in ("kv_to_mamba", "mamba_to_kv"):
            fires.append({
                "tick": r.get("tick"),
                "direction": r.get("xpool_direction"),
                "unmapped": int(r.get("xpool_unmapped_total", 0) or 0),
                "granted": int(r.get("xpool_granted_total", 0) or 0),
                "kv_cap": r.get("xpool_kv_capacity_tokens"),
                "mamba_cap": r.get("xpool_mamba_capacity_tokens"),
                "skipped": r.get("xpool_skipped"),
            })

    # ---- per-request TTFT distribution ----
    valid_ttfts = sorted(
        m["ttft_ms"] for m in metrics if not m.get("error") and m.get("ttft_ms")
    )

    return {
        "cell": cstr,
        "trial": trial,
        "summary": summary,
        "n_ticks": len(budgeter),
        "peak_kv_inst": peak_kv_inst,
        "peak_mamba_inst": peak_mamba_inst,
        "kv_above_high_ticks": kv_above_high_ticks,
        "mamba_above_high_ticks": mamba_above_high_ticks,
        "n_running_max": n_running_max,
        "n_queue_max": n_queue_max,
        "evict_total": evict_total,
        "fires": fires,
        "ttft_n": len(valid_ttfts),
    }


def aggregate(diags: List[dict]) -> Dict[Tuple[int, int], dict]:
    by_cell: Dict[Tuple[int, int], List[dict]] = {}
    for d in diags:
        c = d["cell"]
        # cell is like "L10_L20" → L1 char at index 2, L2 char at index 6
        if c.startswith("L1") and len(c) >= 7:
            cell = (int(c[2]), int(c[6]))
        else:
            continue
        by_cell.setdefault(cell, []).append(d)

    agg = {}
    for cell, runs in by_cell.items():
        agg[cell] = {
            "n_runs": len(runs),
            "peak_kv_avg": statistics.mean(r["peak_kv_inst"] for r in runs),
            "peak_mamba_avg": statistics.mean(r["peak_mamba_inst"] for r in runs),
            "kv_above_high_ticks_avg": statistics.mean(
                r["kv_above_high_ticks"] for r in runs
            ),
            "mamba_above_high_ticks_avg": statistics.mean(
                r["mamba_above_high_ticks"] for r in runs
            ),
            "n_running_max_avg": statistics.mean(r["n_running_max"] for r in runs),
            "n_queue_max_avg": statistics.mean(r["n_queue_max"] for r in runs),
            "evict_total_avg": statistics.mean(r["evict_total"] for r in runs),
            "fires_with_movement_avg": statistics.mean(
                sum(1 for f in r["fires"] if f["unmapped"] > 0) for r in runs
            ),
            "fires_attempted_avg": statistics.mean(
                len(r["fires"]) for r in runs
            ),
            "summaries": [r["summary"] for r in runs if r["summary"]],
        }
        # mean TTFT / P99 / E2E / output_tps from summary
        s = [s for s in agg[cell]["summaries"]]
        if s:
            for key in (
                "mean_ttft_ms",
                "p99_ttft_ms",
                "mean_e2e_ms",
                "p99_e2e_ms",
                "input_tps",
                "output_tps",
                "num_requests_valid",
                "num_errors",
                "max_session_tokens_observed",
            ):
                vals = [x.get(key, 0) for x in s if x.get(key) is not None]
                if vals:
                    agg[cell][key + "_mean"] = statistics.mean(vals)
                    agg[cell][key + "_std"] = (
                        statistics.stdev(vals) if len(vals) > 1 else 0
                    )
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="variance root dir")
    ap.add_argument("--n-trials", type=int, default=None)
    args = ap.parse_args()

    cells = [(0, 0), (1, 0), (0, 1), (1, 1)]
    diags = []
    for cell in cells:
        cstr = f"L1{cell[0]}_L2{cell[1]}"
        for trial in range(1, 100):
            out = os.path.join(args.root, f"trial{trial}_{cstr}")
            if not os.path.isdir(out):
                break
            d = diagnose_cell(args.root, trial, cell)
            diags.append(d)
    agg = aggregate(diags)

    print("\n## Multi-turn variance diagnostic\n")
    print(f"Root: `{args.root}`")
    print(f"Cells discovered: {sorted(agg.keys())}")
    print()
    print(
        "| cell | n_runs | TTFT_mean_ms | TTFT_p99_ms | E2E_mean_ms | "
        "out_TPS | reqs | errors | peak_KV | peak_M | KV>HIGH ticks | "
        "M>HIGH ticks | running_max | queue_max | evict_tot | fires_attempted | "
        "fires_w_mv |"
    )
    print(
        "|------|-------:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|"
        "-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|"
    )
    for cell in sorted(agg.keys()):
        a = agg[cell]
        row = [
            f"({cell[0]},{cell[1]})",
            str(a["n_runs"]),
            f"{a.get('mean_ttft_ms_mean',0):.0f}±{a.get('mean_ttft_ms_std',0):.0f}",
            f"{a.get('p99_ttft_ms_mean',0):.0f}±{a.get('p99_ttft_ms_std',0):.0f}",
            f"{a.get('mean_e2e_ms_mean',0):.0f}±{a.get('mean_e2e_ms_std',0):.0f}",
            f"{a.get('output_tps_mean',0):.0f}±{a.get('output_tps_std',0):.0f}",
            f"{a.get('num_requests_valid_mean',0):.0f}",
            f"{a.get('num_errors_mean',0):.0f}",
            f"{a['peak_kv_avg']:.2f}",
            f"{a['peak_mamba_avg']:.2f}",
            f"{a['kv_above_high_ticks_avg']:.0f}",
            f"{a['mamba_above_high_ticks_avg']:.0f}",
            f"{a['n_running_max_avg']:.0f}",
            f"{a['n_queue_max_avg']:.0f}",
            f"{a['evict_total_avg']:.0f}",
            f"{a['fires_attempted_avg']:.1f}",
            f"{a['fires_with_movement_avg']:.1f}",
        ]
        print("| " + " | ".join(row) + " |")

    print()
    print("## Diagnostic interpretation\n")
    if (0, 0) in agg and (1, 1) in agg:
        a00 = agg[(0, 0)]
        a11 = agg[(1, 1)]
        peak_kv = a00["peak_kv_avg"]
        ticks = a00["kv_above_high_ticks_avg"]
        queue = a00["n_queue_max_avg"]
        evict = a00["evict_total_avg"]
        ttft_diff_ms = (
            a00.get("mean_ttft_ms_mean", 0) - a11.get("mean_ttft_ms_mean", 0)
        )
        ttft_diff_pct = (
            100 * ttft_diff_ms / a00.get("mean_ttft_ms_mean", 1)
            if a00.get("mean_ttft_ms_mean", 0) > 0
            else 0
        )
        print(
            f"- Baseline (0,0) peak KV usage: {peak_kv:.2f} "
            f"(ticks above HIGH=0.5: {ticks:.0f})"
        )
        print(f"- Baseline queue depth max: {queue:.0f}")
        print(f"- Baseline cumulative tree-cache evict: {evict:.0f} tokens")
        print(
            f"- (1,1) joint fires attempted: {a11['fires_attempted_avg']:.1f} "
            f"avg, with movement: {a11['fires_with_movement_avg']:.1f} avg"
        )
        print(
            f"- TTFT mean delta (0,0) → (1,1): {ttft_diff_ms:+.0f} ms "
            f"({ttft_diff_pct:+.1f}%)"
        )
        print()
        # Diagnostic hints
        if peak_kv < 0.95:
            print("⚠️  Baseline KV peak < 0.95 — workload may NOT be truly KV-binding.")
            print("    Increase concurrency, raise turn-input-tokens, or lower mem-fraction.")
        if queue < 5:
            print("⚠️  Baseline queue max < 5 — engine isn't queueing reqs.")
            print("    Without queueing, expanding mamba/KV via L2 has no req to admit.")
            print("    Increase concurrency until queue_max > 10 sustainedly.")
        if a11["fires_with_movement_avg"] < 2:
            print("⚠️  (1,1) avg fires_with_movement < 2 — L2 not firing often enough.")
            print("    Check XPOOL_UNIT, KV_HIGH/MAMBA_HIGH thresholds, COOLDOWN.")
        if abs(ttft_diff_pct) < 3:
            print("⚠️  TTFT delta < 3% — within typical run-to-run noise.")
        if ttft_diff_pct > 5:
            print(
                f"✅  TTFT mean improvement {ttft_diff_pct:+.1f}% — (1,1) outperforms "
                f"baseline beyond noise."
            )


if __name__ == "__main__":
    main()
