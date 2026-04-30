#!/usr/bin/env python3
"""Aggregate Setting 1 (24-h phase-shift, 4-cell ablation) results into a
paper-ready table.

Usage:
    python _aggregate_phase_shift.py <out_dir>

Reads <out_dir>/{cell}_phase_{A,B}_bench.json and {cell}_phase_C_summary.txt
for cells L1{0,1}_L2{0,1}, plus {cell}_budgeter.jsonl for L2=1 cells.
"""

import json
import os
import sys


def read_bench_last(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    lines = [l for l in open(path) if l.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def count_xpool(path: str) -> tuple:
    if not os.path.exists(path):
        return (0, 0, 0)
    k2m = m2k = total = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("xpool_direction") == "kv_to_mamba":
                k2m += 1
            elif d.get("xpool_direction") == "mamba_to_kv":
                m2k += 1
            if "xpool_direction" in d:
                total += 1
    return (total, k2m, m2k)


def main(out_dir: str) -> None:
    print(f"=== Setting 1 (24-h phase-shift) results: {out_dir} ===\n")
    print(f"{'cell':<10} {'phase':<6} {'TPS':>8} {'mean TTFT':>11} "
          f"{'P99 TTFT':>10} {'med E2E':>10} {'xfers':>6}")
    print("-" * 70)
    for L1 in (0, 1):
        for L2 in (0, 1):
            cell = f"L1{L1}_L2{L2}"
            for phase in ("A", "B"):
                d = read_bench_last(f"{out_dir}/{cell}_phase_{phase}_bench.json")
                if not d:
                    print(f"{cell:<10} {phase:<6} {'N/A':>8} "
                          f"{'N/A':>11} {'N/A':>10} {'N/A':>10}")
                    continue
                xfers = "-"
                if L2 == 1 and phase == "B":
                    t, _, _ = count_xpool(f"{out_dir}/{cell}_budgeter.jsonl")
                    xfers = str(t)
                print(f"{cell:<10} {phase:<6} "
                      f"{d.get('input_throughput',0):>8.1f} "
                      f"{d.get('mean_ttft_ms',0):>9.1f}ms "
                      f"{d.get('p99_ttft_ms',0):>8.1f}ms "
                      f"{d.get('median_e2e_latency_ms',0):>8.1f}ms "
                      f"{xfers:>6}")
            # Phase C (multi-turn) summary
            pc = f"{out_dir}/{cell}_phase_C_summary.txt"
            if os.path.exists(pc):
                with open(pc) as f:
                    fields = dict(line.strip().split('=', 1)
                                  for line in f if '=' in line)
                xfers = "-"
                if L2 == 1:
                    t, _, _ = count_xpool(f"{out_dir}/{cell}_budgeter.jsonl")
                    xfers = str(t)
                print(f"{cell:<10} {'C':<6} {'N/A':>8} "
                      f"{'N/A':>11} {'N/A':>10} "
                      f"{float(fields.get('mean_ms',0)):>8.1f}ms {xfers:>6}")
            print()

    # Cross-pool transfer detail
    print("=== Cross-pool transfer summary (L2=1 cells only) ===")
    for L1 in (0, 1):
        cell = f"L1{L1}_L21"
        path = f"{out_dir}/{cell}_budgeter.jsonl"
        if not os.path.exists(path):
            continue
        t, k2m, m2k = count_xpool(path)
        print(f"  {cell}: total={t}  kv→mamba={k2m}  mamba→kv={m2k}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase_shift_v3_1777548459"
    main(out_dir)
