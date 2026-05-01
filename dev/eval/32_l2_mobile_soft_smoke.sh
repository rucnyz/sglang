#!/bin/bash
# L2 mobile-soft smoke — verifies that L2 fires actually move physical
# chunks when SGLANG_ARENA_*_MOBILE_SOFT_CHUNKS is set non-zero.
#
# Background (2026-05-01 afternoon discovery): every prior L2 eval
# committed (q3b-variance, v9-variance, all today's L2-positive search
# runs, etc.) was launched WITHOUT mobile-soft env vars set, so
# init_chunks == static_min_chunks → shared free queue empty at boot
# → every L2 fire returned xpool_unmapped_total=0 / granted_total=0
# (dormant). The paper's "L2 = no-regression" finding is therefore on
# a configuration where L2 physically did nothing.
#
# This script enables mobile-soft (KV and mamba each donate 4 chunks
# to shared queue at boot, ≈ 8 GB total byte-movement room with 1 GB
# chunks) on L2-on cells, runs the v9 4-cell ablation, and checks
# whether fires move chunks (granted_total > 0) and whether the
# joint cell shows any measurable delta vs L1-only.
#
# Run on GPU 2. ~30 min per cell × 4 cells ≈ 2 h.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30900}"
RUN_NAME="${RUN_NAME:-l2-mobile-soft-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[l2-mobile-soft] root=$ROOT gpu=$GPU"

CELLS=("0 0" "1 0" "0 1" "1 1")

idx=0
for pair in "${CELLS[@]}"; do
  set -- $pair
  L1=$1; L2=$2
  cell="L1${L1}_L2${L2}"
  out_dir="$ROOT/$cell"
  mkdir -p "$out_dir"
  port=$((PORT_BASE + idx))
  idx=$((idx + 1))
  echo
  echo "=========================================================="
  echo "[l2-mobile-soft] $cell port=$port (L1=$L1 L2=$L2 mobile-soft=4)"
  echo "=========================================================="
  ONLY_L1=$L1 ONLY_L2=$L2 \
    CUDA_VISIBLE_DEVICES=$GPU PORT=$port OUT_DIR="$out_dir" \
    SGLANG_K_BIG_AUTO_THRESHOLD=0.5 \
    SGLANG_L2_MOBILE_SOFT_KV_CHUNKS=1 \
    SGLANG_L2_MOBILE_SOFT_MAMBA_CHUNKS=0 \
    bash dev/eval/21_setting1_v9_pool_binding.sh \
    2>&1 | tee "$out_dir/runner.log" || echo "[l2-mobile-soft] $cell FAILED"
done

echo
echo "=========================================================="
echo "[l2-mobile-soft] SUMMARY"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$ROOT"
cells = ["L10_L20", "L11_L20", "L10_L21", "L11_L21"]
phases = ["A", "B", "C"]

print(f"\n{'cell':<10}{'phase':<6}{'TPS_in':>8}{'mean_ttft':>11}{'p99_ttft':>11}{'med_e2e':>11}")
print("-"*60)
for cell in cells:
    for phase in phases:
        fp = f"{root}/{cell}/{cell}_phase_{phase}_bench.json"
        if not os.path.exists(fp):
            print(f"{cell:<10}{phase:<6} (no data)")
            continue
        with open(fp) as f:
            lines = [l for l in f if l.strip()]
        d = json.loads(lines[-1])
        print(f"{cell:<10}{phase:<6}{d['input_throughput']:>8.0f}{d['mean_ttft_ms']:>11.1f}{d['p99_ttft_ms']:>11.1f}{d['median_e2e_latency_ms']:>11.1f}")

print()
print("L2 fire stats (granted_total > 0 means real chunk movement):")
print(f"{'cell':<10}{'fires':>7}{'kv2m':>7}{'m2kv':>7}{'granted_sum':>14}{'unmapped_sum':>14}")
print("-"*60)
import re
for cell in ["L10_L21", "L11_L21"]:
    fp = f"{root}/{cell}/{cell}_budgeter.jsonl"
    if not os.path.exists(fp):
        print(f"{cell:<10} (no log)")
        continue
    fires, kv2m, m2kv, granted, unmapped = 0, 0, 0, 0, 0
    with open(fp) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            direction = d.get("xpool_direction")
            if direction in ("kv_to_mamba", "mamba_to_kv"):
                fires += 1
                if direction == "kv_to_mamba": kv2m += 1
                else: m2kv += 1
                granted += d.get("xpool_granted_total", 0)
                unmapped += d.get("xpool_unmapped_total", 0)
    print(f"{cell:<10}{fires:>7}{kv2m:>7}{m2kv:>7}{granted:>14}{unmapped:>14}")

print()
print("Pass: granted_sum > 0 in either L2-on cell → mobile-soft works,")
print("L2 actuator physically moves chunks. Now compare metrics across cells.")
PY
