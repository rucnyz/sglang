#!/bin/bash
# Setting 1 v9-auto gate retuning sweep — addresses the "positive
# interaction term" finding from contribution attribution.
#
# Hypothesis: with default MAMBA_HIGH=0.08, the gate fires on Phase A's
# transient mamba peaks, paying actuator cost on a phase that doesn't
# benefit. Higher threshold should suppress those fires and recover
# Phase A throughput while preserving Phase C improvements.
#
# Sweep MAMBA_HIGH ∈ {0.08, 0.20, 0.50, 0.80} on (L1=1, L2=1) cell
# only. ~12 min/cell × 4 = 48 min wall.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30200}"
RUN_NAME="${RUN_NAME:-gate-retune-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[gate-retune] root=$ROOT gpu=$GPU"

THRESHOLDS=(0.08 0.20 0.50 0.80)
idx=0
for t in "${THRESHOLDS[@]}"; do
  cell="mamba_high_${t}"
  out_dir="$ROOT/$cell"
  mkdir -p "$out_dir"
  port=$((PORT_BASE + idx))
  idx=$((idx + 1))
  echo
  echo "=========================================================="
  echo "[gate-retune] $cell (MAMBA_HIGH=$t) port=$port"
  echo "=========================================================="
  ONLY_L1=1 ONLY_L2=1 \
    CUDA_VISIBLE_DEVICES=$GPU PORT=$port OUT_DIR="$out_dir" \
    SGLANG_K_BIG_AUTO_THRESHOLD=0.5 \
    SGLANG_XPOOL_MAMBA_HIGH=$t \
    bash dev/eval/21_setting1_v9_pool_binding.sh \
    2>&1 | tee "$out_dir/runner.log" || echo "[gate-retune] $cell FAILED"
done

echo
echo "=========================================================="
echo "[gate-retune] SUMMARY ($ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$ROOT"
print(f"\n{'mamba_high':<12}{'phase':<6}{'mean_ttft':>11}{'p99_ttft':>11}{'tps':>10}")
print("-"*52)
for t in ["0.08", "0.20", "0.50", "0.80"]:
    for ph in ["A","B","C"]:
        cell = f"mamba_high_{t}"
        fp = f"{root}/{cell}/L11_L21_phase_{ph}_bench.json"
        if not os.path.exists(fp): continue
        with open(fp) as f:
            lines = [l for l in f if l.strip()]
        d = json.loads(lines[-1])
        print(f"{t:<12}{ph:<6}{d['mean_ttft_ms']:>11.1f}{d['p99_ttft_ms']:>11.1f}{d['input_throughput']:>10.0f}")

# Reference: v9 baseline (0,0)
print("\nReference (3-trial mean from variance run):")
print("  (0,0) Phase A: mean 1818, P99 3929, TPS 82116")
print("  (1,1) Phase A @ MAMBA_HIGH=0.08: mean 3105, P99 9468, TPS 76186")
PY
