#!/bin/bash
# Q3.B 4-cell variance bands — symmetric to 22_setting1_v9_variance.sh
# but for the cold-burst trace (b2_cold_burst.sh).
#
# Q3.B is L2's headline trace per the paper's final framing: tab:q3b-4cell
# shows (1,1) > L1-only on every metric. Variance bands on this table
# mirror what Fig 7 + tab:variance-bands give for v9-auto headline.
#
# 4 cells × 3 trials = 12 runs, each ~5 min wall = ~60 min total.
# Reuses regression_suite/workloads/b2_cold_burst.sh as the per-cell
# bench (build/burst/recovery 3-phase trace, 12K-token GSP + 200 random
# 4K burst + GSP recovery). Cell envs match jobs.py PRELUDE_ENV /
# BASELINE_ENV with L1/L2 toggled per cell.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30400}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-q3b-variance-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[q3b-variance] root=$ROOT gpu=$GPU n_trials=$N_TRIALS"

# Common env (all cells)
common_env=(
  "MEM_FRACTION=0.8"
  "WARMUP_S=300"
)

# Per-cell env — corresponds to PRELUDE_ENV / BASELINE_ENV + L1/L2 toggle
cell_env_L00() { :; }   # baseline = no extra env
cell_env_L10() {        # L1 only (HPB-LRU + K_BIG, no arena/budgeter)
  export SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0
  export SGLANG_K_BIG=8192 SGLANG_K_BIG_AUTO_THRESHOLD=0.5
  export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
}
cell_env_L01() {        # L2 only (arena + budgeter + planner, no L1)
  export SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1
  export SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))
  export SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1
  export SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0
  export SGLANG_XPOOL_KV_HIGH=0.85 SGLANG_XPOOL_KV_LOW=0.40
  export SGLANG_XPOOL_MAMBA_HIGH=0.80 SGLANG_XPOOL_MAMBA_LOW=0.40
  export SGLANG_XPOOL_COOLDOWN=2 SGLANG_XPOOL_EDGE_TRIGGER=1
}
cell_env_L11() {        # Full system
  cell_env_L10
  cell_env_L01
}

CELLS=(L00 L10 L01 L11)
idx=0
for trial in $(seq 1 $N_TRIALS); do
  for cell in "${CELLS[@]}"; do
    out_dir="$ROOT/trial${trial}_${cell}"
    mkdir -p "$out_dir"
    port=$((PORT_BASE + idx))
    idx=$((idx + 1))
    echo
    echo "=========================================================="
    echo "[q3b-variance] trial=$trial cell=$cell port=$port"
    echo "=========================================================="
    metrics_path="$out_dir/metrics.json"
    (
      # subshell for env isolation
      for kv in "${common_env[@]}"; do export "$kv"; done
      cell_env_$cell
      export OUT_DIR="$out_dir" PORT=$port METRICS_PATH=$metrics_path
      export CUDA_VISIBLE_DEVICES=$GPU
      bash dev/eval/regression_suite/workloads/b2_cold_burst.sh \
        2>&1 | tee "$out_dir/runner.log"
    ) || echo "[q3b-variance] $cell trial$trial FAILED"
  done
done

echo
echo "=========================================================="
echo "[q3b-variance] SUMMARY ($ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os, statistics
root = "$ROOT"
n_trials = $N_TRIALS
cells = ["L00", "L10", "L01", "L11"]
phases = ["build", "burst", "recovery"]
metrics = ["mean_ttft_ms", "p99_ttft_ms", "input_throughput", "median_e2e_latency_ms"]

def load(trial, cell, phase):
    fp = f"{root}/trial{trial}_{cell}/{phase}_bench.json"
    if not os.path.exists(fp): return None
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

print(f"\n{'cell':<6}{'phase':<10}{'metric':<22}{'mean':>10}{'std':>9}{'n':>4}")
print("-"*66)
for cell in cells:
    for phase in phases:
        for k in metrics:
            vals = []
            for t in range(1, n_trials + 1):
                d = load(t, cell, phase)
                if d and k in d:
                    vals.append(d[k])
            if not vals: continue
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"{cell:<6}{phase:<10}{k:<22}{m:>10.1f}{s:>9.1f}{len(vals):>4}")
PY
