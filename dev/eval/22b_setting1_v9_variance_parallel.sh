#!/bin/bash
# Parallel 4-cell × 3-trial variance run for v9-auto.
# Distributes 12 cell-runs across 7 GPUs (skipping GPU 2 which often
# has stale state). Two waves: 7 jobs in wave 1, 5 jobs in wave 2.
# Total wall-clock ≈ 2 × 12 min = ~25 min vs. ~150 min sequential.
#
# Use:
#   bash dev/eval/22b_setting1_v9_variance_parallel.sh

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30100}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-v9-variance-parallel-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[v9-variance-parallel] root=$ROOT"

# GPUs to use, in order. Wave assignment is round-robin.
GPUS=(0 1 3 4 5 6 7)
N_GPUS=${#GPUS[@]}

CELLS=("0 0" "1 0" "0 1" "1 1")

# Build the full job list: (trial, L1, L2) tuples in deterministic order.
JOBS=()
for trial in $(seq 1 $N_TRIALS); do
  for pair in "${CELLS[@]}"; do
    set -- $pair
    JOBS+=("$trial $1 $2")
  done
done
N_JOBS=${#JOBS[@]}
echo "[v9-variance-parallel] $N_JOBS jobs across ${N_GPUS} GPUs (${GPUS[@]})"

# Launch jobs in waves; wave size = N_GPUS.
launch_wave() {
  local start=$1 end=$2
  local pids=()
  local i=$start
  while [ $i -lt $end ] && [ $i -lt $N_JOBS ]; do
    set -- ${JOBS[$i]}
    local trial=$1 L1=$2 L2=$3
    local cell="L1${L1}_L2${L2}"
    local out_dir="$ROOT/trial${trial}_${cell}"
    local gpu=${GPUS[$((i % N_GPUS))]}
    local port=$((PORT_BASE + i))
    mkdir -p "$out_dir"
    echo "  [job $i] trial=$trial cell=$cell gpu=$gpu port=$port → $out_dir"
    ONLY_L1=$L1 ONLY_L2=$L2 \
      CUDA_VISIBLE_DEVICES=$gpu PORT=$port OUT_DIR="$out_dir" \
      SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      bash dev/eval/21_setting1_v9_pool_binding.sh \
      > "$out_dir/runner.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
  done
  echo "  waiting on ${#pids[@]} pids: ${pids[@]}"
  for p in "${pids[@]}"; do
    wait $p || echo "  pid $p exited non-zero"
  done
}

# Run waves of N_GPUS jobs each.
i=0
wave=1
while [ $i -lt $N_JOBS ]; do
  echo "=========================================================="
  echo "[v9-variance-parallel] WAVE $wave (jobs $i..$((i+N_GPUS-1)))"
  echo "=========================================================="
  launch_wave $i $((i + N_GPUS))
  i=$((i + N_GPUS))
  wave=$((wave + 1))
done

echo
echo "=========================================================="
echo "[v9-variance-parallel] SUMMARY ($ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os, statistics
root = "$ROOT"
n_trials = $N_TRIALS

phases = ["A", "B", "C"]
cells = [(0,0),(1,0),(0,1),(1,1)]
metrics = ["mean_ttft_ms", "p99_ttft_ms", "input_throughput", "median_e2e_latency_ms"]

def load(trial, cell, phase):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"trial{trial}_{cstr}", f"{cstr}_phase_{phase}_bench.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

print(f"{'phase':<6}{'cell':<8}{'TPS':>10}{'mean_TTFT':>12}{'P99_TTFT':>12}{'mean_E2E':>12}")
for phase in phases:
    for cell in cells:
        cstr = f"({cell[0]},{cell[1]})"
        runs = [load(t, cell, phase) for t in range(1, n_trials+1)]
        runs = [r for r in runs if r is not None]
        if not runs:
            print(f"{phase:<6}{cstr:<8}  (no data)")
            continue
        means = {m: statistics.mean(r.get(m, 0) for r in runs) for m in metrics}
        stds  = {m: (statistics.stdev(r.get(m, 0) for r in runs) if len(runs) > 1 else 0) for m in metrics}
        def fmt(m, unit=""): return f"{means[m]:.1f}±{stds[m]:.1f}{unit}"
        print(f"{phase:<6}{cstr:<8}{fmt('input_throughput'):>10}{fmt('mean_ttft_ms','ms'):>12}{fmt('p99_ttft_ms','ms'):>12}{fmt('median_e2e_latency_ms','ms'):>12}")
PY

echo "[v9-variance-parallel] done"
