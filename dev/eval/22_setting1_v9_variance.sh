#!/bin/bash
# Setting 1 v9-auto variance bands: 4 cells × 3 trials = 12 cell runs.
# Each cell takes ~12 min (5 boot + 3×2 bench + sleeps), total ~150 min.
#
# Reuses 21_setting1_v9_pool_binding.sh as the per-cell driver; iterates
# over (L1, L2) and trial index, writes outputs to per-trial subdirs.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30099}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-v9-variance-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[v9-variance] root=$ROOT gpu=$GPU n_trials=$N_TRIALS"

CELLS=("0 0" "1 0" "0 1" "1 1")
idx=0
for trial in $(seq 1 $N_TRIALS); do
  for pair in "${CELLS[@]}"; do
    set -- $pair
    L1=$1; L2=$2
    cell="L1${L1}_L2${L2}"
    out_dir="$ROOT/trial${trial}_${cell}"
    port=$((PORT_BASE + idx))
    idx=$((idx + 1))
    echo
    echo "=========================================================="
    echo "[v9-variance] trial=$trial cell=$cell port=$port"
    echo "=========================================================="
    ONLY_L1=$L1 ONLY_L2=$L2 \
      CUDA_VISIBLE_DEVICES=$GPU PORT=$port OUT_DIR="$out_dir" \
      SGLANG_K_BIG_AUTO_THRESHOLD=0.5 \
      bash dev/eval/21_setting1_v9_pool_binding.sh \
      2>&1 | tee "$out_dir/runner.log" || echo "[v9-variance] $cell trial$trial FAILED"
  done
done

echo
echo "=========================================================="
echo "[v9-variance] SUMMARY ($ROOT)"
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

print(f"\n{'cell':<10}{'phase':<4}{'metric':<22}{'mean':>10}{'std':>9}{'n':>4}")
print("-" * 64)
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
            cstr = f"L1{cell[0]}_L2{cell[1]}"
            print(f"{cstr:<10}{phase:<4}{k:<22}{m:>10.1f}{s:>9.1f}{len(vals):>4}")
PY
