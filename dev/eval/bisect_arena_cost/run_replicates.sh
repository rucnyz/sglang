#!/bin/bash
# n=500 multi-trial C0-vs-C1 — variance bands. 3 trials each cell sequentially.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOAD="$SCRIPT_DIR/random_workload_n500.sh"

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-33300}"
N_TRIALS="${N_TRIALS:-2}"
RUN_NAME="${RUN_NAME:-replicates-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"
echo "[bisect-replicates] run_root=$RUN_ROOT gpu=$GPU n_trials=$N_TRIALS"

env_common() { export MEM_FRACTION=0.8; export CUDA_VISIBLE_DEVICES=$GPU; }

env_C0_baseline() {
  env_common
  unset SGLANG_ARENA_SHARED SGLANG_ARENA_FROM_BLOB SGLANG_ARENA_CHUNK_BYTES \
        SGLANG_BUDGETER SGLANG_BUDGETER_XPOOL_PLANNER \
        SGLANG_BUDGETER_XPOOL_COORDINATED SGLANG_BUDGETER_TICK_S \
        SGLANG_HPB_LRU SGLANG_K_BIG \
        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE 2>/dev/null || true
}

env_C1_pure_arena() {
  env_C0_baseline
  export SGLANG_ARENA_SHARED=1
  export SGLANG_ARENA_FROM_BLOB=1
  export SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))
}

run_cell() {
  local cell_name="$1" profile_fn="$2" idx="$3"
  local out_dir="$RUN_ROOT/$cell_name"
  mkdir -p "$out_dir"
  local port=$((PORT_BASE + idx))
  echo
  echo "[bisect-replicates] $cell_name (port=$port out=$out_dir)"
  ( $profile_fn
    export OUT_DIR="$out_dir"; export PORT="$port"
    export METRICS_PATH="$out_dir/metrics.json"
    bash "$WORKLOAD" 2>&1 | tee "$out_dir/runner.log"
  ) || echo "[bisect-replicates] $cell_name FAILED"
  if [ -f "$out_dir/metrics.json" ]; then
    echo "[bisect-replicates] $cell_name: $(cat $out_dir/metrics.json)"
  fi
}

# Interleave A B A B A B trials so any time-varying GPU behavior
# (thermal, etc) affects both cells equally.
for trial in $(seq 1 $N_TRIALS); do
  echo
  echo "=========================================================="
  echo "[bisect-replicates] TRIAL $trial / $N_TRIALS"
  echo "=========================================================="
  run_cell "trial${trial}_C0_baseline"  env_C0_baseline  $((trial*2))
  run_cell "trial${trial}_C1_pure_arena" env_C1_pure_arena $((trial*2 + 1))
done

echo
echo "=========================================================="
echo "[bisect-replicates] SUMMARY ($RUN_ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os, statistics
root = "$RUN_ROOT"
n_trials = $N_TRIALS

def load(name):
    p = os.path.join(root, name, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None

c0 = []; c1 = []
for trial in range(1, n_trials + 1):
    d0 = load(f"trial{trial}_C0_baseline")
    d1 = load(f"trial{trial}_C1_pure_arena")
    if d0: c0.append(d0)
    if d1: c1.append(d1)

def stat(arr, key):
    vals = [d.get(key, 0) for d in arr if d]
    if not vals: return (0, 0)
    if len(vals) == 1: return (vals[0], 0)
    return (statistics.mean(vals), statistics.stdev(vals))

print(f"\n{'metric':<18} {'C0 mean±std':>18} {'C1 mean±std':>18} {'delta':>10}")
print("-" * 72)
for k, label in [("input_tps", "input_tps"),
                 ("mean_ttft_ms", "mean_ttft_ms"),
                 ("p99_ttft_ms", "p99_ttft_ms"),
                 ("median_e2e_ms", "median_e2e_ms"),
                 ("mean_e2e_ms", "mean_e2e_ms")]:
    m0, s0 = stat(c0, k); m1, s1 = stat(c1, k)
    d = (m1 - m0) / m0 * 100 if m0 else 0
    print(f"{label:<18} {m0:>10.2f} ± {s0:>5.2f} {m1:>10.2f} ± {s1:>5.2f} "
          f"{d:>+9.2f}%")

print(f"\nN trials: C0={len(c0)} C1={len(c1)}")
PY
