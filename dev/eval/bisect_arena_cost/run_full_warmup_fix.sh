#!/bin/bash
# Test the FULL warmup fix: SGLANG_ARENA_ZERO_INIT_LIVE=1 +
# SGLANG_ARENA_WARMUP=1. The latter is a new model_runner-level hook
# that walks every KV/mamba arena page via tensor.sum() AFTER all
# initialization (CUDA graph capture, autotune) — so TLBs are warm at
# the moment serving starts, not 30s before.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOAD="$SCRIPT_DIR/random_workload_n500.sh"

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-34000}"
N_TRIALS="${N_TRIALS:-5}"
RUN_NAME="${RUN_NAME:-full-warmup-fix-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"
echo "[full-warmup-fix] run_root=$RUN_ROOT gpu=$GPU n_trials=$N_TRIALS"

env_common() { export MEM_FRACTION=0.8; export CUDA_VISIBLE_DEVICES=$GPU; }

env_C0_baseline() {
  env_common
  unset SGLANG_ARENA_SHARED SGLANG_ARENA_FROM_BLOB SGLANG_ARENA_CHUNK_BYTES \
        SGLANG_ARENA_ZERO_INIT_LIVE SGLANG_ARENA_WARMUP \
        SGLANG_BUDGETER SGLANG_BUDGETER_XPOOL_PLANNER \
        SGLANG_BUDGETER_XPOOL_COORDINATED SGLANG_BUDGETER_TICK_S \
        SGLANG_HPB_LRU SGLANG_K_BIG \
        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE 2>/dev/null || true
}

env_C1_full_warmup() {
  env_C0_baseline
  export SGLANG_ARENA_SHARED=1
  export SGLANG_ARENA_FROM_BLOB=1
  export SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))
  export SGLANG_ARENA_ZERO_INIT_LIVE=1   # boot-time fill kernel TLB warm
  export SGLANG_ARENA_WARMUP=1           # post-init streaming-read TLB warm
}

run_cell() {
  local cell_name="$1" profile_fn="$2" idx="$3"
  local out_dir="$RUN_ROOT/$cell_name"
  mkdir -p "$out_dir"
  local port=$((PORT_BASE + idx))
  echo
  echo "[full-warmup-fix] $cell_name (port=$port out=$out_dir)"
  ( $profile_fn
    export OUT_DIR="$out_dir"; export PORT="$port"
    export METRICS_PATH="$out_dir/metrics.json"
    bash "$WORKLOAD" 2>&1 | tee "$out_dir/runner.log"
  ) || echo "[full-warmup-fix] $cell_name FAILED"
  if [ -f "$out_dir/metrics.json" ]; then
    echo "[full-warmup-fix] $cell_name: $(cat $out_dir/metrics.json)"
  fi
}

idx=0
for trial in $(seq 1 $N_TRIALS); do
  echo
  echo "=========================================================="
  echo "[full-warmup-fix] TRIAL $trial / $N_TRIALS"
  echo "=========================================================="
  run_cell "trial${trial}_C0_baseline"     env_C0_baseline    $idx
  idx=$((idx + 1))
  run_cell "trial${trial}_C1_full_warmup"  env_C1_full_warmup $idx
  idx=$((idx + 1))
done

echo
echo "=========================================================="
echo "[full-warmup-fix] SUMMARY ($RUN_ROOT)"
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
    d1 = load(f"trial{trial}_C1_full_warmup")
    if d0: c0.append(d0)
    if d1: c1.append(d1)

def stat(arr, key):
    vals = [d.get(key, 0) for d in arr if d]
    if not vals: return (0, 0)
    if len(vals) == 1: return (vals[0], 0)
    return (statistics.mean(vals), statistics.stdev(vals))

print(f"\n{'metric':<18} {'C0 mean±std':>20} {'C1+full-warmup mean±std':>26} {'delta':>10}")
print("-" * 82)
for k in ["input_tps", "mean_ttft_ms", "p99_ttft_ms", "median_e2e_ms", "mean_e2e_ms"]:
    m0, s0 = stat(c0, k); m1, s1 = stat(c1, k)
    d = (m1 - m0) / m0 * 100 if m0 else 0
    print(f"{k:<18} {m0:>11.2f} ± {s0:>5.2f} {m1:>17.2f} ± {s1:>5.2f} "
          f"{d:>+9.2f}%")

print(f"\nN trials: C0={len(c0)} C1+full-warmup={len(c1)}")
print()
print("References:")
print("  no-pretouch (5T):     C1 55.51 ± 5.79 ms, delta +7.15%")
print("  pretouch only (5T):   C1 52.34 ± 2.53 ms, delta +7.08%")
print("  bench pre-warm (3T):  C1 52.64 ± 0.61 ms, delta +2.98% (gold)")
print()
print("Full-warmup-fix prediction: C1 σ should approach 0.61 ms; mean delta")
print("should narrow toward +3% (the warm-state structural floor).")
PY
