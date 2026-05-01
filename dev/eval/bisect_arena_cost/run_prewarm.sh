#!/bin/bash
# Experiment (a): does pre-warming KV pages reduce arena's trial-to-trial
# variance? If yes → TLB is the source.
#
# 3 trials interleaved: C0 (no-arena baseline) and C1 (pure arena), each
# trial does 200-prompt warmup + 500-prompt timed bench.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOAD="$SCRIPT_DIR/workload_prewarm.sh"

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-33600}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-prewarm-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"
echo "[prewarm-bisect] run_root=$RUN_ROOT gpu=$GPU n_trials=$N_TRIALS"

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
  echo "[prewarm-bisect] $cell_name (port=$port out=$out_dir)"
  ( $profile_fn
    export OUT_DIR="$out_dir"; export PORT="$port"
    export METRICS_PATH="$out_dir/metrics.json"
    bash "$WORKLOAD" 2>&1 | tee "$out_dir/runner.log"
  ) || echo "[prewarm-bisect] $cell_name FAILED"
  if [ -f "$out_dir/metrics.json" ]; then
    echo "[prewarm-bisect] $cell_name: $(cat $out_dir/metrics.json)"
  fi
}

for trial in $(seq 1 $N_TRIALS); do
  echo
  echo "=========================================================="
  echo "[prewarm-bisect] TRIAL $trial / $N_TRIALS"
  echo "=========================================================="
  run_cell "trial${trial}_C0_baseline"  env_C0_baseline  $((trial*2))
  run_cell "trial${trial}_C1_pure_arena" env_C1_pure_arena $((trial*2 + 1))
done

echo
echo "=========================================================="
echo "[prewarm-bisect] SUMMARY ($RUN_ROOT)"
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

print(f"\n{'metric':<18} {'C0 mean±std':>20} {'C1 mean±std':>20} {'delta':>10}")
print("-" * 75)
for k in ["input_tps", "mean_ttft_ms", "p99_ttft_ms", "median_e2e_ms", "mean_e2e_ms"]:
    m0, s0 = stat(c0, k); m1, s1 = stat(c1, k)
    d = (m1 - m0) / m0 * 100 if m0 else 0
    print(f"{k:<18} {m0:>11.2f} ± {s0:>5.2f} {m1:>11.2f} ± {s1:>5.2f} "
          f"{d:>+9.2f}%")

print(f"\nN trials: C0={len(c0)} C1={len(c1)}")
print()
print("Reference (NO-prewarm 5-trial, prior commit 3438f46d5):")
print("  C0 mean_ttft 51.80 ± 1.70 ms")
print("  C1 mean_ttft 55.51 ± 5.79 ms")
print("  delta +7.15%, C1 std 3.4× larger than C0")
print()
print("TLB hypothesis prediction: C1 std drops toward C0's, delta narrows.")
print("If C1 std stays >>3 ms, TLB is NOT the source.")
PY
