#!/bin/bash
# Experiment (e): static_min sweep — directly tests TLB hypothesis.
#
# Hypothesis: GPU TLB pressure from 2 MiB pages on the 25 GiB cuMemMap
# range causes C1's elevated trial-to-trial variance. If true, mapping
# LESS of the pool at boot → fewer TLB entries needed → variance drops.
#
# SGLANG_ARENA_KV_MOBILE_SOFT_CHUNKS=N reserves N chunks of the KV
# arena as "unmapped soft", so only (max_chunks - N) × tokens_per_chunk
# is physically mapped at boot. Default N=0 (full pool mapped).
#
# Sweep: N ∈ {0 (baseline), max/2, 3*max/4, 7*max/8} — mapped fraction
# 100% → 50% → 25% → 12.5%. The pool's engine-visible size scales down
# with static_min, so the workload must fit in the smallest mapped region.
#
# For Qwen3.5-A3B with chunk_bytes=256MiB:
#   tokens_per_chunk ≈ 262144, max_chunks = 9 → KV max = 2.36M tokens
#   N=0:  static_min = 2.36M, mapped 100%
#   N=4:  static_min = 1.31M, mapped ~56%   (≈ default tot_aligned, safe)
#   N=6:  static_min = 786K,  mapped ~33%
#   N=7:  static_min = 524K,  mapped ~22%   (still fits 100×640-token req)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOAD="$SCRIPT_DIR/random_workload_n500.sh"

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-33800}"
N_TRIALS="${N_TRIALS:-2}"
RUN_NAME="${RUN_NAME:-static-min-sweep-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"
echo "[static-min-sweep] run_root=$RUN_ROOT gpu=$GPU"

env_common() { export MEM_FRACTION=0.8; export CUDA_VISIBLE_DEVICES=$GPU; }

env_C1_static_min() {
  env_common
  unset SGLANG_BUDGETER SGLANG_BUDGETER_XPOOL_PLANNER \
        SGLANG_BUDGETER_XPOOL_COORDINATED SGLANG_BUDGETER_TICK_S \
        SGLANG_HPB_LRU SGLANG_K_BIG \
        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE 2>/dev/null || true
  export SGLANG_ARENA_SHARED=1
  export SGLANG_ARENA_FROM_BLOB=1
  export SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))
  export SGLANG_ARENA_KV_MOBILE_SOFT_CHUNKS="$1"
}

run_cell() {
  local cell_name="$1" mobile_n="$2" idx="$3"
  local out_dir="$RUN_ROOT/$cell_name"
  mkdir -p "$out_dir"
  local port=$((PORT_BASE + idx))
  echo
  echo "[static-min-sweep] $cell_name (KV_MOBILE_SOFT_CHUNKS=$mobile_n port=$port)"
  ( env_C1_static_min "$mobile_n"
    export OUT_DIR="$out_dir"; export PORT="$port"
    export METRICS_PATH="$out_dir/metrics.json"
    bash "$WORKLOAD" 2>&1 | tee "$out_dir/runner.log"
  ) || echo "[static-min-sweep] $cell_name FAILED"
  if [ -f "$out_dir/metrics.json" ]; then
    echo "[static-min-sweep] $cell_name: $(cat $out_dir/metrics.json)"
  fi
}

idx=0
for trial in $(seq 1 $N_TRIALS); do
  for mobile_n in 0 4 6 7; do
    cell_name="trial${trial}_mobile${mobile_n}"
    run_cell "$cell_name" "$mobile_n" $idx
    idx=$((idx + 1))
  done
done

echo
echo "=========================================================="
echo "[static-min-sweep] SUMMARY ($RUN_ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os, statistics
root = "$RUN_ROOT"
n_trials = $N_TRIALS

groups = {0: [], 4: [], 6: [], 7: []}
for trial in range(1, n_trials + 1):
    for n in groups:
        p = os.path.join(root, f"trial{trial}_mobile{n}", "metrics.json")
        if os.path.exists(p):
            groups[n].append(json.load(open(p)))

def stat(arr, key):
    vs = [d.get(key, 0) for d in arr]
    if not vs: return (0, 0)
    if len(vs) == 1: return (vs[0], 0)
    return (statistics.mean(vs), statistics.stdev(vs))

print(f"{'mobile_chunks':<14}{'mapped_frac':<12}{'mean_ttft (ms)':<22}{'p99_ttft (ms)':<22}")
print("-" * 70)
for n in sorted(groups):
    arr = groups[n]
    if not arr: continue
    frac = (9 - n) / 9 * 100   # for max_chunks=9
    m_mean, m_std = stat(arr, "mean_ttft_ms")
    p_mean, p_std = stat(arr, "p99_ttft_ms")
    print(f"N={n:<11}{frac:>5.1f}%      "
          f"{m_mean:>7.2f} ± {m_std:>5.2f}      "
          f"{p_mean:>7.2f} ± {p_std:>5.2f}")

print()
print("TLB hypothesis prediction: mean_ttft std DROPS as mobile_chunks ↑")
print("(less mapped → fewer TLB entries → less cold-miss variance)")
PY
