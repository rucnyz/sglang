#!/bin/bash
# Bisection of the arena-on / cudaMalloc gap (cell_00 vs cell_11).
#
# Hypothesis (after fused_moe TLB hypothesis was REFUTED by
# dev/2e/40_arena_kernel_isolation.py): the 5.86% mean / 12.34% P99 TTFT
# gap on B2 cold_burst lives in scheduler-side bookkeeping. We bisect by
# adding L2 layers one at a time on top of the baseline and watching where
# the gap appears. Largest-to-smallest: start at full prelude L2 and
# remove components downward.
#
#   C0_baseline       : no arena, no budgeter (= cell_00)
#   C1_pure_arena     : arena only (cuMemMap range, from_blob tensors,
#                       arena-aware allocator). No budgeter.
#   C2_arena_budget   : + SGLANG_BUDGETER=1 (per-tick snapshot). No planner.
#   C3_arena_planner  : + SGLANG_BUDGETER_XPOOL_PLANNER=1 +
#                       SGLANG_BUDGETER_XPOOL_COORDINATED=1 + thresholds
#                       (= cell_11 minus L1).
#
# L1 (HPB-LRU, K_BIG) is held OFF for ALL cells so the gap we see is
# purely arena/L2 machinery.
#
# Workload: B2 cold_burst (paper headline). Three phases per cell, ~5 min
# each. Total ~20 min wall on a single GPU.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUITE_DIR="$SCRIPT_DIR/../regression_suite"
WORKLOAD="$SUITE_DIR/workloads/b2_cold_burst.sh"

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-33000}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"
echo "[bisect] run_root=$RUN_ROOT gpu=$GPU"

# ---- env profiles ----
# Common to every cell.
env_common() {
  export MEM_FRACTION=0.8
  export CUDA_VISIBLE_DEVICES=$GPU
}

env_C0_baseline() {
  env_common
  unset SGLANG_ARENA_SHARED SGLANG_ARENA_FROM_BLOB SGLANG_ARENA_CHUNK_BYTES \
        SGLANG_BUDGETER SGLANG_BUDGETER_XPOOL_PLANNER \
        SGLANG_BUDGETER_XPOOL_COORDINATED SGLANG_BUDGETER_TICK_S \
        SGLANG_XPOOL_KV_HIGH SGLANG_XPOOL_KV_LOW \
        SGLANG_XPOOL_MAMBA_HIGH SGLANG_XPOOL_MAMBA_LOW \
        SGLANG_XPOOL_COOLDOWN SGLANG_XPOOL_EDGE_TRIGGER \
        SGLANG_HPB_LRU SGLANG_HPB_WINDOW_S SGLANG_K_BIG SGLANG_K_BIG_AUTO_THRESHOLD \
        SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE 2>/dev/null || true
}

env_C1_pure_arena() {
  env_C0_baseline
  export SGLANG_ARENA_SHARED=1
  export SGLANG_ARENA_FROM_BLOB=1
  export SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))
}

env_C2_arena_budget() {
  env_C1_pure_arena
  export SGLANG_BUDGETER=1
  export SGLANG_BUDGETER_TICK_S=2.0
}

env_C3_arena_planner() {
  env_C2_arena_budget
  export SGLANG_BUDGETER_XPOOL_PLANNER=1
  export SGLANG_BUDGETER_XPOOL_COORDINATED=1
  export SGLANG_XPOOL_KV_HIGH=0.85
  export SGLANG_XPOOL_KV_LOW=0.40
  export SGLANG_XPOOL_MAMBA_HIGH=0.80
  export SGLANG_XPOOL_MAMBA_LOW=0.40
  export SGLANG_XPOOL_COOLDOWN=2
  export SGLANG_XPOOL_EDGE_TRIGGER=1
}

run_cell() {
  local cell_name="$1"
  local profile_fn="$2"
  local idx="$3"
  local out_dir="$RUN_ROOT/$cell_name"
  mkdir -p "$out_dir"
  local port=$((PORT_BASE + idx))
  echo
  echo "=========================================================="
  echo "[bisect] $cell_name (port=$port out=$out_dir)"
  echo "=========================================================="

  ( $profile_fn
    export OUT_DIR="$out_dir"
    export PORT="$port"
    export METRICS_PATH="$out_dir/metrics.json"
    env | grep -E "^(SGLANG_|MEM_FRACTION|CUDA_VISIBLE)" | sort \
      > "$out_dir/env.txt"
    bash "$WORKLOAD" 2>&1 | tee "$out_dir/runner.log"
  ) || echo "[bisect] $cell_name FAILED"

  if [ -f "$out_dir/metrics.json" ]; then
    echo "[bisect] $cell_name metrics: $(cat $out_dir/metrics.json)"
  fi
}

# Sequential — one cell at a time on the same GPU.
# Order: largest-to-smallest (C3 down to C0) per user direction
# "从大缩小到小，看看去掉哪些部分这个开销依然存在".
run_cell "C3_arena_planner"  env_C3_arena_planner  0
run_cell "C2_arena_budget"   env_C2_arena_budget   1
run_cell "C1_pure_arena"     env_C1_pure_arena     2
run_cell "C0_baseline"       env_C0_baseline       3

# Summary table.
echo
echo "=========================================================="
echo "[bisect] SUMMARY ($RUN_ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os, glob
root = "$RUN_ROOT"
rows = []
for name in ["C0_baseline","C1_pure_arena","C2_arena_budget","C3_arena_planner"]:
    p = os.path.join(root, name, "metrics.json")
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            rows.append((name, d.get("input_tps",0),
                         d.get("mean_ttft_ms",0), d.get("p99_ttft_ms",0),
                         d.get("median_e2e_ms",0)))
        except Exception as e:
            rows.append((name, f"ERR {e}", "", "", ""))
    else:
        rows.append((name, "MISSING", "", "", ""))
print(f"{'cell':<22}{'tps':>10}{'mean_ttft':>14}{'p99_ttft':>14}{'med_e2e':>14}")
for r in rows:
    print(f"{r[0]:<22}{r[1]!s:>10}{r[2]!s:>14}{r[3]!s:>14}{r[4]!s:>14}")
PY
