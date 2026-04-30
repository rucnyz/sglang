#!/bin/bash
# Same C0..C3 bisection as run.sh, but on the random-prefill workload that
# matches dev/2e/24_arena_from_blob_perf.sh (the original measurement that
# produced the 5.86%/12.34% gap claim in paper §sec:eval-arena-cost).
# B2 cold_burst recovery — used by run.sh — is GSP shared-prefix at RPS=2,
# which is cache-friendly enough that all 4 cells were indistinguishable
# (TPS spread 0.05%). On the random workload, the gap should reproduce
# (if it still exists with current code) and we can localize it.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOAD="$SCRIPT_DIR/random_workload.sh"

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-33100}"
RUN_NAME="${RUN_NAME:-random-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"
echo "[bisect-random] run_root=$RUN_ROOT gpu=$GPU"

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
  echo "[bisect-random] $cell_name (port=$port out=$out_dir)"
  echo "=========================================================="
  ( $profile_fn
    export OUT_DIR="$out_dir"
    export PORT="$port"
    export METRICS_PATH="$out_dir/metrics.json"
    env | grep -E "^(SGLANG_|MEM_FRACTION|CUDA_VISIBLE)" | sort \
      > "$out_dir/env.txt"
    bash "$WORKLOAD" 2>&1 | tee "$out_dir/runner.log"
  ) || echo "[bisect-random] $cell_name FAILED"

  if [ -f "$out_dir/metrics.json" ]; then
    echo "[bisect-random] $cell_name metrics: $(cat $out_dir/metrics.json)"
  fi
}

# Largest-to-smallest order (C3 down to C0).
run_cell "C3_arena_planner"  env_C3_arena_planner  0
run_cell "C2_arena_budget"   env_C2_arena_budget   1
run_cell "C1_pure_arena"     env_C1_pure_arena     2
run_cell "C0_baseline"       env_C0_baseline       3

echo
echo "=========================================================="
echo "[bisect-random] SUMMARY ($RUN_ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os
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
print()
# Compute pct deltas vs C0
if rows[0][1] not in (0, "MISSING") and not isinstance(rows[0][1], str):
    base = rows[0][1]
    print(f"{'cell':<22}{'tps_delta':>14}")
    for r in rows:
        if isinstance(r[1], (int, float)) and r[1]:
            d = (r[1] - base) / base * 100
            print(f"{r[0]:<22}{d:>+13.2f}%")
PY
