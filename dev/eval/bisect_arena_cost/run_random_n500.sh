#!/bin/bash
# n=500 C0-vs-C1 single-pass to nail the arena structural cost magnitude.
# Random 512in/128out RPS=8 — same workload as run_random.sh but n=500
# instead of n=100 so percentiles stabilize. Skip C2/C3 since the n=100
# pass already showed C1≈C2≈C3 within 1.5ms.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKLOAD="$SCRIPT_DIR/random_workload_n500.sh"

GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-33200}"
RUN_NAME="${RUN_NAME:-random-n500-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="$SCRIPT_DIR/runs/$RUN_NAME"
mkdir -p "$RUN_ROOT"
echo "[bisect-n500] run_root=$RUN_ROOT gpu=$GPU"

env_common() { export MEM_FRACTION=0.8; export CUDA_VISIBLE_DEVICES=$GPU; }

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

run_cell() {
  local cell_name="$1" profile_fn="$2" idx="$3"
  local out_dir="$RUN_ROOT/$cell_name"
  mkdir -p "$out_dir"
  local port=$((PORT_BASE + idx))
  echo
  echo "=========================================================="
  echo "[bisect-n500] $cell_name (port=$port out=$out_dir)"
  echo "=========================================================="
  ( $profile_fn
    export OUT_DIR="$out_dir"; export PORT="$port"
    export METRICS_PATH="$out_dir/metrics.json"
    env | grep -E "^(SGLANG_|MEM_FRACTION|CUDA_VISIBLE)" | sort > "$out_dir/env.txt"
    bash "$WORKLOAD" 2>&1 | tee "$out_dir/runner.log"
  ) || echo "[bisect-n500] $cell_name FAILED"
  if [ -f "$out_dir/metrics.json" ]; then
    echo "[bisect-n500] $cell_name metrics: $(cat $out_dir/metrics.json)"
  fi
}

run_cell "C1_pure_arena" env_C1_pure_arena 0
run_cell "C0_baseline"   env_C0_baseline   1

echo
echo "=========================================================="
echo "[bisect-n500] SUMMARY ($RUN_ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$RUN_ROOT"
rows = []
for name in ["C0_baseline","C1_pure_arena"]:
    p = os.path.join(root, name, "metrics.json")
    if os.path.exists(p):
        d = json.load(open(p))
        rows.append((name, d.get("input_tps",0), d.get("mean_ttft_ms",0),
                     d.get("p99_ttft_ms",0), d.get("median_e2e_ms",0)))
print(f"{'cell':<22}{'tps':>10}{'mean_ttft':>14}{'p99_ttft':>14}{'med_e2e':>14}")
for r in rows:
    print(f"{r[0]:<22}{r[1]:>10.1f}{r[2]:>14.2f}{r[3]:>14.2f}{r[4]:>14.2f}")
if len(rows) == 2:
    base = rows[0]; arena = rows[1]
    pct = lambda a, b: (b - a) / a * 100 if a else 0
    print(f"\nC1 vs C0 deltas: tps {pct(base[1],arena[1]):+.2f}% "
          f"mean_ttft {pct(base[2],arena[2]):+.2f}% "
          f"p99_ttft {pct(base[3],arena[3]):+.2f}% "
          f"med_e2e {pct(base[4],arena[4]):+.2f}%")
PY
