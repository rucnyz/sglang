#!/bin/bash
# Admission-pressured workload search for L2-positive demonstration.
#
# Goal: workload heavy enough that stock cache can't evict fast enough
# → produces paused/retracted requests → net-benefit gate sees non-zero
# admission pressure → L2 fires can be justified by avoided re-prefill cost.
#
# Design:
#   - GSP at HIGH RPS (32 vs v9's 8) → many concurrent in-flight requests
#   - 64 groups × 30 prompts × 12K system prompt → distinct mamba snapshots
#     (mamba pool ≈ 362 slots; 64 distinct snapshots × ≥6 simultaneous in
#     flight → potential slot exhaustion)
#   - --gsp-output-len 256 → modest output, dominated by prefill
#
# 4 cells × 1 trial first (smoke test); if successful + L2 fires meaningfully,
# extend to 3-trial variance run.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30600}"
RUN_NAME="${RUN_NAME:-admission-pressured-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[admission-pressured] root=$ROOT gpu=$GPU"

# 4 cells (0,0)/(1,0)/(0,1)/(1,1)
CELLS=("0 0" "1 0" "0 1" "1 1")

idx=0
for pair in "${CELLS[@]}"; do
  set -- $pair
  L1=$1; L2=$2
  cell="L1${L1}_L2${L2}"
  out_dir="$ROOT/$cell"
  mkdir -p "$out_dir"
  port=$((PORT_BASE + idx))
  idx=$((idx + 1))
  log="$out_dir/server.log"

  # Build env stack — same as 21_setting1_v9_pool_binding.sh layering
  extra_env=""
  if [ "$L1" = "1" ]; then
    extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_K_BIG_AUTO_THRESHOLD=0.5 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
  fi
  if [ "$L2" = "1" ]; then
    extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024)) SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0 SGLANG_BUDGETER_LOG=$out_dir/budgeter.jsonl SGLANG_XPOOL_KV_HIGH=0.04 SGLANG_XPOOL_KV_LOW=0.015 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.03 SGLANG_XPOOL_COOLDOWN=2 SGLANG_XPOOL_NET_BENEFIT=1"
  fi
  mem_frac=0.8
  [ "$L2" = "1" ] && mem_frac=0.7

  echo
  echo "=========================================================="
  echo "[admission-pressured] $cell port=$port (L1=$L1 L2=$L2)"
  echo "=========================================================="
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 6

  CUDA_VISIBLE_DEVICES=$GPU nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path Qwen/Qwen3.5-35B-A3B --host 127.0.0.1 --port $port \
      --mem-fraction-static $mem_frac --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  pid=$!
  echo "[$cell] pid=$pid"

  waited=0
  while [ $waited -lt 300 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$port/health" 2>/dev/null)" = "200" ]; then
      echo "[$cell] ready after ${waited}s"; break
    fi
  done

  # Single phase: heavy GSP (admission-pressured)
  # 64 groups × 30 prompts = 1920 requests; RPS=32 means up to ~30s of
  # in-flight. 12K system prompt; many groups → many distinct snapshots.
  echo "[$cell] Phase pressured (GSP heavy)..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $port \
    --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 64 --gsp-prompts-per-group 30 \
    --gsp-system-prompt-len 12000 --gsp-question-len 64 \
    --gsp-output-len 256 \
    --request-rate 32 \
    --output-file "$out_dir/bench.json" \
    >"$out_dir/bench.log" 2>&1 || echo "[$cell] bench failed"

  # Stats from server log + budgeter
  pftot=$(grep -c "Prefill batch" "$log" 2>/dev/null || echo 0)
  retract=$(grep -cE "Retract|num_retract|retracted" "$log" 2>/dev/null || echo 0)
  pause=$(grep -cE "Pause|paused" "$log" 2>/dev/null || echo 0)
  if [ "$L2" = "1" ]; then
    fires=$(grep -c '"xpool_direction":' "$out_dir/budgeter.jsonl" 2>/dev/null | head -1 || echo 0)
    fires_real=$(grep -cE '"xpool_direction": "(kv_to_mamba|mamba_to_kv)"' "$out_dir/budgeter.jsonl" 2>/dev/null | head -1 || echo 0)
    echo "[$cell] prefill=$pftot retract_logs=$retract pause_logs=$pause L2_fires_total=$fires real_fires=$fires_real"
  else
    echo "[$cell] prefill=$pftot retract_logs=$retract pause_logs=$pause"
  fi

  kill -9 $pid 2>/dev/null || true
  sleep 6
done

echo
echo "=========================================================="
echo "[admission-pressured] SUMMARY"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$ROOT"
cells = ["L10_L20", "L11_L20", "L10_L21", "L11_L21"]
print(f"\n{'cell':<10}{'TPS':>10}{'mean_ttft':>11}{'p99_ttft':>11}{'med_e2e':>11}")
print("-"*53)
for cell in cells:
    fp = f"{root}/{cell}/bench.json"
    if not os.path.exists(fp):
        print(f"{cell:<10} (no data)")
        continue
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1])
    print(f"{cell:<10}{d['input_throughput']:>10.0f}{d['mean_ttft_ms']:>11.1f}{d['p99_ttft_ms']:>11.1f}{d['median_e2e_latency_ms']:>11.1f}")

print("\nLook for: L11 better than L10? L01/L11 budgeter shows fires?")
PY
