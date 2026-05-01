#!/bin/bash
# L2-positive search v2: force KV pool overflow by overriding
# --max-running-requests beyond what the default pool can sustain.
#
# Insight from prior run (l2-kv-overflow-20260501-143041 / L10_L20):
#   max_running_requests=120 is pre-computed from KV pool size.
#   With chunked prefill + scheduler cap, engine NEVER lets KV
#   actually overflow — reqs queue (mean_ttft=55s) but never retract.
#   So L2's "extra KV bytes" is invisible to a pre-capped scheduler.
#
# Workaround: pass --max-running-requests 240 (≈2x what pool can hold)
# to force the engine to admit more reqs than KV pool allows. Now KV
# WILL overflow → retract events → L2 mamba_to_kv has something to fix.
#
# Workload sized for short cells (~5-8 min):
#   random 24K × 240 reqs × RPS=48 + output 256
#   240 in-flight cap × 24K = 5.76M tokens >> 1.26M pool capacity
#   → engine MUST retract some reqs

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30800}"
RUN_NAME="${RUN_NAME:-l2-force-kv-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[l2-force-kv] root=$ROOT gpu=$GPU"

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

  extra_env=""
  if [ "$L1" = "1" ]; then
    extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_K_BIG_AUTO_THRESHOLD=0.5 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
  fi
  if [ "$L2" = "1" ]; then
    extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024)) SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0 SGLANG_BUDGETER_LOG=$out_dir/budgeter.jsonl SGLANG_XPOOL_KV_HIGH=0.04 SGLANG_XPOOL_KV_LOW=0.015 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.03 SGLANG_XPOOL_COOLDOWN=2"
  fi
  mem_frac=0.8
  [ "$L2" = "1" ] && mem_frac=0.7

  echo
  echo "=========================================================="
  echo "[l2-force-kv] $cell port=$port (L1=$L1 L2=$L2 mem_frac=$mem_frac)"
  echo "=========================================================="
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 6

  CUDA_VISIBLE_DEVICES=$GPU nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path Qwen/Qwen3.5-35B-A3B --host 127.0.0.1 --port $port \
      --mem-fraction-static $mem_frac --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      --max-running-requests 240 \
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
  if [ $waited -ge 300 ]; then
    echo "[$cell] FAILED to come up — skipping bench"
    kill -9 $pid 2>/dev/null || true
    sleep 6
    continue
  fi

  echo "[$cell] Phase forced-overflow (random 24K × 240 × RPS=48)..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $port \
    --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
    --dataset-name random --num-prompts 240 \
    --random-input-len 24000 --random-output-len 256 \
    --request-rate 48 \
    --output-file "$out_dir/bench.json" \
    >"$out_dir/bench.log" 2>&1 || echo "[$cell] bench failed"

  pftot=$(grep -c "Prefill batch" "$log" 2>/dev/null || echo 0)
  retract=$(grep -cE "Retract|num_retract|retracted|Retraction" "$log" 2>/dev/null || echo 0)
  pause=$(grep -cE "Pause|paused" "$log" 2>/dev/null || echo 0)
  abort=$(grep -cE "abort|aborted" "$log" 2>/dev/null || echo 0)
  if [ "$L2" = "1" ]; then
    fires=$(grep -c '"xpool_direction":' "$out_dir/budgeter.jsonl" 2>/dev/null || echo 0)
    m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$out_dir/budgeter.jsonl" 2>/dev/null || echo 0)
    k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$out_dir/budgeter.jsonl" 2>/dev/null || echo 0)
    echo "[$cell] prefill=$pftot retract=$retract pause=$pause abort=$abort L2_fires=$fires (m2k=$m2k k2m=$k2m)"
  else
    echo "[$cell] prefill=$pftot retract=$retract pause=$pause abort=$abort"
  fi

  kill -9 $pid 2>/dev/null || true
  sleep 6
done

echo
echo "=========================================================="
echo "[l2-force-kv] SUMMARY"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$ROOT"
cells = ["L10_L20", "L11_L20", "L10_L21", "L11_L21"]
print(f"\n{'cell':<10}{'TPS_in':>10}{'TPS_out':>10}{'mean_ttft':>11}{'p99_ttft':>11}{'med_e2e':>11}{'completed':>11}{'concurr':>10}")
print("-"*82)
for cell in cells:
    fp = f"{root}/{cell}/bench.json"
    if not os.path.exists(fp):
        print(f"{cell:<10} (no data)")
        continue
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1])
    print(f"{cell:<10}{d['input_throughput']:>10.0f}{d['output_throughput']:>10.0f}{d['mean_ttft_ms']:>11.1f}{d['p99_ttft_ms']:>11.1f}{d['median_e2e_latency_ms']:>11.1f}{d['completed']:>11}{d['concurrency']:>10.1f}")

print()
print("L2-positive criterion: L2-on cells better mean_ttft / p99_ttft")
print("when baseline shows retract>0 (genuine admission pressure).")
PY
