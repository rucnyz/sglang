#!/bin/bash
# 4-cell validation under the v7 config that finally made L2 fire physically.
#
# All cells: 256 MB chunks, KV mobile=2 (40 chunks donated, mamba mobile=0
# to avoid v3-class CUDA-graph + slot-cap crash), EDGE_TRIGGER=1, KV_HIGH=0.5.
# L2-on cells additionally set SGLANG_XPOOL_NON_BALANCED=1 to use the
# direct kv_to_mamba_chunks(1) path (fits in shared without src.shrink).
#
# Workload: GSP 24 groups × 5 prompts × 6 K, RPS=10 — sized to fit in
# the reduced 524 K KV pool while saturating mamba past 0.08 watermark
# (verified in v7c smoke: peak mamba 0.717, peak KV 0.316).

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-31100}"
RUN_NAME="${RUN_NAME:-l2-validation-4cell-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[l2-validation] root=$ROOT gpu=$GPU"

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
    extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=0 SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))"
    extra_env="$extra_env SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0"
    extra_env="$extra_env SGLANG_BUDGETER_LOG=$out_dir/budgeter.jsonl"
    extra_env="$extra_env SGLANG_XPOOL_KV_HIGH=0.5 SGLANG_XPOOL_KV_LOW=0.05 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.03 SGLANG_XPOOL_COOLDOWN=2"
    extra_env="$extra_env SGLANG_XPOOL_EDGE_TRIGGER=1 SGLANG_XPOOL_NON_BALANCED=1"
    extra_env="$extra_env SGLANG_ARENA_KV_MOBILE_SOFT_CHUNKS=2 SGLANG_ARENA_MAMBA_MOBILE_SOFT_CHUNKS=0"
  fi
  mem_frac=0.8
  [ "$L2" = "1" ] && mem_frac=0.7

  echo
  echo "=========================================================="
  echo "[l2-validation] $cell port=$port (L1=$L1 L2=$L2 mem_frac=$mem_frac)"
  echo "=========================================================="
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 4

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
  while [ $waited -lt 240 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$port/health" 2>/dev/null)" = "200" ]; then
      echo "[$cell] ready after ${waited}s"; break
    fi
  done
  if [ $waited -ge 240 ]; then
    echo "[$cell] FAILED to come up — skipping bench"
    kill -9 $pid 2>/dev/null || true
    sleep 4
    continue
  fi

  echo "[$cell] Phase A medium (24 × 5 × 6K, RPS=10)..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $port \
    --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 24 --gsp-prompts-per-group 5 \
    --gsp-system-prompt-len 6000 --gsp-question-len 64 \
    --gsp-output-len 256 \
    --request-rate 10 \
    --output-file "$out_dir/bench.json" \
    >"$out_dir/bench.log" 2>&1 || echo "[$cell] bench failed"

  pftot=$(grep -c "Prefill batch" "$log" 2>/dev/null || echo 0)
  retract=$(grep -c "KV cache pool is full" "$log" 2>/dev/null || echo 0)
  if [ "$L2" = "1" ]; then
    fires=$(grep -c '"xpool_direction":' "$out_dir/budgeter.jsonl" 2>/dev/null || echo 0)
    granted=$(grep '"xpool_direction":' "$out_dir/budgeter.jsonl" 2>/dev/null | python3 -c "
import sys, json
total = 0
for ln in sys.stdin:
    try: d = json.loads(ln)
    except: continue
    total += d.get('xpool_granted_total', 0)
print(total)
" 2>/dev/null || echo 0)
    echo "[$cell] prefill=$pftot retract=$retract fires=$fires granted=$granted"
  else
    echo "[$cell] prefill=$pftot retract=$retract"
  fi

  kill -9 $pid 2>/dev/null || true
  sleep 4
done

echo
echo "=========================================================="
echo "[l2-validation] SUMMARY"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$ROOT"
cells = ["L10_L20", "L11_L20", "L10_L21", "L11_L21"]
print(f"\n{'cell':<10}{'TPS_in':>10}{'mean_ttft':>11}{'p99_ttft':>11}{'med_e2e':>11}{'completed':>11}")
print("-"*64)
for cell in cells:
    fp = f"{root}/{cell}/bench.json"
    if not os.path.exists(fp):
        print(f"{cell:<10} (no data)")
        continue
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1])
    print(f"{cell:<10}{d['input_throughput']:>10.0f}{d['mean_ttft_ms']:>11.1f}{d['p99_ttft_ms']:>11.1f}{d['median_e2e_latency_ms']:>11.1f}{d['completed']:>11}")

print("\nL2 fire totals (granted_total > 0 = real chunk movement):")
for cell in ["L10_L21", "L11_L21"]:
    fp = f"{root}/{cell}/budgeter.jsonl"
    if not os.path.exists(fp):
        print(f"  {cell}: no log")
        continue
    fires, granted = 0, 0
    with open(fp) as f:
        for ln in f:
            try: d = json.loads(ln)
            except: continue
            if d.get("xpool_direction") in ("kv_to_mamba", "mamba_to_kv"):
                fires += 1
                granted += d.get("xpool_granted_total", 0)
    print(f"  {cell}: {fires} fires, {granted} chunks total moved")
PY
