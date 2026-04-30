#!/bin/bash
# Setting 3.C — composed Layer 1 + Layer 2 effect (Q3.C).
#
# The Setting 1 trace fires only 1 cross-pool transfer per cell, leaving
# no signal to differentiate "decisive" from "thrashing". The A4 sweep
# showed that the Phase 1+2+3 long/short/long trace at τ=0.5s drives 21
# transfers — that's the workload we need.
#
# This script runs 4 cells on the Phase 1+2+3 trace (single GPU per cell,
# parallel across GPUs):
#   (L1=0, L2=1) — Layer 2 only on the engine baseline (recency LRU,
#                  no K_big)
#   (L1=H, L2=1) — Layer 2 on top of HPB LRU (no K_big)
#   (L1=F, L2=1) — Layer 2 on top of full Layer 1 (HPB + K_big=8192)
#   (L1=F, L2=0) — full Layer 1 with no Layer 2 (control)
#
# Reports per-cell:
#   - total cross-pool transfers
#   - reversals (a kv→mamba immediately followed by mamba→kv or vice versa)
#   - prefill batches with cached-token > 0 (cache hit indicator)
#
# Total runtime: ~10 min wall clock (4 cells parallel, ~6 min each).

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
WARMUP_S=${WARMUP_S:-300}
ONLY_CELL=${ONLY_CELL:-}
PORT=${PORT:-30099}
OUT_DIR=${OUT_DIR:-/tmp/setting3c_$$}
mkdir -p "$OUT_DIR"

run_cell() {
  local cell="$1"      # one of: L0_L21 L1H_L21 L1F_L21 L1F_L20
  local extra_env="SGLANG_BUDGETER_TICK_S=0.5 SGLANG_XPOOL_KV_HIGH=0.04 SGLANG_XPOOL_KV_LOW=0.038 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.076 SGLANG_XPOOL_COOLDOWN=2"
  case "$cell" in
    L0_L21)
      extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_LOG=$OUT_DIR/${cell}_budgeter.jsonl"
      ;;
    L1H_L21)
      extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_LOG=$OUT_DIR/${cell}_budgeter.jsonl"
      ;;
    L1F_L21)
      extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_LOG=$OUT_DIR/${cell}_budgeter.jsonl"
      ;;
    L1F_L20)
      extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
      ;;
  esac
  local log="$OUT_DIR/${cell}_server.log"
  echo "=== cell=$cell ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[$cell] ready after ${waited}s"
      break
    fi
  done

  PORT=$PORT MODEL=$MODEL .venv/bin/python <<'PY'
import json, urllib.request, time, threading, os
PORT = os.environ['PORT']; MODEL = os.environ['MODEL']
LONG_BASE = "Compute step by step. " * 250
def fire(prompt):
    data = json.dumps({'model': MODEL, 'prompt': prompt, 'max_tokens': 64, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try: urllib.request.urlopen(req, timeout=300).read()
    except: pass

threads = []
for i in range(30):
    t = threading.Thread(target=fire, args=(LONG_BASE + f' Q{i}: name a fruit:',), daemon=True)
    t.start(); threads.append(t); time.sleep(0.05)
for t in threads: t.join(timeout=300)
time.sleep(8)
threads = []
for i in range(40):
    t = threading.Thread(target=fire, args=(f'Q{i}: name a color:',), daemon=True)
    t.start(); threads.append(t); time.sleep(0.03)
for t in threads: t.join(timeout=180)
time.sleep(8)
LONG2 = "Compute step by step. " * 5500
for i in range(4):
    fire(LONG2 + f' Q{i}: name a fruit:'); time.sleep(1)
PY

  sleep 6
  local jsonl="$OUT_DIR/${cell}_budgeter.jsonl"
  if [ -f "$jsonl" ]; then
    local total=$(grep -c '"xpool_direction":' "$jsonl" 2>/dev/null || echo 0)
    local k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" 2>/dev/null || echo 0)
    local m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" 2>/dev/null || echo 0)
    echo "[$cell] transfers: total=$total kv→mamba=$k2m mamba→kv=$m2k"
  fi
  local hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local pftot=$(grep -c "Prefill batch" "$log" || true)
  echo "[$cell] prefill batches: $pftot, with cached-token > 0: $hit"

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

if [ -n "$ONLY_CELL" ]; then
  run_cell "$ONLY_CELL"
else
  for cell in L0_L21 L1H_L21 L1F_L21 L1F_L20; do
    run_cell "$cell"
  done
  echo
  echo "=== Setting 3.C summary ==="
  for cell in L0_L21 L1H_L21 L1F_L21 L1F_L20; do
    jsonl="$OUT_DIR/${cell}_budgeter.jsonl"
    log="$OUT_DIR/${cell}_server.log"
    total=$(grep -c '"xpool_direction":' "$jsonl" 2>/dev/null || echo 0)
    k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" 2>/dev/null || echo 0)
    m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" 2>/dev/null || echo 0)
    hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
    pftot=$(grep -c "Prefill batch" "$log" || true)
    echo "  $cell: transfers=$total (k2m=$k2m m2k=$m2k); cache hits $hit/$pftot"
  done
fi
