#!/bin/bash
# T6 smoke: T1+T2+T3+T4+T5+T6 boot + 5 generates. Verifies the env
# composition + the admission-time hook wiring doesn't break boot.
# Smoke load doesn't trigger emergency fire (no admission saturation),
# but boot must succeed and the budgeter agent must register itself
# as the singleton.

set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

GPU=${GPU:-2}
PORT=${PORT:-31699}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
LOG=/tmp/t6_smoke_$PORT.log

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 3

echo "[T6 smoke] boot with T1..T6 flags on GPU=$GPU port=$PORT"
SGLANG_ARENA_SHARED=1 \
SGLANG_ARENA_FROM_BLOB=1 \
SGLANG_ALLOCATOR_PLACEMENT_BIAS=1 \
SGLANG_SMART_OVERCAP=1 \
SGLANG_ATOMIC_MIGRATION=1 \
SGLANG_BUDGETER=1 \
SGLANG_BUDGETER_XPOOL_PLANNER=1 \
SGLANG_BUDGETER_XPOOL_COORDINATED=1 \
SGLANG_ADMISSION_TIME_FIRE=1 \
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
CUDA_VISIBLE_DEVICES=$GPU \
  nohup .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph --reasoning-parser qwen3 \
    > "$LOG" 2>&1 &
PID=$!
trap "kill -9 $PID 2>/dev/null || true; pkill -f 'launch_server.*--port $PORT' 2>/dev/null || true" EXIT

waited=0
while [ $waited -lt 600 ]; do
  sleep 5; waited=$((waited+5))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then
    echo "[T6 smoke] ready after ${waited}s"
    break
  fi
done
if [ $waited -ge 600 ]; then
  echo "[T6 smoke] FAIL: server didn't come up in 10 min"
  tail -40 "$LOG"
  exit 1
fi

# Verify T6 admission-time fire log line appeared at boot.
if ! grep -q "T6 admission-time fire enabled" "$LOG"; then
  echo "[T6 smoke] FAIL: T6 admission-time fire log line missing"
  grep -i "T6\|admission" "$LOG" | head
  exit 1
fi
echo "[T6 smoke] T6 boot log line confirmed"

# 5 generates.
for i in 1 2 3 4 5; do
  RESP=$(curl -s --max-time 60 -X POST http://127.0.0.1:$PORT/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":16,"temperature":0}}')
  if [ -z "$RESP" ] || ! echo "$RESP" | grep -q '"text"'; then
    echo "[T6 smoke] FAIL on prompt $i: $RESP"
    exit 1
  fi
done

echo "[T6 smoke] OK — boot succeeded, 5 generates returned, T1..T6 flags compose"
exit 0
