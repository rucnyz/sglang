#!/bin/bash
# T3 smoke: boot with T1 (page-grain) + T2 (placement bias) + T3 (smart
# over-cap selection) all enabled. Verify boot succeeds, serving works,
# and a budgeter / actuator-driven cross-pool fire would consult the
# new path.
#
# This smoke does NOT trigger an actual fire (would need M2-style
# admission saturation). It only checks the integration boots cleanly.

set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

GPU=${GPU:-2}
PORT=${PORT:-31399}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
LOG=/tmp/t3_smoke_$PORT.log

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 3

echo "[T3 smoke] boot with T1+T2+T3 flags on GPU=$GPU port=$PORT"
SGLANG_ARENA_SHARED=1 \
SGLANG_ARENA_FROM_BLOB=1 \
SGLANG_ALLOCATOR_PLACEMENT_BIAS=1 \
SGLANG_SMART_OVERCAP=1 \
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
    echo "[T3 smoke] ready after ${waited}s"
    break
  fi
done
if [ $waited -ge 600 ]; then
  echo "[T3 smoke] FAIL: server didn't come up in 10 min"
  tail -40 "$LOG"
  exit 1
fi

# Verify T2 log line (T3's prerequisite)
if ! grep -q "T2 placement bias active" "$LOG"; then
  echo "[T3 smoke] FAIL: T2 prerequisite log missing"
  exit 1
fi
echo "[T3 smoke] T2 placement bias active confirmed"

# Smoke: 5 short prompts.
for i in 1 2 3 4 5; do
  RESP=$(curl -s --max-time 60 -X POST http://127.0.0.1:$PORT/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":16,"temperature":0}}')
  if [ -z "$RESP" ] || ! echo "$RESP" | grep -q '"text"'; then
    echo "[T3 smoke] FAIL on prompt $i: $RESP"
    exit 1
  fi
done

echo "[T3 smoke] OK — boot succeeded, 5 generates returned, T1+T2+T3 flags compose without crash"
exit 0
