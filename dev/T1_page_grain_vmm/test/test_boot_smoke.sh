#!/bin/bash
# Quick smoke: boot Qwen3.5-35B-A3B with arena-on at 2 MiB pages and
# verify a single generate request returns successfully. Exits non-zero
# on any failure.

set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

GPU=${GPU:-2}
PORT=${PORT:-31199}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
LOG=/tmp/t1_smoke_$PORT.log

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 3

echo "[T1 smoke] booting at SGLANG_ARENA_CHUNK_BYTES=2 MiB on GPU=$GPU port=$PORT"
SGLANG_ARENA_SHARED=1 \
SGLANG_ARENA_FROM_BLOB=1 \
SGLANG_ARENA_CHUNK_BYTES=2097152 \
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
    echo "[T1 smoke] ready after ${waited}s"
    break
  fi
done
if [ $waited -ge 600 ]; then
  echo "[T1 smoke] FAIL: server didn't come up in 10 min"
  tail -40 "$LOG"
  exit 1
fi

echo "[T1 smoke] sending one generate request..."
RESP=$(curl -s --max-time 60 -X POST http://127.0.0.1:$PORT/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":16,"temperature":0}}')

if [ -z "$RESP" ] || ! echo "$RESP" | grep -q '"text"'; then
  echo "[T1 smoke] FAIL: bad response: $RESP"
  exit 1
fi

echo "[T1 smoke] OK — boot succeeded, generate succeeded"
echo "[T1 smoke] response excerpt: $(echo "$RESP" | head -c 200)"
exit 0
