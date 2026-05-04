#!/bin/bash
# T5 smoke: boot with all 5 flags on (T1+T2+T3+T4+T5), verify the
# arena init log shows the expanded max_tokens (T5 effect), and serve
# 5 prompts to confirm the bigger VA reservation doesn't break boot.

set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

GPU=${GPU:-2}
PORT=${PORT:-31599}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
LOG=/tmp/t5_smoke_$PORT.log

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 3

echo "[T5 smoke] boot with T1+T2+T3+T4+T5 flags on GPU=$GPU port=$PORT"
SGLANG_ARENA_SHARED=1 \
SGLANG_ARENA_FROM_BLOB=1 \
SGLANG_ALLOCATOR_PLACEMENT_BIAS=1 \
SGLANG_SMART_OVERCAP=1 \
SGLANG_ATOMIC_MIGRATION=1 \
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
    echo "[T5 smoke] ready after ${waited}s"
    break
  fi
done
if [ $waited -ge 600 ]; then
  echo "[T5 smoke] FAIL: server didn't come up in 10 min"
  tail -40 "$LOG"
  exit 1
fi

# Verify max_tokens is now MUCH bigger than init_tokens (T5 effect).
KV_LINE=$(grep "MultiTensorArena initialized" "$LOG" | grep "n_kinds=2" | head -1)
M_LINE=$(grep "MultiTensorArena initialized" "$LOG" | grep "n_kinds=1" | head -1)

if [ -z "$KV_LINE" ] || [ -z "$M_LINE" ]; then
  echo "[T5 smoke] FAIL: arena init lines missing"
  grep "MultiTensorArena" "$LOG" | head
  exit 1
fi

echo "[T5 smoke] KV arena: $KV_LINE"
echo "[T5 smoke] mamba arena: $M_LINE"

# Extract max_tokens / init_tokens to confirm large headroom.
KV_MAX=$(echo "$KV_LINE" | grep -oE "max_tokens=[0-9]+" | cut -d= -f2)
KV_INIT=$(echo "$KV_LINE" | grep -oE "init_tokens=[0-9]+" | cut -d= -f2)
M_MAX=$(echo "$M_LINE" | grep -oE "max_tokens=[0-9]+" | cut -d= -f2)
M_INIT=$(echo "$M_LINE" | grep -oE "init_tokens=[0-9]+" | cut -d= -f2)

# Headroom = max - init. T5 default = 80 GiB / chunk_size chunks.
KV_HEADROOM=$((KV_MAX - KV_INIT))
M_HEADROOM=$((M_MAX - M_INIT))

# At 2 MiB chunks, KV tokens_per_chunk = 2048; 80 GiB / 2 MiB = 40960 chunks
# = 40960 × 2048 tokens = 83 886 080 tokens.
# Allow a generous lower bound (≥ 10× init_tokens) to confirm T5 fires.
if [ "$KV_HEADROOM" -lt $((10 * KV_INIT)) ]; then
  echo "[T5 smoke] FAIL: KV headroom too small. headroom=${KV_HEADROOM}, init=${KV_INIT}"
  exit 1
fi
echo "[T5 smoke] KV headroom=${KV_HEADROOM} (init=${KV_INIT}, max=${KV_MAX})"
echo "[T5 smoke] mamba headroom=${M_HEADROOM} (init=${M_INIT}, max=${M_MAX})"

# Run 5 generates.
for i in 1 2 3 4 5; do
  RESP=$(curl -s --max-time 60 -X POST http://127.0.0.1:$PORT/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"What is 2+2?","sampling_params":{"max_new_tokens":16,"temperature":0}}')
  if [ -z "$RESP" ] || ! echo "$RESP" | grep -q '"text"'; then
    echo "[T5 smoke] FAIL on prompt $i: $RESP"
    exit 1
  fi
done

echo "[T5 smoke] OK — boot succeeded, 5 generates returned, T1+T2+T3+T4+T5 flags compose"
exit 0
