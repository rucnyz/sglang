#!/bin/bash
# Per-cell driver for the FAN-OUT agent benchmark using genai-bench.
# Many concurrent short-prompt subagent calls. Each Locust user runs a
# request-response loop with short prompts and short decode; with N
# users > mamba pool size, the recurrent-slot pool becomes the
# admission gate while paged KV stays at single-digit utilisation.
# This is paper §motivation §76's fan-out agent regime: 'orchestrator
# dispatching N short sub-task LLM calls in parallel'.
#
# Required env: ONLY_L1, ONLY_L2, CUDA_VISIBLE_DEVICES, PORT, OUT_DIR
# Optional:
#   NUM_CONCURRENCY (400)             Concurrent fan-out subagents
#   TRAFFIC_SCENARIO ("D(256,32)")    Short prompt + short decode per turn
#   MAX_TIME_MIN (5)                  Bench wall-clock minutes
#   MEM_FRAC (0.8)
#   SGLANG_K_BIG_AUTO_THRESHOLD (0.85)
#   SGLANG_XPOOL_KV_HIGH (0.5)
#   SGLANG_XPOOL_MAMBA_HIGH (0.5)
#   SGLANG_XPOOL_UNIT (1)
#   SGLANG_XPOOL_MAMBA_FLUSH_CAP (256)

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

ONLY_L1=${ONLY_L1:?missing}
ONLY_L2=${ONLY_L2:?missing}
PORT=${PORT:-30099}
OUT_DIR=${OUT_DIR:?missing}
mkdir -p "$OUT_DIR"

cell="L1${ONLY_L1}_L2${ONLY_L2}"

unset SGLANG_HPB_LRU SGLANG_HPB_WINDOW_S SGLANG_K_BIG SGLANG_K_BIG_AUTO_THRESHOLD
unset SGLANG_ARENA_SHARED SGLANG_ARENA_FROM_BLOB SGLANG_ARENA_CHUNK_BYTES
unset SGLANG_BUDGETER SGLANG_BUDGETER_XPOOL_PLANNER SGLANG_BUDGETER_XPOOL_COORDINATED
unset SGLANG_BUDGETER_TICK_S SGLANG_BUDGETER_LOG
unset SGLANG_XPOOL_KV_HIGH SGLANG_XPOOL_KV_LOW SGLANG_XPOOL_MAMBA_HIGH SGLANG_XPOOL_MAMBA_LOW SGLANG_XPOOL_COOLDOWN
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0

if [ "$ONLY_L1" = "1" ]; then
  export SGLANG_HPB_LRU=1
  export SGLANG_HPB_WINDOW_S=120.0
  export SGLANG_K_BIG=8192
  export SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85}
fi

if [ "$ONLY_L2" = "1" ]; then
  export SGLANG_ARENA_SHARED=1
  export SGLANG_ARENA_FROM_BLOB=1
  export SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))
  export SGLANG_BUDGETER=1
  export SGLANG_BUDGETER_XPOOL_PLANNER=1
  export SGLANG_BUDGETER_XPOOL_COORDINATED=1
  export SGLANG_BUDGETER_TICK_S=2.0
  export SGLANG_BUDGETER_LOG="$OUT_DIR/${cell}_budgeter.jsonl"
  export SGLANG_XPOOL_KV_HIGH=${SGLANG_XPOOL_KV_HIGH:-0.5}
  export SGLANG_XPOOL_KV_LOW=${SGLANG_XPOOL_KV_LOW:-0.1}
  export SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.5}
  export SGLANG_XPOOL_MAMBA_LOW=${SGLANG_XPOOL_MAMBA_LOW:-0.1}
  export SGLANG_XPOOL_COOLDOWN=2
  export SGLANG_XPOOL_UNIT=${SGLANG_XPOOL_UNIT:-1}
  export SGLANG_XPOOL_MAMBA_FLUSH_CAP=${SGLANG_XPOOL_MAMBA_FLUSH_CAP:-256}
fi

mem_frac=${MEM_FRAC:-0.8}
log="$OUT_DIR/${cell}_server.log"
echo "[$cell] starting server on port $PORT (gpu=$CUDA_VISIBLE_DEVICES)"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 4

nohup .venv/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-35B-A3B --host 127.0.0.1 --port $PORT \
  --mem-fraction-static $mem_frac --log-level info \
  --enforce-piecewise-cuda-graph \
  --reasoning-parser qwen3 \
  > "$log" 2>&1 &
sv_pid=$!
echo "[$cell] server pid=$sv_pid"

waited=0
while [ $waited -lt 240 ]; do
  sleep 10; waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[$cell] ready after ${waited}s"; break
  fi
done
if [ $waited -ge 240 ]; then
  echo "[$cell] FAILED to come up — log tail:"; tail -25 "$log"
  kill -9 $sv_pid 2>/dev/null || true; exit 1
fi

NUM_CONCURRENCY=${NUM_CONCURRENCY:-400}
TRAFFIC_SCENARIO=${TRAFFIC_SCENARIO:-D(256,32)}
MAX_TIME_MIN=${MAX_TIME_MIN:-5}

# Single-turn task — each Locust user issues an independent short
# request, no per-user state across requests. With N users above the
# mamba slot pool's boot capacity (~384 on Qwen3.5-35B-A3B / H200),
# the recurrent-slot pool becomes the admission gate; paged KV stays
# at a few percent of its pool because each request only carries
# ~256 prefill + ~32 decode tokens of state.
echo "[$cell] running genai-bench fan-out (concurrency=$NUM_CONCURRENCY, ${MAX_TIME_MIN}min, scenario=$TRAFFIC_SCENARIO)"
.venv/bin/python -m genai_bench.cli.cli benchmark \
  --api-backend sglang \
  --api-base "http://127.0.0.1:$PORT" \
  --api-key dummy \
  --api-model-name Qwen/Qwen3.5-35B-A3B \
  --model-tokenizer Qwen/Qwen3.5-35B-A3B \
  --task text-to-text \
  --traffic-scenario "$TRAFFIC_SCENARIO" \
  --num-concurrency $NUM_CONCURRENCY \
  --max-time-per-run $MAX_TIME_MIN \
  --max-requests-per-run 1000000 \
  --experiment-folder-name "$OUT_DIR/genai_results" \
  --server-engine SGLang \
  > "$OUT_DIR/${cell}_client.log" 2>&1 || echo "[$cell] client failed"
echo "[$cell] client done"

if [ "$ONLY_L2" = "1" ] && [ -f "$OUT_DIR/${cell}_budgeter.jsonl" ]; then
  total=$(grep -c '"xpool_direction":' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  echo "[$cell] xpool transfers: total=$total k2m=$k2m m2k=$m2k"
fi

kill -9 $sv_pid 2>/dev/null || true
sleep 4
