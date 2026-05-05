#!/bin/bash
# Per-cell driver for the multi-turn long-horizon agent benchmark using
# genai-bench (Locust under the hood). Each Locust user maintains its
# own conversation history across iterations and sends the next request
# only after the previous returns — naturally steady-state, no client-
# side admission burst at t=0 (which is what 42_multiturn_per_cell.sh's
# asyncio-based all-at-once submission caused).
#
# Required env: ONLY_L1, ONLY_L2, CUDA_VISIBLE_DEVICES, PORT, OUT_DIR
# Optional:
#   NUM_CONCURRENCY (14)        Number of concurrent agent users
#   TRAFFIC_SCENARIO ("D(4096,4096)")  Per-turn input/output token counts
#   SESSION_CAP (60000)          Tokens before a session resets to system prompt
#   MAX_TIME_MIN (8)             Bench wall-clock minutes
#   MEM_FRAC (0.8)
#   SGLANG_K_BIG_AUTO_THRESHOLD (0.85)
#   SGLANG_XPOOL_KV_HIGH (0.5), SGLANG_XPOOL_MAMBA_HIGH (0.5)
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

# Cell-specific server env. Same construction as 42_multiturn_per_cell.sh.
unset SGLANG_HPB_LRU SGLANG_HPB_WINDOW_S SGLANG_K_BIG SGLANG_K_BIG_AUTO_THRESHOLD
unset SGLANG_ARENA_SHARED SGLANG_ARENA_FROM_BLOB SGLANG_ARENA_CHUNK_BYTES
unset SGLANG_BUDGETER SGLANG_BUDGETER_XPOOL_PLANNER SGLANG_BUDGETER_XPOOL_COORDINATED
unset SGLANG_BUDGETER_TICK_S SGLANG_BUDGETER_LOG
unset SGLANG_XPOOL_KV_HIGH SGLANG_XPOOL_KV_LOW SGLANG_XPOOL_MAMBA_HIGH SGLANG_XPOOL_MAMBA_LOW SGLANG_XPOOL_COOLDOWN
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0

if [ "$ONLY_L1" = "1" ]; then
  export SGLANG_HPB_LRU=1
  export SGLANG_HPB_WINDOW_S=120.0
  # K_BIG controls heterogeneous-granularity mamba snapshot retention:
  # only depths aligned to K_BIG get a snapshot; non-aligned chunked-
  # prefill stash inserts get suppressed (see mamba_radix_cache.py
  # insert(), the suppressed_mamba branch). For multi-turn long-horizon
  # workloads this is what creates the asymmetric KV-saturated /
  # mamba-idle binding that Layer 2 can act on. Without K_BIG, every
  # chunked-prefill boundary forks a mamba snapshot, so mamba grows
  # ~with prefill volume → both pools co-saturate → no asymmetric
  # binding for L2 to fix.
  export SGLANG_K_BIG=${SGLANG_K_BIG:-16384}
  # Auto-threshold gates K_BIG activation by current mamba usage.
  # Default 0.85 (paper §design-l1): K_BIG only engages when mamba is
  # actually pressured.
  #
  # The earlier always-on (0) default was wrong on this workload: at
  # concurrency=24 mamba peaks ~0.71, never crossing high-water, so
  # always-on K_BIG paid the snapshot-recovery cost on the prefill
  # critical path (~32ms full-stack re-derivation × ~140 events / 5min
  # — every odd-aligned turn drops its snapshot and forces an 8K-token
  # re-run from the previous K_BIG boundary on next hit) without ever
  # earning the "freed mamba HBM" benefit. The variance run at
  # cost-aware-variance-20260502-200501 measured this as TTFT_p99
  # 13054ms (L1-only) vs 8931ms (baseline), +46%. Threshold 0.85 keeps
  # K_BIG inert here and only engages when mamba genuinely saturates
  # (higher concurrency or shorter sessions).
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
  # Cooldown was 2 ticks (4s) under the legacy edge-trigger gate, where fires
  # only happen on state transitions so the short cooldown didn't compound.
  # Under the NB direction-aware gate (paper §sec:design-l2-firegate Eq
  # nb-direction-gate, default ON), NB stays positive across many ticks
  # whenever cost-asymmetry × usage clears α·C_act, so a 2-tick cooldown
  # produces ~10x the desired fire rate (90 fires/10min in NB-mode high-
  # pressure run vs ~22 fires/run in legacy). 30 ticks (60s) lets the
  # gate's effect settle before re-evaluating.
  export SGLANG_XPOOL_COOLDOWN=${SGLANG_XPOOL_COOLDOWN:-30}
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

NUM_CONCURRENCY=${NUM_CONCURRENCY:-14}
TRAFFIC_SCENARIO=${TRAFFIC_SCENARIO:-D(4096,4096)}
SESSION_CAP=${SESSION_CAP:-60000}
MAX_TIME_MIN=${MAX_TIME_MIN:-8}

# genai-bench's Locust-based execution model: spawn N concurrent users,
# each runs a request-response loop indefinitely until the time budget
# is exhausted. No t=0 client-side admission burst (each user only has
# one request in flight at a time). Steady state from second one.
echo "[$cell] running genai-bench multi-turn (concurrency=$NUM_CONCURRENCY, ${MAX_TIME_MIN}min)"
GENAI_BENCH_MT_SESSION_CAP_TOKENS=$SESSION_CAP \
  .venv/bin/python -m genai_bench.cli.cli benchmark \
  --api-backend sglang \
  --api-base "http://127.0.0.1:$PORT" \
  --api-key dummy \
  --api-model-name Qwen/Qwen3.5-35B-A3B \
  --model-tokenizer Qwen/Qwen3.5-35B-A3B \
  --task text-to-text-multi-turn \
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
