#!/bin/bash
# T7 per-cell driver: same workload as 45_swarm_multiturn_per_cell.sh
# but the L2 branch enables the full T1+T3+T4+T6 ideal-mode stack.
# (T2 placement bias is layered with L1 too — see below.)
# (T5 default 80 GiB headroom is implicit; no flag needed.)
#
# Required env: ONLY_L1, ONLY_L2, CUDA_VISIBLE_DEVICES, PORT, OUT_DIR
# Optional:
#   NUM_CONCURRENCY (800, paper §sec:eval-main-swarm)
#   TRAFFIC_SCENARIO (D(256,256))
#   SESSION_CAP (3000)
#   MAX_TIME_MIN (8)
#   MEM_FRAC (0.8)

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

# Full env reset.
unset SGLANG_HPB_LRU SGLANG_HPB_WINDOW_S SGLANG_K_BIG SGLANG_K_BIG_AUTO_THRESHOLD
unset SGLANG_ARENA_SHARED SGLANG_ARENA_FROM_BLOB SGLANG_ARENA_CHUNK_BYTES
unset SGLANG_BUDGETER SGLANG_BUDGETER_XPOOL_PLANNER SGLANG_BUDGETER_XPOOL_COORDINATED
unset SGLANG_BUDGETER_TICK_S SGLANG_BUDGETER_LOG
unset SGLANG_XPOOL_KV_HIGH SGLANG_XPOOL_KV_LOW SGLANG_XPOOL_MAMBA_HIGH SGLANG_XPOOL_MAMBA_LOW SGLANG_XPOOL_COOLDOWN
unset SGLANG_ALLOCATOR_PLACEMENT_BIAS SGLANG_SMART_OVERCAP SGLANG_ATOMIC_MIGRATION SGLANG_ADMISSION_TIME_FIRE
export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0

if [ "$ONLY_L1" = "1" ]; then
  export SGLANG_HPB_LRU=1
  export SGLANG_HPB_WINDOW_S=120.0
  export SGLANG_K_BIG=8192
  export SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85}
fi

if [ "$ONLY_L2" = "1" ]; then
  # T1 substrate (page-grain VMM, default 2 MiB chunks via memory_pool.py).
  export SGLANG_ARENA_SHARED=1
  export SGLANG_ARENA_FROM_BLOB=1
  # T2 placement bias: must be ON when L2 fires care about tail-empty tail.
  export SGLANG_ALLOCATOR_PLACEMENT_BIAS=1
  # T3 smart over-cap: actuator picks free chunks via allocator query.
  export SGLANG_SMART_OVERCAP=1
  # T4 atomic migration: residual-live-page fallback.
  export SGLANG_ATOMIC_MIGRATION=1
  # T6 admission-time on-demand fire.
  export SGLANG_ADMISSION_TIME_FIRE=1
  # Budgeter + planner.
  export SGLANG_BUDGETER=1
  export SGLANG_BUDGETER_XPOOL_PLANNER=1
  export SGLANG_BUDGETER_XPOOL_COORDINATED=1
  export SGLANG_BUDGETER_TICK_S=2.0
  export SGLANG_BUDGETER_LOG="$OUT_DIR/${cell}_budgeter.jsonl"
  export SGLANG_XPOOL_KV_HIGH=${SGLANG_XPOOL_KV_HIGH:-0.5}
  export SGLANG_XPOOL_KV_LOW=${SGLANG_XPOOL_KV_LOW:-0.1}
  export SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.5}
  export SGLANG_XPOOL_MAMBA_LOW=${SGLANG_XPOOL_MAMBA_LOW:-0.1}
  export SGLANG_XPOOL_COOLDOWN=${SGLANG_XPOOL_COOLDOWN:-30}
  export SGLANG_XPOOL_UNIT=${SGLANG_XPOOL_UNIT:-1}
  export SGLANG_XPOOL_MAMBA_FLUSH_CAP=${SGLANG_XPOOL_MAMBA_FLUSH_CAP:-256}
  # T5 headroom: default 80 GiB, no flag override needed. To bump:
  # export SGLANG_ARENA_KV_HEADROOM_BYTES=$((30*1024*1024*1024))
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
  --enable-metrics \
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

NUM_CONCURRENCY=${NUM_CONCURRENCY:-800}
TRAFFIC_SCENARIO=${TRAFFIC_SCENARIO:-D(256,256)}
SESSION_CAP=${SESSION_CAP:-3000}
MAX_TIME_MIN=${MAX_TIME_MIN:-8}

echo "[$cell] running multi-turn swarm (conc=$NUM_CONCURRENCY, ${MAX_TIME_MIN}min, $TRAFFIC_SCENARIO, cap=$SESSION_CAP)"
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

# Counts of the new T3/T4/T6 log lines (proves on-demand fire was hit).
if [ "$ONLY_L2" = "1" ]; then
  t6=$(grep -c "T6 admission-time fire:" "$log" 2>/dev/null || echo 0)
  t6_committed=$(grep -c "T6 admission-time fire.*committed=True" "$log" 2>/dev/null || echo 0)
  t3=$(grep -c "T3 smart over-cap selection:" "$log" 2>/dev/null || echo 0)
  t4=$(grep -c "T4 atomic migration:" "$log" 2>/dev/null || echo 0)
  echo "[$cell] log line counts: T3=$t3 T4=$t4 T6=$t6 T6_committed=$t6_committed"
fi

if [ "$ONLY_L2" = "1" ] && [ -f "$OUT_DIR/${cell}_budgeter.jsonl" ]; then
  total=$(grep -c '"xpool_direction":' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  echo "[$cell] budgeter fires: total=$total k2m=$k2m m2k=$m2k"
fi

kill -9 $sv_pid 2>/dev/null || true
sleep 4
