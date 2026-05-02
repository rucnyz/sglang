#!/bin/bash
# Per-cell driver for the FAN-OUT demo workload (paper §motivation §76).
# Many concurrent short-prompt subagent calls — mamba slots saturate
# at boot capacity (~384 on Qwen3.5-35B-A3B / H200) while paged KV
# stays at <15% utilisation. Without L2: requests queue once mamba
# fills. With L2: kv_to_mamba transfer expands mamba slot capacity
# (e.g., 384→512), engine admits more concurrent subagents → TPS up.
#
# Workload: random 256-token input + 256-token output prompts, RPS=120,
# 800 prompts. Per-req lifetime ~2-3s decode → peak concurrency 240-360
# subagents → mamba demand exceeds boot 384 → fan-out regime.
#
# Used by 39b_fanout_variance_parallel.sh.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
ONLY_L1=${ONLY_L1:?missing}
ONLY_L2=${ONLY_L2:?missing}
OUT_DIR=${OUT_DIR:?missing}
mkdir -p "$OUT_DIR"

cell="L1${ONLY_L1}_L2${ONLY_L2}"
extra_env=""
if [ "$ONLY_L1" = "1" ]; then
  extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
  if [ -n "${SGLANG_K_BIG_AUTO_THRESHOLD:-}" ]; then
    extra_env="$extra_env SGLANG_K_BIG_AUTO_THRESHOLD=$SGLANG_K_BIG_AUTO_THRESHOLD"
  fi
fi
if [ "$ONLY_L2" = "1" ]; then
  chunk_bytes=${SGLANG_ARENA_CHUNK_BYTES:-$((256*1024*1024))}
  # Asymmetric thresholds favoring kv→mamba detection: KV stays low
  # naturally; mamba is the binding pool. KV_LOW high enough that
  # KV's true low usage triggers the LOW state.
  kv_hi=${SGLANG_XPOOL_KV_HIGH:-0.5}
  kv_lo=${SGLANG_XPOOL_KV_LOW:-0.15}
  m_hi=${SGLANG_XPOOL_MAMBA_HIGH:-0.5}
  m_lo=${SGLANG_XPOOL_MAMBA_LOW:-0.1}
  extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=$chunk_bytes SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0 SGLANG_BUDGETER_LOG=$OUT_DIR/${cell}_budgeter.jsonl SGLANG_XPOOL_KV_HIGH=$kv_hi SGLANG_XPOOL_KV_LOW=$kv_lo SGLANG_XPOOL_MAMBA_HIGH=$m_hi SGLANG_XPOOL_MAMBA_LOW=$m_lo SGLANG_XPOOL_COOLDOWN=2"
fi
log="$OUT_DIR/${cell}_server.log"
echo "=== fanout cell=$cell ($extra_env) ==="

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

mem_frac="${MEM_FRAC:-0.8}"

nohup env $extra_env \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
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
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[$cell] ready after ${waited}s"; break
  fi
done
if [ $waited -ge 240 ]; then
  echo "[$cell] FAILED to come up — log tail:"; tail -25 "$log"
  kill -9 $pid 2>/dev/null || true; exit 1
fi

echo "[$cell] Fan-out workload (256+32 random, max-concurrency=400)..."
# Bypass RPS-induced compute backpressure by using --max-concurrency
# (submit-as-many-as-engine-admits) with short decode (32 tokens, ~640
# ms per req). At max-concurrency=400 the engine's mamba-slot pool is
# the sole admission gate; without L2 capacity stays at boot 384,
# concurrency stalls at 384, queueing rises. With L2 mamba expands
# (384 → 512+), more concurrent admissions, lower per-req queue time.
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random \
  --random-input-len 256 --random-output-len 32 \
  --random-range-ratio 1.0 \
  --num-prompts 1200 \
  --max-concurrency 400 \
  --request-rate inf \
  --output-file "$OUT_DIR/${cell}_bench.json" \
  >"$OUT_DIR/${cell}_bench.log" 2>&1 || echo "[$cell] bench failed"
echo "[$cell] bench done"

if [ "$ONLY_L2" = "1" ]; then
  total=$(grep -c '"xpool_direction":' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  echo "[$cell] xpool transfers: total=$total k2m=$k2m m2k=$m2k"
fi

kill -9 $pid 2>/dev/null || true
sleep 4
