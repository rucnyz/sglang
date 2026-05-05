#!/bin/bash
# Per-cell driver for the LONG-HORIZON demo workload (paper §motivation
# §76). Few concurrent very-long-context prompts — paged KV pool fills
# (each request takes 16K-32K tokens × low-tens of concurrent reqs ≈
# 500K+ tokens) while DeltaNet recurrent-slot pool sits at single-digit
# utilisation (one slot per request × ~20 concurrent reqs = ~5%).
# Without L2: requests queue once KV fills. With L2: mamba_to_kv
# transfer expands KV capacity (e.g., 800K→1M tokens) by reclaiming
# unused mamba slots, engine admits more concurrent long-horizon
# sessions → TPS up.
#
# Workload: random 16K-token input + 16-token output prompts, RPS=2,
# 60 prompts. Per-req lifetime ~30s prefill+decode → peak concurrency
# ~20-30 long-horizon sessions → ~400K-600K KV tokens demand vs.
# boot ~880K → ~50-70% baseline KV utilisation, room for L2 to grow KV.
#
# Used by 40b_longhorizon_variance_parallel.sh.

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
  # Asymmetric thresholds favoring mamba→kv detection: mamba stays low
  # naturally; KV is the binding pool. MAMBA_LOW high enough that
  # mamba's true low usage triggers the LOW state.
  kv_hi=${SGLANG_XPOOL_KV_HIGH:-0.5}
  kv_lo=${SGLANG_XPOOL_KV_LOW:-0.1}
  m_hi=${SGLANG_XPOOL_MAMBA_HIGH:-0.5}
  m_lo=${SGLANG_XPOOL_MAMBA_LOW:-0.15}
  cooldown=${SGLANG_XPOOL_COOLDOWN:-30}
  extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=$chunk_bytes SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0 SGLANG_BUDGETER_LOG=$OUT_DIR/${cell}_budgeter.jsonl SGLANG_XPOOL_KV_HIGH=$kv_hi SGLANG_XPOOL_KV_LOW=$kv_lo SGLANG_XPOOL_MAMBA_HIGH=$m_hi SGLANG_XPOOL_MAMBA_LOW=$m_lo SGLANG_XPOOL_COOLDOWN=$cooldown"
  # CSIGMA_DEVICE / CSIGMA_MODEL skipped — informational only and may
  # contain spaces ("NVIDIA H200") that break env var=val arg parsing.
  for v in SGLANG_CSIGMA_KV_ALPHA SGLANG_CSIGMA_KV_BETA SGLANG_CSIGMA_KV_GAMMA SGLANG_CSIGMA_M_ALPHA SGLANG_CSIGMA_M_BETA SGLANG_CSIGMA_LSTAR SGLANG_CSIGMA_JSON SGLANG_XPOOL_NB_DIRECTION_AWARE SGLANG_XPOOL_COST_LOG SGLANG_XPOOL_DEFAULT_L; do
    eval "val=\${$v:-}"
    if [ -n "$val" ]; then extra_env="$extra_env $v=$val"; fi
  done
fi
log="$OUT_DIR/${cell}_server.log"
echo "=== longhorizon cell=$cell ($extra_env) ==="

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

mem_frac="${MEM_FRAC:-0.8}"

mamba_ratio_arg=""
if [ -n "${MAMBA_RATIO:-}" ]; then
  mamba_ratio_arg="--mamba-full-memory-ratio $MAMBA_RATIO"
fi
nohup env $extra_env \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static $mem_frac --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    $mamba_ratio_arg \
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

echo "[$cell] Long-horizon workload (16K+16 random, RPS=4)..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random \
  --random-input-len ${LONGHORIZON_INPUT:-16384} --random-output-len ${LONGHORIZON_OUTPUT:-16} \
  --random-range-ratio 1.0 \
  --num-prompts ${NUM_PROMPTS:-120} \
  --request-rate 4 \
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
