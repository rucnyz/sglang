#!/bin/bash
# Setting 4 follow-up — dual-saturation workload to exercise the
# SGLANG_XPOOL_QDEPTH_TRIGGER saturation+queue fallback rule.
#
# The previous Setting 4 e2e validation (Phase 1+2+3 trace) didn't push
# both pools into saturation simultaneously: KV stayed <1% while mamba
# saturated. We need a workload that pushes BOTH:
#   - Long-context prompts (4K each) → fills KV
#   - At very high concurrency → many in-flight prompts → fills mamba
#
# Two arms:
#   legacy:  SGLANG_XPOOL_QDEPTH_TRIGGER=0 (legacy V≈usage rule only)
#   qdepth:  SGLANG_XPOOL_QDEPTH_TRIGGER=4 (saturation+queue fallback)

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
ARM=${ARM:-legacy}
OUT_DIR=${OUT_DIR:?missing}
mkdir -p "$OUT_DIR"

extra_env="SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=0.5 SGLANG_XPOOL_KV_HIGH=0.04 SGLANG_XPOOL_KV_LOW=0.038 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.076 SGLANG_XPOOL_COOLDOWN=2"
case "$ARM" in
  legacy) extra_env="$extra_env SGLANG_XPOOL_QDEPTH_TRIGGER=0" ;;
  qdepth) extra_env="$extra_env SGLANG_XPOOL_QDEPTH_TRIGGER=4" ;;
esac
log="$OUT_DIR/${ARM}_server.log"
jsonl="$OUT_DIR/${ARM}_budgeter.jsonl"
extra_env="$extra_env SGLANG_BUDGETER_LOG=$jsonl"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env $extra_env \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.7 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
pid=$!
echo "[$ARM] pid=$pid"

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10; waited=$((waited+10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[$ARM] ready after ${waited}s"; break
  fi
done

# Dual-saturation workload: 4K-token random prompts at RPS=64.
# 300 prompts → 300×4096 = 1.2M tokens KV pressure; 300 active reqs hit
# mamba pool's 361-slot capacity → both saturate concurrently.
echo "[$ARM] running dual-saturation bench (random 4K, RPS=64)..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 300 \
  --random-input-len 4096 --random-output-len 64 \
  --request-rate 64 \
  --output-file "$OUT_DIR/${ARM}_bench.json" \
  >"$OUT_DIR/${ARM}_bench.log" 2>&1 || echo "[$ARM] bench failed"
echo "[$ARM] bench done"

sleep 6
total=$(grep -c '"xpool_direction":' "$jsonl" 2>/dev/null || echo 0)
k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" 2>/dev/null || echo 0)
m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" 2>/dev/null || echo 0)
sat_q=$(grep -c '"xpool_plan_reason": "saturation+queue' "$jsonl" 2>/dev/null || echo 0)
echo "[$ARM] transfers: total=$total k2m=$k2m m2k=$m2k saturation_queue_decisions=$sat_q"

kill -9 $pid 2>/dev/null || true
sleep 5
