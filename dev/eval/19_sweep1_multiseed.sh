#!/bin/bash
# Sweep 1 multi-seed variance bands.
#
# Re-run Sweep 1 (KV↔DN ratio sweep) at 3 different seeds to get error
# bars on the 1.91× throughput swing claim. Single ratio + single seed
# per invocation; outer driver fans 3 seeds × 5 ratios across GPUs.
#
# Use: SEED=$s RATIO=$r CUDA_VISIBLE_DEVICES=$g PORT=$p OUT_DIR=$d \
#      dev/eval/19_sweep1_multiseed.sh

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
NUM_PROMPTS=${NUM_PROMPTS:-500}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-256}
RPS=${RPS:-32}
RATIO=${RATIO:-0.5}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-/tmp/sweep1_seed${SEED}_$$}
mkdir -p "$OUT_DIR"

label="ratio${RATIO}_seed${SEED}"
log="$OUT_DIR/${label}_server.log"
bench_out="$OUT_DIR/${label}_bench.json"
echo "[$label] out=$OUT_DIR"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup .venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
  --mem-fraction-static 0.8 --log-level info \
  --enforce-piecewise-cuda-graph \
  --reasoning-parser qwen3 \
  --mamba-full-memory-ratio "$RATIO" \
  --random-seed "$SEED" \
  >"$log" 2>&1 &
pid=$!

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[$label] ready after ${waited}s"
    break
  fi
done

.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts $NUM_PROMPTS \
  --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
  --request-rate $RPS \
  --seed "$SEED" \
  --output-file "$bench_out" \
  >"$OUT_DIR/${label}_bench.log" 2>&1 || echo "[$label] bench FAILED"
echo "[$label] bench done"

kill -9 $pid 2>/dev/null || true
sleep 5
