#!/usr/bin/env bash
# Run sglang with SGLANG_GAP_TRACE_LOG enabled + drive CC trace replay.
# Records per-batch GPU/CPU gap distribution.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/gap_trace}
GPU=${GPU:-3}
PORT=${PORT:-30077}
NUM_CONCURRENCY=${NUM_CONCURRENCY:-14}
MAX_TIME_MIN=${MAX_TIME_MIN:-4}
TRACES_FILE=${TRACES_FILE:-/scratch/yuzhou/projects/vllm-songyang/dev/interlayer/cc_long_traces.jsonl}

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/server.log" "$OUT_DIR/gap.jsonl" "$OUT_DIR/bench.json" "$OUT_DIR/bench.log"

pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
sleep 3

echo "[gap-trace] boot (C=$NUM_CONCURRENCY, ${MAX_TIME_MIN}min replay)"
CUDA_VISIBLE_DEVICES=$GPU \
    SGLANG_GAP_TRACE_LOG=$OUT_DIR/gap.jsonl \
    nohup $VENV -m sglang.launch_server \
        --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
        --tp 1 --mem-fraction-static 0.55 \
        --max-running-requests 256 \
        --max-mamba-cache-size 256 \
        --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
        --log-level info > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

waited=0
while [ $waited -lt 600 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
        echo "[gap-trace] ready after ${waited}s"; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[gap-trace] server died"; tail -25 "$OUT_DIR/server.log"; exit 1
    fi
done
[ $waited -ge 600 ] && { kill -9 $SERVER_PID; echo "[gap-trace] TIMEOUT"; exit 1; }

echo "[gap-trace] CC replay for ${MAX_TIME_MIN} min"
$VENV dev/eval/main/cc_trace_replay.py \
    --api-base "http://127.0.0.1:$PORT" \
    --model "$MODEL_DIR" \
    --traces "$TRACES_FILE" \
    --num-concurrency "$NUM_CONCURRENCY" \
    --max-time-min "$MAX_TIME_MIN" \
    --max-tokens 1024 \
    --min-turns 15 --min-chars 30000 \
    --output-file "$OUT_DIR/bench.json" \
    > "$OUT_DIR/bench.log" 2>&1 || echo "[gap-trace] bench rc=$?"

kill -9 $SERVER_PID 2>/dev/null
sleep 3

echo
echo "=== gap distribution ==="
$VENV dev/interlayer/0_batch_boundary_fire/step2_scheduler_idle_gap_trace/analyze_gap_trace.py "$OUT_DIR/gap.jsonl"
