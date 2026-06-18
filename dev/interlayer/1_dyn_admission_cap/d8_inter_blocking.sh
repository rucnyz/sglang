#!/usr/bin/env bash
# Reproduce the D8 v4 crash with CUDA_LAUNCH_BLOCKING=1 to get a sync
# trace of the illegal memory access. Inter phase only.

set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_DIR=$(ls -d "$HUB/models--Qwen--Qwen3.5-9B/snapshots/"* | head -1)

OUT_DIR=${OUT_DIR:-/tmp/d8_debug}
GPU=${GPU:-3}
PORT=${PORT:-30077}
mkdir -p "$OUT_DIR"

pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
sleep 3

echo "[D8-debug] boot with CUDA_LAUNCH_BLOCKING=1"
CUDA_VISIBLE_DEVICES=$GPU \
    CUDA_LAUNCH_BLOCKING=1 \
    SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 \
    SGLANG_HIMA_LOG="$OUT_DIR/budgeter.jsonl" \
    SGLANG_XPOOL_MAMBA_HIGH=0.50 \
nohup $VENV -m sglang.launch_server \
    --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
    --tp 1 --mem-fraction-static 0.70 \
    --max-running-requests 256 --max-mamba-cache-size 100 \
    --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
    --log-level info > "$OUT_DIR/server.log" 2>&1 &
SV_PID=$!

waited=0
while [ $waited -lt 600 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
        echo "[D8-debug] ready after ${waited}s"
        break
    fi
    if ! kill -0 $SV_PID 2>/dev/null; then
        echo "[D8-debug] server died early"; tail -30 "$OUT_DIR/server.log"
        exit 1
    fi
done

# Short workload — just enough to trigger the crash (D8 v4 crashed
# within ~10s of bench start). 30s should be plenty.
echo "[D8-debug] short workload 30s"
$VENV -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 1024 \
    --request-rate 32 --num-prompts 960 \
    --output-file "$OUT_DIR/bench.json" \
    > "$OUT_DIR/bench.log" 2>&1 || echo "[D8-debug] bench rc=$?"

# Pull server log
kill -9 $SV_PID 2>/dev/null
sleep 2

echo
echo "=== Last 80 lines of server.log ==="
tail -80 "$OUT_DIR/server.log"
