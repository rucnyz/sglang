#!/bin/bash
# Command 1 of 2: boot a BASELINE sglang server (LRU eviction, no cross-pool
# Budgeter) for a waste measurement. The static mamba/KV split is sglang's
# DEFAULT (mamba_full_memory_ratio = 0.9); we never override it with
# --max-mamba-cache-size. The optional <ratio> arg sweeps the design's own split
# knob (--mamba-full-memory-ratio) to find the static-best split per workload;
# omit it to run the out-of-box default.
# Usage: serve.sh <gpu> <port> [ctxlen] [mamba_full_memory_ratio]
set -u
GPU=$1; PORT=$2; CTXLEN=${3:-262144}; RATIO=${4:-}
MODEL=Qwen/Qwen3.5-9B
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
LOG=${SERVE_LOG:-/tmp/waste_server_${PORT}.log}
RATIO_FLAG=""; [ -n "$RATIO" ] && RATIO_FLAG="--mamba-full-memory-ratio $RATIO"

# SIGKILL any prior server on this port, then wait for release.
PRIOR=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1)
[ -n "$PRIOR" ] && kill -9 "$PRIOR" 2>/dev/null
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
sleep 1

CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VENV -m sglang.launch_server \
  --model-path $MODEL --host 127.0.0.1 --port $PORT --tp 1 \
  --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
  --mem-fraction-static 0.45 --enable-cache-report --disable-overlap-schedule \
  --context-length $CTXLEN --radix-eviction-policy lru $RATIO_FLAG \
  --log-level info > "$LOG" 2>&1 &
echo "[serve] gpu=$GPU port=$PORT ctxlen=$CTXLEN ratio=${RATIO:-default} pid=$! log=$LOG"
