#!/usr/bin/env bash
# Baseline V4-Flash launch with HiCache DISABLED. Same model / TP / EP / GPU
# layout as launch_sglang_v4flash.sh, but no hierarchical KV cache, no mooncake
# backend — only the device-side radix prefix cache that always ships with
# sglang. Use for HiCache-on vs HiCache-off A/B comparisons.

set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL_PATH="deepseek-ai/DeepSeek-V4-Flash"
HOST="0.0.0.0"
PORT=30000

LOG="$AGINFER_LOGS/sglang_v4flash_nohicache.log"
rotate_log "$LOG"
echo "[launch_sglang_nohicache] GPUs=$AGINFER_GPUS MODEL=$MODEL_PATH PORT=$PORT"
echo "[launch_sglang_nohicache] logging to $LOG"

MAX_TOTAL_TOKENS_ARG=""
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
    MAX_TOTAL_TOKENS_ARG="--max-total-tokens $MAX_TOTAL_TOKENS"
fi
MAX_RUNNING_ARG=""
if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
    MAX_RUNNING_ARG="--max-running-requests $MAX_RUNNING_REQUESTS"
fi

CUDA_VISIBLE_DEVICES="$AGINFER_GPUS" \
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --tp 2 --ep 2 \
    --moe-a2a-backend none \
    --moe-runner-backend deep_gemm \
    --mem-fraction-static 0.85 \
    --context-length 65536 \
    $MAX_TOTAL_TOKENS_ARG \
    $MAX_RUNNING_ARG \
    --reasoning-parser deepseek-r1 \
    --trust-remote-code \
    >>"$LOG" 2>&1
