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

# Use the same log filename as the HiCache launcher so run_k.sh's
# `grep Uvicorn` waits on the right file regardless of variant.
# (Was sglang_v4flash_nohicache.log — different filename caused
#  Run J to silently time out waiting for sglang.)
LOG="$AGINFER_LOGS/sglang_v4flash.log"
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

TP="${SGLANG_TP:-2}"
EP="${SGLANG_EP:-$TP}"

# Same aginfer webhook wiring as the HiCache variant — admission's
# pressure trigger must work in Run J too, otherwise the daemon's
# admission layer is silently a no-op in the HiCache-OFF ablation.
# `${VAR-default}` (no colon) so explicit empty string opts out.
AGINFER_NOTIFY_URL="${AGINFER_NOTIFY_URL-http://127.0.0.1:9100}"
AGINFER_THETA_HI="${AGINFER_THETA_HI:-0.85}"
AGINFER_THETA_CRIT="${AGINFER_THETA_CRIT:-0.95}"
AGINFER_HEARTBEAT_S="${AGINFER_HEARTBEAT_S:-5.0}"
AGINFER_FLAGS=()
if [[ -n "$AGINFER_NOTIFY_URL" ]]; then
    AGINFER_FLAGS=(
        --aginfer-notify-url "$AGINFER_NOTIFY_URL"
        --aginfer-theta-hi "$AGINFER_THETA_HI"
        --aginfer-theta-crit "$AGINFER_THETA_CRIT"
        --aginfer-heartbeat-s "$AGINFER_HEARTBEAT_S"
    )
fi

SGLANG_KV_POLICY_MODULE="${SGLANG_KV_POLICY_MODULE:-}" \
PYTHONPATH="$AGINFER_ROOT:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$AGINFER_GPUS" \
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --tp "$TP" --ep "$EP" \
    --moe-a2a-backend none \
    --moe-runner-backend deep_gemm \
    --mem-fraction-static 0.85 \
    --context-length 65536 \
    $MAX_TOTAL_TOKENS_ARG \
    $MAX_RUNNING_ARG \
    "${AGINFER_FLAGS[@]}" \
    --reasoning-parser deepseek-r1 \
    --trust-remote-code \
    --enable-metrics \
    --enable-cache-report \
    --log-requests \
    --log-requests-level 0 \
    --log-requests-format json \
    --random-seed "${SGLANG_RANDOM_SEED:-42}" \
    >>"$LOG" 2>&1
