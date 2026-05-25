#!/usr/bin/env bash
# Smoke test: launch SGLang with a tiny model + HiCache + Mooncake.
# Goal: verify the whole pipeline (mooncake_master <-> sglang HiCache backend)
# works end-to-end before pulling V4-Flash, which takes ~minutes to load.
#
# Model: Qwen/Qwen3-0.6B (small, fast to load, easy to verify).

set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL_PATH="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
HOST="127.0.0.1"
PORT="${SMOKE_PORT:-30001}"

LOG="$AGINFER_LOGS/sglang_smoke.log"
rotate_log "$LOG"
echo "[smoke] GPU=${AGINFER_GPUS%%,*} MODEL=$MODEL_PATH PORT=$PORT"
echo "[smoke] logging to $LOG"

MOONCAKE_EXTRA=$(cat <<EOF
{
  "master_server_address": "127.0.0.1:50051",
  "local_hostname": "localhost",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": "16gb",
  "local_buffer_size": "1gb"
}
EOF
)

# Single GPU (first one from AGINFER_GPUS, default 5)
CUDA_VISIBLE_DEVICES="${AGINFER_GPUS%%,*}" \
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --tp 1 \
    --mem-fraction-static 0.80 \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --hicache-write-policy write_through_selective \
    --hicache-storage-backend mooncake \
    --hicache-storage-prefetch-policy best_effort \
    --hicache-storage-backend-extra-config "$MOONCAKE_EXTRA" \
    --trust-remote-code \
    >>"$LOG" 2>&1
