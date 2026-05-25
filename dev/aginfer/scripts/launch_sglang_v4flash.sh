#!/usr/bin/env bash
# Launch SGLang serving DeepSeek-V4-Flash on GPUs 5,6 with TP=2, EP=2.
# Full 4-tier HiCache: HBM <-> DRAM <-> Mooncake (DRAM + SSD spill) via the
# UnifiedRadixCache + L3 HiStorage framework on PR #26062.
# Prerequisite: start_mooncake_master.sh is running on port 50051.

set -euo pipefail
source "$(dirname "$0")/env.sh"

MODEL_PATH="deepseek-ai/DeepSeek-V4-Flash"
# Bind to all interfaces so harbor agents in Docker containers can reach the
# server at host.docker.internal:30000 (= 172.17.0.1 from the bridge).
HOST="0.0.0.0"
PORT=30000

LOG="$AGINFER_LOGS/sglang_v4flash.log"
rotate_log "$LOG"
echo "[launch_sglang] GPUs=$AGINFER_GPUS MODEL=$MODEL_PATH PORT=$PORT"
echo "[launch_sglang] logging to $LOG"

# Mooncake L3: tcp + P2PHANDSHAKE (single-node, no metadata svc).
# global_segment_size: 200GB DRAM contributed.
# enable_ssd_offload: spill to /scratch/yuzhou/mooncake_ssd.
MOONCAKE_EXTRA=$(cat <<EOF
{
  "master_server_address": "127.0.0.1:50051",
  "local_hostname": "localhost",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": "200gb",
  "local_buffer_size": "4gb"
}
EOF
)

# Optional manual KV-pool cap to force device-side eviction (so the HiCache
# DRAM tier actually gets exercised). Empty = use sglang's default ~10M tokens.
MAX_TOTAL_TOKENS_ARG=""
if [[ -n "${MAX_TOTAL_TOKENS:-}" ]]; then
    MAX_TOTAL_TOKENS_ARG="--max-total-tokens $MAX_TOTAL_TOKENS"
fi
MAX_RUNNING_ARG=""
if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
    MAX_RUNNING_ARG="--max-running-requests $MAX_RUNNING_REQUESTS"
fi
EVICTION_POLICY_ARG=""
if [[ -n "${EVICTION_POLICY:-}" ]]; then
    EVICTION_POLICY_ARG="--radix-eviction-policy $EVICTION_POLICY"
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
    $EVICTION_POLICY_ARG \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --hicache-write-policy write_through_selective \
    --hicache-storage-backend mooncake \
    --hicache-storage-prefetch-policy best_effort \
    --hicache-storage-backend-extra-config "$MOONCAKE_EXTRA" \
    --reasoning-parser deepseek-r1 \
    --trust-remote-code \
    --enable-metrics \
    >>"$LOG" 2>&1
