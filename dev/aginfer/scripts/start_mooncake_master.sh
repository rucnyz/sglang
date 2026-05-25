#!/usr/bin/env bash
# Start Mooncake master service.
# - 50051 RPC, 9053 metrics. Single-node setup; sglang config uses
#   metadata_server=P2PHANDSHAKE so no separate metadata service is needed.

set -euo pipefail
source "$(dirname "$0")/env.sh"

LOG="$AGINFER_LOGS/mooncake_master.log"
rotate_log "$LOG"
echo "[start_mooncake_master] MOONCAKE_MASTER=$MOONCAKE_MASTER  log=$LOG"

exec mooncake_master \
    --eviction_high_watermark_ratio=0.95 \
    --metrics_port=9053 \
    --enable_offload=true \
    --offload_on_evict=true \
    >>"$LOG" 2>&1
