#!/usr/bin/env bash
# Start Mooncake master service.
# - 50051 RPC, 9053 metrics. Single-node setup; sglang config uses
#   metadata_server=P2PHANDSHAKE so no separate metadata service is needed.

set -euo pipefail
source "$(dirname "$0")/env.sh"

LOG="${MOONCAKE_LOG_FILE:-$AGINFER_LOGS/mooncake_master.log}"
rotate_log "$LOG"
# RPC + metrics ports overridable so two masters run in parallel (instance B
# passes MOONCAKE_RPC_PORT / MOONCAKE_METRICS_PORT, and points its sglang at
# MOONCAKE_MASTER=127.0.0.1:<MOONCAKE_RPC_PORT>).
echo "[start_mooncake_master] MOONCAKE_MASTER=$MOONCAKE_MASTER  rpc=${MOONCAKE_RPC_PORT:-50051}  metrics=${MOONCAKE_METRICS_PORT:-9053}  log=$LOG"

exec mooncake_master \
    --eviction_high_watermark_ratio=0.95 \
    --rpc_port="${MOONCAKE_RPC_PORT:-50051}" \
    --metrics_port="${MOONCAKE_METRICS_PORT:-9053}" \
    --enable_offload=true \
    --offload_on_evict=true \
    >>"$LOG" 2>&1
