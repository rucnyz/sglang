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
# Port + log overridable so two stacks can run in parallel on disjoint GPU
# pairs (instance B passes SGLANG_PORT / SGLANG_LOG_FILE / MOONCAKE_MASTER).
PORT="${SGLANG_PORT:-30000}"

LOG="${SGLANG_LOG_FILE:-$AGINFER_LOGS/sglang_v4flash.log}"
rotate_log "$LOG"
echo "[launch_sglang] GPUs=$AGINFER_GPUS MODEL=$MODEL_PATH PORT=$PORT"
echo "[launch_sglang] logging to $LOG"

# Mooncake L3: tcp + P2PHANDSHAKE (single-node, no metadata svc).
# global_segment_size: 200GB DRAM contributed.
# enable_ssd_offload: spill to /scratch/yuzhou/mooncake_ssd.
MOONCAKE_EXTRA=$(cat <<EOF
{
  "master_server_address": "${MOONCAKE_MASTER:-127.0.0.1:50051}",
  "local_hostname": "localhost",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": "${HICACHE_STORE_SIZE:-200gb}",
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

TP="${SGLANG_TP:-2}"
EP="${SGLANG_EP:-$TP}"

# Explicit nccl/dist port so two parallel stacks don't collide on sglang's
# port-derived internal TCP ports (a +100 HTTP offset was NOT enough — the
# second stack's HTTP bind landed on the first's nccl/dist range).  Unset =
# sglang's default (random) for the single-instance path.
NCCL_ARG=()
if [[ -n "${SGLANG_NCCL_PORT:-}" ]]; then
    NCCL_ARG=(--nccl-port "$SGLANG_NCCL_PORT")
fi

# aginfer T5 webhook target.  sglang's AginferWebhookFirer POSTs
# memory_pressure / pressure_resolved transitions here.  Default
# matches launch_daemon.sh's listening port (9100).
#
# Use `${VAR-default}` (no colon) so an *explicitly empty* env var
# (``AGINFER_NOTIFY_URL="" bash launch_sglang_v4flash.sh``) opts
# OUT of webhook firing — needed by no-daemon baseline runners
# (run_lru.sh, run_thunderagent.sh, run_direct.sh) so sglang
# doesn't spam connection-refused retries.
AGINFER_NOTIFY_URL="${AGINFER_NOTIFY_URL-http://127.0.0.1:9100}"
# Threshold knobs (sglang side).  Must match daemon-side
# admission_controller theta_hi / theta_lo for the design's
# "sglang fires → daemon acts" handshake to be consistent.
AGINFER_THETA_HI="${AGINFER_THETA_HI:-0.85}"
AGINFER_THETA_LO="${AGINFER_THETA_LO:-0.70}"
AGINFER_THETA_CRIT="${AGINFER_THETA_CRIT:-0.95}"
AGINFER_HEARTBEAT_S="${AGINFER_HEARTBEAT_S:-5.0}"

# Only forward aginfer flags when notify URL is non-empty (avoid
# firing webhook at no-one in baseline runners).
AGINFER_FLAGS=()
if [[ -n "$AGINFER_NOTIFY_URL" ]]; then
    AGINFER_FLAGS=(
        --aginfer-notify-url "$AGINFER_NOTIFY_URL"
        --aginfer-theta-hi "$AGINFER_THETA_HI"
        --aginfer-theta-lo "$AGINFER_THETA_LO"
        --aginfer-theta-crit "$AGINFER_THETA_CRIT"
        --aginfer-heartbeat-s "$AGINFER_HEARTBEAT_S"
    )
fi

# aginfer eviction-scorer plug-in.  Format: 'pkg.mod:callable' (loaded by
# UnifiedRadixCache._load_eviction_scorer).  Unset / empty = stock LRU.
# Run H sets SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:ours_greedy_score.
SGLANG_KV_POLICY_MODULE="${SGLANG_KV_POLICY_MODULE:-}" \
PYTHONPATH="$AGINFER_ROOT:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$AGINFER_GPUS" \
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    "${NCCL_ARG[@]}" \
    --tp "$TP" --ep "$EP" \
    --moe-a2a-backend none \
    --moe-runner-backend deep_gemm \
    --mem-fraction-static 0.85 \
    --context-length 65536 \
    $MAX_TOTAL_TOKENS_ARG \
    $MAX_RUNNING_ARG \
    $EVICTION_POLICY_ARG \
    --enable-hierarchical-cache \
    --hicache-ratio "${HICACHE_RATIO:-1.5}" \
    --hicache-write-policy write_through_selective \
    --hicache-storage-backend mooncake \
    --hicache-storage-prefetch-policy best_effort \
    --hicache-storage-backend-extra-config "$MOONCAKE_EXTRA" \
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
