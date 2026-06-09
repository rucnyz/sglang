#!/usr/bin/env bash
# Launch ThunderAgent as a TR-mode (capacity-scheduling) proxy in front of the
# already-running sglang V4-Flash server. Harbor agent containers send their
# chat-completion requests here (port 9000); ThunderAgent inspects each
# request's program_id, tracks per-program KV-cache usage on the backend, and
# pauses/resumes programs to keep the backend within capacity.
#
# Requires:
#   - sglang V4-Flash up on 30000 (launch_sglang_v4flash.sh), --enable-metrics on
#   - ThunderAgent installed in agsched env (pip install -e on /scratch/yuzhou/projects/ThunderAgent)
set -euo pipefail
source "$(dirname "$0")/env.sh"

HOST="${TA_HOST:-0.0.0.0}"
PORT="${TA_PORT:-9000}"
# Address used by ThunderAgent itself to reach sglang; localhost works here
# because the proxy runs on the same host as sglang.
BACKEND_URL="${TA_BACKEND_URL:-http://127.0.0.1:30000}"
PROFILE_DIR="${TA_PROFILE_DIR:-$AGINFER_LOGS/thunderagent_profile}"
mkdir -p "$PROFILE_DIR"

LOG="$AGINFER_LOGS/thunderagent.log"
rotate_log "$LOG"
echo "[launch_thunderagent] host=$HOST:$PORT backend=$BACKEND_URL"
echo "[launch_thunderagent] logging to $LOG"

thunderagent \
    --host "$HOST" \
    --port "$PORT" \
    --backends "$BACKEND_URL" \
    --backend-type sglang \
    --router tr \
    --metrics \
    --scheduler-interval 2.0 \
    --acting-token-weight 1.0 \
    --profile \
    --profile-dir "$PROFILE_DIR" \
    >>"$LOG" 2>&1
