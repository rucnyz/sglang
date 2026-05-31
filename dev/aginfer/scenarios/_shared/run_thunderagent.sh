#!/usr/bin/env bash
# T9 — ThunderAgent baseline (G_now): sglang LRU + TA proxy on :9200.
# Harbor → TA → sglang :30000.  No SGLANG_KV_POLICY_MODULE (TA does the
# scheduling at the proxy level via BFD pause/resume).
#
# Same temp=0.0 seed=42 -l 32 -n 32 as the other arms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

RESULTS_DIR="$AGINFER_RESULTS/run_TA_now${RUN_K_RESULTS_TAG:+_${RUN_K_RESULTS_TAG}}"
mkdir -p "$RESULTS_DIR"

MOONCAKE_LOG="$RESULTS_DIR/mooncake_master.log"
SGLANG_LOG="$RESULTS_DIR/sglang.log"
TA_LOG="$RESULTS_DIR/thunderagent.log"
HARBOR_LOG="$RESULTS_DIR/harbor.log"

MOONCAKE_PID=""
SGLANG_PID=""
TA_PID=""

cleanup() {
    set +e
    echo "[ta] cleanup..."
    [[ -n "${TA_PID:-}" ]] && kill "$TA_PID" 2>/dev/null
    [[ -n "${SGLANG_PID:-}" ]] && kill "$SGLANG_PID" 2>/dev/null
    [[ -n "${MOONCAKE_PID:-}" ]] && kill "$MOONCAKE_PID" 2>/dev/null
    sleep 3
    # IMPORTANT: `-f thunderagent` would match this very script's path
    # (/scratch/.../verify/t9/run_thunderagent.sh) → self-SIGKILL → cleanup
    # stops mid-flight → next cycle's GPU pre-flight HALTs.  Use specific
    # patterns that match only the actual TA process cmdline.
    pkill -9 -f "thunderagent --host" 2>/dev/null  # actual TA bin invocation
    pkill -9 -f "python.*-m ThunderAgent" 2>/dev/null  # alternative entry
    pkill -9 -f "python.*sglang\\.launch_server" 2>/dev/null
    pkill -9 -f "sglang::" 2>/dev/null
    pkill -9 -f "mooncake_master" 2>/dev/null
    sleep 2
    for _ in $(seq 1 10); do
        if ! ps -eo state,comm 2>/dev/null | awk '$1=="Z" && $2~/^sglang/' | grep -q .; then
            break
        fi
        sleep 1
    done
    docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'instance_|swebenchpro' | xargs -r docker kill 2>/dev/null
    docker network prune -f 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

echo "[ta] pre-flight GPU check..."
IFS=',' read -ra _GPU_LIST <<< "$AGINFER_GPUS"
for gpu in "${_GPU_LIST[@]}"; do
    used_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used_mb > 1024 )); then
        echo "[ta] HALT — GPU $gpu has $used_mb MiB" >&2
        exit 1
    fi
done

echo "[ta] starting mooncake_master..."
bash "$AGINFER_DIR/scripts/start_mooncake_master.sh" >"$MOONCAKE_LOG" 2>&1 &
MOONCAKE_PID=$!
sleep 3

echo "[ta] starting sglang (LRU)..."
unset SGLANG_KV_POLICY_MODULE
SGLANG_LOG_REAL="$AGINFER_LOGS/sglang_v4flash.log"
[[ -e "$SGLANG_LOG_REAL" ]] && mv "$SGLANG_LOG_REAL" "${SGLANG_LOG_REAL}.ta_prev"

# TA is the baseline arm — no aginfer daemon involved.
# Empty AGINFER_NOTIFY_URL keeps sglang from firing the webhook
# at :9100 (nothing listening → connection-refused retry spam).
AGINFER_NOTIFY_URL="" \
    bash "$AGINFER_DIR/scripts/launch_sglang_v4flash.sh" >"$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

for i in $(seq 1 300); do
    if grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
        echo "[ta] sglang Uvicorn up"
        break
    fi
    sleep 2
done
if ! grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
    echo "[ta] sglang never started" >&2
    exit 4
fi

echo "[ta] starting ThunderAgent on :9200 → sglang :30000..."
TA_PROFILE_DIR="$RESULTS_DIR/thunderagent_profile"
mkdir -p "$TA_PROFILE_DIR"
TA_HOST=0.0.0.0 TA_PORT=9200 \
    TA_BACKEND_URL="http://127.0.0.1:30000" \
    TA_PROFILE_DIR="$TA_PROFILE_DIR" \
    bash "$AGINFER_DIR/scripts/launch_thunderagent.sh" >"$TA_LOG" 2>&1 &
TA_PID=$!
# Wait for TA to be listening.  TA proxies most paths to sglang;
# `/v1/models` returns 404 because TA strips `/v1/` then sglang
# doesn't serve `/models`.  Just confirm the port answers (any HTTP
# status counts — connection-refused means not up).
for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:9200/v1/models" 2>/dev/null || echo "")
    if [[ -n "$code" && "$code" != "000" ]]; then
        echo "[ta] ThunderAgent up on :9200 (HTTP $code from readiness probe)"
        break
    fi
    sleep 2
done
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:9200/v1/models" 2>/dev/null || echo "")
if [[ -z "$code" || "$code" == "000" ]]; then
    echo "[ta] ThunderAgent never started; see $AGINFER_LOGS/thunderagent.log" >&2
    tail -50 "$AGINFER_LOGS/thunderagent.log" >&2 || true
    exit 6
fi

echo "[ta] starting harbor (→ TA :9200)..."
HARBOR_RESULTS="$RESULTS_DIR/harbor_jobs"
mkdir -p "$HARBOR_RESULTS"

HARBOR_N_TASKS="${SMOKE_N_TASKS:-32}"
HARBOR_N_CONCURRENT="${SMOKE_N_CONCURRENT:-32}"
HARBOR_MAX_TURNS="${SMOKE_MAX_TURNS:-200}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
(cd /scratch/yuzhou/projects/harbor && \
    harbor run \
        -p datasets/swebenchpro \
        -a terminus-2 \
        -m openai/deepseek-ai/DeepSeek-V4-Flash \
        --ak api_base=http://172.17.0.1:9200/v1 \
        --ak api_key="${OPENAI_API_KEY}" \
        --ak max_turns="${HARBOR_MAX_TURNS}" \
        --ak temperature=0.0 \
        --ak seed=42 \
        -l "${HARBOR_N_TASKS}" \
        -n "${HARBOR_N_CONCURRENT}" \
        -k 1 \
        --jobs-dir "$HARBOR_RESULTS" \
    >"$HARBOR_LOG" 2>&1) || HARBOR_EXIT=$?

echo "[ta] harbor exit code: ${HARBOR_EXIT:-0}"

if [[ -e "$SGLANG_LOG_REAL" ]]; then
    cp -- "$SGLANG_LOG_REAL" "$RESULTS_DIR/sglang_v4flash.log"
fi

echo "[ta] DONE"
exit 0
