#!/usr/bin/env bash
# T9 — LRU baseline (F'_now): sglang default eviction, NO inline scorer,
# NO daemon.  Harbor → sglang :30000 directly.
#
# Same temp=0.0 seed=42 -l 32 -n 32 as matrix / H'_now.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

RESULTS_DIR="$AGINFER_RESULTS/run_LRU_now${RUN_K_RESULTS_TAG:+_${RUN_K_RESULTS_TAG}}"
mkdir -p "$RESULTS_DIR"

MOONCAKE_LOG="$RESULTS_DIR/mooncake_master.log"
SGLANG_LOG="$RESULTS_DIR/sglang.log"
HARBOR_LOG="$RESULTS_DIR/harbor.log"

MOONCAKE_PID=""
SGLANG_PID=""

cleanup() {
    set +e
    echo "[lru] cleanup..."
    [[ -n "${SGLANG_PID:-}" ]] && kill "$SGLANG_PID" 2>/dev/null
    [[ -n "${MOONCAKE_PID:-}" ]] && kill "$MOONCAKE_PID" 2>/dev/null
    sleep 3
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

echo "[lru] pre-flight GPU check on $AGINFER_GPUS..."
IFS=',' read -ra _GPU_LIST <<< "$AGINFER_GPUS"
for gpu in "${_GPU_LIST[@]}"; do
    used_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used_mb > 1024 )); then
        echo "[lru] HALT — GPU $gpu has $used_mb MiB" >&2
        exit 1
    fi
    echo "[lru]   GPU $gpu: $used_mb MiB used (OK)"
done

echo "[lru] starting mooncake_master..."
bash "$AGINFER_DIR/scripts/start_mooncake_master.sh" >"$MOONCAKE_LOG" 2>&1 &
MOONCAKE_PID=$!
sleep 3

echo "[lru] starting sglang (LRU, no SGLANG_KV_POLICY_MODULE)..."
# Explicitly unset so launch script falls back to stock LRU heap key.
unset SGLANG_KV_POLICY_MODULE

SGLANG_LOG_REAL="$AGINFER_LOGS/sglang_v4flash.log"
[[ -e "$SGLANG_LOG_REAL" ]] && mv "$SGLANG_LOG_REAL" "${SGLANG_LOG_REAL}.lru_prev"

bash "$AGINFER_DIR/scripts/launch_sglang_v4flash.sh" >"$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

echo "[lru] waiting for sglang Uvicorn..."
for i in $(seq 1 300); do
    if grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
        echo "[lru] sglang Uvicorn up"
        break
    fi
    sleep 2
done
if ! grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
    echo "[lru] sglang never started" >&2
    exit 4
fi

# LRU invariant: kv_policy_loaded should be the *default* (no module).
# Launch script logs kv_policy_loaded=default_lru when SGLANG_KV_POLICY_MODULE
# is empty.  Match either "default_lru" OR the absence of an aginfer module.
if grep -E "kv_policy_loaded=baselines\\." "$SGLANG_LOG_REAL" >/dev/null 2>&1; then
    echo "[lru] HALT — kv_policy_loaded is aginfer (should be LRU)" >&2
    exit 5
fi
echo "[lru]   ✓ kv_policy_loaded=LRU (no aginfer module)"

echo "[lru] starting harbor (direct → sglang :30000)..."
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
        --ak api_base=http://172.17.0.1:30000/v1 \
        --ak api_key="${OPENAI_API_KEY}" \
        --ak max_turns="${HARBOR_MAX_TURNS}" \
        --ak temperature=0.0 \
        --ak seed=42 \
        -l "${HARBOR_N_TASKS}" \
        -n "${HARBOR_N_CONCURRENT}" \
        -k 1 \
        --jobs-dir "$HARBOR_RESULTS" \
    >"$HARBOR_LOG" 2>&1) || HARBOR_EXIT=$?

echo "[lru] harbor exit code: ${HARBOR_EXIT:-0}"

if [[ -e "$SGLANG_LOG_REAL" ]]; then
    cp -- "$SGLANG_LOG_REAL" "$RESULTS_DIR/sglang_v4flash.log"
fi

echo "[lru] DONE"
exit 0
