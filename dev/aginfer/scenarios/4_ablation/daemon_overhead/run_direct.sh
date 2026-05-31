#!/usr/bin/env bash
# T9 — re-run Run H' under CURRENT settings (no daemon, harbor → sglang
# directly).
#
# Discriminator:
#   H'_now ≈ 1389 s (matrix baseline) → daemon proxy NOT the cause of
#                                       the H' 885 s ↔ matrix gap;
#                                       gap is from sglang/agent/dataset
#                                       changes since the original H'.
#   H'_now ≈ 885 s  (original H')    → daemon proxy IS the cause; daemon
#                                       overhead = ~500 s/trial.
#
# Differs from run_k.sh kv_off only in: no daemon is started; harbor's
# api_base points at sglang :30000 instead of daemon :9100.
#
# Same temperature=0.0 seed=42, same -l 32 -n 32, same sglang HEAD.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

RESULTS_DIR="$AGINFER_RESULTS/run_H_prime_now${RUN_K_RESULTS_TAG:+_${RUN_K_RESULTS_TAG}}"
mkdir -p "$RESULTS_DIR"

MOONCAKE_LOG="$RESULTS_DIR/mooncake_master.log"
SGLANG_LOG="$RESULTS_DIR/sglang.log"
HARBOR_LOG="$RESULTS_DIR/harbor.log"

MOONCAKE_PID=""
SGLANG_PID=""

cleanup() {
    set +e
    echo "[h_prime] cleanup..."
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
    # Reap orphaned networks: Docker's default bridge address pool has
    # ~30 /24 slots; harbor allocates one per trial.  Across many runs,
    # if killed mid-flight (no `down`), the networks linger.  After ~30
    # the pool is exhausted and `docker compose up` fails with
    # "all predefined address pools have been fully subnetted".
    docker network prune -f 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

echo "[h_prime] pre-flight GPU check on $AGINFER_GPUS..."
IFS=',' read -ra _GPU_LIST <<< "$AGINFER_GPUS"
for gpu in "${_GPU_LIST[@]}"; do
    used_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used_mb > 1024 )); then
        echo "[h_prime] HALT — GPU $gpu has $used_mb MiB used" >&2
        exit 1
    fi
    echo "[h_prime]   GPU $gpu: $used_mb MiB used (OK)"
done

echo "[h_prime] starting mooncake_master..."
bash "$AGINFER_DIR/scripts/start_mooncake_master.sh" >"$MOONCAKE_LOG" 2>&1 &
MOONCAKE_PID=$!
sleep 3

echo "[h_prime] starting sglang..."
export SGLANG_KV_POLICY_MODULE="baselines.sglang_adapter:ours_greedy_score"

SGLANG_LOG_REAL="$AGINFER_LOGS/sglang_v4flash.log"
[[ -e "$SGLANG_LOG_REAL" ]] && mv "$SGLANG_LOG_REAL" "${SGLANG_LOG_REAL}.h_prime_prev"

# direct_sglang arm (no daemon) — explicit empty webhook URL so
# sglang doesn't fire at the absent daemon on :9100.
AGINFER_NOTIFY_URL="" \
    bash "$AGINFER_DIR/scripts/launch_sglang_v4flash.sh" >"$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!

echo "[h_prime] waiting for sglang Uvicorn on :30000 (up to 600 s)..."
for i in $(seq 1 300); do
    if grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
        echo "[h_prime] sglang Uvicorn up"
        break
    fi
    sleep 2
done
if ! grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
    echo "[h_prime] sglang never started" >&2
    exit 4
fi

if ! grep -E "kv_policy_loaded=baselines.sglang_adapter:ours_greedy_score" "$SGLANG_LOG_REAL" >/dev/null; then
    echo "[h_prime] HALT — kv_policy_loaded did not match" >&2
    exit 5
fi
echo "[h_prime]   ✓ kv_policy_loaded=ours_greedy_score"

# No daemon — harbor talks directly to sglang :30000.

echo "[h_prime] starting harbor (direct → sglang :30000, NO daemon)..."
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

echo "[h_prime] harbor exit code: ${HARBOR_EXIT:-0}"

if [[ -e "$SGLANG_LOG_REAL" ]]; then
    cp -- "$SGLANG_LOG_REAL" "$RESULTS_DIR/sglang_v4flash.log"
    echo "[h_prime] copied sglang log → $RESULTS_DIR/sglang_v4flash.log"
fi

echo "[h_prime] DONE"
exit 0
