#!/usr/bin/env bash
# T9 Run K orchestrator — runs ONE variant end-to-end.
#
# Usage:
#   bash verify/t9/run_k.sh <full|ka|J>
#
# Variants:
#   full — kv_scheduler ON + admission_controller ON + HiCache ON
#   ka   — kv_scheduler ON + admission_controller OFF + HiCache ON
#   J    — kv_scheduler ON + admission_controller ON + HiCache OFF
#
# Each variant: ~45 min wall-clock (32 harbor trials).
# Total for all three: ~3 h GPU time.
#
# Tears down mooncake / sglang / daemon on exit (via trap).

set -euo pipefail

VARIANT="${1:?usage: run_k.sh <full|ka|J>}"
case "$VARIANT" in
    full|ka|J) ;;
    *)
        echo "[run_k] invalid variant: $VARIANT (expected: full|ka|J)" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

# ---- variant config ----
case "$VARIANT" in
    full)
        HICACHE_FLAG="--enable-hierarchical-cache"
        DAEMON_KV="enabled"
        DAEMON_ADMISSION="enabled"
        ;;
    ka)
        HICACHE_FLAG="--enable-hierarchical-cache"
        DAEMON_KV="enabled"
        DAEMON_ADMISSION="disabled"
        ;;
    J)
        HICACHE_FLAG=""  # HiCache OFF
        DAEMON_KV="enabled"
        DAEMON_ADMISSION="enabled"
        ;;
esac

RESULTS_DIR="$AGINFER_RESULTS/run_K_${VARIANT}"
mkdir -p "$RESULTS_DIR"

MOONCAKE_LOG="$RESULTS_DIR/mooncake_master.log"
SGLANG_LOG="$RESULTS_DIR/sglang.log"
DAEMON_LOG="$RESULTS_DIR/daemon.log"
HARBOR_LOG="$RESULTS_DIR/harbor.log"

# Pids — populated as we launch.
MOONCAKE_PID=""
SGLANG_PID=""
DAEMON_PID=""

# ---- cleanup ----
cleanup() {
    set +e
    echo "[run_k:$VARIANT] cleanup..."
    [[ -n "${DAEMON_PID:-}" ]] && kill "$DAEMON_PID" 2>/dev/null
    [[ -n "${SGLANG_PID:-}" ]] && kill "$SGLANG_PID" 2>/dev/null
    [[ -n "${MOONCAKE_PID:-}" ]] && kill "$MOONCAKE_PID" 2>/dev/null
    sleep 3
    [[ -n "${DAEMON_PID:-}" ]] && kill -9 "$DAEMON_PID" 2>/dev/null
    [[ -n "${SGLANG_PID:-}" ]] && kill -9 "$SGLANG_PID" 2>/dev/null
    [[ -n "${MOONCAKE_PID:-}" ]] && kill -9 "$MOONCAKE_PID" 2>/dev/null
    # Harbor docker leftovers.
    docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'instance_|swebenchpro' | xargs -r docker kill 2>/dev/null
}
trap cleanup EXIT INT TERM

# ---- 1. mooncake_master (skip for J) ----
if [[ -n "$HICACHE_FLAG" ]]; then
    echo "[run_k:$VARIANT] starting mooncake_master..."
    bash "$AGINFER_DIR/scripts/start_mooncake_master.sh" >"$MOONCAKE_LOG" 2>&1 &
    MOONCAKE_PID=$!
    sleep 3
    if ! kill -0 "$MOONCAKE_PID" 2>/dev/null; then
        echo "[run_k:$VARIANT] mooncake_master died; see $MOONCAKE_LOG" >&2
        exit 3
    fi
    echo "[run_k:$VARIANT] mooncake_master up (pid=$MOONCAKE_PID)"
fi

# ---- 2. sglang ----
echo "[run_k:$VARIANT] starting sglang (TP=$SGLANG_TP, GPUs=$AGINFER_GPUS, HiCache=${HICACHE_FLAG:-OFF})..."
if [[ -n "$HICACHE_FLAG" ]]; then
    bash "$AGINFER_DIR/scripts/launch_sglang_v4flash.sh" >"$SGLANG_LOG" 2>&1 &
else
    bash "$AGINFER_DIR/scripts/launch_sglang_v4flash_nohicache.sh" >"$SGLANG_LOG" 2>&1 &
fi
SGLANG_PID=$!

# Wait for sglang Uvicorn listener.
SGLANG_LOG_REAL="$AGINFER_LOGS/sglang_v4flash.log"  # The launch scripts log here.
echo "[run_k:$VARIANT] waiting for sglang Uvicorn on :30000 (up to 600 s)..."
for i in $(seq 1 300); do
    if grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
        echo "[run_k:$VARIANT] sglang Uvicorn up"
        break
    fi
    sleep 2
done
if ! grep -q "Uvicorn running on http" "$SGLANG_LOG_REAL" 2>/dev/null; then
    echo "[run_k:$VARIANT] sglang never started; see $SGLANG_LOG_REAL" >&2
    exit 4
fi

# ---- 3. Startup invariants (T9 README §"Pre-run startup invariants") ----
echo "[run_k:$VARIANT] checking startup invariants..."

# 3a. kv_policy_loaded grep — MUST be the ours_greedy_score module.
if ! grep -E "kv_policy_loaded=baselines.sglang_adapter:ours_greedy_score" "$SGLANG_LOG_REAL" >/dev/null; then
    echo "[run_k:$VARIANT] HALT — kv_policy_loaded did not match" >&2
    grep "kv_policy_loaded=" "$SGLANG_LOG_REAL" | head -1 >&2 || echo "  (line not found at all)" >&2
    exit 5
fi
echo "[run_k:$VARIANT]   ✓ kv_policy_loaded=baselines.sglang_adapter:ours_greedy_score"

# 3b. tree_cache invariant.  Best-effort grep — sglang doesn't have a
# single canonical "UnifiedRadixCache" log line, but
# SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 must be set (env, not log).
if [[ "${SGLANG_ENABLE_UNIFIED_RADIX_TREE:-0}" != "1" ]]; then
    echo "[run_k:$VARIANT] HALT — SGLANG_ENABLE_UNIFIED_RADIX_TREE != 1" >&2
    exit 6
fi
echo "[run_k:$VARIANT]   ✓ SGLANG_ENABLE_UNIFIED_RADIX_TREE=1"

# 3c. HiCache invariant per variant.
if [[ -n "$HICACHE_FLAG" ]]; then
    if ! grep -q "enable-hierarchical-cache\|HiRadixCache\|hicache" "$SGLANG_LOG_REAL"; then
        echo "[run_k:$VARIANT] HALT — HiCache expected ON but no marker found" >&2
        exit 7
    fi
    echo "[run_k:$VARIANT]   ✓ HiCache ON"
else
    if grep -q "HiRadixCache" "$SGLANG_LOG_REAL"; then
        echo "[run_k:$VARIANT] HALT — HiCache expected OFF but HiRadixCache marker found" >&2
        exit 7
    fi
    echo "[run_k:$VARIANT]   ✓ HiCache OFF (variant J)"
fi

# ---- 4. daemon ----
echo "[run_k:$VARIANT] starting aginfer-daemon (kv=$DAEMON_KV admission=$DAEMON_ADMISSION)..."
PYTHONPATH="$AGINFER_DIR:${PYTHONPATH:-}" \
    python -m daemon.main \
        --sglang-base-url=http://127.0.0.1:30000 \
        --port=9100 \
        --kv-scheduler="$DAEMON_KV" \
        --admission-controller="$DAEMON_ADMISSION" \
        >"$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!

# Wait for daemon ready.
for i in $(seq 1 30); do
    if grep -q "Uvicorn running on http://0.0.0.0:9100" "$DAEMON_LOG" 2>/dev/null; then
        break
    fi
    sleep 1
done
if ! grep -q "Uvicorn running on http://0.0.0.0:9100" "$DAEMON_LOG" 2>/dev/null; then
    echo "[run_k:$VARIANT] daemon never started; see $DAEMON_LOG" >&2
    exit 8
fi

# 4a. Daemon startup invariant.
if ! grep -E "kv_scheduler=$DAEMON_KV admission_controller=$DAEMON_ADMISSION" "$DAEMON_LOG" >/dev/null; then
    echo "[run_k:$VARIANT] HALT — daemon startup invariant did not match" >&2
    grep "kv_scheduler=" "$DAEMON_LOG" | head -1 >&2
    exit 9
fi
echo "[run_k:$VARIANT]   ✓ daemon: kv_scheduler=$DAEMON_KV admission_controller=$DAEMON_ADMISSION"

# ---- 5. harbor run ----
echo "[run_k:$VARIANT] starting harbor (32 trials, swebenchpro/terminus-2)..."
HARBOR_RESULTS="$RESULTS_DIR/harbor_jobs"
mkdir -p "$HARBOR_RESULTS"

(cd /scratch/yuzhou/projects/harbor && \
    harbor run \
        -p datasets/swebenchpro \
        -a terminus-2 \
        -m openai/deepseek-ai/DeepSeek-V4-Flash \
        --ak api_base=http://172.17.0.1:9100/v1 \
        --ak max_turns=200 \
        -n 32 \
        --jobs-dir "$HARBOR_RESULTS" \
    >"$HARBOR_LOG" 2>&1) || HARBOR_EXIT=$?

echo "[run_k:$VARIANT] harbor exit code: ${HARBOR_EXIT:-0}"
echo "[run_k:$VARIANT] harbor results: $HARBOR_RESULTS"
echo "[run_k:$VARIANT] DONE — variant complete"

# Cleanup happens via trap.
exit 0
