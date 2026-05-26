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

VARIANT="${1:?usage: run_k.sh <full|ka|J|kv_off>}"
case "$VARIANT" in
    full|ka|J|kv_off) ;;
    *)
        echo "[run_k] invalid variant: $VARIANT (expected: full|ka|J|kv_off)" >&2
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
    kv_off)
        # Diagnostic: kv_scheduler OFF + admission OFF + HiCache ON.
        # Daemon proxies requests + EventRouter no-ops on events; no
        # migrate POSTs, no pause/resume.  Inline scorer (sglang's
        # drive_eviction) still uses ours_greedy_score.  Approximates
        # Run H' + daemon proxy hop.  If kv_off mean ≈ Run H' 885 s,
        # kv_scheduler is the source of the K-full / K-a slowdown.
        # If kv_off mean ≈ 1550 s, the inline scorer's V_u (paper §7)
        # itself is the source — escalate to T11 (empirical p_hat).
        HICACHE_FLAG="--enable-hierarchical-cache"
        DAEMON_KV="disabled"
        DAEMON_ADMISSION="disabled"
        ;;
esac

RESULTS_DIR="$AGINFER_RESULTS/run_K_${VARIANT}${RUN_K_RESULTS_TAG:+_${RUN_K_RESULTS_TAG}}"
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
# Killing the top-level launch script's pid does NOT reap sglang's
# scheduler subprocess (TP shards, ray workers) — they keep holding
# GPU memory and the NEXT Run K hits CUDA OOM.  Match by command
# pattern to catch the full tree.
cleanup() {
    set +e
    echo "[run_k:$VARIANT] cleanup..."
    # SIGTERM by pid first (clean shutdown if it honors it).
    [[ -n "${DAEMON_PID:-}" ]] && kill "$DAEMON_PID" 2>/dev/null
    [[ -n "${SGLANG_PID:-}" ]] && kill "$SGLANG_PID" 2>/dev/null
    [[ -n "${MOONCAKE_PID:-}" ]] && kill "$MOONCAKE_PID" 2>/dev/null
    sleep 3
    # Force-kill the full process trees.  Match by cmdline substring
    # since Linux truncates comm to 15 chars (sglang::scheduler →
    # sglang::schedul) — a pkill on the full name silently misses.
    pkill -9 -f "daemon.main" 2>/dev/null
    pkill -9 -f "sglang" 2>/dev/null      # catches launch_server, srt, schedul, detoken, tp
    pkill -9 -f "mooncake_master" 2>/dev/null
    sleep 2
    # Drain any remaining zombies (PPID=1, state=Z) — the kernel
    # reclaims the GPU buffer only after init reaps them, so without
    # the wait the next Run K may hit phantom-OOM.  Up to 10 s.
    for _ in $(seq 1 10); do
        if ! ps -eo state,comm 2>/dev/null | awk '$1=="Z" && $2~/^sglang/' | grep -q .; then
            break
        fi
        sleep 1
    done
    # Harbor docker leftovers.  Pipe failure (no match) is fine.
    docker ps --format '{{.Names}}' 2>/dev/null | grep -E 'instance_|swebenchpro' | xargs -r docker kill 2>/dev/null
    # End cleanup with explicit success so the EXIT trap doesn't
    # propagate a cosmetic non-zero from the last pipe (docker ps |
    # grep with no match) into the script's exit code.
    return 0
}
trap cleanup EXIT INT TERM

# ---- pre-flight: confirm our GPUs are actually free ----
# nvidia-smi reports `pid, used_memory` even for zombie processes; a
# launch into a "0 MB free" GPU just OOMs after model load.  Use the
# per-GPU `memory.used` view from nvidia-smi which (post-zombie-reap)
# accurately reflects actually-claimable VRAM.
echo "[run_k:$VARIANT] pre-flight GPU check on $AGINFER_GPUS..."
IFS=',' read -ra _GPU_LIST <<< "$AGINFER_GPUS"
for gpu in "${_GPU_LIST[@]}"; do
    used_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( used_mb > 1024 )); then
        echo "[run_k:$VARIANT] HALT — GPU $gpu has $used_mb MiB used (need < 1024 MiB)" >&2
        echo "[run_k:$VARIANT] check zombies: ps -eo state,pid,comm | awk '\$1==\"Z\"'" >&2
        echo "[run_k:$VARIANT] or other users' jobs on this GPU." >&2
        exit 1
    fi
    echo "[run_k:$VARIANT]   GPU $gpu: $used_mb MiB used (OK)"
done

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
# T9 README §"For ALL variants": ours_greedy_score scorer is the
# load-bearing inline path; without it, the floor argument breaks.
# Export here (NOT in env.sh) so the env stays local to Run K.
export SGLANG_KV_POLICY_MODULE="baselines.sglang_adapter:ours_greedy_score"

# Pre-rotate the launch-script's internal log so our grep-wait doesn't
# race the launch-script's own rotate_log and match stale content.
# (Bug observed on the second run after a halt: orchestrator's grep
# saw the prior run's "Uvicorn running" line before the launch script
# had a chance to wipe the log.)
SGLANG_LOG_REAL="$AGINFER_LOGS/sglang_v4flash.log"
[[ -e "$SGLANG_LOG_REAL" ]] && mv "$SGLANG_LOG_REAL" "${SGLANG_LOG_REAL}.run_k_prev"

if [[ -n "$HICACHE_FLAG" ]]; then
    bash "$AGINFER_DIR/scripts/launch_sglang_v4flash.sh" >"$SGLANG_LOG" 2>&1 &
else
    bash "$AGINFER_DIR/scripts/launch_sglang_v4flash_nohicache.sh" >"$SGLANG_LOG" 2>&1 &
fi
SGLANG_PID=$!

# Wait for sglang Uvicorn listener.
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
HARBOR_RESULTS="$RESULTS_DIR/harbor_jobs"
mkdir -p "$HARBOR_RESULTS"

# litellm runs on the HOST (not in docker as I first thought), so its
# OPENAI_API_KEY check reads from THIS shell's env, not from --ae.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
# Harbor flag map (harbor run --help):
#   -n / --n-concurrent : how many trials run in parallel (default 4)
#   -l / --n-tasks      : TOTAL trial cap (NOT what -n does)
#   -k / --n-attempts   : retries per trial (default 1)
#
# Swebenchpro dataset has 32 tasks.  We want all 32, fully parallel:
#   N_TASKS=32, N_CONCURRENT=32, N_ATTEMPTS=1.
#
# Smoke override: SMOKE_N_TASKS=2 SMOKE_N_CONCURRENT=2 SMOKE_MAX_TURNS=20
# → pipeline-validation run in ~25 min wall.
HARBOR_N_TASKS="${SMOKE_N_TASKS:-32}"
HARBOR_N_CONCURRENT="${SMOKE_N_CONCURRENT:-32}"
HARBOR_MAX_TURNS="${SMOKE_MAX_TURNS:-200}"
HARBOR_ATTEMPTS="${SMOKE_ATTEMPTS:-1}"

echo "[run_k:$VARIANT] starting harbor (n_tasks=${HARBOR_N_TASKS}, concurrent=${HARBOR_N_CONCURRENT}, max_turns=${HARBOR_MAX_TURNS}, swebenchpro/terminus-2)..."

(cd /scratch/yuzhou/projects/harbor && \
    harbor run \
        -p datasets/swebenchpro \
        -a terminus-2 \
        -m openai/deepseek-ai/DeepSeek-V4-Flash \
        --ak api_base=http://172.17.0.1:9100/v1 \
        --ak api_key="${OPENAI_API_KEY}" \
        --ak max_turns="${HARBOR_MAX_TURNS}" \
        --ak temperature=0.0 \
        --ak seed=42 \
        -l "${HARBOR_N_TASKS}" \
        -n "${HARBOR_N_CONCURRENT}" \
        -k "${HARBOR_ATTEMPTS}" \
        --jobs-dir "$HARBOR_RESULTS" \
    >"$HARBOR_LOG" 2>&1) || HARBOR_EXIT=$?

echo "[run_k:$VARIANT] harbor exit code: ${HARBOR_EXIT:-0}"
echo "[run_k:$VARIANT] harbor results: $HARBOR_RESULTS"

# Copy the real sglang log into the cycle's results dir (the launch
# script writes to $AGINFER_LOGS/sglang_v4flash.log; the orchestrator's
# stdout-redirected $SGLANG_LOG only catches wrapper output).  Needed
# for offline TTFT / cache-hit analysis (parse_matrix.py).
if [[ -e "$SGLANG_LOG_REAL" ]]; then
    cp -- "$SGLANG_LOG_REAL" "$RESULTS_DIR/sglang_v4flash.log"
    echo "[run_k:$VARIANT] copied sglang log → $RESULTS_DIR/sglang_v4flash.log"
fi

echo "[run_k:$VARIANT] DONE — variant complete"

# Cleanup happens via trap.
exit 0
