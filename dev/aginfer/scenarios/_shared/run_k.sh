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

VARIANT="${1:?usage: run_k.sh <full|ka|J|kv_off|a3|a3_kvoff>}"
case "$VARIANT" in
    full|ka|J|kv_off|a3|a3_kvoff) ;;
    *)
        echo "[run_k] invalid variant: $VARIANT (expected: full|ka|J|kv_off|a3|a3_kvoff)" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

# ---- variant config ----
# A few variants need extra `--ak <k>=<v>` pairs sent to harbor; the
# a3 variant uses this to cap completion tokens.  Default is empty.
declare -a EXTRA_AK_OPTS=()
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
    a3)
        # "A3" — push workload into HBM pressure regime.  Same daemon
        # config as `full` (kv ON + admission ON + HiCache ON) but with:
        #   1. --max-total-tokens = 256K (default ~10M → too loose)
        #   2. --ak llm_call_kwargs={"max_tokens":4096} caps runaway
        #
        # See verify/t9/results/N3_A3_PLAN.md for the rationale and
        # expected daemon counters (specifically: HBM occ peak > 0.85,
        # memory_pressure events > 0, admission pauses > 0, migrate_post
        # count > 0).  If still 0 under a3, escalate to G10 fix.
        HICACHE_FLAG="--enable-hierarchical-cache"
        DAEMON_KV="enabled"
        DAEMON_ADMISSION="enabled"
        # 256K-token KV pool → forces HBM into pressure regime.
        export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
        # 4k-cap completion → kills runaway 60k-token decode tail.
        # parse_kwargs JSON-decodes the value into a dict.
        EXTRA_AK_OPTS=("--ak" 'llm_call_kwargs={"max_tokens":4096}')
        ;;
    a3_kvoff)
        # A3 BASELINE: identical workload regime to `a3` (256K pool + 4k
        # completion cap + HiCache + inline ours_greedy_score scorer) but the
        # daemon's kv_scheduler + admission_controller are OFF — it proxies
        # requests and no-ops on events, issuing NO migrate/pause/resume.
        # Isolates the DAEMON's scheduling effect: a3 vs a3_kvoff under the
        # same pressure = the value-gated daemon's contribution (do-no-harm
        # ⇒ a3 ≈ a3_kvoff within noise).
        HICACHE_FLAG="--enable-hierarchical-cache"
        DAEMON_KV="disabled"
        DAEMON_ADMISSION="disabled"
        export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
        EXTRA_AK_OPTS=("--ak" 'llm_call_kwargs={"max_tokens":4096}')
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
    #
    # IMPORTANT: do NOT match bare "sglang" with -f because our script
    # path /scratch/.../sglang/dev/aginfer/verify/... contains "sglang",
    # so `pkill -f sglang` would SIGKILL this very script and skip the
    # rest of cleanup.  Use specific patterns instead.
    pkill -9 -f "daemon\\.main" 2>/dev/null
    pkill -9 -f "python.*sglang\\.launch_server" 2>/dev/null  # parent process
    pkill -9 -f "sglang::" 2>/dev/null                         # scheduler / detokenizer / TP children
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
    # Reap orphaned networks (Docker's default bridge pool has ~30
    # /24 slots; without `docker compose down`, networks linger across
    # killed runs and eventually `up` fails with
    # "all predefined address pools have been fully subnetted").
    docker network prune -f 2>/dev/null
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

# ---- 1b. aginfer-daemon (BEFORE sglang) ----
# #208 ordering fix: sglang's launch-time bootstrap_thresholds_into_server_
# args (T22 #165) GETs the daemon's /aginfer/thresholds and HALTS if the
# daemon is unreachable ("daemon must be up before sglang").  So the daemon
# MUST start first.  It tolerates a not-yet-up sglang at boot
# (cold_start_probe logs + continues).  The historical A3 data predates
# #165; the old sglang-then-daemon order now deadlocks.
echo "[run_k:$VARIANT] starting aginfer-daemon (kv=$DAEMON_KV admission=$DAEMON_ADMISSION)..."
PYTHONPATH="$AGINFER_DIR:${PYTHONPATH:-}" \
    python -m daemon.main \
        --sglang-base-url=http://127.0.0.1:${SGLANG_PORT:-30000} \
        --port=${DAEMON_PORT:-9100} \
        --kv-scheduler="$DAEMON_KV" \
        --admission-controller="$DAEMON_ADMISSION" \
        >"$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!
for i in $(seq 1 30); do
    if grep -q "Uvicorn running on http://0.0.0.0:${DAEMON_PORT:-9100}" "$DAEMON_LOG" 2>/dev/null; then
        break
    fi
    sleep 1
done
if ! grep -q "Uvicorn running on http://0.0.0.0:${DAEMON_PORT:-9100}" "$DAEMON_LOG" 2>/dev/null; then
    echo "[run_k:$VARIANT] daemon never started; see $DAEMON_LOG" >&2
    exit 8
fi
if ! grep -E "kv_scheduler=$DAEMON_KV admission_controller=$DAEMON_ADMISSION" "$DAEMON_LOG" >/dev/null; then
    echo "[run_k:$VARIANT] HALT — daemon startup invariant did not match" >&2
    grep "kv_scheduler=" "$DAEMON_LOG" | head -1 >&2
    exit 9
fi
echo "[run_k:$VARIANT]   ✓ daemon: kv_scheduler=$DAEMON_KV admission_controller=$DAEMON_ADMISSION"

# ---- 2. sglang ----
echo "[run_k:$VARIANT] starting sglang (TP=${SGLANG_TP:-2}, GPUs=$AGINFER_GPUS, HiCache=${HICACHE_FLAG:-OFF})..."
# T9 README §"For ALL variants": ours_greedy_score scorer is the
# load-bearing inline path; without it, the floor argument breaks.
# Export here (NOT in env.sh) so the env stays local to Run K.
# #230: respect a pre-set value so the eviction-characterization arms can
# swap the inline scorer (lru_score / const_v_u_score / empty=stock LRU)
# without forking run_k.sh.  Unset → the production default below.
export SGLANG_KV_POLICY_MODULE="${SGLANG_KV_POLICY_MODULE:-baselines.sglang_adapter:ours_greedy_score}"

# Pre-rotate the launch-script's internal log so our grep-wait doesn't
# race the launch-script's own rotate_log and match stale content.
# (Bug observed on the second run after a halt: orchestrator's grep
# saw the prior run's "Uvicorn running" line before the launch script
# had a chance to wipe the log.)
SGLANG_LOG_REAL="${SGLANG_LOG_FILE:-$AGINFER_LOGS/sglang_v4flash.log}"
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

# 3b. write_through_loaded grep (#178 T9 parity) — MUST be the default
# hit_count trigger.  The daemon V_u-aware write-through is deferred
# (T27 #188); this pins that it is NOT yet active.  Update to the
# aginfer module here when #188 wires SGLANG_WRITE_THROUGH_MODULE.
if ! grep -E "write_through_loaded=default_hitcount" "$SGLANG_LOG_REAL" >/dev/null; then
    echo "[run_k:$VARIANT] HALT — write_through_loaded did not match" >&2
    grep "write_through_loaded=" "$SGLANG_LOG_REAL" | head -1 >&2 || echo "  (line not found at all)" >&2
    exit 5
fi
echo "[run_k:$VARIANT]   ✓ write_through_loaded=default_hitcount"

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

# ---- 4. daemon: started in §1b (BEFORE sglang) for the T22 bootstrap
#         ordering (#208).  DAEMON_PID is live here for cleanup. ----

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
        --ak api_base=http://172.17.0.1:${DAEMON_PORT:-9100}/v1 \
        --ak api_key="${OPENAI_API_KEY}" \
        --ak max_turns="${HARBOR_MAX_TURNS}" \
        --ak temperature=0.0 \
        --ak seed=42 \
        "${EXTRA_AK_OPTS[@]}" \
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
