#!/usr/bin/env bash
# Request-bounded N-rep paired cc A/B for the ZERO-DOWNSIDE check.
#
# Motivation (#290/#268): the time-bounded cc replay (MAX_SESSIONS=0) confounds
# cache_hit — off and inter complete DIFFERENT session sets within the time cap,
# so cache_hit = Σcached/Σprompt is computed over different populations. Two
# runs with near-identical fires can then look opposite:
#   d290_run  : 2x m2k, drained 5+6 cached pages, off cache_hit 0.629 -> inter 0.708 (+7.86pp)
#   d290_run2 : 2x m2k, drained 1+2 cached pages, off cache_hit 0.695 -> inter 0.451 (-24.4pp)
# run2 drained FEWER cached pages yet "regressed" more => the swing is replay
# variance, not the mechanism.
#
# This wrapper request-bounds the replay (MAX_SESSIONS=N: both cells process the
# IDENTICAL session set) and runs N_REPS paired off/inter A/Bs, then asserts
# zero-downside (no metric regresses beyond a noise band) via
# validate_zero_downside.py. Sources the real H200 profile (kappa_M=0), LPB on.
#
# Knobs default to run1's moderate fire config (PAGES_PER_FIRE=8, COOLDOWN=6,
# NB_MARGIN=1.2) so fires actually happen; override via env to stress-test
# (e.g. run2's aggressive 4/3/1.1) -- zero-downside should hold either way.
set -uo pipefail
cd "$(dirname "$0")"

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
source /scratch/yuzhou/projects/sglang/dev/eval/cost_model/profiles/Qwen_Qwen3.5-9B_H200.sh

export SGLANG_BUDGETER_PAGES_PER_FIRE=${SGLANG_BUDGETER_PAGES_PER_FIRE:-8}
# cooldown/amortize in WALL SECONDS (#302 τ-invariance): old COOLDOWN=6
# ticks at TICK_S=2 = 12 s. cooldown_min_s == amortize_horizon_s here.
export SGLANG_XPOOL_COOLDOWN_S=${SGLANG_XPOOL_COOLDOWN_S:-12}
export SGLANG_XPOOL_NB_MARGIN=${SGLANG_XPOOL_NB_MARGIN:-1.2}
export EVICTION_POLICY=lpb

# N_SESSIONS small enough that BOTH cells fully complete all N within
# MAX_TIME_MIN (else the cells finish different fractions and the rep is
# excluded by validate_zero_downside.py's completion-parity guard). 40
# sessions did NOT complete in 30 min on this node, so use 10 with a generous
# cap.
N_SESSIONS=${N_SESSIONS:-10}
N_REPS=${N_REPS:-3}
MAX_TIME_MIN=${MAX_TIME_MIN:-45}
GPU=${GPU:-0}
PORT=${PORT:-30090}
TAG=${TAG:-d290_zd}
# Shrink MAX_MAMBA_CACHE (e.g. 64) to make the mamba pool the binding
# constraint, so the Budgeter actually fires under pressure (the mamba-bound /
# #299 / #285 case). Default 256 = the mamba-slack regime.
MAX_MAMBA_CACHE=${MAX_MAMBA_CACHE:-256}
NUM_CONCURRENCY=${NUM_CONCURRENCY:-22}

dirs=()
for rep in $(seq 1 "$N_REPS"); do
    OUT=/tmp/${TAG}_rep${rep}
    dirs+=("$OUT")
    echo "=== [zero_downside] rep $rep/$N_REPS -> $OUT (N_SESSIONS=$N_SESSIONS) ==="
    GPU="$GPU" PORT="$PORT" OUT_DIR="$OUT" \
        NUM_CONCURRENCY="$NUM_CONCURRENCY" MAX_MAMBA_CACHE="$MAX_MAMBA_CACHE" \
        MAX_SESSIONS="$N_SESSIONS" MAX_TIME_MIN="$MAX_TIME_MIN" \
        bash /scratch/yuzhou/projects/sglang/dev/interlayer/4_e2e/cc_traces_headline/run_cc.sh \
        > "$OUT.log" 2>&1 || echo "[zero_downside] rep $rep rc=$?"
done

echo
echo "=== zero-downside validation (N=$N_REPS paired, request-bounded) ==="
"$VENV" /scratch/yuzhou/projects/sglang/dev/interlayer/4_e2e/cc_zero_downside/validate_zero_downside.py \
    --out-dirs "${dirs[@]}"

echo
echo "=== harvest structured result into repo (dev/interlayer/4_e2e/results/) ==="
# Self-contained JSON with per-rep off/inter cached/prompt tokens + cache_hit
# (summed over ALL hourly metrics logs), tps, ttft, fires, parity, deltas +
# median — so results are inspectable in-repo and never re-parsed from raw logs.
"$VENV" /scratch/yuzhou/projects/sglang/dev/interlayer/4_e2e/results/harvest_run.py \
    --out-dir "/tmp/$TAG" --tag "$TAG" \
    --meta "{\"MAX_MAMBA_CACHE\":$MAX_MAMBA_CACHE,\"N_SESSIONS\":$N_SESSIONS,\"N_REPS\":$N_REPS,\"NUM_CONCURRENCY\":$NUM_CONCURRENCY,\"eviction\":\"${EVICTION_POLICY:-lpb}\",\"queue_wait_us\":\"${SGLANG_XPOOL_QUEUE_WAIT_US:-125000}\"}" \
    || echo "[zero_downside] harvest rc=$?"
