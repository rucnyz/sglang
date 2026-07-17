#!/usr/bin/env bash
# Real-cc-trace concurrency sweep (NO artificial mamba starve — realistic
# mamba 256). Two goals:
#   (1) no-regression: at the known-neutral point (concurrency 14) on must
#       not regress off on any metric.
#   (2) find a win regime: as concurrency rises, more distinct sessions
#       compete for the mamba cache; if the hot working set exceeds the
#       pool, off cache_hit drops (mamba binds) and the mechanism should
#       recover it. Reports off vs on cache_hit + pool usage per point.
# First pass is time-bounded (fast regime-finding); the chosen point gets
# a request-bounded N=3 confirm afterwards.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

BASE=${BASE:-/tmp/d10_conc}
CONCS=(${CONCS:-14 64 128})
MAMBA=${MAMBA:-256}
TMIN=${TMIN:-10}

pids=(); i=0
for c in "${CONCS[@]}"; do
    gpu=$((4+i)); port=$((30110+i)); i=$((i+1))
    NUM_CONCURRENCY=$c MAX_MAMBA_CACHE=$MAMBA MAX_TIME_MIN=$TMIN \
        OUT_DIR="$BASE/c$c" GPU=$gpu PORT=$port \
        bash dev/interlayer/4_e2e/cc_traces_headline/run_cc.sh \
        > "$BASE.c$c.log" 2>&1 &
    pids+=($!)
    echo "[conc-sweep] launched concurrency=$c on GPU $gpu port $port"
    sleep 5
done
for p in "${pids[@]}"; do wait $p || true; done

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
echo
echo "=== concurrency sweep: per-point off-vs-on (cache_hit + usage) ==="
for c in "${CONCS[@]}"; do
    echo "--- concurrency=$c ---"
    $VENV dev/interlayer/4_e2e/cc_traces_headline/validate_cc.py \
        --out-dir "$BASE/c$c" 2>&1 | grep -E "off:|inter:|cache_hit|mean_ttft|out_tps|WIN|PASS|FAIL" | head -12
done
echo "[conc-sweep] done"
