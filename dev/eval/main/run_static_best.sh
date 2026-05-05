#!/bin/bash
# SGLang static-best baseline (paper §sec:eval-static-best, tab:static-best).
# Sweeps --mamba-full-memory-ratio across {0.3, 0.5, 0.7, 0.9} on a target
# regime; no L_intra, no L_inter. Per-phase optimum is the static-best
# baseline column in tab:main-cross-model.
#
# Required env: MODEL TP GPU_LIST REGIME (m1|m2|m3) PORT OUT_DIR
# Optional:     RATIOS ("0.3 0.5 0.7 0.9")

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

source dev/eval/main/_common.sh
require_env MODEL; require_env TP; require_env GPU_LIST
require_env REGIME; require_env PORT; require_env OUT_DIR

RATIOS=${RATIOS:-"0.3 0.5 0.7 0.9"}
mkdir -p "$OUT_DIR"

# Force stock cell (no L_intra, no L_inter). Export so the inner runner sees them.
export INTRA=0
export INTER=0

for ratio in $RATIOS; do
    sub_out="$OUT_DIR/static_best_r${ratio}"
    mkdir -p "$sub_out"
    echo "[static-best] regime=$REGIME ratio=$ratio out=$sub_out"

    OUT_DIR="$sub_out" \
    EXTRA_LAUNCH_FLAGS="--mamba-full-memory-ratio $ratio" \
    CELL_LABEL_OVERRIDE="static_best_r${ratio}" \
        bash dev/eval/main/run_${REGIME}.sh
done

echo "[static-best] sweep done over ratios: $RATIOS"
