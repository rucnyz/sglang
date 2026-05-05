#!/bin/bash
# M2 swarm regime (paper §sec:eval-main-swarm).
# 800 sub-agents × 2-5 turns × ≤256 in/out, 3K cap, 480s wall. Recurrent-bound.
#
# Required env: MODEL TP GPU_LIST INTRA INTER PORT OUT_DIR
# Optional:     MEM_FRAC NUM_CONCURRENCY TURN_INPUT TURN_OUTPUT
#               SESSION_CAP MAX_TIME_S STAGGER_S MEASURE_AFTER_S

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

source dev/eval/main/_common.sh
require_env MODEL; require_env TP; require_env GPU_LIST
require_env INTRA; require_env INTER; require_env PORT; require_env OUT_DIR
mkdir -p "$OUT_DIR"

# Workload knobs (paper §sec:eval-main-swarm defaults)
NUM_CONCURRENCY=${NUM_CONCURRENCY:-800}
TURN_INPUT=${TURN_INPUT:-256}
TURN_OUTPUT=${TURN_OUTPUT:-256}
SESSION_CAP=${SESSION_CAP:-3000}
MAX_TIME_S=${MAX_TIME_S:-480}
STAGGER_S=${STAGGER_S:-0.05}     # 50ms stagger × 800 = 40s ramp-up
MEASURE_AFTER_S=${MEASURE_AFTER_S:-60.0}

apply_cell_env
boot_sglang || { teardown_sglang; exit 1; }

cell=$(cell_label)
echo "[$cell] M2 swarm: conc=$NUM_CONCURRENCY input=$TURN_INPUT output=$TURN_OUTPUT cap=$SESSION_CAP time=${MAX_TIME_S}s"

.venv/bin/python dev/eval/multiturn_client.py \
    --api-base "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --num-concurrency "$NUM_CONCURRENCY" \
    --turn-input-tokens "$TURN_INPUT" \
    --turn-output-tokens "$TURN_OUTPUT" \
    --session-cap-tokens "$SESSION_CAP" \
    --max-time-s "$MAX_TIME_S" \
    --stagger-s "$STAGGER_S" \
    --measure-after-s "$MEASURE_AFTER_S" \
    --output-dir "$OUT_DIR" \
    > "$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed (see client.log)"

[ -f "$OUT_DIR/multiturn_summary.json" ] && cp "$OUT_DIR/multiturn_summary.json" "$OUT_DIR/bench.json"

emit_xpool_summary
teardown_sglang
echo "[$cell] M2 done -> $OUT_DIR"
