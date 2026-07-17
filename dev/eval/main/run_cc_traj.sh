#!/bin/bash
#
# run_cc_traj.sh — Claude Code trajectory replay regime.
#
# Boots an SGLang server (using the same _common.sh boot path as run_m1/m2/m3)
# and drives it with cc_trace_replay.py against the extracted >=100K-token
# Claude Code traces in dev/eval/datasets/cc_long_traces.jsonl. Output bench.json
# is genai-bench-compatible so aggregate.py and Table 1's real-workload row
# can consume it without changes.
#
# Required env: MODEL TP GPU_LIST INTRA INTER PORT OUT_DIR
# Optional:     MEM_FRAC NUM_CONCURRENCY MAX_TIME_MIN MAX_TOKENS
#               TRACES_FILE MIN_TURNS MIN_CHARS

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

source dev/eval/main/_common.sh
require_env MODEL; require_env TP; require_env GPU_LIST
require_env INTRA; require_env INTER; require_env PORT; require_env OUT_DIR
mkdir -p "$OUT_DIR"

NUM_CONCURRENCY=${NUM_CONCURRENCY:-14}
MAX_TIME_MIN=${MAX_TIME_MIN:-10}
MAX_TOKENS=${MAX_TOKENS:-1024}
TRACES_FILE=${TRACES_FILE:-dev/eval/datasets/cc_long_traces.jsonl}
MIN_TURNS=${MIN_TURNS:-15}
MIN_CHARS=${MIN_CHARS:-30000}    # filter on user-only chars; full ctx (user+asst growth) is much larger

apply_cell_env
boot_sglang || { teardown_sglang; exit 1; }

cell=$(cell_label)
echo "[$cell] CC-traj replay: conc=$NUM_CONCURRENCY traces=$TRACES_FILE max_time=${MAX_TIME_MIN}min max_tokens=$MAX_TOKENS"

.venv/bin/python dev/eval/main/cc_trace_replay.py \
    --api-base "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --traces "$TRACES_FILE" \
    --num-concurrency "$NUM_CONCURRENCY" \
    --max-time-min "$MAX_TIME_MIN" \
    --max-tokens "$MAX_TOKENS" \
    --min-turns "$MIN_TURNS" \
    --min-chars "$MIN_CHARS" \
    --output-file "$OUT_DIR/bench.json" \
    > "$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed (see client.log)"

emit_xpool_summary
teardown_sglang
echo "[$cell] CC-traj done -> $OUT_DIR"
