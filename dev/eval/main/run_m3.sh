#!/bin/bash
# M3 phase-shift trace (paper §sec:eval-headline-v9).
# Phase A: GSP shared-prefix, RPS=8, recurrent-bound (16x10 prompts).
# Phase B: random 8K prompts, RPS=4, KV-bound.
# Phase C: random 4K prompts, RPS=8, mixed.
# Phases concatenated back-to-back; each phase ~PHASE_DURATION_S seconds.
#
# Required env: MODEL TP GPU_LIST INTRA INTER PORT OUT_DIR
# Optional:     PHASE_DURATION_S (default 160 → ~480s total per trial)
#               GSP_GROUPS GSP_PROMPTS_PER_GROUP

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

source dev/eval/main/_common.sh
require_env MODEL; require_env TP; require_env GPU_LIST
require_env INTRA; require_env INTER; require_env PORT; require_env OUT_DIR
mkdir -p "$OUT_DIR"

PHASE_DURATION_S=${PHASE_DURATION_S:-160}
GSP_GROUPS=${GSP_GROUPS:-16}
GSP_PROMPTS_PER_GROUP=${GSP_PROMPTS_PER_GROUP:-10}

apply_cell_env
boot_sglang || { teardown_sglang; exit 1; }

cell=$(cell_label)
echo "[$cell] M3 phase-shift: 3 phases × ${PHASE_DURATION_S}s each"

run_phase() {
    local phase=$1; shift
    local out="$OUT_DIR/phase${phase}_bench.json"
    local lg="$OUT_DIR/phase${phase}_bench.log"
    echo "[$cell] phase $phase running..."
    .venv/bin/python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port "$PORT" \
        --model "$MODEL" --tokenizer "$MODEL" \
        --output-file "$out" \
        "$@" \
        >"$lg" 2>&1 || echo "[$cell] phase $phase bench failed (see $lg)"
}

# Phase A: GSP shared-prefix, RPS=8, recurrent-bound (system prompt builds
# many short snapshots → mamba pool fills; KV stays light).
run_phase A \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$GSP_GROUPS" --gsp-prompts-per-group "$GSP_PROMPTS_PER_GROUP" \
    --gsp-system-prompt-len 6000 --gsp-question-len 64 --gsp-output-len 256 \
    --request-rate 8 \
    --num-prompts $((GSP_GROUPS * GSP_PROMPTS_PER_GROUP * PHASE_DURATION_S * 8 / 480))

sleep 5

# Phase B: random 8K prompts, RPS=4, KV-bound (long prefill fills KV pool).
run_phase B \
    --dataset-name random --random-input-len 8000 --random-output-len 64 \
    --request-rate 4 \
    --num-prompts $((PHASE_DURATION_S * 4))

sleep 5

# Phase C: random 4K prompts, RPS=8, mixed (cooldown / steady-state).
run_phase C \
    --dataset-name random --random-input-len 4000 --random-output-len 64 \
    --request-rate 8 \
    --num-prompts $((PHASE_DURATION_S * 8))

# Phase C bench is the headline metric (paper Table tab:headline-v9 highlights
# Phase-C P99). Symlink it as bench.json so aggregate.py picks it up.
[ -f "$OUT_DIR/phaseC_bench.json" ] && cp "$OUT_DIR/phaseC_bench.json" "$OUT_DIR/bench.json"

emit_xpool_summary
teardown_sglang
echo "[$cell] M3 done -> $OUT_DIR"
