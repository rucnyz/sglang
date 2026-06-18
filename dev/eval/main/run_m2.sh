#!/bin/bash
# M2 swarm regime (paper §sec:eval-main-swarm).
# 800 sub-agents × 2-5 turns × ≤256 in/out, 3K cap, 480-600s wall.
# Recurrent-bound. Uses genai-bench (Locust) — text-to-text-multi-turn
# task with D(256,256) traffic scenario.
#
# Required env: MODEL TP GPU_LIST INTRA INTER PORT OUT_DIR
# Optional:     MEM_FRAC NUM_CONCURRENCY TRAFFIC_SCENARIO
#               SESSION_CAP MAX_TIME_MIN

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
TRAFFIC_SCENARIO=${TRAFFIC_SCENARIO:-D(256,256)}
SESSION_CAP=${SESSION_CAP:-5000}
MAX_TIME_MIN=${MAX_TIME_MIN:-10}    # paper says 480s; give slack to 10min

apply_cell_env
boot_sglang || { teardown_sglang; exit 1; }

cell=$(cell_label)
echo "[$cell] M2 swarm (genai-bench): conc=$NUM_CONCURRENCY scenario=$TRAFFIC_SCENARIO cap=$SESSION_CAP time=${MAX_TIME_MIN}min"

GENAI_BENCH_MT_SESSION_CAP_TOKENS=$SESSION_CAP \
    .venv/bin/python -m genai_bench.cli.cli benchmark \
    --api-backend sglang \
    --api-base "http://127.0.0.1:$PORT" \
    --api-key dummy \
    --api-model-name "$MODEL" \
    --model-tokenizer "$MODEL" \
    --task text-to-text-multi-turn \
    --traffic-scenario "$TRAFFIC_SCENARIO" \
    --num-concurrency $NUM_CONCURRENCY \
    --max-time-per-run $MAX_TIME_MIN \
    --max-requests-per-run 1000000 \
    --experiment-folder-name "$OUT_DIR/genai_results" \
    --server-engine SGLang \
    > "$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed (see client.log)"

SUMMARY=$(find "$OUT_DIR/genai_results" -maxdepth 2 -name "D*_text-to-text-multi-turn_*.json" 2>/dev/null | head -1)
if [ -n "$SUMMARY" ]; then
    .venv/bin/python -c "
import json, sys
d = json.load(open('$SUMMARY'))
m = d.get('aggregated_metrics', d)
s = m.get('stats', {})
ttft = s.get('ttft', {})
e2e = s.get('e2e_latency', {})
out = {
    'wall_s': m.get('run_duration', 0),
    'num_concurrency': m.get('num_concurrency'),
    'num_requests_total': m.get('num_requests'),
    'num_requests_valid': m.get('num_completed_requests'),
    'num_errors': m.get('num_error_requests', 0),
    'mean_ttft_ms': ttft.get('mean', 0) * 1000,
    'p50_ttft_ms': ttft.get('p50', 0) * 1000,
    'p99_ttft_ms': ttft.get('p99', 0) * 1000,
    'mean_e2e_ms': e2e.get('mean', 0) * 1000,
    'p50_e2e_ms': e2e.get('p50', 0) * 1000,
    'p99_e2e_ms': e2e.get('p99', 0) * 1000,
    'output_tps': m.get('mean_output_throughput_tokens_per_s', 0),
    'input_tps': m.get('mean_input_throughput_tokens_per_s', 0),
    'requests_per_second': m.get('requests_per_second', 0),
    'error_rate': m.get('error_rate', 0),
}
json.dump(out, open('$OUT_DIR/bench.json', 'w'), indent=2)
print('normalized', out['num_requests_valid'], 'reqs')
" >> "$OUT_DIR/client.log" 2>&1
fi

emit_xpool_summary
teardown_sglang
echo "[$cell] M2 done -> $OUT_DIR"
