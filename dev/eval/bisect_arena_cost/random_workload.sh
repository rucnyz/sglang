#!/bin/bash
# Random-prefill bisection workload — matches dev/2e/24_arena_from_blob_perf.sh
# (the original measurement that produced the 5.86%/12.34% structural-cost
# headline in paper §sec:eval-arena-cost). 100 prompts, random 512 input /
# 128 output, request-rate 8, MEM_FRACTION=0.8 — prefill-pressured workload
# without a shared prefix, which is the regime where the cost was visible.
#
# Caller (run_random.sh) sets PORT, OUT_DIR, METRICS_PATH, MEM_FRACTION,
# CUDA_VISIBLE_DEVICES + the four C0..C3 env profiles.

set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../regression_suite/workloads/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

echo "[bisect] random-prefill bench (512in/128out, RPS=8, 100 prompts)"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 100 \
  --random-input-len 512 --random-output-len 128 \
  --request-rate 8 \
  --output-file "$OUT_DIR/bench.json" \
  > "$OUT_DIR/bench.log" 2>&1 || echo "[bisect] bench failed"

teardown_server

emit_metrics_from_bench "$OUT_DIR/bench.json" "$METRICS_PATH"
echo "[bisect] random metrics: $(cat $METRICS_PATH)"
