#!/bin/bash
# Closed-loop concurrency=8 experiment (d): replace --request-rate 8 with
# --max-concurrency 8 to remove Poisson arrival noise. If P99 spread
# narrows AND C1 mean variance stays similar to RPS-mode, then Poisson
# arrivals were the dominant variance source. If C1 mean variance is
# still 3× C0, the arena layer's per-request cost itself is bursty.

set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../regression_suite/workloads/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

echo "[concur] timed bench (500 prompts, 512in/128out, max-concurrency=8)"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 500 \
  --random-input-len 512 --random-output-len 128 \
  --max-concurrency 8 \
  --output-file "$OUT_DIR/bench.json" \
  > "$OUT_DIR/bench.log" 2>&1 || echo "[concur] bench failed"

teardown_server
emit_metrics_from_bench "$OUT_DIR/bench.json" "$METRICS_PATH"
echo "[concur] metrics: $(cat $METRICS_PATH)"
