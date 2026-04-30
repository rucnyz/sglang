#!/bin/bash
# R2 — steady-state GSP shared-prefix (long, ~4 min).
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"

boot_server

BENCH_JSON="$OUT_DIR/bench.json"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 16 --gsp-prompts-per-group 30 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 4 \
  --output-file "$BENCH_JSON" \
  >"$OUT_DIR/bench.log" 2>&1 || echo "[suite-runner] bench failed"

teardown_server
emit_metrics_from_bench "$BENCH_JSON" "$METRICS_PATH"
echo "[suite-runner] R2 metrics: $(cat "$METRICS_PATH")"
