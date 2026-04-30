#!/bin/bash
# n=500 random-prefill workload.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../regression_suite/workloads/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

echo "[bisect] random-prefill bench n=500 (512in/128out, RPS=8)"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 500 \
  --random-input-len 512 --random-output-len 128 \
  --request-rate 8 \
  --output-file "$OUT_DIR/bench.json" \
  > "$OUT_DIR/bench.log" 2>&1 || echo "[bisect] bench failed"

teardown_server
emit_metrics_from_bench "$OUT_DIR/bench.json" "$METRICS_PATH"
echo "[bisect] metrics: $(cat $METRICS_PATH)"
