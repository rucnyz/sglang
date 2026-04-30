#!/bin/bash
# B2 — cold-burst (build → burst → recovery). Q3.B-style.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

# Three back-to-back benches: GSP build, random burst, GSP recovery.
echo "[suite-runner] Phase 1 build (GSP)"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 8 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 2 \
  --output-file "$OUT_DIR/build_bench.json" \
  >"$OUT_DIR/build.log" 2>&1 || echo "build phase failed"

echo "[suite-runner] Phase 2 burst (random 4K)"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 200 \
  --random-input-len 4096 --random-output-len 64 \
  --request-rate 8 \
  --output-file "$OUT_DIR/burst_bench.json" \
  >"$OUT_DIR/burst.log" 2>&1 || echo "burst phase failed"

echo "[suite-runner] Phase 3 recovery (GSP)"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 8 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 2 \
  --output-file "$OUT_DIR/recovery_bench.json" \
  >"$OUT_DIR/recovery.log" 2>&1 || echo "recovery phase failed"

teardown_server

# Emit metrics from the recovery phase (the headline metric in Q3.B).
emit_metrics_from_bench "$OUT_DIR/recovery_bench.json" "$METRICS_PATH"
echo "[suite-runner] B2 metrics: $(cat "$METRICS_PATH")"
