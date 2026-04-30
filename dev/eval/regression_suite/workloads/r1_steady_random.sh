#!/bin/bash
# R1 — steady-state random workload (regression test).
# Layer 2 should NOT touch this; pass = TPS in [95%, 105%] of baseline.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

# Layer-2 budgeter log goes here so the metrics emitter can count transfers.
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"

EXTRA_FLAGS="--mamba-full-memory-ratio 0.5"
boot_server

BENCH_JSON="$OUT_DIR/bench.json"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 800 \
  --random-input-len 1024 --random-output-len 256 \
  --request-rate 32 \
  --output-file "$BENCH_JSON" \
  >"$OUT_DIR/bench.log" 2>&1 || echo "[suite-runner] bench failed (will still emit metrics)"

teardown_server
emit_metrics_from_bench "$BENCH_JSON" "$METRICS_PATH"
echo "[suite-runner] R1 metrics: $(cat "$METRICS_PATH")"
