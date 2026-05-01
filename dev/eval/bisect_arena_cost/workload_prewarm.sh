#!/bin/bash
# Pre-warm experiment (a): touch all KV/mamba pages via a 200-prompt
# warmup bench BEFORE the timed n=500 bench. Hypothesis: GPU TLB on the
# 25 GiB cuMemMap range with 2 MiB pages is the variance source — if so,
# pre-warming should make C1 trial-to-trial variance collapse toward the
# C0 baseline (rock-solid σ=1.70 ms on mean TTFT).
#
# Caller (run_prewarm.sh) sets PORT, OUT_DIR, METRICS_PATH, MEM_FRACTION,
# CUDA_VISIBLE_DEVICES + the C0/C1 env profile.

set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../regression_suite/workloads/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

# Pre-warmup: 200 prompts with 4K input length to touch a wide range of
# KV pages. Larger inputs than the timed bench so we exercise more of
# the pool. RPS=8 same as timed bench so saturation pattern matches.
echo "[prewarm] WARMUP bench (200 prompts, 4096in/64out, RPS=8) — discarded"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 200 \
  --random-input-len 4096 --random-output-len 64 \
  --request-rate 8 \
  --output-file "$OUT_DIR/warmup_bench.json" \
  > "$OUT_DIR/warmup.log" 2>&1 || echo "[prewarm] warmup bench failed"

# Brief settle so warmup requests fully drain. TLB entries persist
# across the freed-pages window because no other process competes for
# them on this idle GPU.
sleep 5

echo "[prewarm] TIMED bench (500 prompts, 512in/128out, RPS=8)"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 500 \
  --random-input-len 512 --random-output-len 128 \
  --request-rate 8 \
  --output-file "$OUT_DIR/bench.json" \
  > "$OUT_DIR/bench.log" 2>&1 || echo "[prewarm] timed bench failed"

teardown_server
emit_metrics_from_bench "$OUT_DIR/bench.json" "$METRICS_PATH"
echo "[prewarm] metrics: $(cat $METRICS_PATH)"
