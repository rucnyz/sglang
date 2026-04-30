#!/bin/bash
# R3 — LoRA-skewed (Qwen3-4B + 32 LoRA, ml=8). Non-mamba model.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Disable Qwen3.5-mamba-only flags BEFORE sourcing _common.sh — they break
# the LoRA Triton dispatch on Qwen3-4B (assert x.shape[-1] == K).
export MAMBA_FLAGS=""
source "$SCRIPT_DIR/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"

# Override model to Qwen3-4B for LoRA serving.
MODEL=${MODEL:-Qwen/Qwen3-4B}
LORA_DIR=${LORA_DIR:-/scratch/yuzhou/.cache/synthetic_loras/qwen3-4b-r16}

LORA_PATHS=""
for i in $(seq 0 31); do
  LORA_PATHS="$LORA_PATHS lora_${i}=${LORA_DIR}/lora_${i}"
done

EXTRA_FLAGS="--enable-lora --max-loras-per-batch 8 --max-lora-rank 16 --lora-paths $LORA_PATHS"
boot_server

LORA_NAMES=""
for i in $(seq 0 31); do
  LORA_NAMES="$LORA_NAMES lora_${i}"
done

BENCH_JSON="$OUT_DIR/bench.json"
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 600 \
  --random-input-len 512 --random-output-len 128 \
  --request-rate 16 \
  --lora-name $LORA_NAMES \
  --output-file "$BENCH_JSON" \
  >"$OUT_DIR/bench.log" 2>&1 || echo "[suite-runner] bench failed"

teardown_server
emit_metrics_from_bench "$BENCH_JSON" "$METRICS_PATH"
echo "[suite-runner] R3 metrics: $(cat "$METRICS_PATH")"
