#!/bin/bash
# R3 — LoRA-skewed (Qwen3-4B + 32 LoRA, ml=8). Non-mamba model.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Set MODEL and disable mamba-only flags BEFORE sourcing _common.sh.
# _common.sh defaults MODEL to Qwen3.5-35B-A3B with ${MODEL:-...}; if we
# don't set it first, our override later is a no-op.
export MODEL=${MODEL:-Qwen/Qwen3-4B}
export MAMBA_FLAGS=""
# Disable arena+budgeter on the LoRA workload: (a) Qwen3-4B has 36 layers × 2
# kinds = 72 KV subpools, exceeding arena_multi64.so's hardcoded 64-subpool limit;
# (b) no mamba pool exists, so cross-pool transfer is N/A. We still want this
# job in the prelude arm to verify L2 stays silent on non-mamba workloads.
export SGLANG_ARENA_SHARED=0
export SGLANG_ARENA_FROM_BLOB=0
export SGLANG_BUDGETER_XPOOL_PLANNER=0
export SGLANG_BUDGETER_XPOOL_COORDINATED=0
source "$SCRIPT_DIR/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"

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
