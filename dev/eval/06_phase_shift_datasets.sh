#!/bin/bash
# Setting 1 (24-h phase-shift trace) — dataset preparation step.
#
# Generates three JSONL datasets (Phase A/B/C) using the pd_exp tooling,
# saved to dev/eval/datasets/. Run this ONCE before the headline trace
# script.
#
# Phase A: classification + multi-LoRA + shared system prompt — alpaca-source,
#          input ~512, output ~16, 1500 prompts.
# Phase B: short-form rerank — sharegpt-source, input ~512, output ~8, 4000 prompts.
# Phase C: long-context multi-turn chat — wildchat, 200 conversations × 8 turns,
#          input grows turn-by-turn from 1K to 16K, output ~256-2K.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/aproj/vllm/pd_exp:/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

DATASET_DIR=${DATASET_DIR:-/scratch/yuzhou/projects/sglang/dev/eval/datasets}
mkdir -p "$DATASET_DIR"
MODEL_TOK=${MODEL_TOK:-Qwen/Qwen3.5-35B-A3B}

# Phase A — classification (alpaca, short).
echo "=== generating Phase A ==="
.venv/bin/python /data/yuzhou/projects/aproj/vllm/pd_exp/serve/generate_distribution_shift_dataset.py \
  --model "$MODEL_TOK" \
  --num-prompts-per-phase 1500 \
  --phases "512:16" \
  --variance 0.1 \
  --source-dataset alpaca \
  --output "$DATASET_DIR/phase_a.jsonl" \
  --seed 7 || { echo "Phase A FAIL"; exit 1; }
echo "Phase A done: $DATASET_DIR/phase_a.jsonl"

# Phase B — short-form rerank (sharegpt, K∈[4,16]).
echo "=== generating Phase B ==="
.venv/bin/python /data/yuzhou/projects/aproj/vllm/pd_exp/serve/generate_distribution_shift_dataset.py \
  --model "$MODEL_TOK" \
  --num-prompts-per-phase 4000 \
  --phases "512:8" \
  --variance 0.5 \
  --source-dataset sharegpt \
  --output "$DATASET_DIR/phase_b.jsonl" \
  --seed 17 || { echo "Phase B FAIL"; exit 1; }
echo "Phase B done: $DATASET_DIR/phase_b.jsonl"

# Phase C — wildchat multi-turn.
echo "=== generating Phase C ==="
.venv/bin/python /data/yuzhou/projects/aproj/vllm/pd_exp/multiturn/export_dataset.py \
  --dataset wildchat \
  --model "$MODEL_TOK" \
  --num-conversations 200 \
  --min-turns 8 \
  --output "$DATASET_DIR/phase_c.json" \
  || { echo "Phase C FAIL (likely needs HF download for wildchat)"; }
echo "Phase C done (or partial): $DATASET_DIR/phase_c.json"

echo
echo "=== datasets prepared ==="
ls -la "$DATASET_DIR/"
