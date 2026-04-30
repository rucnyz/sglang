#!/bin/bash
# B3 — long multi-turn workload using sglang's built-in --gsp-num-turns.
#
# True multi-turn session benchmark: each session shares a 12K-token system
# prefix; per-turn the user appends a question and the assistant replies,
# accumulating the full conversation history into the next turn's context.
# 16 sessions concurrent × 8 turns each = 128 total requests, with per-
# session context growing from ~13K (turn 1) to ~24K (turn 8) tokens.
#
# This is the regime the paper's Layer 1 (HPB-LRU prefix retention across
# rounds) and Layer 2 (cross-pool reallocation as KV grows monotonically
# through the trace) both target. Replaces the prior random α/β concurrency-
# shift dispatcher (which had no shared prefix and no multi-turn semantics).
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

# Built-in multi-turn via sglang.bench_serving + gsp-num-turns.
# - 16 sessions (gsp-num-groups=16, prompts-per-group=1)
# - 12K-token shared system prefix per session
# - 8 turns per session, 1K user question + 1K assistant reply per turn
# - Backend MUST be one of {sglang-oai-chat, vllm-chat, lmdeploy-chat}
#   for multi-turn message accumulation; we use sglang-oai-chat.
.venv/bin/python -m sglang.bench_serving \
    --backend sglang-oai-chat --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 16 --gsp-prompts-per-group 1 \
    --gsp-system-prompt-len 12000 \
    --gsp-question-len 1024 \
    --gsp-output-len 1024 \
    --gsp-num-turns 8 \
    --request-rate 4 \
    --output-file "$OUT_DIR/bench.json" \
    > "$OUT_DIR/bench.log" 2>&1 || echo "[suite-runner] B3 bench failed"

teardown_server

emit_metrics_from_bench "$OUT_DIR/bench.json" "$METRICS_PATH"
echo "[suite-runner] B3 metrics: $(cat "$METRICS_PATH")"
