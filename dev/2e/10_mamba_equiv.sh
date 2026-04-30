#!/bin/bash
# Phase 2e.5.3 — E2E equivalence: SGLANG_MAMBA_PERLAYER=0 vs =1.
#
# Boots two SGLang servers in sequence (so they don't share GPU), sends the
# same set of prompts at temperature=0, captures completions, compares
# token-by-token. Pass criterion: identical token sequences for every
# prompt across the two arms.
#
# Hybrid model: Qwen3.5-35B-A3B (cached). DeltaNet hybrid -> exercises mamba pool.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-900}    # large hybrid-model first-launch can take 10+ min
OUT_DIR=/tmp/mamba_equiv_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR  warmup_max=${WARMUP_S}s"

PROMPTS=(
  "The capital of France is"
  "Once upon a time"
  "Q: 2 + 2 ="
  "def fibonacci(n):"
)
MAX_TOKENS=20

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "perlayer" ]; then
    extra_env="SGLANG_MAMBA_PERLAYER=1"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  echo "=== arm=$arm ==="
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 3
  nohup env $extra_env .venv/bin/python -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.85 --log-level info \
    >"$log" 2>&1 &
  local pid=$!
  echo "server pid=$pid log=$log; waiting up to ${WARMUP_S}s for ready"
  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q .; then
      echo "ready after ${waited}s"
      break
    fi
  done
  if ! curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q .; then
    echo "[$arm] server failed to start (waited ${WARMUP_S}s); tail of log:"
    tail -20 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi
  local out="$OUT_DIR/${arm}_completions.txt"
  : >"$out"
  for prompt in "${PROMPTS[@]}"; do
    .venv/bin/python -c "
import sys, json, urllib.request
data = json.dumps({
    'model': '$MODEL', 'prompt': '''$prompt''',
    'max_tokens': $MAX_TOKENS, 'temperature': 0,
}).encode()
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
    data=data, headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req, timeout=60)
print(json.loads(r.read())['choices'][0]['text'])
" >>"$out" 2>&1
  done
  kill -9 $pid 2>/dev/null || true
  sleep 5
}

run_arm stacked
run_arm perlayer

echo "=== diff ==="
if diff -q "$OUT_DIR/stacked_completions.txt" "$OUT_DIR/perlayer_completions.txt" >/dev/null; then
  echo "PASS: completions are identical across arms"
  echo "--- example output (first 5 lines) ---"
  head -5 "$OUT_DIR/stacked_completions.txt"
else
  echo "FAIL: completions differ"
  diff "$OUT_DIR/stacked_completions.txt" "$OUT_DIR/perlayer_completions.txt"
  exit 1
fi
