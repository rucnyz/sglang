#!/bin/bash
# Phase 2e.5.6.3 verification — capacity-coordinated cross-pool transfer
# preserves byte-identical engine output.
#
# Same shape as 16_kv_mamba_xfer_equiv.sh but arm B has
# SGLANG_BUDGETER_XPOOL_COORDINATED=1 set, which causes
# CrossPoolTransferActuator to call into KVArenaActuator and
# MambaArenaActuator before/after each chunk move so the engine's
# allocators actually learn about the capacity changes.
#
# Pass criterion (all of):
#   1. arm B logs "coordinated=True (kv_act=True, mamba_act=True)"
#   2. each transfer log line is followed by KVArenaActuator and
#      MambaPool.set_capacity_slots updates (allocator caps moved)
#   3. arm B's completions are byte-identical to arm A baseline
#   4. no segfault / no leak detection trip during serving

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-600}
OUT_DIR=/tmp/kv_mamba_xfer_coord_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR  GPU=$CUDA_VISIBLE_DEVICES  warmup_max=${WARMUP_S}s"

PROMPTS=(
  "The capital of France is"
  "Once upon a time"
  "Q: 2 + 2 ="
  "def fibonacci(n):"
  "List three primes:"
)
MAX_TOKENS=24

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "shared_xpool_coord" ]; then
    extra_env="SGLANG_ARENA_SHARED=1 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_DEMO=1 SGLANG_BUDGETER_XPOOL_UNIT=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0 SGLANG_BUDGETER_LOG=$OUT_DIR/${arm}_budgeter.jsonl"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  local out="$OUT_DIR/${arm}_completions.txt"
  : >"$out"

  echo "=== arm=$arm ==="
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 5

  nohup env $extra_env .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
  local pid=$!
  echo "[$arm] server pid=$pid log=$log; waiting ${WARMUP_S}s for ready"
  echo "--- live server log ---"
  tail -F "$log" 2>/dev/null &
  local tailer=$!

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      kill $tailer 2>/dev/null || true
      wait $tailer 2>/dev/null || true
      echo
      echo "--- ready after ${waited}s ---"
      break
    fi
  done
  if kill -0 $tailer 2>/dev/null; then
    kill $tailer 2>/dev/null || true
    wait $tailer 2>/dev/null || true
  fi

  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
    echo "[$arm] server failed to become ready in ${WARMUP_S}s"
    tail -30 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  local i=0
  for prompt in "${PROMPTS[@]}"; do
    i=$((i + 1))
    .venv/bin/python -c "
import json, urllib.request
data = json.dumps({
    'model': '$MODEL', 'prompt': '''$prompt''',
    'max_tokens': $MAX_TOKENS, 'temperature': 0,
}).encode()
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
    data=data, headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req, timeout=300)
print(json.loads(r.read())['choices'][0]['text'])
" >>"$out" 2>&1
    sleep 3
  done

  echo "[$arm] completions captured to $out"
  if [ "$arm" = "shared_xpool_coord" ]; then
    # Verify coordinated mode actually engaged.
    local coord_line=$(grep "coordinated=True" "$log" | head -1 || true)
    if [ -z "$coord_line" ]; then
      echo "FAIL: arm B did not log coordinated=True (kv_act + mamba_act)"
      kill -9 $pid 2>/dev/null || true
      return 1
    fi
    echo "[$arm] coordination engaged: $coord_line"

    # Verify coordinator actually moved allocator caps (KV side).
    local kv_cap_lines=$(grep -c "Allocator.set_capacity_pages:" "$log" || true)
    local mamba_cap_lines=$(grep -c "MambaPool.set_capacity_slots:" "$log" || true)
    echo "[$arm] capacity-update events: KV=$kv_cap_lines, mamba=$mamba_cap_lines"
    if [ "${kv_cap_lines:-0}" -lt 1 ] || [ "${mamba_cap_lines:-0}" -lt 1 ]; then
      echo "FAIL: coordinated mode didn't actually invoke per-pool capacity updates"
      kill -9 $pid 2>/dev/null || true
      return 1
    fi

    local jsonl="$OUT_DIR/${arm}_budgeter.jsonl"
    local k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" || true)
    local m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" || true)
    echo "[$arm] xpool transfers: kv→mamba=$k2m mamba→kv=$m2k"
    if [ "${k2m:-0}" -lt 1 ] || [ "${m2k:-0}" -lt 1 ]; then
      echo "FAIL: arm B did not fire any transfers"
      kill -9 $pid 2>/dev/null || true
      return 1
    fi
  fi

  # Inspect server log for runtime errors.
  local err_lines=$(awk '/Server started/,/Shutting down/' "$log" \
    | grep -iE "memory leak|RuntimeError|Traceback|CUDA error" || true)
  if [ -n "$err_lines" ]; then
    echo "FAIL: errors found in [$arm] server log during serving:"
    echo "$err_lines"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  kill -TERM $pid 2>/dev/null || true
  sleep 5
  kill -9 $pid 2>/dev/null || true
  sleep 3
}

run_arm baseline
run_arm shared_xpool_coord

echo "=== diff ==="
if diff -q "$OUT_DIR/baseline_completions.txt" "$OUT_DIR/shared_xpool_coord_completions.txt" >/dev/null; then
  echo "PASS: completions byte-identical between baseline and shared+xpool+coordinated arms"
  echo "--- example output (first 6 lines from baseline) ---"
  head -6 "$OUT_DIR/baseline_completions.txt"
else
  echo "FAIL: completions differ (coordinated mode introduced divergence)"
  diff "$OUT_DIR/baseline_completions.txt" "$OUT_DIR/shared_xpool_coord_completions.txt"
  exit 1
fi
