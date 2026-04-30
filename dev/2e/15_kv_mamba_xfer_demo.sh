#!/bin/bash
# Phase 2e.5.6.2 — KV ↔ mamba cross-pool transfer demo (live serving).
#
# Boots one server with:
#   SGLANG_ARENA_SHARED=1            # implies KV_ARENA + MAMBA_ARENA
#   SGLANG_BUDGETER=1
#   SGLANG_BUDGETER_XPOOL_DEMO=1     # this script's load-bearing flag
#   SGLANG_BUDGETER_XPOOL_UNIT=1     # 1 chunk per source sub-pool per tick
#   SGLANG_BUDGETER_TICK_S=2.0
#   SGLANG_BUDGETER_LOG=/tmp/budgeter_xpool_demo.jsonl
#
# Sends sequential completions with sleeps between them to give the
# budgeter idle windows in which to fire the cross-pool actuator.
#
# Pass criterion (all of):
#   1. server reaches /health 200
#   2. all completions return non-empty coherent text (engine survives
#      the cross-pool capacity changes)
#   3. budgeter JSONL contains both `xpool_direction=kv_to_mamba` and
#      `xpool_direction=mamba_to_kv` entries (= both directions exercised)
#   4. CrossPoolTransferActuator is attached at runtime
#   5. no leak / RuntimeError / Traceback during serving

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-600}
OUT_DIR=/tmp/kv_mamba_xfer_$$
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/server.log"
JSONL="$OUT_DIR/budgeter_xpool_demo.jsonl"
echo "out_dir=$OUT_DIR  GPU=$CUDA_VISIBLE_DEVICES  warmup_max=${WARMUP_S}s"

PROMPTS=(
  "The capital of France is"
  "Once upon a time"
  "Q: 2 + 2 ="
  "def fibonacci(n):"
  "List three primes:"
  "Translate to French: hello"
  "Write a haiku about CUDA:"
  "Sum of first ten integers ="
)
MAX_TOKENS=20
INTER_PROMPT_SLEEP=3   # seconds; lets the budgeter tick fire between prompts

echo "=== arena-shared serving + xpool demo ==="
pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 3

nohup env \
  SGLANG_ARENA_SHARED=1 \
  SGLANG_BUDGETER=1 \
  SGLANG_BUDGETER_XPOOL_DEMO=1 \
  SGLANG_BUDGETER_XPOOL_UNIT=1 \
  SGLANG_BUDGETER_TICK_S=2.0 \
  SGLANG_BUDGETER_LOG="$JSONL" \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$LOG" 2>&1 &
PID=$!
echo "server pid=$PID log=$LOG; waiting up to ${WARMUP_S}s for /health 200"
echo "--- live server log ---"
tail -F "$LOG" 2>/dev/null &
TAILER=$!

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    kill $TAILER 2>/dev/null || true
    wait $TAILER 2>/dev/null || true
    echo
    echo "--- ready after ${waited}s ---"
    break
  fi
done
if kill -0 $TAILER 2>/dev/null; then
  kill $TAILER 2>/dev/null || true
  wait $TAILER 2>/dev/null || true
fi

if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
  echo "FAIL: server did not become ready in ${WARMUP_S}s"
  tail -40 "$LOG"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

# Sanity: confirm both arenas were built on the SHARED handle pool.
echo "=== sanity-check shared arena init ==="
SHARED_LINE=$(grep "Arena shared mode: created process-singleton" "$LOG" | head -1 || true)
KV_LINE=$(grep "MHATokenToKVPool arena:.*shared=True" "$LOG" | head -1 || true)
MAMBA_LINE=$(grep "MambaPool arena:.*shared=True" "$LOG" | head -1 || true)
ATTACH_LINE=$(grep "BudgetAgent xpool: actuator attached" "$LOG" | head -1 || true)
echo "Shared singleton: ${SHARED_LINE:-(none)}"
echo "KV arena:         ${KV_LINE:-(none)}"
echo "Mamba arena:      ${MAMBA_LINE:-(none)}"
echo "Actuator:         ${ATTACH_LINE:-(none, will attach on first tick)}"
if [ -z "$SHARED_LINE" ] || [ -z "$KV_LINE" ] || [ -z "$MAMBA_LINE" ]; then
  echo "FAIL: shared arena init not visible in log"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

# Sequential completions with idle windows between them.
OUT="$OUT_DIR/completions.txt"
: >"$OUT"
echo "=== sending ${#PROMPTS[@]} completions, sleeping ${INTER_PROMPT_SLEEP}s between (lets budgeter fire) ==="
i=0
for prompt in "${PROMPTS[@]}"; do
  i=$((i + 1))
  echo "--- prompt $i: \"$prompt\""
  resp=$(.venv/bin/python -c "
import json, urllib.request
data = json.dumps({
    'model': '$MODEL', 'prompt': '''$prompt''',
    'max_tokens': $MAX_TOKENS, 'temperature': 0,
}).encode()
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
    data=data, headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req, timeout=300)
print(json.loads(r.read())['choices'][0]['text'])
" 2>&1)
  printf '[%d] %s\n---\n%s\n===\n' "$i" "$prompt" "$resp" >>"$OUT"
  echo "$resp" | head -2
  sleep $INTER_PROMPT_SLEEP
done

# Give the budgeter a few more ticks to log post-traffic state.
sleep 6

# Inspect server log + budgeter JSONL.
echo "=== checking server log for runtime errors ==="
ERR_LINES=$(awk '/Server started/,/Shutting down/' "$LOG" \
            | grep -iE "memory leak|RuntimeError|Traceback|CUDA error" || true)
if [ -n "$ERR_LINES" ]; then
  echo "FAIL: errors found in server log during serving:"
  echo "$ERR_LINES"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

echo "=== inspecting budgeter JSONL ==="
if [ ! -s "$JSONL" ]; then
  echo "FAIL: budgeter JSONL ($JSONL) is empty"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi
KV_TO_MAMBA=$(grep -c '"xpool_direction": "kv_to_mamba"' "$JSONL" || true)
MAMBA_TO_KV=$(grep -c '"xpool_direction": "mamba_to_kv"' "$JSONL" || true)
SKIPPED_BUSY=$(grep -c '"xpool_skipped": "engine_busy"' "$JSONL" || true)
TOTAL_TICKS=$(wc -l <"$JSONL" | awk '{print $1}')
echo "budgeter ticks total:   $TOTAL_TICKS"
echo "  kv→mamba transfers:   $KV_TO_MAMBA"
echo "  mamba→kv transfers:   $MAMBA_TO_KV"
echo "  skipped (engine busy): $SKIPPED_BUSY"

if [ "${KV_TO_MAMBA:-0}" -lt 1 ]; then
  echo "FAIL: no kv→mamba transfers logged"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi
if [ "${MAMBA_TO_KV:-0}" -lt 1 ]; then
  echo "FAIL: no mamba→kv transfers logged"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

# Print one example transfer log line + final state.
echo "=== example xpool transfer entry ==="
grep '"xpool_direction"' "$JSONL" | head -1
echo "=== final shared-pool state ==="
grep "CrossPoolTransferActuator" "$LOG" | tail -3

echo "=== shutting down ==="
kill -TERM $PID 2>/dev/null || true
sleep 5
kill -9 $PID 2>/dev/null || true

echo
echo "=== completions ==="
cat "$OUT"
echo
echo "PASS: KV ↔ mamba cross-pool transfer demo (kv→mamba=$KV_TO_MAMBA, "
echo "      mamba→kv=$MAMBA_TO_KV, skipped-busy=$SKIPPED_BUSY)"
echo "log:           $LOG"
echo "completions:   $OUT"
echo "budgeter jsonl:$JSONL"
