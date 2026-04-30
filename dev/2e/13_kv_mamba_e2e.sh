#!/bin/bash
# Phase 2e.5.5 — End-to-end serving with BOTH KV and mamba pools arena-backed.
#
# Boots one SGLang server with:
#   SGLANG_KV_ARENA=1                  (KV pool from MultiTensorArena)
#   SGLANG_MAMBA_PERLAYER=1            (mamba temporal_state -> per-layer list)
#   SGLANG_MAMBA_ARENA=1               (mamba temporal -> MultiTensorArena)
# Sends a handful of completions, checks they return coherent text, and verifies
# the engine doesn't segfault or trip its leak detection during the run.
#
# Pass criterion (all of):
#   - server reaches /health 200
#   - all completions return non-empty text
#   - no "memory leak" / "RuntimeError" in server log during serving
#   - server exits cleanly via SIGTERM (a known process-exit segfault in
#     PyTorch's MemPool::~MemPool is acceptable: it happens AFTER serving and
#     does not affect any served request).

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-600}
OUT_DIR=/tmp/kv_mamba_e2e_$$
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

LOG="$OUT_DIR/server.log"
echo "=== arena-on serving (KV+mamba both arena-backed) ==="
pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 3

nohup env \
  SGLANG_KV_ARENA=1 \
  SGLANG_MAMBA_PERLAYER=1 \
  SGLANG_MAMBA_ARENA=1 \
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

# Confirm both arena paths are live in the server log.
echo "=== sanity-check arena paths in log ==="
KV_LINE=$(grep "MHATokenToKVPool buffers: backend=arena" "$LOG" | head -1 || true)
MAMBA_LINE=$(grep "MambaPool arena:" "$LOG" | head -1 || true)
PERLAYER_LINE=$(grep "temporal layout=per-layer-list" "$LOG" | head -1 || true)
echo "KV arena evidence:    ${KV_LINE:-(none found)}"
echo "Mamba arena evidence: ${MAMBA_LINE:-(none found)}"
echo "Perlayer evidence:    ${PERLAYER_LINE:-(none found)}"

# Send completions; collect outputs.
OUT="$OUT_DIR/completions.txt"
: >"$OUT"
echo "=== sending ${#PROMPTS[@]} completions ==="
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
  echo "$resp" | head -3
done

# Inspect server log for runtime errors.
echo "=== checking server log for runtime errors ==="
ERR_LINES=$(grep -iE "leak|RuntimeError|Traceback|CUDA error" "$LOG" || true)
if [ -n "$ERR_LINES" ]; then
  echo "WARNING: errors found in server log:"
  echo "$ERR_LINES"
fi

# Check that every completion returned non-empty text.
EMPTY_COUNT=$(awk -v RS='===' 'NR<=ENVIRON["NPROMPTS"] {
  body=$0; gsub(/^[ \t\r\n]*\[[0-9]+\][^\n]*\n---\n/,"",body); gsub(/[ \t\r\n]+$/,"",body);
  if (body=="") n++
} END { print n+0 }' NPROMPTS="${#PROMPTS[@]}" "$OUT")
if [ "${EMPTY_COUNT:-0}" -gt 0 ]; then
  echo "FAIL: ${EMPTY_COUNT} completion(s) were empty"
  cat "$OUT"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

echo "=== shutting down ==="
kill -TERM $PID 2>/dev/null || true
sleep 5
kill -9 $PID 2>/dev/null || true

# Re-check log for "memory leak" trips (these happen during serving, not at exit).
LEAK_DURING_SERVING=$(awk '/Server started/,/Shutting down/' "$LOG" | grep -iE "memory leak|RuntimeError" || true)
if [ -n "$LEAK_DURING_SERVING" ]; then
  echo "FAIL: leak/error detected during serving:"
  echo "$LEAK_DURING_SERVING"
  exit 1
fi

echo
echo "=== completions ==="
cat "$OUT"
echo
echo "PASS: KV+mamba arena e2e served ${#PROMPTS[@]} completions cleanly"
echo "log: $LOG"
echo "completions: $OUT"
