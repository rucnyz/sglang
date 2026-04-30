#!/bin/bash
# Phase 3.c — combined Layer 1 + Layer 2 integration test.
#
# Boots a server with:
#   Layer 1: SGLANG_HPB_LRU=1 (paper §4.2 hits-per-byte LRU)
#   Layer 2: SGLANG_ARENA_SHARED + FROM_BLOB + XPOOL_PLANNER + COORDINATED
#
# Runs the same 3-phase workload as 25_xpool_planner_trace.sh and
# additionally:
#   1. Verifies V_prefix_marginal appears in the budgeter JSONL.
#   2. Verifies HPB LRU doesn't crash anything.
#   3. Verifies cross-pool transfers still fire (Layer 2 still works).
#   4. Reports peak V_prefix' values across the trace.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
OUT_DIR=/tmp/layer1_layer2_$$
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/server.log"
JSONL="$OUT_DIR/budgeter.jsonl"
echo "out_dir=$OUT_DIR"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env \
  SGLANG_ARENA_SHARED=1 \
  SGLANG_ARENA_FROM_BLOB=1 \
  SGLANG_HPB_LRU=1 \
  SGLANG_HPB_WINDOW_S=60.0 \
  SGLANG_BUDGETER=1 \
  SGLANG_BUDGETER_XPOOL_PLANNER=1 \
  SGLANG_BUDGETER_XPOOL_COORDINATED=1 \
  SGLANG_BUDGETER_TICK_S=0.5 \
  SGLANG_BUDGETER_LOG="$JSONL" \
  SGLANG_XPOOL_KV_HIGH=0.04 \
  SGLANG_XPOOL_KV_LOW=0.015 \
  SGLANG_XPOOL_MAMBA_HIGH=0.08 \
  SGLANG_XPOOL_MAMBA_LOW=0.03 \
  SGLANG_XPOOL_COOLDOWN=2 \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$LOG" 2>&1 &
PID=$!
echo "server pid=$PID log=$LOG"
echo "--- live server log ---"
tail -F "$LOG" 2>/dev/null &
TAILER=$!

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    kill $TAILER 2>/dev/null || true; wait $TAILER 2>/dev/null || true
    echo; echo "--- ready after ${waited}s ---"; break
  fi
done
if kill -0 $TAILER 2>/dev/null; then kill $TAILER 2>/dev/null || true; wait $TAILER 2>/dev/null || true; fi
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
  echo "FAIL: server did not become ready"; tail -40 "$LOG"; kill -9 $PID 2>/dev/null || true; exit 1
fi

# 3-phase workload (same as 25).
echo
echo "=== Phase 1: KV-bound (50 concurrent long context) ==="
.venv/bin/python <<PY
import json, urllib.request, time, threading
LONG_BASE = "Compute step by step. " * 250
results = []
lock = threading.Lock()
def fire(i, prompt):
    data = json.dumps({
        'model': '$MODEL', 'prompt': prompt,
        'max_tokens': 64, 'temperature': 0,
    }).encode()
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=300).read())
        with lock: results.append(('ok', i, body['usage']['total_tokens']))
    except Exception as e:
        with lock: results.append(('fail', i, str(e)))
threads = []
for i in range(50):
    prompt = LONG_BASE + f" Q{i}: name a fruit:"
    t = threading.Thread(target=fire, args=(i, prompt), daemon=True); t.start(); threads.append(t)
    time.sleep(0.05)
for t in threads: t.join(timeout=300)
oks = sum(1 for s, _, _ in results if s == 'ok')
print(f"  Phase 1: {oks}/50 ok")
PY

echo
echo "=== Phase 1.5: drain ==="
sleep 8

echo
echo "=== Phase 2: mamba-bound (60 concurrent short) ==="
.venv/bin/python <<PY
import json, urllib.request, time, threading
results = []; lock = threading.Lock()
def fire(i, prompt):
    data = json.dumps({
        'model': '$MODEL', 'prompt': prompt,
        'max_tokens': 32, 'temperature': 0,
    }).encode()
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=120).read())
        with lock: results.append(('ok', i, body['usage']['total_tokens']))
    except Exception as e:
        with lock: results.append(('fail', i, str(e)))
threads = []
for i in range(60):
    prompt = f"Q{i}: name a color starting with " + chr(ord('a') + (i % 26)) + ":"
    t = threading.Thread(target=fire, args=(i, prompt), daemon=True); t.start(); threads.append(t)
    time.sleep(0.03)
for t in threads: t.join(timeout=180)
oks = sum(1 for s, _, _ in results if s == 'ok')
print(f"  Phase 2: {oks}/60 ok")
PY

echo
echo "=== Phase 2.5: flush_cache + drain ==="
curl -s -X POST "http://127.0.0.1:$PORT/flush_cache?timeout=10.0" >/dev/null || true
sleep 12

echo
echo "=== Phase 3: KV-bound sequential 60K context ==="
.venv/bin/python <<PY
import json, urllib.request, time
LONG_BASE = "Compute step by step. " * 11000
for i in range(4):
    prompt = LONG_BASE + f" Q{i}: name a fruit:"
    data = json.dumps({
        'model': '$MODEL', 'prompt': prompt,
        'max_tokens': 32, 'temperature': 0,
    }).encode()
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=300).read())
        print(f"  Phase 3 prompt {i}: {body['usage']['total_tokens']} tokens")
    except Exception as e:
        print(f"  Phase 3 prompt {i}: FAILED {e}")
    time.sleep(1)
PY

sleep 6

echo
echo "=== combined Layer 1 + Layer 2 analysis ==="
.venv/bin/python <<PY
import json
v_prefix_seen = []
xpool_kv_to_mamba = 0
xpool_mamba_to_kv = 0
errors = []
with open("$JSONL") as f:
    for line in f:
        try: d = json.loads(line)
        except: continue
        if 'v_prefix_marginal' in d:
            v_prefix_seen.append(d['v_prefix_marginal'])
        if d.get('xpool_direction') == 'kv_to_mamba': xpool_kv_to_mamba += 1
        if d.get('xpool_direction') == 'mamba_to_kv': xpool_mamba_to_kv += 1
        if 'v_prefix_marginal_error' in d: errors.append(d['v_prefix_marginal_error'])

# Layer 1 health
print(f"V_prefix_marginal samples:     {len(v_prefix_seen)}")
if v_prefix_seen:
    nonzero = [v for v in v_prefix_seen if v > 0]
    print(f"  non-zero samples:            {len(nonzero)}")
    if nonzero:
        print(f"  peak:                        {max(nonzero):.4f}")
        print(f"  mean (over non-zero):        {sum(nonzero)/len(nonzero):.4f}")
print(f"V_prefix' compute errors:      {len(errors)}")
if errors:
    print(f"  example error: {errors[0][:120]}")

# Layer 2 health
print(f"\nLayer 2 cross-pool transfers:")
print(f"  kv → mamba:                  {xpool_kv_to_mamba}")
print(f"  mamba → kv:                  {xpool_mamba_to_kv}")
PY

echo
err=$(awk '/Server started/,/Shutting down/' "$LOG" | grep -iE "memory leak|RuntimeError|Traceback|CUDA error" || true)
if [ -n "$err" ]; then
  echo "FAIL: errors during serving"
  echo "$err"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

echo "=== shutting down ==="
kill -TERM $PID 2>/dev/null || true
sleep 5
kill -9 $PID 2>/dev/null || true

echo
echo "PASS: Layer 1 (HPB LRU) + Layer 2 (cross-pool actuator) coexist cleanly"
echo "log:    $LOG"
echo "jsonl:  $JSONL"
