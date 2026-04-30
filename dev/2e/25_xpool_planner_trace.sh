#!/bin/bash
# Phase 2e.5.6.3.c — headline trace: workload-driven cross-pool transfer.
#
# Scenario:
#   Phase 1 (KV-bound):  ~30s of long-context prompts (input=2048, output=64).
#                        KV pool fills up; planner should detect kv_high &
#                        mamba_low and trigger mamba_to_kv transfer(s).
#   Phase 2 (mamba-bound): ~30s of many concurrent short prompts (input=64,
#                        output=64, RPS=20). Mamba slot pressure rises;
#                        planner should detect mamba_high & kv_low and
#                        trigger kv_to_mamba transfer(s).
#
# Pass criterion (qualitative):
#   - planner sees both phases (different usage_kv / usage_mamba)
#   - at least one mamba_to_kv decision in Phase 1
#   - at least one kv_to_mamba decision in Phase 2
#   - completions remain coherent through the transition (no crash, no leak)

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
OUT_DIR=/tmp/xpool_planner_trace_$$
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/server.log"
JSONL="$OUT_DIR/budgeter.jsonl"
echo "out_dir=$OUT_DIR  warmup_max=${WARMUP_S}s"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env \
  SGLANG_ARENA_SHARED=1 \
  SGLANG_ARENA_FROM_BLOB=1 \
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
  echo "server failed to start"; tail -40 "$LOG"; kill -9 $PID 2>/dev/null || true; exit 1
fi

# Sanity.
grep "BudgetAgent xpool: actuator attached" "$LOG" | head -1 || echo "(actuator not yet)"
grep "CrossPoolPlanner:" "$LOG" | head -1 || echo "(planner not yet)"

# Phase 1: KV-bound workload (long context, many concurrent so KV pool
# really fills). 50 concurrent × ~1500-token prompts × 64 output ≈
# 80K tokens in flight at peak ≈ 6% of 1.3M cap → above kv_high=0.05.
echo
echo "=== Phase 1: KV-bound (long-context, 50 concurrent) ==="
.venv/bin/python <<PY
import json, urllib.request, time, random, threading
random.seed(0)
LONG_BASE = "Compute step by step. " * 250  # ~1500 tokens of dummy context
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
        tokens = body['usage']['total_tokens']
        with lock: results.append(('ok', i, tokens))
    except Exception as e:
        with lock: results.append(('fail', i, str(e)))
threads = []
for i in range(50):
    prompt = LONG_BASE + f" Q{i}: name a fruit:"
    t = threading.Thread(target=fire, args=(i, prompt), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(0.05)
for t in threads:
    t.join(timeout=300)
oks = sum(1 for s, _, _ in results if s == 'ok')
fails = sum(1 for s, _, _ in results if s == 'fail')
total_tokens = sum(t for s, _, t in results if s == 'ok' and isinstance(t, int))
print(f"  KV-bound batch: {oks} ok, {fails} fail, {total_tokens} total tokens")
PY

echo
echo "=== Phase 1.5: idle (drain in-flight, let cooldowns clear) ==="
sleep 8

echo
echo "=== Phase 2: mamba-bound (many short, 60 concurrent) ==="
.venv/bin/python <<PY
import json, urllib.request, time, threading
results = []
lock = threading.Lock()
def fire(i, prompt):
    data = json.dumps({
        'model': '$MODEL', 'prompt': prompt,
        'max_tokens': 32, 'temperature': 0,
    }).encode()
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=120).read())
        tokens = body['usage']['total_tokens']
        with lock:
            results.append(('ok', i, tokens))
    except Exception as e:
        with lock:
            results.append(('fail', i, str(e)))

# 60 concurrent prompts → at peak ~60/361 = 17% mamba slot usage,
# above mamba_high=0.08; should trigger kv_to_mamba.
threads = []
for i in range(60):
    prompt = f"Q{i}: name a color starting with " + chr(ord('a') + (i % 26)) + ":"
    t = threading.Thread(target=fire, args=(i, prompt), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(0.03)
for t in threads:
    t.join(timeout=180)
oks = sum(1 for s, _, _ in results if s == 'ok')
fails = sum(1 for s, _, _ in results if s == 'fail')
print(f"  mamba-bound batch: {oks} ok, {fails} fail")
PY

echo
echo "=== Phase 2.5: idle drain + flush_cache (clear radix-held mamba slots) ==="
curl -s -X POST "http://127.0.0.1:$PORT/flush_cache?timeout=10.0" >/dev/null || true
sleep 12

# Phase 3: KV-only pressure. After Phase 2 the KV pool was drained
# repeatedly (down to ~327K tokens). Single large-context requests
# pushed sequentially keep mamba at 1 slot (≪ mamba_low) while KV
# usage rises above kv_high. Should trigger mamba_to_kv.
echo
echo "=== Phase 3: KV-bound (large-context, sequential, 1 in flight) ==="
.venv/bin/python <<PY
import json, urllib.request, time, random
random.seed(1)
# 60K-context prompt: enough that even on a 1.3M KV pool, single-prompt
# in-flight KV usage crosses kv_high=0.04 (= 52K tokens). "Compute step
# by step. " ≈ 6 tokens × 11000 = 66000 tokens.
LONG_BASE = "Compute step by step. " * 11000
for i in range(4):
    prompt = LONG_BASE + f" Question {i}: name a fruit:"
    data = json.dumps({
        'model': '$MODEL', 'prompt': prompt,
        'max_tokens': 32, 'temperature': 0,
    }).encode()
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=300).read())
        tokens = body['usage']['total_tokens']
        print(f"  KV-bound seq prompt {i}: {tokens} tokens")
    except Exception as e:
        print(f"  KV-bound seq prompt {i}: FAILED {e}")
    time.sleep(1)
PY

# Let final ticks settle.
sleep 6

# Inspect plan decisions.
echo
echo "=== planner decisions ==="
.venv/bin/python <<PY
import json
from collections import Counter
direction_count = Counter()
phase_buckets = {'kv_high': 0, 'mamba_high': 0, 'both_band': 0, 'cooldown': 0, 'busy_skip': 0, 'no_data': 0}
plan_lines = []
exec_lines = []
with open("$JSONL") as f:
    for line in f:
        try: d = json.loads(line)
        except: continue
        if 'xpool_plan_direction' not in d: continue
        plan_lines.append(d)
        direction_count[d.get('xpool_plan_direction', 'none')] += 1
        if d.get('xpool_plan_executed'):
            exec_lines.append(d)
            continue
        reason = d.get('xpool_plan_reason', '')
        if 'cooldown' in reason: phase_buckets['cooldown'] += 1
        elif 'within band' in reason: phase_buckets['both_band'] += 1
        elif d.get('xpool_plan_skipped') == 'engine_busy': phase_buckets['busy_skip'] += 1
print(f"total plan ticks: {len(plan_lines)}")
print(f"  direction breakdown: {dict(direction_count)}")
print(f"  reason buckets: {phase_buckets}")
print(f"  executed transfers: {len(exec_lines)}")
if exec_lines:
    print("\n  first / last executed:")
    for d in exec_lines[:1] + ([exec_lines[-1]] if len(exec_lines) > 1 else []):
        print(f"    ts={d.get('ts')} dir={d.get('xpool_direction')} "
              f"kv_cap={d.get('xpool_kv_capacity_tokens')} "
              f"mamba_cap={d.get('xpool_mamba_capacity_tokens')} "
              f"reason={d.get('xpool_plan_reason')[:80]}")
print("\n  example usage signals over time:")
# Print at fixed strides.
n = max(1, len(plan_lines) // 10)
for d in plan_lines[::n]:
    print(f"    ts={d.get('ts')} kv={d.get('xpool_plan_usage_kv', 0):.3f} "
          f"mamba={d.get('xpool_plan_usage_mamba', 0):.3f} "
          f"dir={d.get('xpool_plan_direction')} "
          f"reason={d.get('xpool_plan_reason', '')[:60]}")
PY

echo
echo "=== checking server log for errors ==="
err=$(awk '/Server started/,/Shutting down/' "$LOG" | grep -iE "memory leak|RuntimeError|Traceback|CUDA error" || true)
if [ -n "$err" ]; then
  echo "FAIL: errors during serving"
  echo "$err"
  kill -9 $PID 2>/dev/null || true; exit 1
fi

echo
echo "=== shutting down ==="
kill -TERM $PID 2>/dev/null || true
sleep 5
kill -9 $PID 2>/dev/null || true

echo
echo "PASS: planner trace captured"
echo "log:    $LOG"
echo "jsonl:  $JSONL"
