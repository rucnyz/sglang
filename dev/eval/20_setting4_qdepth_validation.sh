#!/bin/bash
# Setting 4 follow-up — validate the saturation+queue-depth rule end-to-end.
#
# Runs the Phase 1+2+3 long/short/long stress trace (same as A3/A4) twice:
#   arm a: SGLANG_XPOOL_QDEPTH_TRIGGER=0 (legacy, V≈usage only)
#   arm b: SGLANG_XPOOL_QDEPTH_TRIGGER=4 (new saturation+queue rule active)
#
# The legacy rule fires when one pool is high AND the other is low. The
# new rule additionally fires when one pool is saturated AND the queue is
# non-trivial — even if the other pool is also above its high watermark.
# Expectation: the new rule should fire at LEAST as many transfers as
# legacy on this workload, with extra transfers landing in the saturation
# regime that legacy misses.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
ARM=${ARM:-legacy}  # legacy or qdepth
OUT_DIR=${OUT_DIR:-/tmp/setting4_qd_$$}
mkdir -p "$OUT_DIR"

extra_env="SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=0.5 SGLANG_XPOOL_KV_HIGH=0.04 SGLANG_XPOOL_KV_LOW=0.038 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.076 SGLANG_XPOOL_COOLDOWN=2"
case "$ARM" in
  legacy)
    extra_env="$extra_env SGLANG_XPOOL_QDEPTH_TRIGGER=0"
    ;;
  qdepth)
    extra_env="$extra_env SGLANG_XPOOL_QDEPTH_TRIGGER=4"
    ;;
esac
log="$OUT_DIR/${ARM}_server.log"
jsonl="$OUT_DIR/${ARM}_budgeter.jsonl"
extra_env="$extra_env SGLANG_BUDGETER_LOG=$jsonl"
echo "[$ARM] env: $extra_env"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env $extra_env \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
pid=$!
echo "[$ARM] pid=$pid"

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[$ARM] ready after ${waited}s"
    break
  fi
done

# Phase 1+2+3 stress trace identical to A3/A4.
PORT=$PORT MODEL=$MODEL .venv/bin/python <<'PY'
import json, urllib.request, time, threading, os
PORT = os.environ['PORT']; MODEL = os.environ['MODEL']
LONG_BASE = "Compute step by step. " * 250
def fire(prompt):
    data = json.dumps({'model': MODEL, 'prompt': prompt, 'max_tokens': 64, 'temperature': 0}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try: urllib.request.urlopen(req, timeout=300).read()
    except: pass

threads = []
for i in range(30):
    t = threading.Thread(target=fire, args=(LONG_BASE + f' Q{i}: name a fruit:',), daemon=True)
    t.start(); threads.append(t); time.sleep(0.05)
for t in threads: t.join(timeout=300)
time.sleep(8)
threads = []
for i in range(40):
    t = threading.Thread(target=fire, args=(f'Q{i}: name a color:',), daemon=True)
    t.start(); threads.append(t); time.sleep(0.03)
for t in threads: t.join(timeout=180)
time.sleep(8)
LONG2 = "Compute step by step. " * 5500
for i in range(4):
    fire(LONG2 + f' Q{i}: name a fruit:'); time.sleep(1)
PY

sleep 6
total=$(grep -c '"xpool_direction":' "$jsonl" 2>/dev/null || echo 0)
k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" 2>/dev/null || echo 0)
m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" 2>/dev/null || echo 0)
sat_q=$(grep -c '"xpool_plan_reason": "saturation+queue' "$jsonl" 2>/dev/null || echo 0)
echo "[$ARM] transfers: total=$total kv_to_mamba=$k2m mamba_to_kv=$m2k saturation_queue_decisions=$sat_q"

kill -9 $pid 2>/dev/null || true
sleep 5
