#!/bin/bash
# Ablation A4 — Layer 2 control interval (τ).
#
# Paper §6.7 spec: "Sweep τ ∈ {5, 15, 30, 60, 300}s". Those values target
# the 24-hour trace. For our compressed bench we sweep {0.5, 1, 2, 5, 15}
# seconds — fast enough that something can be observed within ~3 minutes
# of serving.
#
# Mechanism: SGLANG_BUDGETER_TICK_S sets the planner's poll interval.
# Reports number of transfers fired and per-transfer ms-from-trigger
# latency.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
OUT_DIR=/tmp/a4_tau_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_point() {
  local tau="$1"
  local label="tau${tau}"
  local log="$OUT_DIR/${label}_server.log"
  local jsonl="$OUT_DIR/${label}_budgeter.jsonl"
  echo "=== τ=$tau ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env \
    SGLANG_ARENA_SHARED=1 \
    SGLANG_ARENA_FROM_BLOB=1 \
    SGLANG_BUDGETER=1 \
    SGLANG_BUDGETER_XPOOL_PLANNER=1 \
    SGLANG_BUDGETER_XPOOL_COORDINATED=1 \
    SGLANG_BUDGETER_TICK_S=$tau \
    SGLANG_BUDGETER_LOG="$jsonl" \
    SGLANG_XPOOL_KV_HIGH=0.04 \
    SGLANG_XPOOL_KV_LOW=0.038 \
    SGLANG_XPOOL_MAMBA_HIGH=0.08 \
    SGLANG_XPOOL_MAMBA_LOW=0.076 \
    SGLANG_XPOOL_COOLDOWN=2 \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[τ=$tau] ready after ${waited}s"
      break
    fi
  done

  # Reuse the same Phase 1+2+3 trace as A3.
  .venv/bin/python <<PY
import json, urllib.request, time, threading
LONG_BASE = "Compute step by step. " * 250
def fire(prompt):
    data = json.dumps({'model': '$MODEL', 'prompt': prompt, 'max_tokens': 64, 'temperature': 0}).encode()
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try: urllib.request.urlopen(req, timeout=300).read()
    except: pass

threads = []
for i in range(30):
    t = threading.Thread(target=fire, args=(LONG_BASE + f' Q{i}: name a fruit:',), daemon=True)
    t.start(); threads.append(t)
    time.sleep(0.05)
for t in threads: t.join(timeout=300)
time.sleep(8)
threads = []
for i in range(40):
    t = threading.Thread(target=fire, args=(f'Q{i}: name a color:',), daemon=True)
    t.start(); threads.append(t)
    time.sleep(0.03)
for t in threads: t.join(timeout=180)
time.sleep(8)
LONG2 = "Compute step by step. " * 5500
for i in range(4):
    fire(LONG2 + f' Q{i}: name a fruit:')
    time.sleep(1)
PY

  sleep 6
  kill -9 $pid 2>/dev/null || true
  sleep 5

  local transfers=$(grep -c '"xpool_direction":' "$jsonl" 2>/dev/null || echo 0)
  local k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" 2>/dev/null || echo 0)
  local m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" 2>/dev/null || echo 0)
  echo "[τ=$tau] transfers: total=$transfers, kv_to_mamba=$k2m, mamba_to_kv=$m2k"
}

for tau in 0.5 1 2 5 15; do
  run_point "$tau" || echo "[τ=$tau] FAILED, continuing"
done

echo
echo "=== A4 control-interval summary ==="
.venv/bin/python <<PY
import os
out = "$OUT_DIR"
print(f"\n{'τ (s)':>6} {'total':>7} {'kv→mamba':>10} {'mamba→kv':>10}")
print('-' * 40)
for tau in (0.5, 1, 2, 5, 15):
    path = f"{out}/tau{tau}_budgeter.jsonl"
    if not os.path.exists(path):
        print(f"{tau:>6} {'N/A':>7}")
        continue
    total = k2m = m2k = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if '"xpool_direction":' not in line: continue
            total += 1
            if '"kv_to_mamba"' in line: k2m += 1
            elif '"mamba_to_kv"' in line: m2k += 1
    print(f"{tau:>6} {total:>7} {k2m:>10} {m2k:>10}")
PY
