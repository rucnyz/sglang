#!/bin/bash
# Phase 2e.5.6.3.b — diagnostic: PyTorch profiler trace of prefill
# under baseline vs arena modes.
#
# Boots each server in turn, warms it up by sending two prompts (so
# Triton kernels are JIT-compiled), then triggers /start_profile,
# sends ONE prompt, /stop_profile. Output: chrome trace JSON per arm.
#
# After both runs, dump the top-10 GPU kernels by total time for each
# arm so we can see WHERE the +6% TTFT is going.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
GPUS=${GPUS:-3}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}

OUT_DIR=/tmp/arena_profile_$$
mkdir -p "$OUT_DIR"
PROFILE_DIR=$OUT_DIR/profiles
mkdir -p "$PROFILE_DIR"

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "arena" ]; then
    extra_env="SGLANG_ARENA_SHARED=1"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  local arm_profile_dir="$PROFILE_DIR/$arm"
  mkdir -p "$arm_profile_dir"

  echo "=== arm=$arm ==="
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env CUDA_VISIBLE_DEVICES="$GPUS" $extra_env \
    SGLANG_TORCH_PROFILER_DIR="$arm_profile_dir" \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --tensor-parallel-size 1 \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!
  echo "[$arm] pid=$pid"

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[$arm] ready after ${waited}s"
      break
    fi
  done
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
    echo "[$arm] server failed"
    tail -30 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  # Warmup with 3 prompts so Triton kernels are compiled and steady
  # state is reached before profiling.
  echo "[$arm] warming up Triton cache (3 prompts)"
  for warm_prompt in "Hello world" "Solve: 1+1=" "Capital of France:"; do
    .venv/bin/python -c "
import json, urllib.request
data = json.dumps({
    'model': '$MODEL', 'prompt': '''$warm_prompt''',
    'max_tokens': 16, 'temperature': 0,
}).encode()
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
    data=data, headers={'Content-Type': 'application/json'})
urllib.request.urlopen(req, timeout=120).read()
print('  warm prompt done')
"
    sleep 1
  done

  # Start profile, send one prompt, stop.
  echo "[$arm] starting profile..."
  curl -s -X POST "http://127.0.0.1:$PORT/start_profile" \
    -H "Content-Type: application/json" \
    -d '{"with_stack": true, "record_shapes": false}'
  echo
  sleep 1

  .venv/bin/python -c "
import json, urllib.request
data = json.dumps({
    'model': '$MODEL',
    'prompt': 'Tell me a joke about CUDA in 50 words:',
    'max_tokens': 64, 'temperature': 0,
}).encode()
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
    data=data, headers={'Content-Type': 'application/json'})
print(json.loads(urllib.request.urlopen(req, timeout=120).read())['choices'][0]['text'][:120])
"
  sleep 2

  curl -s -X POST "http://127.0.0.1:$PORT/stop_profile" || true
  echo
  echo "[$arm] profile saved to $arm_profile_dir"
  sleep 6   # let trace be flushed to disk

  kill -TERM $pid 2>/dev/null || true
  sleep 5
  kill -9 $pid 2>/dev/null || true
  sleep 3
}

run_arm baseline
run_arm arena

echo
echo "=== profile output trees ==="
echo "--- baseline ---"
ls -la "$PROFILE_DIR/baseline" 2>/dev/null || echo "(empty)"
echo "--- arena ---"
ls -la "$PROFILE_DIR/arena" 2>/dev/null || echo "(empty)"

echo
echo "=== top-10 GPU kernels per arm ==="
ARM_DIR_BASE=$PROFILE_DIR .venv/bin/python <<'PY'
import json, os, glob, gzip
from collections import defaultdict

base = os.environ["ARM_DIR_BASE"]

def walk_traces(arm):
    arm_dir = os.path.join(base, arm)
    files = []
    for root, _, names in os.walk(arm_dir):
        for n in names:
            if n.endswith(".pt.trace.json") or n.endswith(".pt.trace.json.gz"):
                files.append(os.path.join(root, n))
    return sorted(files)

def parse_trace(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            d = json.load(f)
    else:
        with open(path) as f:
            d = json.load(f)
    return d.get("traceEvents", [])

def top_kernels(events):
    by_name = defaultdict(lambda: [0, 0])  # [count, total_us]
    for ev in events:
        cat = ev.get("cat", "")
        if cat == "kernel":   # GPU kernel events
            name = ev.get("name", "")
            dur = ev.get("dur", 0)
            by_name[name][0] += 1
            by_name[name][1] += dur
    rows = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:15]
    return rows

for arm in ("baseline", "arena"):
    print(f"\n--- {arm} ---")
    traces = walk_traces(arm)
    if not traces:
        print("(no traces found)")
        continue
    print(f"trace files: {[os.path.basename(t) for t in traces]}")
    for tf in traces:
        events = parse_trace(tf)
        print(f"\n[{os.path.basename(tf)}] events={len(events)}")
        rows = top_kernels(events)
        if not rows:
            print("  (no GPU kernel events found)")
            continue
        total = sum(r[1][1] for r in rows)
        print(f"  top kernels (total {total/1000:.2f} ms):")
        for name, (cnt, dur) in rows:
            display = name if len(name) < 78 else name[:75] + "..."
            print(f"    {cnt:>4}× {dur/1000:>9.3f} ms  {display}")
PY
