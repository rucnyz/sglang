#!/bin/bash
# Phase 3.a eval v3 — cold-burst with real-text prompts that genuinely
# cross chunked_prefill_size=8192.
#
# Findings from v2 diagnosis: shared-prefix prompts using repeated
# strings tokenize to far fewer tokens than their character count
# (compression). With <8192 tokens, no chunk-boundary snapshot is
# inserted, so MambaRadixCache.match returns 0 even on identical
# prompts. To make cache hits actually fire, prompts must:
#   1. tokenize to MORE than 8192 tokens (so a chunk-boundary snapshot
#      with mamba_value is created during chunked prefill);
#   2. be diverse text (not repeats) so the tokenizer doesn't compress.
#
# This script uses a chunk of real source code (scheduler.py, ~25K
# tokens) as the system prefix.
#
# Pass criterion: HPB Pulse 2 mean latency materially lower than recency
# Pulse 2 (= cache survived the cold burst on HPB but not on recency).

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
OUT_DIR=/tmp/hpb_realtext_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "hpb" ]; then
    extra_env="SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  local results="$OUT_DIR/${arm}_results.txt"
  : >"$results"
  echo "=== arm=$arm ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
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
    echo "FAIL: server did not become ready"
    tail -30 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  RESULTS=$results .venv/bin/python <<PY
import json, urllib.request, time, os, threading, statistics, random
random.seed(0)
PORT = $PORT
RESULTS = os.environ["RESULTS"]

# Real-text prefix: take a chunk of SGLang source as natural diverse
# text (no tokenizer compression). ~12000 lines of code clipped to
# ~50000 chars → tokenizes to roughly 12-15K tokens (above
# chunked_prefill_size=8192, so a chunk-boundary snapshot WILL be
# created during chunked prefill).
with open("/data/yuzhou/projects/sglang/python/sglang/srt/managers/scheduler.py") as f:
    SYS_PREFIX = f.read()[:55000]
print(f"  system prefix: {len(SYS_PREFIX)} chars")

def fire(prompt, mark, results):
    t0 = time.time()
    data = json.dumps({
        'model': '$MODEL', 'prompt': prompt,
        'max_tokens': 8, 'temperature': 0,
        'stream': False,
    }).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=300).read())
        elapsed = (time.time() - t0) * 1000
        results.append((mark, elapsed, body['usage']['total_tokens']))
    except Exception as e:
        results.append((mark, -1, str(e)))

# Pulse 1: build cache. 8 prompts sharing real source-code prefix.
print(">> Pulse 1 (build): 8 prompts sharing real-text prefix (~50K chars)")
results = []
for i in range(8):
    prompt = SYS_PREFIX + f"\n# Q{i}: explain in one word what this code does:"
    fire(prompt, 'pulse1', results)
    time.sleep(0.5)
p1 = [r[1] for r in results if r[0] == 'pulse1' and r[1] > 0]
p1_tokens = [r[2] for r in results if r[0] == 'pulse1' and r[1] > 0]
print(f"  pulse1: n={len(p1)}, mean_latency={statistics.mean(p1):.1f} ms, "
      f"prompt_tokens={p1_tokens[0] if p1_tokens else '?'}")

# Cold burst: 30 unique-prefix prompts. Each is a different chunk of
# scheduler.py so they cross chunk_boundary too, flooding the cache.
print(">> Cold burst: 30 unique 50K-char prefixes (~30s)")
threads = []
SCHED = open("/data/yuzhou/projects/sglang/python/sglang/srt/managers/scheduler.py").read()
for i in range(30):
    # Different starting offset → different prefix.
    start = (i * 1500) % max(1, len(SCHED) - 55000)
    UNIQUE = SCHED[start:start + 55000]
    t = threading.Thread(target=fire, args=(UNIQUE + f"\n# burst Q{i}:", 'burst', results), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(1.0)  # 1 RPS — long-prefill is heavy
for t in threads:
    t.join(timeout=300)
b = [r[1] for r in results if r[0] == 'burst' and r[1] > 0]
print(f"  burst: n={len(b)}, mean_latency={statistics.mean(b):.1f} ms")

# Idle drain.
time.sleep(8)

# Pulse 2: same prefix as Pulse 1; expect cache hit if HPB preserved.
print(">> Pulse 2 (post-burst): 8 prompts with same prefix as Pulse 1")
for i in range(8):
    prompt = SYS_PREFIX + f"\n# Q-recovered{i}: name a fruit:"
    fire(prompt, 'pulse2', results)
    time.sleep(0.3)
p2 = [r[1] for r in results if r[0] == 'pulse2' and r[1] > 0]
print(f"  pulse2: n={len(p2)}, mean_latency={statistics.mean(p2):.1f} ms")

with open(RESULTS, 'w') as f:
    f.write(f"# arm = {os.environ.get('SGLANG_HPB_LRU', '0')}\n")
    f.write(f"prompt_tokens = {p1_tokens[0] if p1_tokens else 0}\n")
    f.write(f"pulse1_mean_ms = {statistics.mean(p1):.2f}\n")
    f.write(f"burst_mean_ms = {statistics.mean(b):.2f}\n")
    f.write(f"pulse2_mean_ms = {statistics.mean(p2):.2f}\n")
PY

  # Sanity: count cached_token > 0 in the prefill log.
  hit_lines=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  echo "[$arm] prefill batches with cached-token > 0: $hit_lines"

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

run_arm recency
run_arm hpb

echo
echo "=== compare ==="
.venv/bin/python <<'PY'
import os
def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"): continue
        k, _, v = line.partition("=")
        d[k.strip()] = float(v.strip())
    return d
import sys
out_dir = os.environ.get('OUT_DIR') or sys.argv[1] if len(sys.argv) > 1 else None
PY
.venv/bin/python <<PY
def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"): continue
        k, _, v = line.partition("=")
        d[k.strip()] = float(v.strip())
    return d
r = load("$OUT_DIR/recency_results.txt")
h = load("$OUT_DIR/hpb_results.txt")
print(f"\n{'metric':<20} {'recency':>10} {'hpb':>10} {'delta%':>10}")
print('-' * 55)
for k in ('prompt_tokens', 'pulse1_mean_ms', 'burst_mean_ms', 'pulse2_mean_ms'):
    rv = r.get(k); hv = h.get(k)
    if rv is None or hv is None:
        print(f"{k:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        continue
    delta = (hv - rv) / rv * 100 if rv else float('inf')
    marker = ""
    if k == 'pulse2_mean_ms':
        marker = "  ←HEADLINE" if delta < -2 else "  (no benefit)" if abs(delta) < 2 else "  (worse)"
    print(f"{k:<20} {rv:>10.2f} {hv:>10.2f} {delta:>+9.2f}%{marker}")
PY
