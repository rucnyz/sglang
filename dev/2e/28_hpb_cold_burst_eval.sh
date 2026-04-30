#!/bin/bash
# Phase 3.a eval — empirical cold-burst test (paper §4.2 narrative).
#
# Reproduces the failure mode and the fix on a real Qwen3.5-35B-A3B
# instance:
#   t=0..30s   "Pulse 1": 30 prompts sharing a 1500-token system prefix.
#               Cache builds up; system-prompt prefix node accumulates hits.
#   t=30..50s  "Cold burst": 60 unique-prefix prompts at 3 RPS.
#               Recency LRU evicts the (older, but high-hit) system prompt.
#               HPB LRU preserves it (low priority for cold leaves).
#   t=50..80s  "Pulse 2": 30 prompts with same system prefix.
#               Without HPB: cache miss → high TTFT (re-prefill).
#               With HPB:    cache hit  → low TTFT (prefix matched).
#
# We run two arms back-to-back (recency LRU vs HPB LRU) on the same
# server config and compare Pulse 2 mean / median TTFT.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
OUT_DIR=/tmp/hpb_cold_burst_$$
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
# Need a system prefix LONGER than chunked_prefill_size (=8192) so the
# engine actually inserts into the radix cache via cache_unfinished_req
# at the chunk boundary. ~12000 tokens of repeated text:
SYS_PREFIX = "You are a helpful assistant. Always be concise. " * 600  # ~12000 tokens

def fire(prompt, mark, results):
    t0 = time.time()
    data = json.dumps({
        'model': '$MODEL', 'prompt': prompt,
        'max_tokens': 16, 'temperature': 0,
        'stream': False,
    }).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=300).read())
        elapsed = (time.time() - t0) * 1000
        # SGLang exposes per-request 'usage' but not TTFT in the openai
        # response. Use total request latency as a proxy; for short max_tokens
        # this is dominated by prefill (= TTFT-equivalent for our test).
        results.append((mark, elapsed, body['usage']['total_tokens']))
    except Exception as e:
        results.append((mark, -1, str(e)))

# Pulse 1: build the cache. ~12K-token system prefix sent 10 times
# (smaller count because each request now does a real 12K prefill;
# we don't need many — one is enough to populate the radix).
print(">> Pulse 1 (build): 10 prompts sharing 12K-token system prefix")
results = []
for i in range(10):
    prompt = SYS_PREFIX + f"Question {i}: name a fruit:"
    fire(prompt, 'pulse1', results)
    time.sleep(0.3)
p1 = [r[1] for r in results if r[0] == 'pulse1' and r[1] > 0]
print(f"  pulse1: n={len(p1)}, mean_latency={statistics.mean(p1):.1f} ms, "
      f"median={statistics.median(p1):.1f} ms")

# Cold burst: many unique prompts, each 12K+ tokens, so they push hard
# on the cache and force eviction. We use 50 burst prompts at 2 RPS.
print(">> Cold burst: 50 unique 12K-prefix prompts (~25s)")
threads = []
for i in range(50):
    # Each burst prompt has a unique 12K-token prefix. Floods the cache.
    UNIQUE = (f"Topic {i}: " + " ".join(f"word{i}{j}" for j in range(2000))
              + f" -- answer: what color is {i}?")
    t = threading.Thread(target=fire, args=(UNIQUE, 'burst', results), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(0.5)  # 2 RPS
for t in threads:
    t.join(timeout=300)
b = [r[1] for r in results if r[0] == 'burst' and r[1] > 0]
print(f"  burst: n={len(b)}, mean_latency={statistics.mean(b):.1f} ms")

# Idle drain.
time.sleep(8)

# Pulse 2: should hit the cache if the system prefix survived the burst.
print(">> Pulse 2 (post-burst): 10 prompts with same system prefix as Pulse 1")
for i in range(10):
    prompt = SYS_PREFIX + f"Quiz {i}: capital of country {i % 5}?"
    fire(prompt, 'pulse2', results)
    time.sleep(0.3)
p2 = [r[1] for r in results if r[0] == 'pulse2' and r[1] > 0]
print(f"  pulse2: n={len(p2)}, mean_latency={statistics.mean(p2):.1f} ms, "
      f"median={statistics.median(p2):.1f} ms")

with open(RESULTS, 'w') as f:
    f.write(f"# arm = {os.environ.get('SGLANG_HPB_LRU', '0')}\n")
    f.write(f"pulse1_mean_ms = {statistics.mean(p1):.2f}\n")
    f.write(f"pulse1_median_ms = {statistics.median(p1):.2f}\n")
    f.write(f"burst_mean_ms = {statistics.mean(b):.2f}\n")
    f.write(f"pulse2_mean_ms = {statistics.mean(p2):.2f}\n")
    f.write(f"pulse2_median_ms = {statistics.median(p2):.2f}\n")
    f.write(f"# pulse2 is the headline number — system-prefix re-use after cold burst\n")
PY

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

run_arm recency
run_arm hpb

echo
echo "=== compare ==="
.venv/bin/python <<PY
import os
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
print(f"\n{'metric':<25} {'recency':>10} {'hpb':>10} {'delta%':>10}")
print('-' * 60)
for k in ('pulse1_mean_ms', 'pulse1_median_ms', 'burst_mean_ms',
         'pulse2_mean_ms', 'pulse2_median_ms'):
    rv = r.get(k); hv = h.get(k)
    if rv is None or hv is None:
        print(f"{k:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        continue
    delta = (hv - rv) / rv * 100 if rv else float('inf')
    marker = ""
    if 'pulse2' in k:
        marker = "  ←HEADLINE" if delta < -2 else "  (no benefit)" if abs(delta) < 2 else "  (worse)"
    print(f"{k:<25} {rv:>10.2f} {hv:>10.2f} {delta:>+9.2f}%{marker}")
print("\nHEADLINE: pulse2_mean_ms compares post-burst latency.")
print("If HPB row is materially lower than recency row, paper §4.2 reproduces:")
print("  HPB preserves the system-prompt big page across the cold burst.")
PY
