#!/bin/bash
# Phase 3.a eval v4 — cold-burst with REAL ShareGPT prompts.
#
# ShareGPT V3 dataset (locally cached at
# /data/yuzhou/.cache/huggingface/hub/...) has thousands of multi-turn
# conversations in natural English. We pick one whose concatenated
# content tokenizes to >8192 tokens (so chunked prefill creates a
# snapshot node) and use it as the shared "system" prefix.
#
# Workload:
#   Pulse 1: 8 different downstream questions about the document
#            (each with doc as prefix). After turn 1, the chunk-boundary
#            snapshot is created and subsequent turns hit at depth=8192.
#   Cold burst: 25 OTHER ShareGPT conversations as unique prefixes.
#            Floods the radix; recency-LRU evicts the original snapshot
#            when mamba slots saturate.
#   Pulse 2: 8 more downstream questions about the original document.
#            HPB-LRU expects cache HIT (snapshot preserved); recency-LRU
#            expects cache MISS (snapshot evicted by burst).
#
# Pass criterion: HPB Pulse 2 mean latency materially lower than recency
# Pulse 2 (the headline §4.2 effect).

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
SHAREGPT=${SHAREGPT:-/data/yuzhou/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json}
OUT_DIR=/tmp/hpb_sharegpt_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"
echo "sharegpt_path=$SHAREGPT"

if [ ! -f "$SHAREGPT" ]; then
  echo "FAIL: ShareGPT dataset not found at $SHAREGPT"
  exit 1
fi

# Pre-process the dataset once: pick one long convo as "document" and
# 25 short convos as cold-burst prompts. Save to JSON for both arms to
# reuse identically.
WORKLOAD="$OUT_DIR/workload.json"
SHAREGPT=$SHAREGPT WORKLOAD=$WORKLOAD .venv/bin/python <<'PY'
import json, os, random
random.seed(42)

src = json.load(open(os.environ["SHAREGPT"]))
print(f"loaded {len(src)} conversations")

# Find one convo whose 1st-turn content is long (~10-20K chars
# = roughly 7-15K tokens for natural English).
doc = None
for conv in src:
    convs = conv.get("conversations") or []
    if not convs:
        continue
    first = convs[0].get("value", "")
    if 12000 <= len(first) <= 30000:
        doc = first
        break
if doc is None:
    # Concatenate consecutive turns from one convo until ~15K chars.
    for conv in src:
        convs = conv.get("conversations") or []
        joined = "\n\n".join(c.get("value", "") for c in convs)
        if 12000 <= len(joined) <= 30000:
            doc = joined
            break
assert doc is not None, "couldn't find long enough ShareGPT convo"
print(f"document chars: {len(doc)}")

# Cold-burst: 25 OTHER convos' first turns, length 6-12K each.
burst = []
for conv in src:
    if len(burst) >= 25:
        break
    convs = conv.get("conversations") or []
    if not convs:
        continue
    first = convs[0].get("value", "")
    if 6000 <= len(first) <= 12000 and first != doc:
        burst.append(first)
print(f"burst prompts: {len(burst)}, lengths {[len(b) for b in burst[:5]]}")

json.dump({"document": doc, "burst": burst}, open(os.environ["WORKLOAD"], "w"))
print(f"workload saved to {os.environ['WORKLOAD']}")
PY

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

  WORKLOAD=$WORKLOAD RESULTS=$results PORT=$PORT MODEL=$MODEL .venv/bin/python <<'PY'
import json, urllib.request, time, os, threading, statistics, random
random.seed(0)
PORT = int(os.environ["PORT"])
MODEL = os.environ["MODEL"]
RESULTS = os.environ["RESULTS"]
WORKLOAD = os.environ["WORKLOAD"]

w = json.load(open(WORKLOAD))
DOC = w["document"]
BURST = w["burst"]
print(f"  document chars: {len(DOC)}, burst count: {len(BURST)}")

def fire(prompt, mark, results):
    t0 = time.time()
    data = json.dumps({
        'model': MODEL, 'prompt': prompt,
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

# Pulse 1: 8 questions sharing the document.
print(">> Pulse 1 (build): 8 questions about the document")
results = []
for i in range(8):
    prompt = DOC + f"\n\n---\nQ{i}: in one word, what is this about?"
    fire(prompt, 'pulse1', results)
    time.sleep(0.3)
p1 = [r[1] for r in results if r[0] == 'pulse1' and r[1] > 0]
p1_tokens = [r[2] for r in results if r[0] == 'pulse1' and r[1] > 0]
print(f"  pulse1: n={len(p1)}, mean_latency={statistics.mean(p1):.1f} ms, "
      f"prompt_tokens={p1_tokens[0] if p1_tokens else '?'}")

# Cold burst: 25 unique full ShareGPT prompts in parallel.
print(f">> Cold burst: {len(BURST)} unique ShareGPT prompts")
threads = []
for i, b_prompt in enumerate(BURST):
    t = threading.Thread(target=fire, args=(b_prompt, 'burst', results), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(0.5)
for t in threads:
    t.join(timeout=300)
b_lat = [r[1] for r in results if r[0] == 'burst' and r[1] > 0]
print(f"  burst: n={len(b_lat)}, mean_latency={statistics.mean(b_lat):.1f} ms")

time.sleep(8)

# Pulse 2: 8 more questions about the same document.
print(">> Pulse 2 (post-burst): 8 questions about the document")
for i in range(8):
    prompt = DOC + f"\n\n---\nQ-recovered{i}: name something interesting:"
    fire(prompt, 'pulse2', results)
    time.sleep(0.3)
p2 = [r[1] for r in results if r[0] == 'pulse2' and r[1] > 0]
print(f"  pulse2: n={len(p2)}, mean_latency={statistics.mean(p2):.1f} ms")

with open(RESULTS, 'w') as f:
    f.write(f"# arm = {os.environ.get('SGLANG_HPB_LRU', '0')}\n")
    f.write(f"prompt_tokens = {p1_tokens[0] if p1_tokens else 0}\n")
    f.write(f"pulse1_mean_ms = {statistics.mean(p1):.2f}\n")
    f.write(f"burst_mean_ms = {statistics.mean(b_lat):.2f}\n")
    f.write(f"pulse2_mean_ms = {statistics.mean(p2):.2f}\n")
PY

  hit_lines=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  total_prefills=$(grep -c "Prefill batch" "$log" || true)
  echo "[$arm] prefill batches: $total_prefills, with cached-token > 0: $hit_lines"

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

run_arm recency
run_arm hpb

echo
echo "=== compare ==="
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
print(f"\n{'metric':<22} {'recency':>10} {'hpb':>10} {'delta%':>10}")
print('-' * 60)
for k in ('prompt_tokens', 'pulse1_mean_ms', 'burst_mean_ms', 'pulse2_mean_ms'):
    rv = r.get(k); hv = h.get(k)
    if rv is None or hv is None:
        print(f"{k:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        continue
    delta = (hv - rv) / rv * 100 if rv else float('inf')
    marker = ""
    if k == 'pulse2_mean_ms':
        if delta < -5:
            marker = "  ←HEADLINE PASS"
        elif abs(delta) < 5:
            marker = "  (no benefit, paper effect not reproduced)"
        else:
            marker = "  ←HPB SLOWER"
    print(f"{k:<22} {rv:>10.2f} {hv:>10.2f} {delta:>+9.2f}%{marker}")
PY
