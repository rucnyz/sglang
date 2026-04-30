#!/bin/bash
# Phase 3.a eval v5 — cold-burst with vLLM-style 20K-token long documents.
#
# Adapts vLLM's official benchmark_long_document_qa_throughput.py
# prompt-construction (str(i) + "hi "*20000 → ~20K tokens, no
# tokenizer compression because "hi" is one token) to SGLang's serving
# endpoint. Each document is well above chunked_prefill_size=8192 so
# chunk-boundary mamba_value snapshots ARE created.
#
# Workload:
#   Pulse 1: 8 unique 20K-token docs, sent once each.
#   Cold burst: 30 OTHER unique 20K-token docs (different ids).
#   Pulse 2: same 8 docs from Pulse 1, sent once each.
#
# Pass criterion (all):
#   - Pulse 1 has #cached-token=0 (cold start)
#   - At least some burst requests have #cached-token=0 (new docs)
#   - Pulse 2 should hit the chunk-boundary snapshots IF the cache
#     eviction policy preserved them.
#   - HPB Pulse 2 mean latency materially LOWER than recency Pulse 2.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
DOC_TOKENS=${DOC_TOKENS:-20000}
NUM_DOCS=${NUM_DOCS:-8}
NUM_BURST=${NUM_BURST:-30}
OUT_DIR=/tmp/hpb_longdoc_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR  doc_tokens=$DOC_TOKENS  num_docs=$NUM_DOCS  num_burst=$NUM_BURST"

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "hpb" ]; then
    extra_env="SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=300.0"
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

  RESULTS=$results PORT=$PORT MODEL=$MODEL DOC_TOKENS=$DOC_TOKENS NUM_DOCS=$NUM_DOCS NUM_BURST=$NUM_BURST .venv/bin/python <<'PY'
import json, urllib.request, time, os, threading, statistics
PORT = int(os.environ["PORT"])
MODEL = os.environ["MODEL"]
RESULTS = os.environ["RESULTS"]
DOC_TOKENS = int(os.environ["DOC_TOKENS"])
NUM_DOCS = int(os.environ["NUM_DOCS"])
NUM_BURST = int(os.environ["NUM_BURST"])

def make_doc(doc_id: str) -> str:
    """Same construction as vLLM's benchmark_long_document_qa_throughput
    (each "hi " is exactly 1 token in Qwen tokenizer). The doc_id at
    the front prevents cross-doc prefix sharing, matching vLLM's
    intent."""
    return doc_id + " ".join(["hi"] * DOC_TOKENS) + "\n\n# answer in one word:"

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
        body = json.loads(urllib.request.urlopen(req, timeout=600).read())
        elapsed = (time.time() - t0) * 1000
        results.append((mark, elapsed, body['usage']['total_tokens']))
    except Exception as e:
        results.append((mark, -1, str(e)))

# Pulse 1: 8 unique docs, sent once each.
print(f">> Pulse 1 (build): {NUM_DOCS} unique 20K-token docs")
results = []
docs = [make_doc(f"doc{i:03d}") for i in range(NUM_DOCS)]
for i, prompt in enumerate(docs):
    fire(prompt, 'pulse1', results)
    print(f"    pulse1 doc {i}: latency={results[-1][1]:.1f}ms tokens={results[-1][2] if results[-1][1] > 0 else 'FAIL'}")
p1 = [r[1] for r in results if r[0] == 'pulse1' and r[1] > 0]
p1_tokens = [r[2] for r in results if r[0] == 'pulse1' and r[1] > 0]
print(f"  pulse1: n={len(p1)}, mean_latency={statistics.mean(p1):.1f} ms, "
      f"prompt_tokens={p1_tokens[0] if p1_tokens else '?'}")

# Cold burst: 30 unique docs, sent in parallel at moderate rate.
print(f">> Cold burst: {NUM_BURST} unique 20K-token docs")
threads = []
burst_docs = [make_doc(f"burst{i:03d}") for i in range(NUM_BURST)]
for i, prompt in enumerate(burst_docs):
    t = threading.Thread(target=fire, args=(prompt, 'burst', results), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(0.5)
for t in threads:
    t.join(timeout=600)
b_lat = [r[1] for r in results if r[0] == 'burst' and r[1] > 0]
print(f"  burst: n={len(b_lat)}, mean_latency={statistics.mean(b_lat):.1f} ms")

time.sleep(8)

# Pulse 2: same 8 docs as Pulse 1.
print(f">> Pulse 2 (post-burst): {NUM_DOCS} same docs as Pulse 1")
for i, prompt in enumerate(docs):
    fire(prompt, 'pulse2', results)
    print(f"    pulse2 doc {i}: latency={results[-1][1]:.1f}ms tokens={results[-1][2] if results[-1][1] > 0 else 'FAIL'}")
p2 = [r[1] for r in results if r[0] == 'pulse2' and r[1] > 0]
print(f"  pulse2: n={len(p2)}, mean_latency={statistics.mean(p2):.1f} ms")

with open(RESULTS, 'w') as f:
    f.write(f"# arm = {os.environ.get('SGLANG_HPB_LRU', '0')}\n")
    f.write(f"prompt_tokens = {p1_tokens[0] if p1_tokens else 0}\n")
    f.write(f"pulse1_mean_ms = {statistics.mean(p1):.2f}\n")
    f.write(f"pulse1_median_ms = {statistics.median(p1):.2f}\n")
    f.write(f"burst_mean_ms = {statistics.mean(b_lat):.2f}\n")
    f.write(f"pulse2_mean_ms = {statistics.mean(p2):.2f}\n")
    f.write(f"pulse2_median_ms = {statistics.median(p2):.2f}\n")
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
for k in ('prompt_tokens', 'pulse1_mean_ms', 'pulse1_median_ms',
          'burst_mean_ms', 'pulse2_mean_ms', 'pulse2_median_ms'):
    rv = r.get(k); hv = h.get(k)
    if rv is None or hv is None:
        print(f"{k:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        continue
    delta = (hv - rv) / rv * 100 if rv else float('inf')
    marker = ""
    if k.startswith('pulse2'):
        if delta < -5:
            marker = "  ←HPB FASTER"
        elif abs(delta) < 5:
            marker = "  (no benefit)"
        else:
            marker = "  ←HPB SLOWER"
    print(f"{k:<22} {rv:>10.2f} {hv:>10.2f} {delta:>+9.2f}%{marker}")
print()
print("Headline: pulse2 — same docs as pulse1, after a cold burst that")
print("may evict their snapshots under recency LRU. HPB should preserve")
print("the high-hit pulse1 docs and produce lower pulse2 latency.")
PY
