#!/bin/bash
# Quality preservation Q2 — seeded sampling determinism at temperature=1.0.
#
# Q1 verified that at temperature=0 the prelude system is bit-exact to
# the engine baseline. This Q2 extension checks the next-most-stringent
# claim: with the same seed, sampled decoding (temperature=1.0) is also
# bit-exact, because the sampling RNG state is deterministic given the
# seed. If Layer 1's eviction or Layer 2's cross-pool transfers
# contaminate the per-request RNG, this would break.
#
# Sends 50 prompts × 3 seeds (0, 7, 42) at temperature=1.0 to two server
# configurations. Pass criterion: 100% byte-identical outputs across
# arms for the same (prompt, seed) pair.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/q2_seeded_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

PROMPTS_FILE="$OUT_DIR/prompts.jsonl"
.venv/bin/python <<PY
import json
prompts = [
    "Explain the concept of recursion in computer science.",
    "What are three differences between SQL and NoSQL databases?",
    "Write a Python function that returns the nth Fibonacci number.",
    "Describe the architecture of a Transformer model.",
    "What does a kernel scheduler do in an operating system?",
    "Explain why CAP theorem matters for distributed systems.",
    "How does CUDA differ from OpenCL?",
    "Define the difference between batch and stream processing.",
    "What is the role of attention in deep learning models?",
    "Explain how garbage collection works in JVM.",
    "How does TCP differ from UDP at a protocol level?",
    "What is a hash table and what is its average lookup complexity?",
    "Describe the Raft consensus algorithm in three sentences.",
    "Explain how deduplication works in a backup system.",
    "What is the difference between latency and throughput?",
    "Why is column-oriented storage better for analytical queries?",
    "Explain the read-copy-update synchronization primitive.",
    "What is the role of the page table in a CPU's MMU?",
    "How does asynchronous I/O differ from synchronous I/O?",
    "Describe the role of the L2 cache in a multi-core CPU.",
    "What is a memory barrier and when do you need one?",
    "Explain how a B-tree is balanced during insertion.",
    "Define MapReduce and give one example.",
    "What is the role of the loopback adapter in TCP/IP?",
    "Explain the difference between a process and a thread.",
    "What is virtual memory and why is it useful?",
    "Describe the operation of an LRU cache.",
    "What is checkpointing in distributed systems?",
    "Explain how a Bloom filter trades space for false positives.",
    "Define eventual consistency in distributed databases.",
    "What is the role of the BIOS in a PC?",
    "Explain the difference between RISC and CISC.",
    "What is a write-ahead log?",
    "Describe how DHT routing works in a P2P network.",
    "Explain the difference between mutex and semaphore.",
    "What is the role of an inode in Unix file systems?",
    "Define a thread pool and explain its benefits.",
    "What is the difference between unicast and multicast?",
    "Explain how CSMA/CD works in Ethernet.",
    "What is a graph traversal and give two examples.",
    "Describe the difference between hash join and merge join.",
    "Explain how copy-on-write fork() works.",
    "What is a fence in lock-free programming?",
    "Define eventual consistency vs strong consistency.",
    "What does the OSI model's transport layer do?",
    "Explain the difference between a DAG and a tree.",
    "What is the consensus protocol used by Bitcoin?",
    "Describe how Merkle trees enable verification.",
    "What is the role of a distributed file system?",
    "Explain the trade-offs between row and column storage.",
]
with open("$PROMPTS_FILE", "w") as f:
    for p in prompts:
        f.write(json.dumps({"prompt": p}) + "\n")
print(f"wrote {len(prompts)} prompts")
PY

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "prelude" ]; then
    extra_env="SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=1073741824 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_TICK_S=2.0"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  local outputs="$OUT_DIR/${arm}_outputs.jsonl"
  echo "=== arm=$arm ($extra_env) ==="

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

  PORT=$PORT MODEL=$MODEL PROMPTS_FILE=$PROMPTS_FILE OUT=$outputs \
    .venv/bin/python <<'PY'
import json, urllib.request, os
PORT = os.environ['PORT']; MODEL = os.environ['MODEL']
prompts = [json.loads(l)['prompt'] for l in open(os.environ['PROMPTS_FILE']) if l.strip()]
results = []
for i, p in enumerate(prompts):
    for seed in (0, 7, 42):
        body = json.dumps({'model': MODEL, 'prompt': p,
                           'max_tokens': 64, 'temperature': 1.0,
                           'top_p': 0.95, 'seed': seed}).encode()
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                headers={'Content-Type': 'application/json'}),
                timeout=120).read())
            results.append({'prompt_idx': i, 'seed': seed,
                            'text': r['choices'][0]['text']})
        except Exception as e:
            results.append({'prompt_idx': i, 'seed': seed,
                            'error': str(e)})
with open(os.environ['OUT'], 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print(f"  wrote {len(results)} (prompt, seed) outputs")
PY

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

run_arm default
run_arm prelude

echo
echo "=== Q2 seeded-sampling comparison ==="
.venv/bin/python <<PY
import json
out = "$OUT_DIR"
def load(name):
    return [json.loads(l) for l in open(f"{out}/{name}_outputs.jsonl") if l.strip()]
default = load("default")
prelude = load("prelude")
n = min(len(default), len(prelude))
identical = 0
diffs = []
for i in range(n):
    d, p = default[i], prelude[i]
    if d.get("error") or p.get("error"):
        diffs.append(("error", i, d.get("error", "")[:100], p.get("error", "")[:100]))
        continue
    if d['text'] == p['text']:
        identical += 1
    else:
        diffs.append(("diverge", i, d['text'][:80], p['text'][:80]))
print(f"comparisons: {n}, byte-identical: {identical}/{n} ({100*identical/n:.1f}%)")
if diffs:
    print(f"first {min(3, len(diffs))} divergences:")
    for tag, i, d, p in diffs[:3]:
        meta = default[i] if i < len(default) else {}
        print(f"  [{tag}] (prompt={meta.get('prompt_idx','?')}, seed={meta.get('seed','?')})")
        print(f"    default: {d!r}")
        print(f"    prelude: {p!r}")
PY
