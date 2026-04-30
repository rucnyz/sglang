#!/bin/bash
# Quality preservation Q3 — per-task classification accuracy.
#
# Sends 100 simple multiple-choice questions to two server arms at
# temperature=0 and computes per-arm accuracy by checking whether the
# generated answer contains the correct letter (A/B/C/D).
#
# Pass criterion: accuracy delta < 1 percentage point between default and
# full prelude. (At temperature=0 with seed=0 we expect EXACTLY the same
# accuracy because Q1 already showed byte-identity, but this gives a
# downstream-task quality number for the paper.)

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/q3_classify_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

QUESTIONS_FILE="$OUT_DIR/questions.jsonl"
.venv/bin/python <<PY
import json
qs = [
    ("Which of the following is a valid HTTP status code for 'Not Found'?\nA) 200\nB) 301\nC) 404\nD) 500\nAnswer:", "C"),
    ("Which sorting algorithm has worst-case time complexity O(n log n)?\nA) Bubble sort\nB) Merge sort\nC) Insertion sort\nD) Selection sort\nAnswer:", "B"),
    ("Which programming paradigm is Haskell primarily associated with?\nA) Procedural\nB) Object-oriented\nC) Functional\nD) Logic\nAnswer:", "C"),
    ("What does SQL stand for?\nA) Standard Query Language\nB) Structured Query Language\nC) Sequential Query Language\nD) Server Query Language\nAnswer:", "B"),
    ("Which of these is NOT a JavaScript primitive type?\nA) string\nB) number\nC) array\nD) boolean\nAnswer:", "C"),
    ("In which year was Python first released?\nA) 1989\nB) 1991\nC) 1995\nD) 2000\nAnswer:", "B"),
    ("Which data structure uses LIFO ordering?\nA) Queue\nB) Stack\nC) Heap\nD) Tree\nAnswer:", "B"),
    ("Which keyword is used to define a constant in JavaScript?\nA) var\nB) let\nC) const\nD) static\nAnswer:", "C"),
    ("What is the time complexity of binary search?\nA) O(n)\nB) O(log n)\nC) O(n log n)\nD) O(n^2)\nAnswer:", "B"),
    ("Which of the following is a NoSQL database?\nA) PostgreSQL\nB) MongoDB\nC) MySQL\nD) Oracle\nAnswer:", "B"),
    ("What does TCP stand for?\nA) Transfer Control Protocol\nB) Transmission Control Protocol\nC) Transit Control Protocol\nD) Transport Control Protocol\nAnswer:", "B"),
    ("Which OSI layer handles routing?\nA) Physical\nB) Data Link\nC) Network\nD) Transport\nAnswer:", "C"),
    ("Which of these is a NoSQL database type?\nA) Document\nB) Relational\nC) Hierarchical\nD) Network\nAnswer:", "A"),
    ("What does CPU stand for?\nA) Central Processing Unit\nB) Computer Processing Unit\nC) Central Program Unit\nD) Computer Program Unit\nAnswer:", "A"),
    ("Which symbol is used for single-line comments in Python?\nA) //\nB) /* */\nC) #\nD) --\nAnswer:", "C"),
    ("Which of these is NOT an HTTP method?\nA) GET\nB) POST\nC) FETCH\nD) DELETE\nAnswer:", "C"),
    ("Which company developed the Go programming language?\nA) Microsoft\nB) Google\nC) Apple\nD) Facebook\nAnswer:", "B"),
    ("Which Linux command lists files?\nA) ls\nB) dir\nC) list\nD) show\nAnswer:", "A"),
    ("What does API stand for?\nA) Application Programming Interface\nB) Application Process Interface\nC) Algorithm Programming Interface\nD) Advanced Programming Interface\nAnswer:", "A"),
    ("Which language is primarily used for iOS development?\nA) Java\nB) Swift\nC) Kotlin\nD) C#\nAnswer:", "B"),
    ("Which of these is a hash function?\nA) AES\nB) RSA\nC) SHA-256\nD) DES\nAnswer:", "C"),
    ("What does HTML stand for?\nA) HyperText Markup Language\nB) HighText Markup Language\nC) Hyperlinks and Text Markup Language\nD) Home Tool Markup Language\nAnswer:", "A"),
    ("Which of these is a JavaScript framework?\nA) Django\nB) React\nC) Flask\nD) Rails\nAnswer:", "B"),
    ("What is 2^10?\nA) 512\nB) 1000\nC) 1024\nD) 2048\nAnswer:", "C"),
    ("Which sorting algorithm is in-place and stable?\nA) Quicksort\nB) Mergesort\nC) Insertion sort\nD) Heapsort\nAnswer:", "C"),
    ("Which port does HTTPS use by default?\nA) 80\nB) 443\nC) 8080\nD) 22\nAnswer:", "B"),
    ("What is the Big-O of inserting at the end of a dynamic array (amortized)?\nA) O(1)\nB) O(log n)\nC) O(n)\nD) O(n log n)\nAnswer:", "A"),
    ("Which of these languages is statically typed?\nA) Python\nB) Java\nC) JavaScript\nD) Ruby\nAnswer:", "B"),
    ("Which command is used to clone a Git repository?\nA) git pull\nB) git fetch\nC) git clone\nD) git init\nAnswer:", "C"),
    ("What does JSON stand for?\nA) JavaScript Object Notation\nB) Java Standard Object Notation\nC) JavaScript Online Notation\nD) Java Script Open Notation\nAnswer:", "A"),
    ("Which paradigm does SQL primarily use?\nA) Imperative\nB) Declarative\nC) Object-oriented\nD) Functional\nAnswer:", "B"),
    ("What does RAM stand for?\nA) Random Access Memory\nB) Read Access Memory\nC) Rapid Access Memory\nD) Read-Allocated Memory\nAnswer:", "A"),
    ("Which of these is a graph traversal algorithm?\nA) Dijkstra's\nB) BFS\nC) Floyd-Warshall\nD) All of the above\nAnswer:", "D"),
    ("Which port does SSH typically use?\nA) 21\nB) 22\nC) 23\nD) 25\nAnswer:", "B"),
    ("What does GPU stand for?\nA) Graphics Processing Unit\nB) General Processing Unit\nC) Graphical Programming Unit\nD) Game Processing Unit\nAnswer:", "A"),
    ("Which of these is a containerization tool?\nA) Docker\nB) Vagrant\nC) Ansible\nD) Terraform\nAnswer:", "A"),
    ("Which is a type of join in SQL?\nA) INNER\nB) OUTER\nC) CROSS\nD) All of the above\nAnswer:", "D"),
    ("Which language uses indentation to define code blocks?\nA) Java\nB) Python\nC) C\nD) Go\nAnswer:", "B"),
    ("What does VPN stand for?\nA) Virtual Private Network\nB) Verified Private Network\nC) Virtual Public Network\nD) Verified Public Network\nAnswer:", "A"),
    ("Which AWS service is for object storage?\nA) EC2\nB) S3\nC) RDS\nD) Lambda\nAnswer:", "B"),
    ("Which protocol is used for sending email?\nA) HTTP\nB) FTP\nC) SMTP\nD) SSH\nAnswer:", "C"),
    ("Which paradigm does Prolog use?\nA) Object-oriented\nB) Functional\nC) Logic\nD) Procedural\nAnswer:", "C"),
    ("Which company developed Kubernetes originally?\nA) Microsoft\nB) Google\nC) Red Hat\nD) IBM\nAnswer:", "B"),
    ("What is the smallest unit of digital information?\nA) bit\nB) byte\nC) word\nD) nibble\nAnswer:", "A"),
    ("Which of these is a relational database?\nA) Redis\nB) Cassandra\nC) PostgreSQL\nD) Neo4j\nAnswer:", "C"),
    ("Which Linux signal terminates a process by default?\nA) SIGSTOP\nB) SIGTERM\nC) SIGCONT\nD) SIGUSR1\nAnswer:", "B"),
    ("Which year was the C language created?\nA) 1969\nB) 1972\nC) 1980\nD) 1985\nAnswer:", "B"),
    ("Which of these is NOT a CSS unit?\nA) px\nB) em\nC) rem\nD) sx\nAnswer:", "D"),
    ("Which is a memory-allocation function in C?\nA) malloc\nB) alloc\nC) memnew\nD) new\nAnswer:", "A"),
    ("Which of these is a CDN?\nA) Cloudflare\nB) Github\nC) Slack\nD) Notion\nAnswer:", "A"),
]
with open("$QUESTIONS_FILE", "w") as f:
    for q, ans in qs:
        f.write(json.dumps({"q": q, "answer": ans}) + "\n")
print(f"wrote {len(qs)} questions to $QUESTIONS_FILE")
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

  PORT=$PORT MODEL=$MODEL QUESTIONS_FILE=$QUESTIONS_FILE OUT=$outputs \
    .venv/bin/python <<'PY'
import json, urllib.request, os
PORT = os.environ['PORT']; MODEL = os.environ['MODEL']
qs = [json.loads(l) for l in open(os.environ['QUESTIONS_FILE']) if l.strip()]
results = []
for i, item in enumerate(qs):
    body = json.dumps({'model': MODEL, 'prompt': item['q'],
                       'max_tokens': 16, 'temperature': 0,
                       'seed': 0}).encode()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            f'http://127.0.0.1:{PORT}/v1/completions', data=body,
            headers={'Content-Type': 'application/json'}),
            timeout=120).read())
        text = r['choices'][0]['text']
    except Exception as e:
        text = f"ERROR: {e}"
    results.append({'idx': i, 'expected': item['answer'], 'text': text})
with open(os.environ['OUT'], 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print(f"  wrote {len(results)} answers")
PY

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

run_arm default
run_arm prelude

echo
echo "=== Q3 classification accuracy comparison ==="
.venv/bin/python <<PY
import json, re
out = "$OUT_DIR"
def load(name):
    return [json.loads(l) for l in open(f"{out}/{name}_outputs.jsonl") if l.strip()]
def grade(records):
    correct = 0
    for r in records:
        text = r['text'].strip().upper()
        # Look for first occurrence of A/B/C/D (possibly inside parens or after a colon).
        m = re.search(r'\b([ABCD])\b', text)
        if m and m.group(1) == r['expected']:
            correct += 1
    return correct
default = load("default")
prelude = load("prelude")
n = min(len(default), len(prelude))
d_correct = grade(default[:n])
p_correct = grade(prelude[:n])
print(f"sample size: {n}")
print(f"default accuracy: {d_correct}/{n} ({100*d_correct/n:.1f}%)")
print(f"prelude accuracy: {p_correct}/{n} ({100*p_correct/n:.1f}%)")
print(f"delta: {d_correct - p_correct} answers ({100*(d_correct - p_correct)/n:+.1f} pp)")
identical = sum(1 for d, p in zip(default[:n], prelude[:n]) if d['text'] == p['text'])
print(f"byte-identical answers: {identical}/{n}")
PY
