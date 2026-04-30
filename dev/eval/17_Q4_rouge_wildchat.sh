#!/bin/bash
# Quality preservation Q4 — ROUGE-L against reference (wildchat as XSum substitute).
#
# XSum isn't local; we use wildchat assistant responses as references
# (already in dev/eval/datasets/phase_c.json). Sends each conversation's
# first user message at temperature=1.0 with seeds 0/7/42 to default
# vs full prelude (90 samples per arm), then computes ROUGE-L F1 of
# each generated response against the wildchat reference assistant
# response.
#
# Pass criterion: mean ROUGE-L between arms within 1 percentage point.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/q4_rouge_$$
mkdir -p "$OUT_DIR"

# Build prompts file from wildchat phase_c.json: take first user message
# and the assistant's first reply as reference.
PROMPTS_FILE="$OUT_DIR/prompts.jsonl"
.venv/bin/python <<PY
import json
data = json.load(open("/scratch/yuzhou/projects/sglang/dev/eval/datasets/phase_c.json"))
out = []
for conv in data:
    msgs = conv.get("messages", [])
    if len(msgs) < 2: continue
    user = msgs[0].get("content", "") if msgs[0].get("role") == "user" else ""
    assistant = msgs[1].get("content", "") if msgs[1].get("role") == "assistant" else ""
    if not user or not assistant: continue
    if len(user) < 50 or len(user) > 4000: continue
    if len(assistant) < 50 or len(assistant) > 2000: continue
    out.append({"prompt": user, "reference": assistant})
    if len(out) == 30: break
with open("$PROMPTS_FILE", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(out)} (prompt, reference) pairs to $PROMPTS_FILE")
PY

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "prelude" ]; then
    extra_env="SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=1073741824 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_TICK_S=2.0"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  local outputs="$OUT_DIR/${arm}_outputs.jsonl"
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
prompts = [json.loads(l) for l in open(os.environ['PROMPTS_FILE']) if l.strip()]
results = []
for i, item in enumerate(prompts):
    for seed in (0, 7, 42):
        body = json.dumps({'model': MODEL, 'prompt': item['prompt'],
                           'max_tokens': 256, 'temperature': 1.0,
                           'top_p': 0.95, 'seed': seed}).encode()
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                headers={'Content-Type': 'application/json'}),
                timeout=180).read())
            text = r['choices'][0]['text']
        except Exception as e:
            text = ""
        results.append({'idx': i, 'seed': seed, 'reference': item['reference'], 'text': text})
with open(os.environ['OUT'], 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print(f"  wrote {len(results)} outputs")
PY

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

run_arm default
run_arm prelude

echo
echo "=== Q4 ROUGE-L wildchat-reference comparison ==="
.venv/bin/python <<PY
import json
import statistics
from rouge_score import rouge_scorer
out = "$OUT_DIR"
def load(name):
    return [json.loads(l) for l in open(f"{out}/{name}_outputs.jsonl") if l.strip()]
default = load("default")
prelude = load("prelude")
n = min(len(default), len(prelude))
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
d_scores, p_scores = [], []
for i in range(n):
    d_text = default[i].get('text', '')
    p_text = prelude[i].get('text', '')
    ref = default[i].get('reference', '')
    if not ref: continue
    d_score = scorer.score(ref, d_text)['rougeL'].fmeasure if d_text else 0.0
    p_score = scorer.score(ref, p_text)['rougeL'].fmeasure if p_text else 0.0
    d_scores.append(d_score)
    p_scores.append(p_score)
print(f"sample size: {len(d_scores)} (~30 prompts × 3 seeds)")
print(f"default mean ROUGE-L: {statistics.mean(d_scores):.4f}, std {statistics.stdev(d_scores):.4f}")
print(f"prelude mean ROUGE-L: {statistics.mean(p_scores):.4f}, std {statistics.stdev(p_scores):.4f}")
print(f"delta: {statistics.mean(p_scores) - statistics.mean(d_scores):+.4f}")
print(f"|delta| / std_default: {abs(statistics.mean(p_scores) - statistics.mean(d_scores)) / max(statistics.stdev(d_scores), 1e-6):.3f}")
# Paired t-test would be the formal statistic; report KS as a sanity check.
try:
    from scipy.stats import ks_2samp, ttest_rel
    ks = ks_2samp(d_scores, p_scores)
    tt = ttest_rel(d_scores, p_scores)
    print(f"KS test on ROUGE-L distributions: stat={ks.statistic:.3f}, p={ks.pvalue:.3f}")
    print(f"paired t-test:                    stat={tt.statistic:.3f}, p={tt.pvalue:.3f}")
except ImportError:
    pass
PY
