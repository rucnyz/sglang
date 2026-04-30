#!/bin/bash
# Phase 3.a eval v6 — use SGLang's built-in generated-shared-prefix
# (GSP) dataset to test paper §4.2's HPB LRU vs recency LRU.
#
# GSP generates groups of prompts where each group shares a long
# system prompt; multiple "questions" extend the system prompt with
# different short suffixes. With gsp-system-prompt-len > 8192, the
# system prefix crosses the chunked-prefill boundary and a snapshot
# node is created in MambaRadixCache. Subsequent same-group requests
# should HIT that snapshot.
#
# Workload (per arm):
#   gsp-num-groups=8 (groups of shared system prompts)
#   gsp-prompts-per-group=10 (10 questions per group)
#   gsp-system-prompt-len=12000 (real-text system prompt > 8192)
#   gsp-question-len=64 (short downstream question)
#   total: 80 prompts dispatched at moderate request-rate.
#
# Both arms see identical prompts (same seed). The difference is
# eviction policy when the cache fills up. With 8 groups × ~13K tokens
# each = ~104K tokens of system prompts trying to fit. KV pool cap is
# ~1.3M, so no eviction in steady state — but mamba pool has only 361
# slots. Eight 13K-token prompts allocate up to 8 mamba slots; if
# they overlap with concurrent burst-style traffic, mamba slots will
# rotate.
#
# Pass criterion: HPB cache_hit_rate ≥ recency cache_hit_rate, with
# HPB Pulse 2 latencies materially lower.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
GSP_GROUPS=${GSP_GROUPS:-8}
GSP_PER_GROUP=${GSP_PER_GROUP:-10}
GSP_SYSPROMPT_LEN=${GSP_SYSPROMPT_LEN:-12000}
GSP_QUESTION_LEN=${GSP_QUESTION_LEN:-64}
RPS=${RPS:-2}
OUT_DIR=/tmp/hpb_gsp_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"
echo "GSP: groups=$GSP_GROUPS x per_group=$GSP_PER_GROUP, sys_len=$GSP_SYSPROMPT_LEN, q_len=$GSP_QUESTION_LEN"

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "hpb" ]; then
    extra_env="SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=300.0"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  local bench_out="$OUT_DIR/${arm}_bench.json"
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

  echo "[$arm] running bench_serving generated-shared-prefix..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups $GSP_GROUPS \
    --gsp-prompts-per-group $GSP_PER_GROUP \
    --gsp-system-prompt-len $GSP_SYSPROMPT_LEN \
    --gsp-question-len $GSP_QUESTION_LEN \
    --request-rate $RPS \
    --output-file "$bench_out" \
    >"$OUT_DIR/${arm}_bench.log" 2>&1
  echo "[$arm] bench done"

  local hit_lines=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total_prefills=$(grep -c "Prefill batch" "$log" || true)
  echo "[$arm] prefill batches: $total_prefills, with cached-token > 0: $hit_lines"

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

run_arm recency
run_arm hpb

echo
echo "=== compare ==="
.venv/bin/python <<PY
import json, os
def load(p):
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

r = load("$OUT_DIR/recency_bench.json")
h = load("$OUT_DIR/hpb_bench.json")

if r is None or h is None:
    print(f"MISSING: recency={r is not None} hpb={h is not None}")
    raise SystemExit(1)

keys = [
    ("input_throughput", "input toks/s"),
    ("output_throughput", "output toks/s"),
    ("mean_ttft_ms", "mean TTFT (ms)"),
    ("median_ttft_ms", "median TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_tpot_ms", "mean TPOT (ms)"),
    ("median_tpot_ms", "median TPOT (ms)"),
    ("median_e2e_latency_ms", "median E2E (ms)"),
]
print(f"\n{'metric':<22} {'recency':>10} {'hpb':>10} {'delta%':>10}")
print('-' * 60)
for k, label in keys:
    rv = r.get(k); hv = h.get(k)
    if rv is None or hv is None:
        print(f"{label:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        continue
    delta = (hv - rv) / rv * 100 if rv else float('inf')
    marker = ""
    if "ttft" in k or "tpot" in k or "e2e" in k:
        if delta < -3:
            marker = "  ←HPB FASTER"
        elif abs(delta) < 3:
            marker = "  (no benefit)"
        else:
            marker = "  ←HPB SLOWER"
    print(f"{label:<22} {rv:>10.2f} {hv:>10.2f} {delta:>+9.2f}%{marker}")
PY
