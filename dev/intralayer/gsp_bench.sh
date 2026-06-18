#!/bin/bash
# LPB-vs-LRU bench on SGLang's built-in generated-shared-prefix (GSP)
# dataset. Adapted from prelude/2e/32_hpb_gsp_bench.sh (commit
# 7c6828c9a, which reported -19.77% mean TTFT for HPB-vs-recency on
# this exact workload).
#
# With gsp-system-prompt-len > 8192, the shared system prefix crosses
# the chunked-prefill boundary and a mamba snapshot node is created
# in MambaRadixCache. 80 prompts (8 groups × 10 per group) all share
# their group's system prompt — subsequent requests in a group HIT
# the snapshot. The workload differs from our dev/intralayer Path A in
# that:
#   - many independent shared prefixes (8) compete for mamba slots
#     instead of one anchor
#   - per-request question is short → re-use is the dominant cost
#     factor (not new prefill)
#   - real bench_serving HTTP traffic, not offline Engine API
#
# Outputs land in dev/intralayer/runs/sglang_gsp/.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/scratch/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
GSP_GROUPS=${GSP_GROUPS:-8}
GSP_PER_GROUP=${GSP_PER_GROUP:-10}
GSP_SYSPROMPT_LEN=${GSP_SYSPROMPT_LEN:-12000}
GSP_QUESTION_LEN=${GSP_QUESTION_LEN:-64}
RPS=${RPS:-2}
N_TRIAL=${N_TRIAL:-1}

OUT_BASE=/scratch/yuzhou/projects/vllm-songyang/dev/intralayer/runs/sglang_gsp
mkdir -p "$OUT_BASE"
echo "out_base=$OUT_BASE"
echo "GSP: groups=$GSP_GROUPS x per_group=$GSP_PER_GROUP, sys_len=$GSP_SYSPROMPT_LEN, q_len=$GSP_QUESTION_LEN, trials=$N_TRIAL"

run_arm() {
  local arm="$1"
  local trial="$2"
  # LPB on/off is a CLI flag (--radix-eviction-policy lpb), #181.
  local extra_env=""
  local evict_policy="lru"
  if [ "$arm" = "lpb" ]; then
    extra_env="SGLANG_LPB_WINDOW_S=300.0"
    evict_policy="lpb"
  fi
  local log="$OUT_BASE/${arm}_t${trial}_server.log"
  local bench_out="$OUT_BASE/${arm}_t${trial}_bench.json"
  echo "=== arm=$arm trial=$trial ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static 0.8 --log-level info \
      --radix-eviction-policy "$evict_policy" \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!
  echo "[$arm t${trial}] pid=$pid"

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[$arm t${trial}] ready after ${waited}s"
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

  echo "[$arm t${trial}] running bench_serving generated-shared-prefix..."
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
    >"$OUT_BASE/${arm}_t${trial}_bench.log" 2>&1
  echo "[$arm t${trial}] bench done"

  local hit_lines=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total_prefills=$(grep -c "Prefill batch" "$log" || true)
  echo "[$arm t${trial}] prefill batches: $total_prefills, with cached-token > 0: $hit_lines"

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

for trial in $(seq 1 $N_TRIAL); do
  run_arm recency "$trial"
  run_arm lpb "$trial"
done

echo
echo "=== compare (all trials) ==="
.venv/bin/python <<PY
import json, os, statistics
OUT_BASE="$OUT_BASE"
N_TRIAL=int("$N_TRIAL")

def load(p):
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

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
recency_vals = {k: [] for k, _ in keys}
lpb_vals = {k: [] for k, _ in keys}
for t in range(1, N_TRIAL + 1):
    r = load(f"{OUT_BASE}/recency_t{t}_bench.json")
    h = load(f"{OUT_BASE}/lpb_t{t}_bench.json")
    if r and h:
        for k, _ in keys:
            if k in r: recency_vals[k].append(r[k])
            if k in h: lpb_vals[k].append(h[k])

print(f"\n{'metric':<22} {'recency mean':>15} {'lpb mean':>15} {'Δ%':>10}")
print('-' * 70)
for k, label in keys:
    rs = recency_vals[k]
    hs = lpb_vals[k]
    if not rs or not hs:
        print(f"{label:<22} {'N/A':>15} {'N/A':>15} {'N/A':>10}")
        continue
    rm = statistics.mean(rs); hm = statistics.mean(hs)
    delta = (hm - rm) / rm * 100 if rm else float('inf')
    marker = ""
    if "ttft" in k or "tpot" in k or "e2e" in k:
        if delta < -3:
            marker = "  ← LPB FASTER"
        elif abs(delta) < 3:
            marker = "  (tied)"
        else:
            marker = "  ← LPB SLOWER"
    print(f"{label:<22} {rm:>15.2f} {hm:>15.2f} {delta:>+9.2f}%{marker}")
PY
