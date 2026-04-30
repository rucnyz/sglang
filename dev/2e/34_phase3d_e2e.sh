#!/bin/bash
# Phase 3.d e2e — heterogeneous granularity (SGLANG_K_BIG=8192) under
# live serving on Qwen3.5-35B-A3B with the GSP workload.
#
# Hypothesis: with K_big=8192, only chunked-prefill snapshots (at depth
# 8192) carry mamba_value. Final-prompt leaves at non-aligned depths
# become tombstones. The match-walk falls back to nearest big-page
# ancestor. Since GSP system_prompt_len=12000 > 8192, the cache should
# hit the depth-8192 ancestor on every same-system-prompt query —
# behavior should be identical or better than non-K_big.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
OUT_DIR=/tmp/phase3d_e2e_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "kbig8192" ]; then
    extra_env="SGLANG_K_BIG=8192 SGLANG_HPB_LRU=1"
  elif [ "$arm" = "kbig8192_only" ]; then
    extra_env="SGLANG_K_BIG=8192"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  local bench_out="$OUT_DIR/${arm}_bench.json"
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
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
    echo "FAIL: server did not become ready"
    tail -30 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  echo "[$arm] running GSP bench..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 8 \
    --gsp-prompts-per-group 10 \
    --gsp-system-prompt-len 12000 \
    --gsp-question-len 64 \
    --request-rate 2 \
    --output-file "$bench_out" \
    >"$OUT_DIR/${arm}_bench.log" 2>&1
  echo "[$arm] bench done"

  local hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total=$(grep -c "Prefill batch" "$log" || true)
  echo "[$arm] prefill batches: $total, with cached-token > 0: $hit"

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

run_arm baseline
run_arm kbig8192_only      # heterogeneous granularity alone
run_arm kbig8192           # heterogeneous + HPB (full Layer 1)

echo
echo "=== Phase 3.d e2e summary (K_big=8192 alone vs full Layer 1 vs baseline) ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'arm':<20} {'input TPS':>10} {'mean TTFT':>12} {'median TTFT':>13} {'mean TPOT':>11} {'median E2E':>12}")
print('-' * 82)
for arm in ('baseline', 'kbig8192_only', 'kbig8192'):
    p = f"{out}/{arm}_bench.json"
    if not os.path.exists(p):
        print(f"{arm:<20} (MISSING)")
        continue
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    print(f"{arm:<20} {d.get('input_throughput', 0):>10.1f} "
          f"{d.get('mean_ttft_ms', 0):>10.2f}ms {d.get('median_ttft_ms', 0):>11.2f}ms "
          f"{d.get('mean_tpot_ms', 0):>10.2f}ms {d.get('median_e2e_latency_ms', 0):>11.1f}ms")

print()
print("Expected: kbig8192_only and kbig8192 should be at least as fast as baseline.")
print("Expected: kbig8192 (full Layer 1) >= kbig8192_only (heterogeneous alone).")
PY
