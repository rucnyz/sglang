#!/bin/bash
# Setting 3.B — Layer 1 V_prefix' stability under cold-burst (Q3.B).
#
# Compares two LRU policies on Qwen3.5-35B-A3B with K_big=8192 (the
# heterogeneous-granularity tree):
#   recency  — recency-LRU (partial Layer 1)
#   hpb      — hits-per-byte LRU (full Layer 1)
#
# Workload: build → cold-burst → recovery
#   Phase 1 (build)    : GSP shared-prefix, RPS=2, 80 prompts, ~40s
#   Phase 2 (burst)    : random un-shared prompts, RPS=8, 200 prompts, ~25s
#   Phase 3 (recovery) : GSP shared-prefix, RPS=2, 80 prompts, ~40s
#
# Per-phase metrics: mean / P99 TTFT, mean / median E2E, prefix-cache hit
# rate (from server log). Expect: HPB preserves shared-prefix snapshots
# during the burst (priority based on hits-per-byte), so Phase 3 hit rate
# stays high. Recency LRU evicts the shared-prefix snapshots during the
# burst, so Phase 3 has to rebuild.
#
# Total runtime: ~6 min/arm × 2 arms = ~12 min.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/setting3b_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_arm() {
  local arm="$1"
  local extra_env="SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
  if [ "$arm" = "hpb" ]; then
    extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0"
  fi
  # arm=recency: only K_big set; HPB LRU off; matches the partial-Layer-1
  # configuration the paper §6.3 Q3.B describes.

  local log="$OUT_DIR/${arm}_server.log"
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

  for phase in build burst recovery; do
    echo "[$arm] Phase $phase ..."
    local bench_out="$OUT_DIR/${arm}_${phase}_bench.json"
    local bench_log="$OUT_DIR/${arm}_${phase}_bench.log"
    if [ "$phase" = "burst" ]; then
      .venv/bin/python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name random \
        --num-prompts 200 \
        --random-input-len 4096 --random-output-len 64 \
        --request-rate 8 \
        --output-file "$bench_out" \
        >"$bench_log" 2>&1 || echo "[$arm] $phase bench error"
    else
      .venv/bin/python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name generated-shared-prefix \
        --gsp-num-groups 8 --gsp-prompts-per-group 10 \
        --gsp-system-prompt-len 12000 --gsp-question-len 64 \
        --gsp-output-len 256 \
        --request-rate 2 \
        --output-file "$bench_out" \
        >"$bench_log" 2>&1 || echo "[$arm] $phase bench error"
    fi
    echo "[$arm] Phase $phase done"
    sleep 5
  done

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

run_arm recency
run_arm hpb

echo
echo "=== Setting 3.B cold-burst summary ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'arm':<8} {'phase':<10} {'TPS':>8} {'mean TTFT':>11} {'P99 TTFT':>10} {'med E2E':>10}")
print('-' * 60)
for arm in ('recency', 'hpb'):
    for phase in ('build', 'burst', 'recovery'):
        p = f"{out}/{arm}_{phase}_bench.json"
        if not os.path.exists(p):
            print(f"{arm:<8} {phase:<10} {'N/A':>8}")
            continue
        with open(p) as f:
            lines = [l for l in f if l.strip()]
        d = json.loads(lines[-1]) if lines else {}
        print(f"{arm:<8} {phase:<10} "
              f"{d.get('input_throughput',0):>8.1f} "
              f"{d.get('mean_ttft_ms',0):>9.1f}ms "
              f"{d.get('p99_ttft_ms',0):>8.1f}ms "
              f"{d.get('median_e2e_latency_ms',0):>8.1f}ms")
    print()
PY
