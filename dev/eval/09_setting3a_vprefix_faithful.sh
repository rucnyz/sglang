#!/bin/bash
# Setting 3.A — Layer 1 V_prefix' faithful slope (Q3.A subset).
#
# Compares three Mamba prefix-cache configurations on Qwen3.5-35B-A3B
# under the SGLang generated-shared-prefix workload:
#
#   default       — MambaRadixCache, page_size=1, no_buffer (engine default)
#   extra_buffer  — MambaRadixCache, page_size=8K, mamba_scheduler_strategy=extra_buffer
#   layer1        — Layer 1 (HPB LRU + K_big=8192) on page_size=1
#
# Reports: input TPS, mean/P99 TTFT, median E2E, prefix-cache hit-rate
# (from server log).
#
# Total runtime: ~5 min/arm × 3 arms = ~15 min.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/setting3a_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_arm() {
  local arm="$1"
  local extra_env=""
  local extra_flags=""
  case "$arm" in
    default)
      # Engine default: MambaRadixCache, page_size=1, no_buffer.
      extra_flags="--page-size 1"
      ;;
    extra_buffer)
      # Alternative: extra_buffer with page_size=8192.
      extra_flags="--mamba-scheduler-strategy extra_buffer --page-size 8192"
      ;;
    layer1)
      # Layer 1: HPB LRU + K_big=8192 on default page_size=1.
      extra_env="SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
      extra_flags="--page-size 1"
      ;;
  esac
  local log="$OUT_DIR/${arm}_server.log"
  local bench_out="$OUT_DIR/${arm}_bench.json"
  echo "=== arm=$arm (env=$extra_env flags=$extra_flags) ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      $extra_flags \
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

  echo "[$arm] running GSP bench..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups 8 --gsp-prompts-per-group 10 \
    --gsp-system-prompt-len 12000 --gsp-question-len 64 \
    --gsp-output-len 256 \
    --request-rate 2 \
    --output-file "$bench_out" \
    >"$OUT_DIR/${arm}_bench.log" 2>&1
  echo "[$arm] bench done"

  local hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total=$(grep -c "Prefill batch" "$log" || true)
  echo "[$arm] prefill batches: $total, with cached-token > 0: $hit"

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

run_arm default
run_arm extra_buffer
run_arm layer1

echo
echo "=== Setting 3.A V_prefix' faithful summary ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'arm':<14} {'input TPS':>10} {'mean TTFT':>11} {'P99 TTFT':>10} {'med E2E':>10}")
print('-' * 60)
for arm in ('default', 'extra_buffer', 'layer1'):
    p = f"{out}/{arm}_bench.json"
    if not os.path.exists(p):
        print(f"{arm:<14} {'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>10}")
        continue
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    print(f"{arm:<14} {d.get('input_throughput',0):>10.1f} "
          f"{d.get('mean_ttft_ms',0):>10.1f}ms {d.get('p99_ttft_ms',0):>9.1f}ms "
          f"{d.get('median_e2e_latency_ms',0):>9.1f}ms")
PY
