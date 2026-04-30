#!/bin/bash
# Ablation A2 — Layer 1 big-page granularity (K_big sweep).
#
# Sweep SGLANG_K_BIG ∈ {0, 2048, 4096, 8192, 16384} on the SGLang
# generated-shared-prefix (GSP) workload (the same shape paper §4.2
# describes: groups of prompts share a long system prefix). K_small
# is implicitly 512 from the page_size default. K_big=0 is the baseline
# (full-granularity snapshots, the engine default).
#
# Metric: prefix-cache hit rate, mean TTFT, P99 TTFT, snapshot-memory
# footprint (mamba pool eviction count from logs).
#
# Total runtime: ~5 min/arm × 5 arms = ~25 min.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/a2_kbig_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_arm() {
  local kbig="$1"
  local arm="kbig${kbig}"
  local extra_env="SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0"
  if [ "$kbig" != "0" ]; then
    extra_env="$extra_env SGLANG_K_BIG=$kbig SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
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

  # Capture prefix-cache hit-rate evidence from server log.
  local hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total=$(grep -c "Prefill batch" "$log" || true)
  echo "[$arm] prefill batches: $total, with cached-token > 0: $hit"

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

run_arm 0
run_arm 2048
run_arm 4096
run_arm 8192
run_arm 16384

echo
echo "=== A2 K_big sweep summary ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'kbig':<8} {'input TPS':>10} {'mean TTFT':>11} {'P99 TTFT':>10} {'med E2E':>10}")
print('-' * 55)
for k in (0, 2048, 4096, 8192, 16384):
    arm = f"kbig{k}"
    p = f"{out}/{arm}_bench.json"
    if not os.path.exists(p):
        print(f"{k:<8} {'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>10}")
        continue
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    print(f"{k:<8} {d.get('input_throughput',0):>10.1f} "
          f"{d.get('mean_ttft_ms',0):>10.1f}ms {d.get('p99_ttft_ms',0):>9.1f}ms "
          f"{d.get('median_e2e_latency_ms',0):>9.1f}ms")
PY
