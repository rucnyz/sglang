#!/bin/bash
# Setting 2.3 (paper Sweep 3): V_prefix on Qwen3-8B with multi-turn
# shared-prefix traffic. Sweep mem_fraction_static.
#
# Expected per paper: prefix-cache hit rate stays at ~75.8% across all
# 5 points → V_prefix is FLAT on naive RadixCache (the failure mode
# the paper highlights).

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
# GSP simulates the multi-turn shared-prefix workload paper §6.3 wants.
GSP_GROUPS=${GSP_GROUPS:-32}
GSP_PER_GROUP=${GSP_PER_GROUP:-6}
GSP_SYSPROMPT_LEN=${GSP_SYSPROMPT_LEN:-1024}
GSP_QUESTION_LEN=${GSP_QUESTION_LEN:-128}
RPS=${RPS:-8}
OUT_DIR=/tmp/sweep_prefix_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_point() {
  local mem_frac="$1"
  local log="$OUT_DIR/mf${mem_frac}_server.log"
  local bench_out="$OUT_DIR/mf${mem_frac}_bench.json"
  echo "=== mem_fraction_static=$mem_frac ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static "$mem_frac" --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
  local pid=$!
  echo "[mf=$mem_frac] pid=$pid"

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[mf=$mem_frac] ready after ${waited}s"
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

  echo "[mf=$mem_frac] running GSP bench..."
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
    >"$OUT_DIR/mf${mem_frac}_bench.log" 2>&1
  echo "[mf=$mem_frac] bench done"

  local hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total=$(grep -c "Prefill batch" "$log" || true)
  echo "[mf=$mem_frac] prefill batches: $total, with cached-token > 0: $hit"

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

for mem_frac in 0.30 0.40 0.50 0.65 0.80; do
  run_point "$mem_frac" || echo "[mf=$mem_frac] FAILED, continuing"
done

echo
echo "=== Sweep 3 (V_prefix on Qwen3-8B) summary ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'mem_frac':>9} {'input TPS':>10} {'mean TTFT (ms)':>15} {'cache hit rate %':>17}")
print('-' * 60)
for mf in (0.30, 0.40, 0.50, 0.65, 0.80):
    p = f"{out}/mf{mf}_bench.json"
    if not os.path.exists(p):
        print(f"{mf:>9.2f} {'N/A':>10} {'N/A':>15} {'N/A':>17}")
        continue
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    input_tps = d.get('input_throughput', 0)
    mean_ttft = d.get('mean_ttft_ms', 0)
    # Cache hit rate from prefill log:
    log = f"{out}/mf{mf}_server.log"
    if os.path.exists(log):
        import subprocess
        total = subprocess.run(['grep', '-c', 'Prefill batch', log], capture_output=True, text=True).stdout.strip()
        hit_lines = subprocess.run(['grep', 'Prefill batch', log], capture_output=True, text=True).stdout
        hit = sum(1 for line in hit_lines.split('\n') if 'cached-token' in line and 'cached-token: 0' not in line)
        hit_rate = (hit / max(1, int(total or 0))) * 100 if total else 0
    else:
        hit_rate = 0
    print(f"{mf:>9.2f} {input_tps:>10.0f} {mean_ttft:>15.2f} {hit_rate:>16.1f}%")
print()
print("Paper Table 3 reference: hit rate ~75.8% across all 5 points (V_prefix flat).")
PY
