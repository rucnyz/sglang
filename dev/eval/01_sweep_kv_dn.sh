#!/bin/bash
# Setting 2.1 (paper Sweep 1): KV ↔ DeltaNet on Qwen3.5-35B-A3B,
# prefill-heavy traffic. Sweep mamba_full_memory_ratio.
#
# Paper Table \ref{tab:sweep1} target numbers (ref):
#   ratio | input TPS | mean TTFT (s) | mamba_usage | full_token_usage
#    0.1  |  3039     | 69.9          | 0.66        | 0.008
#    0.9  |  7648     | 13.6          | 0.66        | 0.064

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-256}
RPS=${RPS:-32}
OUT_DIR=/tmp/sweep_kv_dn_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

run_point() {
  local ratio="$1"
  local log="$OUT_DIR/ratio${ratio}_server.log"
  local bench_out="$OUT_DIR/ratio${ratio}_bench.json"
  echo "=== ratio=$ratio ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    --mamba-full-memory-ratio "$ratio" \
    >"$log" 2>&1 &
  local pid=$!
  echo "[ratio=$ratio] pid=$pid"

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[ratio=$ratio] ready after ${waited}s"
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

  echo "[ratio=$ratio] running bench..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --num-prompts $NUM_PROMPTS \
    --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --request-rate $RPS \
    --output-file "$bench_out" \
    >"$OUT_DIR/ratio${ratio}_bench.log" 2>&1
  echo "[ratio=$ratio] bench done"

  # Extract mamba_usage / full_token_usage peaks from server log.
  local mamba_peak=$(grep -oE "mamba usage: [0-9.]+" "$log" | awk '{print $3}' | sort -rn | head -1 || echo "n/a")
  local kv_peak=$(grep -oE "full token usage: [0-9.]+" "$log" | awk '{print $4}' | sort -rn | head -1 || echo "n/a")
  echo "[ratio=$ratio] mamba_usage_peak=$mamba_peak  full_token_usage_peak=$kv_peak"

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

for ratio in 0.1 0.3 0.5 0.7 0.9; do
  run_point "$ratio" || echo "[ratio=$ratio] FAILED, continuing"
done

echo
echo "=== Sweep 1 (KV ↔ DN) summary ==="
.venv/bin/python <<PY
import json, os, glob, statistics
out = "$OUT_DIR"
print(f"\n{'ratio':>6} {'input TPS':>10} {'output TPS':>11} {'mean TTFT (s)':>15} {'P99 TTFT (s)':>14} {'median E2E (s)':>16}")
print('-' * 75)
for ratio in (0.1, 0.3, 0.5, 0.7, 0.9):
    p = f"{out}/ratio{ratio}_bench.json"
    if not os.path.exists(p):
        print(f"{ratio:>6.1f} {'N/A':>10} {'N/A':>11} {'N/A':>15} {'N/A':>14} {'N/A':>16}")
        continue
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    input_tps = d.get('input_throughput', 0)
    output_tps = d.get('output_throughput', 0)
    mean_ttft = (d.get('mean_ttft_ms', 0) or 0) / 1000
    p99_ttft = (d.get('p99_ttft_ms', 0) or 0) / 1000
    e2e = (d.get('median_e2e_latency_ms', 0) or 0) / 1000
    print(f"{ratio:>6.1f} {input_tps:>10.0f} {output_tps:>11.0f} {mean_ttft:>15.2f} {p99_ttft:>14.2f} {e2e:>16.2f}")
print()
print("Paper Table 1 reference (paper):")
print("  0.1: input TPS 3039, mean TTFT 69.9s")
print("  0.9: input TPS 7648, mean TTFT 13.6s")
PY
