#!/bin/bash
# Setting 2.2 (paper Sweep 2): V_LoRA on Qwen3-4B + 32 rank-16 adapters.
# Sweep max_loras_per_batch in {1, 2, 4, 8, 16, 32}.
#
# Paper Table 2 reference:
#   max_loras=1  → input TPS 5652, mean TTFT 7047ms
#   max_loras=32 → input TPS 7556, mean TTFT 74ms
# Across the sweep: 95× TTFT swing, KV usage <2%.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3-4B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
LORA_DIR=${LORA_DIR:-/scratch/yuzhou/.cache/synthetic_loras/qwen3-4b-r16}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
RPS=${RPS:-32}
OUT_DIR=/tmp/sweep_lora_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR  lora_dir=$LORA_DIR"

# Build the --lora-paths argument: 32 adapters as name=path pairs.
LORA_ARGS=""
for i in $(seq 0 31); do
  LORA_ARGS="$LORA_ARGS lora_$i=$LORA_DIR/lora_$i"
done

run_point() {
  local max_loras="$1"
  local log="$OUT_DIR/ml${max_loras}_server.log"
  local bench_out="$OUT_DIR/ml${max_loras}_bench.json"
  echo "=== max_loras_per_batch=$max_loras ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enable-lora \
    --max-loras-per-batch "$max_loras" \
    --max-lora-rank 16 \
    --lora-paths $LORA_ARGS \
    >"$log" 2>&1 &
  local pid=$!

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[ml=$max_loras] ready after ${waited}s"
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

  echo "[ml=$max_loras] running bench..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --num-prompts $NUM_PROMPTS \
    --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --request-rate $RPS \
    --lora-name lora_0 lora_1 lora_2 lora_3 lora_4 lora_5 lora_6 lora_7 lora_8 lora_9 lora_10 lora_11 lora_12 lora_13 lora_14 lora_15 lora_16 lora_17 lora_18 lora_19 lora_20 lora_21 lora_22 lora_23 lora_24 lora_25 lora_26 lora_27 lora_28 lora_29 lora_30 lora_31 \
    --output-file "$bench_out" \
    >"$OUT_DIR/ml${max_loras}_bench.log" 2>&1
  echo "[ml=$max_loras] bench done"

  local kv_peak=$(grep -oE "full token usage: [0-9.]+" "$log" | awk '{print $4}' | sort -rn | head -1 || echo "n/a")
  echo "[ml=$max_loras] kv_usage_peak=$kv_peak"

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

for ml in 1 2 4 8 16 32; do
  run_point "$ml" || echo "[ml=$ml] FAILED, continuing"
done

echo
echo "=== Sweep 2 (KV ↔ LoRA) summary ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'max_loras':>10} {'input TPS':>10} {'mean TTFT (ms)':>15} {'P99 TTFT (ms)':>14}")
print('-' * 55)
for ml in (1, 2, 4, 8, 16, 32):
    p = f"{out}/ml{ml}_bench.json"
    if not os.path.exists(p):
        print(f"{ml:>10} {'N/A':>10} {'N/A':>15} {'N/A':>14}")
        continue
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    print(f"{ml:>10} {d.get('input_throughput', 0):>10.0f} "
          f"{d.get('mean_ttft_ms', 0):>15.1f} {d.get('p99_ttft_ms', 0):>14.1f}")
print()
print("Paper Table 2 reference:")
print("  ml=1  → input TPS 5652, mean TTFT 7047ms")
print("  ml=32 → input TPS 7556, mean TTFT 74ms (95× TTFT swing)")
PY
