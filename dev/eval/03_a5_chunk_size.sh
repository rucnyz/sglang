#!/bin/bash
# Ablation A5: VMM chunk size sweep on the cross-pool actuator.
#
# Paper §6.7: "Sweep VMM chunk size in {64MB, 256MB, 1GB}. Expectation:
# smaller chunks reduce wasted bytes on shrink/grow at the cost of more
# bitmap overhead; 256MB is the default."
#
# Independent test: this also re-runs the perf bench from 2e.5.6.3.b at
# different chunk sizes to see if the +6-7% TTFT regression scales with
# chunk granularity (it might, due to mixed cuMemMap/cudaMalloc HBM
# channel-interleaving).
#
# Sweep: chunk_bytes ∈ {64*1024*1024, 256*1024*1024, 1024*1024*1024}.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
NUM_PROMPTS=${NUM_PROMPTS:-100}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
RPS=${RPS:-8}
OUT_DIR=/tmp/a5_chunk_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

# NOTE: chunk_size is currently hard-coded to 64MB in
# multi_tensor_arena.py and memory_pool.py. We expose it via env
# SGLANG_ARENA_CHUNK_BYTES; the engine wiring honors it where set.
# If env is unset, the default 64MB is used.

run_arm() {
  local label="$1"
  local chunk="$2"
  local extra_env=""
  if [ "$label" != "baseline" ]; then
    extra_env="SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=$chunk"
  fi
  local log="$OUT_DIR/${label}_server.log"
  local bench_out="$OUT_DIR/${label}_bench.json"
  echo "=== $label (chunk_bytes=$chunk) ==="

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
  echo "[$label] pid=$pid"

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[$label] ready after ${waited}s"
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

  echo "[$label] running bench..."
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --num-prompts $NUM_PROMPTS \
    --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --request-rate $RPS \
    --output-file "$bench_out" \
    >"$OUT_DIR/${label}_bench.log" 2>&1
  echo "[$label] bench done"
  kill -9 $pid 2>/dev/null || true
  sleep 5
}

run_arm baseline 0
run_arm chunk64MB 67108864
run_arm chunk256MB 268435456
run_arm chunk1GB 1073741824

echo
echo "=== A5 chunk-size summary ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'arm':<14} {'input TPS':>10} {'mean TTFT':>11} {'P99 TTFT':>10} {'mean TPOT':>11} {'median E2E':>12}")
print('-' * 75)
for arm in ('baseline', 'chunk64MB', 'chunk256MB', 'chunk1GB'):
    p = f"{out}/{arm}_bench.json"
    if not os.path.exists(p):
        print(f"{arm:<14} {'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>11} {'N/A':>12}")
        continue
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    print(f"{arm:<14} {d.get('input_throughput', 0):>10.1f} "
          f"{d.get('mean_ttft_ms', 0):>10.2f}ms {d.get('p99_ttft_ms', 0):>9.2f}ms "
          f"{d.get('mean_tpot_ms', 0):>10.2f}ms {d.get('median_e2e_latency_ms', 0):>11.1f}ms")
PY
