#!/bin/bash
# Phase 2e.5.4 — Performance regression bench: SGLANG_MAMBA_PERLAYER=0 vs =1.
#
# Boots two SGLang servers in sequence, runs sglang.bench_serving with the
# random workload on each, captures throughput / TTFT / TPOT, computes the
# delta. Pass criterion: all metrics within ±2% of baseline (stacked).
#
# Usage:
#   MODEL=Qwen/Qwen3.5-35B-A3B GPUS=3 TP=1 ./11_mamba_perf.sh        # default
#   MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct GPUS=4,5 TP=2 ./11_mamba_perf.sh

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
GPUS=${GPUS:-3}
TP=${TP:-1}
PORT=${PORT:-30099}
NUM_PROMPTS=${NUM_PROMPTS:-100}
RPS=${RPS:-8}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
MEM_FRAC=${MEM_FRAC:-0.8}
WARMUP_S=${WARMUP_S:-360}    # how long to wait for warmup (large hybrid models can take 5+ min)

OUT_DIR=/tmp/mamba_perf_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR model=$MODEL TP=$TP GPUS=$GPUS prompts=$NUM_PROMPTS rps=$RPS"

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "perlayer" ]; then
    extra_env="SGLANG_MAMBA_PERLAYER=1"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  echo "=== arm=$arm ==="
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6
  nohup env CUDA_VISIBLE_DEVICES="$GPUS" $extra_env .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size $TP \
    --mem-fraction-static $MEM_FRAC --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
  local pid=$!
  echo "server pid=$pid log=$log; waiting up to ${WARMUP_S}s for ready"
  echo "--- live server log ---"
  tail -F "$log" 2>/dev/null &
  local tailer=$!
  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      kill $tailer 2>/dev/null || true
      wait $tailer 2>/dev/null || true
      echo
      echo "--- ready after ${waited}s ---"
      break
    fi
  done
  if kill -0 $tailer 2>/dev/null; then
    kill $tailer 2>/dev/null || true
    wait $tailer 2>/dev/null || true
  fi
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
    echo "[$arm] server failed to start (waited ${WARMUP_S}s)"
    tail -30 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  echo "running bench for arm=$arm..."
  local bench_out="$OUT_DIR/${arm}_bench.json"
  .venv/bin/python -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random --num-prompts $NUM_PROMPTS \
    --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
    --request-rate $RPS \
    --output-file "$bench_out" \
    >"$OUT_DIR/${arm}_bench.log" 2>&1
  echo "bench done for arm=$arm"
  kill -9 $pid 2>/dev/null || true
  sleep 6
}

run_arm stacked
run_arm perlayer

echo "=== compare ==="
OUT_DIR=$OUT_DIR .venv/bin/python <<'PY'
import json, sys, os
out = os.environ["OUT_DIR"]
def load(arm):
    path = f"{out}/{arm}_bench.json"
    if not os.path.exists(path):
        return None
    # bench_serving writes JSON-lines; take the last entry.
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

s = load("stacked"); p = load("perlayer")
if s is None or p is None:
    print(f"MISSING bench output: stacked={s is not None}, perlayer={p is not None}")
    sys.exit(1)
print(f"\n{'metric':<28} {'stacked':>12} {'perlayer':>12} {'delta%':>10}")
print("-" * 64)
keys = [
    ("input_throughput", "input toks/s"),
    ("output_throughput", "output toks/s"),
    ("mean_ttft_ms", "mean TTFT (ms)"),
    ("median_ttft_ms", "median TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_tpot_ms", "mean TPOT (ms)"),
    ("median_tpot_ms", "median TPOT (ms)"),
    ("p99_tpot_ms", "P99 TPOT (ms)"),
    ("median_e2e_latency_ms", "median E2E (ms)"),
]
# Regression direction: throughput is "lower is worse" (sign +), latency is
# "higher is worse" (sign -). We only fail on regressions, not improvements.
# Map: key -> +1 if perlayer-lower-than-stacked is regression (latency),
#            -1 if perlayer-lower-than-stacked is improvement (throughput).
direction = {
    "input_throughput": -1,
    "output_throughput": -1,
    "mean_ttft_ms": +1,
    "median_ttft_ms": +1,
    "p99_ttft_ms": +1,
    "mean_tpot_ms": +1,
    "median_tpot_ms": +1,
    "p99_tpot_ms": +1,
    "median_e2e_latency_ms": +1,
}
worst_regression = 0.0
for k, label in keys:
    sv = s.get(k); pv = p.get(k)
    if sv is None or pv is None:
        print(f"{label:<28} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
        continue
    if sv == 0:
        delta = float("inf")
    else:
        delta = (pv - sv) / sv * 100
    # Regression: signed by direction. Throughput ↓ = regression (positive sign);
    # latency ↑ = regression (positive sign).
    sign = direction.get(k, +1)
    regression = sign * delta if sign > 0 else -sign * delta
    if regression > worst_regression:
        worst_regression = regression
    marker = ""
    if regression > 2.0:
        marker = " ←REG"
    elif delta < 0 and sign > 0 or delta > 0 and sign < 0:
        marker = " (better)"
    print(f"{label:<28} {sv:>12.2f} {pv:>12.2f} {delta:>+9.2f}%{marker}")
print()
print(f"worst regression: {worst_regression:+.2f}% (improvements not counted)")
if worst_regression > 2.0:
    print(f"FAIL: worst regression {worst_regression:.2f}% > 2.0%")
    sys.exit(1)
print("PASS: no metric regressed by more than 2.0% (improvements allowed)")
PY
