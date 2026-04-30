#!/bin/bash
# Phase 2e.5.6.3.b — verify Cost 1 hypothesis: mem_fraction_static
# interaction.
#
# Theory: arena mode reserves KV/mamba VA outside PyTorch's caching
# allocator. With --mem-fraction-static 0.8, PyTorch reserves 80% of
# GPU but doesn't actually populate most of it (KV/mamba bypassed).
# Total memory = PyTorch 80% + arena ~50 GB > GPU capacity → PyTorch
# growth-and-purge cycles → +6% TTFT in steady state.
#
# Test: same bench but --mem-fraction-static 0.5 for BOTH baseline
# and arena_only. If at 0.5 the gap between baseline and arena_only
# is smaller (or zero), config-interaction is confirmed.

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
MEM_FRAC=${MEM_FRAC:-0.5}     # << HALF the default; the variable under test
WARMUP_S=${WARMUP_S:-360}

OUT_DIR=/tmp/arena_lowmem_perf_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR  mem_fraction_static=$MEM_FRAC"

run_arm() {
  local arm="$1"
  local extra_env=""
  if [ "$arm" = "arena_only" ]; then
    extra_env="SGLANG_ARENA_SHARED=1"
  fi
  local log="$OUT_DIR/${arm}_server.log"
  echo "=== arm=$arm ==="
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env CUDA_VISIBLE_DEVICES="$GPUS" $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --tensor-parallel-size $TP \
      --mem-fraction-static $MEM_FRAC --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!
  echo "[$arm] pid=$pid log=$log"
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
    echo "[$arm] server failed at mem_frac=$MEM_FRAC"
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

run_arm baseline
run_arm arena_only

# Compare against the previous run at mem_fraction_static=0.8.
PREV_BASELINE=${PREV_BASELINE:-/tmp/xpool_perf_1301932/baseline_bench.json}
PREV_ARENA=${PREV_ARENA:-/tmp/arena_only_perf_*/arena_only_bench.json}

# Resolve glob.
PREV_ARENA_RESOLVED=$(ls $PREV_ARENA 2>/dev/null | head -1 || true)

echo "=== compare 4-way (mem_frac=0.8 prior vs mem_frac=$MEM_FRAC current) ==="
PREV_BASELINE=$PREV_BASELINE PREV_ARENA=$PREV_ARENA_RESOLVED \
  CURR_BASELINE="$OUT_DIR/baseline_bench.json" \
  CURR_ARENA="$OUT_DIR/arena_only_bench.json" \
  MEM_FRAC=$MEM_FRAC \
  .venv/bin/python <<'PY'
import json, os, sys

def load(p):
    if not p or not os.path.exists(p):
        return None
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

prev_b = load(os.environ.get("PREV_BASELINE"))
prev_a = load(os.environ.get("PREV_ARENA"))
curr_b = load(os.environ.get("CURR_BASELINE"))
curr_a = load(os.environ.get("CURR_ARENA"))
mf = os.environ["MEM_FRAC"]

if curr_b is None or curr_a is None:
    print("MISSING current bench files")
    sys.exit(1)

keys = [
    ("input_throughput", "input toks/s"),
    ("mean_ttft_ms", "mean TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_tpot_ms", "mean TPOT (ms)"),
    ("p99_tpot_ms", "P99 TPOT (ms)"),
    ("median_e2e_latency_ms", "median E2E (ms)"),
]

def cell(x): return f"{x:>10.2f}" if x is not None else f"{'N/A':>10}"
def delta(av, bv):
    if av in (None, 0) or bv is None: return None
    return (bv - av) / av * 100
def dcell(x): return f"{x:>+8.2f}%" if x is not None else f"{'N/A':>9}"

mf08 = "0.8"
print(f"\n{'metric':<22} {'B@'+mf08:>10} {'A@'+mf08:>10} "
      f"{'B@'+mf:>10} {'A@'+mf:>10} "
      f"{'A@'+mf08+'-B@'+mf08:>10} {'A@'+mf+'-B@'+mf:>10}")
print("-" * 92)
for k, label in keys:
    pb = prev_b.get(k) if prev_b else None
    pa = prev_a.get(k) if prev_a else None
    cb = curr_b.get(k); ca = curr_a.get(k)
    d_prev = delta(pb, pa)
    d_curr = delta(cb, ca)
    print(f"{label:<22} {cell(pb)} {cell(pa)} {cell(cb)} {cell(ca)} "
          f"{dcell(d_prev)} {dcell(d_curr)}")

print(f"\nB = baseline (no flags), A = SGLANG_ARENA_SHARED=1 (no xpool)")
print(f"Hypothesis: at mem_frac={mf} (lower) the A-vs-B regression should shrink")
print(f"            because PyTorch+arena no longer over-reserve GPU memory.")
PY
