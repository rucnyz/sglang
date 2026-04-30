#!/bin/bash
# Phase 2e.5.6.3.b — diagnostic: arena-only (no xpool transfers) perf.
#
# 18_kv_mamba_xfer_perf.sh found a +5-13% latency regression for
# baseline -> shared+xpool+coordinated. To pin the cost: this script
# runs ONLY arm C = SGLANG_ARENA_SHARED=1 with NO xpool/coord, against
# an existing baseline bench JSON.
#
# If arm C is clean (~0% delta) → cost is the transfers (cuMemMap
#   stalls during runtime), not the arena. Fix is policy-side.
# If arm C also regresses → arena alloc path itself is paying overhead
#   (capping logic, MemPool indirection). Fix is mechanism-side.
#
# Pre-req: existing baseline JSON. Pass via env BASELINE_JSON=... .

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
WARMUP_S=${WARMUP_S:-360}

BASELINE_JSON=${BASELINE_JSON:-/tmp/xpool_perf_1301932/baseline_bench.json}
COORD_JSON=${COORD_JSON:-/tmp/xpool_perf_1301932/shared_xpool_coord_bench.json}

if [ ! -s "$BASELINE_JSON" ]; then
  echo "BASELINE_JSON missing or empty: $BASELINE_JSON"
  exit 1
fi

OUT_DIR=/tmp/arena_only_perf_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR  reusing baseline=$BASELINE_JSON  coord=$COORD_JSON"

# Boot ARM C: SGLANG_ARENA_SHARED=1 only. No xpool, no budgeter.
arm=arena_only
log="$OUT_DIR/${arm}_server.log"
echo "=== arm=$arm (SGLANG_ARENA_SHARED=1, no xpool) ==="
pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env CUDA_VISIBLE_DEVICES="$GPUS" \
  SGLANG_ARENA_SHARED=1 \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size $TP \
    --mem-fraction-static $MEM_FRAC --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
PID=$!
echo "[$arm] pid=$PID; waiting up to ${WARMUP_S}s"
echo "--- live server log ---"
tail -F "$log" 2>/dev/null &
TAILER=$!
waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    kill $TAILER 2>/dev/null || true
    wait $TAILER 2>/dev/null || true
    echo
    echo "--- ready after ${waited}s ---"
    break
  fi
done
if kill -0 $TAILER 2>/dev/null; then
  kill $TAILER 2>/dev/null || true
  wait $TAILER 2>/dev/null || true
fi
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
  echo "[$arm] server failed to start"
  tail -30 "$log"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

bench_out="$OUT_DIR/${arm}_bench.json"
echo "running bench for arm=$arm..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts $NUM_PROMPTS \
  --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
  --request-rate $RPS \
  --output-file "$bench_out" \
  >"$OUT_DIR/${arm}_bench.log" 2>&1
echo "bench done"

# Sanity: verify arena built but no xpool transfers.
shared_line=$(grep "Arena shared mode" "$log" | head -1 || true)
xpool_attached=$(grep "BudgetAgent xpool: actuator attached" "$log" | head -1 || true)
echo "shared engaged: ${shared_line:-(none)}"
echo "xpool attached: ${xpool_attached:-(none, expected since BUDGETER_XPOOL_DEMO unset)}"
if [ -n "$xpool_attached" ]; then
  echo "FAIL: xpool actuator attached but should not have been (this run is for isolating arena overhead)"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

kill -9 $PID 2>/dev/null || true
sleep 3

echo "=== compare 3-way ==="
BASELINE_JSON=$BASELINE_JSON COORD_JSON=$COORD_JSON ARENA_JSON=$bench_out \
  .venv/bin/python <<'PY'
import json, os, sys

def load(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

a = load(os.environ["BASELINE_JSON"])
c = load(os.environ["ARENA_JSON"])
b = load(os.environ.get("COORD_JSON", ""))

if a is None or c is None:
    print(f"MISSING: baseline={a is not None} arena_only={c is not None}")
    sys.exit(1)

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

def delta(av, bv):
    if av in (None, 0) or bv is None:
        return None
    return (bv - av) / av * 100

print(f"\n{'metric':<28} {'baseline':>10} {'arena_only':>11} {'shared+xc':>10} "
      f"{'A→C %':>9} {'A→B %':>9} {'C→B %':>9}")
print("-" * 96)
for k, label in keys:
    av = a.get(k); cv = c.get(k); bv = b.get(k) if b else None
    dac = delta(av, cv); dab = delta(av, bv) if bv else None
    dcb = delta(cv, bv) if bv else None
    cell = lambda x: f"{x:>+8.2f}%" if x is not None else f"{'N/A':>9}"
    bvs = f"{bv:>10.2f}" if bv is not None else f"{'N/A':>10}"
    print(f"{label:<28} {av:>10.2f} {cv:>11.2f} {bvs} "
          f"{cell(dac)} {cell(dab) if dab is not None else cell(None):>9} "
          f"{cell(dcb) if dcb is not None else cell(None):>9}")

print("\nA = baseline (no flags)")
print("C = SGLANG_ARENA_SHARED=1 only (arena built, NO xpool transfers)")
print("B = SGLANG_ARENA_SHARED + xpool + coord (the regressing config)")
print("\nA→C reveals the arena/allocator overhead alone.")
print("A→B is the total cost (arena + transfers).")
print("C→B isolates the transfer/coordinator cost.")
PY
