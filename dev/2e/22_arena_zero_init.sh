#!/bin/bash
# Phase 2e.5.6.3.b — verify hypothesis (2): pages allocated via torch.empty
# (no first-touch write) take a slow path on first GPU read.
#
# Test: SGLANG_ARENA_ZERO_INIT_LIVE=1 makes the arena's MultiTensorArena
# explicitly zero the live region after torch.empty. Same physical state
# baseline gets via torch.zeros.

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

OUT_DIR=/tmp/arena_zeroinit_$$
mkdir -p "$OUT_DIR"
log="$OUT_DIR/arena_zeroinit_server.log"
echo "out_dir=$OUT_DIR"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env CUDA_VISIBLE_DEVICES="$GPUS" \
  SGLANG_ARENA_SHARED=1 \
  SGLANG_ARENA_KV_HEADROOM_CHUNKS=4 \
  SGLANG_ARENA_MAMBA_HEADROOM_CHUNKS=4 \
  SGLANG_ARENA_ZERO_INIT_LIVE=1 \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size $TP \
    --mem-fraction-static $MEM_FRAC --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
PID=$!
echo "pid=$PID log=$log"
echo "--- live server log ---"
tail -F "$log" 2>/dev/null &
TAILER=$!
waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    kill $TAILER 2>/dev/null || true; wait $TAILER 2>/dev/null || true
    echo; echo "--- ready after ${waited}s ---"
    break
  fi
done
if kill -0 $TAILER 2>/dev/null; then kill $TAILER 2>/dev/null || true; wait $TAILER 2>/dev/null || true; fi
if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
  echo "server failed"; tail -30 "$log"; kill -9 $PID 2>/dev/null || true; exit 1
fi

bench_out="$OUT_DIR/arena_zeroinit_bench.json"
echo "running bench..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts $NUM_PROMPTS \
  --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
  --request-rate $RPS \
  --output-file "$bench_out" \
  >"$OUT_DIR/arena_zeroinit_bench.log" 2>&1
echo "bench done"
kill -9 $PID 2>/dev/null || true

echo "=== compare 4-way at mem_frac=$MEM_FRAC ==="
BASELINE_JSON=${BASELINE_JSON:-/tmp/xpool_perf_1301932/baseline_bench.json}
ARENA_DEFAULT_JSON=${ARENA_DEFAULT_JSON:-/tmp/arena_only_perf_1371334/arena_only_bench.json}
ARENA_NOHEAD_JSON=${ARENA_NOHEAD_JSON:-/tmp/arena_no_headroom_*/arena_no_headroom_bench.json}
ARENA_NOHEAD_RES=$(ls $ARENA_NOHEAD_JSON 2>/dev/null | head -1)

BASELINE_JSON=$BASELINE_JSON \
  ARENA_DEFAULT_JSON=$ARENA_DEFAULT_JSON \
  ARENA_NOHEAD_JSON=$ARENA_NOHEAD_RES \
  ARENA_ZEROINIT_JSON=$bench_out \
  .venv/bin/python <<'PY'
import json, os, sys

def load(p):
    if not p or not os.path.exists(p):
        return None
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

a = load(os.environ["BASELINE_JSON"])
d = load(os.environ.get("ARENA_DEFAULT_JSON"))
n = load(os.environ.get("ARENA_NOHEAD_JSON"))
z = load(os.environ["ARENA_ZEROINIT_JSON"])

if a is None or z is None:
    print("MISSING json"); sys.exit(1)

keys = [
    ("input_throughput", "input toks/s"),
    ("mean_ttft_ms", "mean TTFT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("mean_tpot_ms", "mean TPOT (ms)"),
    ("p99_tpot_ms", "P99 TPOT (ms)"),
    ("median_e2e_latency_ms", "median E2E (ms)"),
]

def pct(av, bv):
    if av in (None, 0) or bv is None: return None
    return (bv - av) / av * 100

def cell(v): return f"{v:>9.2f}" if v is not None else f"{'N/A':>9}"
def pcell(v): return f"{v:>+8.2f}%" if v is not None else f"{'N/A':>9}"

print(f"\n{'metric':<22} {'A=baseline':>10} {'D=def4HR':>9} {'N=noHR':>9} {'Z=zeroLive':>11} "
      f"{'D vs A':>9} {'N vs A':>9} {'Z vs A':>9}")
print("-" * 110)
for k, label in keys:
    av = a.get(k); dv = d.get(k) if d else None; nv = n.get(k) if n else None; zv = z.get(k)
    pda = pct(av, dv); pna = pct(av, nv); pza = pct(av, zv)
    print(f"{label:<22} {av:>10.2f} {cell(dv)} {cell(nv)} {zv:>11.2f} "
          f"{pcell(pda)} {pcell(pna)} {pcell(pza)}")
print("\nA = baseline, D = +4 chunks headroom (default), N = 0 headroom, Z = 4 chunks + zero-init live")
print("If Z close to A: first-touch / page-init is the cost.")
print("If Z still regresses: cost is elsewhere (allocator path / kernel launch / capture mismatch).")
PY
