#!/bin/bash
# Phase 2e.5.6.3.b — verify Cost 1 root cause: arena growth headroom
# changes engine-visible tensor shape → Triton kernel shape-specialization
# overhead.
#
# Three-way:
#   D = SGLANG_ARENA_SHARED=1 + headroom=0 (no headroom, tensor shape
#       matches baseline). Compared to existing baseline + arena_only
#       at default (4-chunk) headroom.
#
# If D's TTFT matches baseline (close to 0% delta), the root cause is
# the headroom-driven shape mismatch.
# If D still regresses ~6%, root cause is something else (allocator
# callback overhead, MemPool indirection, etc.).

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

OUT_DIR=/tmp/arena_no_headroom_$$
mkdir -p "$OUT_DIR"
log="$OUT_DIR/arena_no_headroom_server.log"

echo "out_dir=$OUT_DIR  mem_fraction_static=$MEM_FRAC headroom=0"
pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env CUDA_VISIBLE_DEVICES="$GPUS" \
  SGLANG_ARENA_SHARED=1 \
  SGLANG_ARENA_KV_HEADROOM_CHUNKS=0 \
  SGLANG_ARENA_MAMBA_HEADROOM_CHUNKS=0 \
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
  echo "server failed at headroom=0"
  tail -30 "$log"
  kill -9 $PID 2>/dev/null || true
  exit 1
fi

# Confirm tensor shape matches baseline.
KV_LINE=$(grep "MHATokenToKVPool arena: tot_tokens" "$log" | head -1 || true)
MAMBA_LINE=$(grep "MambaPool arena: tot=" "$log" | head -1 || true)
SSM_SIZE=$(grep "ssm_state size:" "$log" | head -1 || true)
echo "KV arena init/max:    $KV_LINE"
echo "Mamba arena init/max: $MAMBA_LINE"
echo "ssm_state size:       $SSM_SIZE"

bench_out="$OUT_DIR/arena_no_headroom_bench.json"
echo "running bench..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts $NUM_PROMPTS \
  --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN \
  --request-rate $RPS \
  --output-file "$bench_out" \
  >"$OUT_DIR/arena_no_headroom_bench.log" 2>&1
echo "bench done"

kill -9 $PID 2>/dev/null || true

echo "=== compare 3-way at mem_frac=$MEM_FRAC ==="
BASELINE_JSON=${BASELINE_JSON:-/tmp/xpool_perf_1301932/baseline_bench.json}
ARENA_DEFAULT_JSON=${ARENA_DEFAULT_JSON:-/tmp/arena_only_perf_1371334/arena_only_bench.json}

BASELINE_JSON=$BASELINE_JSON \
  ARENA_DEFAULT_JSON=$ARENA_DEFAULT_JSON \
  ARENA_NOHEAD_JSON=$bench_out \
  .venv/bin/python <<'PY'
import json, os, sys

def load(p):
    if not p or not os.path.exists(p):
        return None
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

a = load(os.environ["BASELINE_JSON"])
d = load(os.environ["ARENA_DEFAULT_JSON"])
n = load(os.environ["ARENA_NOHEAD_JSON"])

if a is None or n is None:
    print(f"MISSING: baseline={a is not None}, no_head={n is not None}")
    sys.exit(1)

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

print(f"\n{'metric':<22} {'A=baseline':>11} {'D=defaultHR':>12} {'N=noheadHR':>11} "
      f"{'D vs A':>9} {'N vs A':>9}")
print("-" * 86)
for k, label in keys:
    av = a.get(k); dv = d.get(k) if d else None; nv = n.get(k)
    pda = pct(av, dv); pna = pct(av, nv)
    cell = lambda v: f"{v:>10.2f}" if v is not None else f"{'N/A':>11}"
    pcell = lambda v: f"{v:>+8.2f}%" if v is not None else f"{'N/A':>9}"
    print(f"{label:<22} {av:>11.2f} {cell(dv):>12} {nv:>11.2f} {pcell(pda)} {pcell(pna)}")
print("\nA = baseline (no flags)")
print("D = SGLANG_ARENA_SHARED=1, default 4-chunk headroom (the regressing config)")
print("N = SGLANG_ARENA_SHARED=1, headroom=0 (tensor shape matches baseline)")
print("\nIf N close to A: shape-specialization is the root cause.")
print("If N still regresses ~6%: root cause is elsewhere (allocator callback?).")
PY
