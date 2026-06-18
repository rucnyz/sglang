#!/usr/bin/env bash
# D8 bisection: ARENA tensors ON, BUDGETER OFF, NO FIRES.
#
# Hypothesis test:
#   if arena (cuMemMap-backed tensor) alone causes -4% TPOT,
#   this run will show ~ same regression as full D8 inter.
#   if neutral, the regression is from budgeter+fires combination.
#
# Same workload + model as D8_saturated.sh, shorter duration
# (90s instead of 180s) to fit in profiling time budget.

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d8_arena_only}
GPU=${GPU:-3}
PORT=${PORT:-30077}
WORKLOAD_S=${WORKLOAD_S:-90}
mkdir -p "$OUT_DIR"

run_phase() {
    local label="$1"           # off / arena_only
    local mode="$2"            # plain / arena
    local log="$OUT_DIR/${label}.server.log"
    local bench="$OUT_DIR/${label}.bench.json"
    local benchlog="$OUT_DIR/${label}.bench.log"
    rm -f "$log" "$bench" "$benchlog"

    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 4

    local env_arena=""
    if [ "$mode" = "arena" ]; then
        # Arena tensors ON, HiMA OFF → no fires, but mamba+kv pools
        # use cuMemMap-backed tensors (same path as inter).
        env_arena="SGLANG_ARENA_SHARED=1 SGLANG_MAMBA_ARENA=1"
    fi

    echo "[D8/$label] boot mode=$mode env_arena=$env_arena"
    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_arena \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp 1 --mem-fraction-static 0.70 \
            --max-running-requests 256 \
            --max-mamba-cache-size 100 \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --log-level info > "$log" 2>&1 &
    local pid=$!

    local waited=0
    while [ $waited -lt 600 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[D8/$label] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[D8/$label] server died"; tail -30 "$log"; return 1
        fi
    done
    [ $waited -ge 600 ] && { echo "[D8/$label] TIMEOUT"; kill -9 $pid; return 1; }

    echo "[D8/$label] driving workload for ${WORKLOAD_S}s"
    $VENV -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
        --dataset-name random \
        --random-input-len 256 --random-output-len 1024 \
        --request-rate 32 \
        --num-prompts $((WORKLOAD_S * 32)) \
        --output-file "$bench" \
        > "$benchlog" 2>&1 || echo "[D8/$label] bench rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4
}

run_phase off        plain  || { echo "[D8] off phase failed"; exit 1; }
run_phase arena_only arena  || { echo "[D8] arena_only phase failed"; exit 1; }

echo
echo "=== Bisection summary ==="
$VENV - <<'PY'
import json
for f in ['off','arena_only']:
    try:
        d = json.load(open(f'/tmp/d8_arena_only/{f}.bench.json'))
        print(f'== {f} ==')
        print(f"  duration {d['duration']:.2f}s  rps {d['request_throughput']:.3f}")
        print(f"  TPOT mean {d['mean_tpot_ms']:.3f} ms (p99 {d['p99_tpot_ms']:.3f})")
        print(f"  TTFT mean {d['mean_ttft_ms']:.0f} ms")
        print(f"  out_throughput {d['output_throughput']:.0f} tok/s")
    except FileNotFoundError:
        print(f'  no {f}.bench.json')

# If both files present, compute delta
try:
    o = json.load(open('/tmp/d8_arena_only/off.bench.json'))
    a = json.load(open('/tmp/d8_arena_only/arena_only.bench.json'))
    print()
    print(f"Δ TPOT:        {(a['mean_tpot_ms']/o['mean_tpot_ms']-1)*100:+.2f}%")
    print(f"Δ throughput:  {(a['request_throughput']/o['request_throughput']-1)*100:+.2f}%")
    print(f"Δ duration:    {(a['duration']/o['duration']-1)*100:+.2f}%")
except FileNotFoundError:
    pass
PY
