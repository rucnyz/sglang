#!/usr/bin/env bash
# burst_recovery — Admitter handles a sudden burst synchronously.
#
# design.md §burst_recovery (the headline SLO claim for the Admitter):
#   - Phase A (60s, RPS=2):   low-rate workload; Budgeter shrinks one
#                              pool toward its working set
#   - Phase B (10s, RPS=128): sudden burst of mamba-needing reqs
#   - Compare queue_p99 during Phase B's first 5s against a baseline
#     run with `inter` disabled.
#   - Pass: queue_p99_phase_B_inter ≤ queue_p99_phase_B_baseline × 1.10
#
# NOTE on direction limitation: design.md assumed the Admitter handles
# both pool directions. Phase 5's `decide_for_req` is currently
# dst='kv', src='mamba' only — so the "burst" must be a KV-bound burst
# (long inputs that pressure KV), not the spec's mamba-bound burst.
# Adjusted accordingly; super-capacity burst (pending) is left for Phase 8.
#
# Workload pattern adapted for Phase 5:
#   - Phase A (60s, RPS=2, INPUT_LEN=256): KV-light cruise, mamba working
#     set ~RPS×output_duration. Lets Budgeter shrink KV pool / mamba may
#     also drift down.
#   - Phase B (10s, RPS=128, INPUT_LEN=2048): sudden KV-saturating burst
#     where Admitter must fire mamba→KV synchronously to keep up.

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d11_run}
GPU=${GPU:-3}
TP=${TP:-1}
PORT=${PORT:-30077}

# Phase knobs
PHASE_A_S=${PHASE_A_S:-60}
PHASE_A_RPS=${PHASE_A_RPS:-2}
PHASE_A_INPUT=${PHASE_A_INPUT:-256}
PHASE_A_OUTPUT=${PHASE_A_OUTPUT:-256}

PHASE_B_S=${PHASE_B_S:-10}
PHASE_B_RPS=${PHASE_B_RPS:-128}
PHASE_B_INPUT=${PHASE_B_INPUT:-2048}
PHASE_B_OUTPUT=${PHASE_B_OUTPUT:-256}

MEM_FRACTION=${MEM_FRACTION:-0.45}
MAMBA_CAP=${MAMBA_CAP:-512}
MAX_RUNNING=${MAX_RUNNING:-256}

mkdir -p "$OUT_DIR"

run_phase_pair() {
    local label="$1"             # off / inter
    local enable_inter="$2"      # 0 / 1
    local log="$OUT_DIR/${label}.server.log"
    local benchA="$OUT_DIR/${label}.phaseA.bench.json"
    local benchB="$OUT_DIR/${label}.phaseB.bench.json"
    local benchAlog="$OUT_DIR/${label}.phaseA.bench.log"
    local benchBlog="$OUT_DIR/${label}.phaseB.bench.log"
    local budg="$OUT_DIR/${label}.budgeter.jsonl"
    local adm="$OUT_DIR/${label}.admitter.jsonl"

    rm -f "$log" "$benchA" "$benchB" "$benchAlog" "$benchBlog" "$budg" "$adm"
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    # CUDA_LAUNCH_BLOCKING=1: even after fixes, inter mode crashes
    # with async "illegal memory access" — need the true synchronous
    # crash site to know what's actually wrong.
    local env_str="CUDA_LAUNCH_BLOCKING=1"
    if [ "$enable_inter" = "1" ]; then
        env_str="$env_str \
                 SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_HIMA_LOG=$budg \
                 SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.50} \
                 SGLANG_HIMA_ADMITTER_LOG=$adm \
                 SGLANG_XPOOL_QUEUE_WAIT_US=${SGLANG_XPOOL_QUEUE_WAIT_US:-125000}"
    fi

    echo "[burst_recovery/$label] boot (inter=$enable_inter)"
    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_str \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp $TP --mem-fraction-static "$MEM_FRACTION" \
            --max-running-requests "$MAX_RUNNING" \
            --max-mamba-cache-size "$MAMBA_CAP" \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --log-level info > "$log" 2>&1 &
    local pid=$!

    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[burst_recovery/$label] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[burst_recovery/$label] server died"; tail -25 "$log"; return 1
        fi
    done
    [ $waited -ge 1200 ] && { echo "[burst_recovery/$label] TIMEOUT"; kill -9 $pid; return 1; }

    echo "[burst_recovery/$label] Phase A: RPS=$PHASE_A_RPS input=$PHASE_A_INPUT for ${PHASE_A_S}s"
    $VENV -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
        --dataset-name random \
        --random-input-len "$PHASE_A_INPUT" --random-output-len "$PHASE_A_OUTPUT" \
        --request-rate "$PHASE_A_RPS" \
        --num-prompts $((PHASE_A_S * PHASE_A_RPS)) \
        --output-file "$benchA" \
        > "$benchAlog" 2>&1 || echo "[burst_recovery/$label] phaseA rc=$?"

    echo "[burst_recovery/$label] Phase B: BURST RPS=$PHASE_B_RPS input=$PHASE_B_INPUT for ${PHASE_B_S}s"
    $VENV -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
        --dataset-name random \
        --random-input-len "$PHASE_B_INPUT" --random-output-len "$PHASE_B_OUTPUT" \
        --request-rate "$PHASE_B_RPS" \
        --num-prompts $((PHASE_B_S * PHASE_B_RPS)) \
        --output-file "$benchB" \
        > "$benchBlog" 2>&1 || echo "[burst_recovery/$label] phaseB rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4
}

run_phase_pair off   0 || { echo "[burst_recovery] off failed"; exit 1; }
run_phase_pair inter 1 || { echo "[burst_recovery] inter failed"; exit 1; }

echo
echo "=== burst_recovery validation ==="
$VENV dev/interlayer/4_e2e/burst_recovery/validate_burst.py --out-dir "$OUT_DIR"
