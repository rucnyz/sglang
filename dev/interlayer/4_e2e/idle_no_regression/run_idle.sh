#!/usr/bin/env bash
# idle_no_regression — idle workload: no fire, no regression (negative test).
#
# Boots sglang twice on the same model + same idle workload:
#   - INTER=0: baseline (budgeter disabled)
#   - INTER=1: with budgeter enabled (Path-A actuator wired)
#
# Drives a low-RPS workload where both pools stay <50% loaded.
# Captures bench.json from each run + budgeter.jsonl from INTER=1.
# validate_idle.py asserts:
#   (a) INTER=1: fire_count ≤ 1 (idle shouldn't trigger fires)
#   (b) |mean_TTFT_inter - mean_TTFT_off| / mean_TTFT_off ≤ 0.02
#
# Each phase ~3 min wall. Total ~6 min.
#
# Usage:
#   GPU=3 PORT=30077 OUT_DIR=/tmp/d8b_run \
#       bash dev/interlayer/4_e2e/idle_no_regression/run_idle.sh

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
# MODEL_NAME defaults to Qwen3.5-9B; override to e.g. Qwen3.5-35B-A3B
# or Qwen3.5-122B-A10B to measure fire-overhead scaling with model size.
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d8b_run}
GPU=${GPU:-3}        # comma-separated for TP>1, e.g. "3,4"
TP=${TP:-1}
PORT=${PORT:-30077}
# WORKLOAD_S=180 → 720 prompts; tightens TTFT mean noise to ~3%
# (sqrt(720) ≈ 27 vs sqrt(240) ≈ 15) so the 2% regression bound
# is statistically meaningful, not noise-bound.
WORKLOAD_S=${WORKLOAD_S:-180}
# MEM_FRAC=0.7 → mamba pool ~787 slots on Qwen3.5-9B; with RPS=4
# random-1024-in/128-out and ~5s avg req lifetime, mamba steady-state
# load stays under 50% (no radix-snapshot saturation that would
# legitimately trigger the planner — that's NOT "idle").
MEM_FRAC=${MEM_FRAC:-0.7}
mkdir -p "$OUT_DIR"

run_phase() {
    local label="$1"            # "off" or "inter"
    local enable_budgeter="$2"  # "0" or "1"
    local log="$OUT_DIR/${label}.server.log"
    local bench="$OUT_DIR/${label}.bench.json"
    local benchlog="$OUT_DIR/${label}.bench.log"
    local budg="$OUT_DIR/${label}.budgeter.jsonl"
    rm -f "$log" "$bench" "$benchlog" "$budg"

    echo "[idle_no_regression] phase $label (budgeter=$enable_budgeter): boot"
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    local env_budget=""
    if [ "$enable_budgeter" = "1" ]; then
        env_budget="SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_HIMA_LOG=$budg"
    fi

    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_budget \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp $TP --mem-fraction-static $MEM_FRAC \
            --max-running-requests 256 \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --log-level info > "$log" 2>&1 &
    local pid=$!

    # Boot timeout 20 min for big models (122B can take 10+ min)
    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[idle_no_regression/$label] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[idle_no_regression/$label] server died"; tail -20 "$log" >&2; return 1
        fi
    done
    [ $waited -ge 1200 ] && { echo "[idle_no_regression/$label] TIMEOUT"; kill -9 $pid; return 1; }

    # Idle workload: RPS=4, short outputs, both pools should stay <50%.
    # Use random dataset so KV gets some load + mamba gets one slot per req.
    # Number of prompts = RPS × WORKLOAD_S.
    echo "[idle_no_regression/$label] idle workload RPS=4 for ${WORKLOAD_S}s"
    $VENV -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
        --dataset-name random \
        --random-input-len 1024 --random-output-len 128 \
        --request-rate 4 \
        --num-prompts $((WORKLOAD_S * 4)) \
        --output-file "$bench" \
        > "$benchlog" 2>&1 || echo "[idle_no_regression/$label] bench rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4
}

run_phase off   0 || { echo "[idle_no_regression] off phase failed"; exit 1; }
run_phase inter 1 || { echo "[idle_no_regression] inter phase failed"; exit 1; }

echo
echo "=== Validation ==="
$VENV dev/interlayer/4_e2e/idle_no_regression/validate_idle.py --out-dir "$OUT_DIR"
