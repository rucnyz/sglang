#!/usr/bin/env bash
# saturated_bubble — saturated single-pool: bubble harvest +10% throughput.
#
# Validates the core selling point of the interlayer mechanism: under
# a workload where one pool is admission-bottlenecked (mamba here),
# enabling the budgeter should harvest "bubbles" (cross-pool capacity
# that the bottlenecked side can borrow), measurably raising throughput.
#
# Compares 2 phases on identical workload:
#   off   : no SGLANG_HIMA (no actuator, no cross-pool transfers)
#   inter : SGLANG_HIMA=1 (HiMA enabled, fires move chunks)
#
# Workload (same as D7 v5 — saturated mamba via forced small pool):
#   - mem-fraction=0.70, --max-mamba-cache-size=100, max-running=256
#   - random dataset, RPS=32, input=256, output=1024
#   - WORKLOAD_S=180 (longer than byte_transfer so steady-state dominates ramp)
#
# Both phases use SGLANG_XPOOL_MAMBA_HIGH=0.50 so the planner can
# trigger fires on byte_transfer's sustained 0.66 active reading (same rationale
# as byte_transfer — sglang queue not visible + phantom-active capping).
#
# validate_saturated.py asserts:
#   (a) inter completed_reqs/s >= off completed_reqs/s × 1.10
#       (the +10% throughput bound from design.md headline)
#   (b) inter total_completed >= off total_completed (covers the case
#       where RPS-driven completion doesn't fully reflect harvest)
#   (c) inter has ≥ 1 non-aborted fire (otherwise we're measuring noise)

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d8_run}
GPU=${GPU:-3}
TP=${TP:-1}
PORT=${PORT:-30077}
WORKLOAD_S=${WORKLOAD_S:-180}
mkdir -p "$OUT_DIR"

run_phase() {
    local label="$1"            # off / inter
    local enable_budgeter="$2"  # 0 / 1
    local log="$OUT_DIR/${label}.server.log"
    local bench="$OUT_DIR/${label}.bench.json"
    local benchlog="$OUT_DIR/${label}.bench.log"
    local budg="$OUT_DIR/${label}.budgeter.jsonl"
    rm -f "$log" "$bench" "$benchlog" "$budg"

    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    echo "[saturated_bubble/$label] boot (budgeter=$enable_budgeter)"
    local env_budget=""
    if [ "$enable_budgeter" = "1" ]; then
        env_budget="SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_HIMA_LOG=$budg SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.50}"
    fi

    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_budget \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp $TP --mem-fraction-static 0.70 \
            --max-running-requests 256 \
            --max-mamba-cache-size 100 \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --log-level info > "$log" 2>&1 &
    local pid=$!

    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[saturated_bubble/$label] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[saturated_bubble/$label] server died"; tail -25 "$log"; return 1
        fi
    done
    [ $waited -ge 1200 ] && { echo "[saturated_bubble/$label] TIMEOUT"; kill -9 $pid; return 1; }

    echo "[saturated_bubble/$label] driving mamba-saturating workload for ${WORKLOAD_S}s"
    $VENV -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
        --dataset-name random \
        --random-input-len 256 --random-output-len 1024 \
        --request-rate 32 \
        --num-prompts $((WORKLOAD_S * 32)) \
        --output-file "$bench" \
        > "$benchlog" 2>&1 || echo "[saturated_bubble/$label] bench rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4
}

run_phase off   0 || { echo "[saturated_bubble] off phase failed"; exit 1; }
run_phase inter 1 || { echo "[saturated_bubble] inter phase failed"; exit 1; }

echo
echo "=== Validation ==="
$VENV dev/interlayer/4_e2e/saturated_bubble/validate_saturated.py --out-dir "$OUT_DIR"
