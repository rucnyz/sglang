#!/usr/bin/env bash
# alternating_saturation — adversarial alternating-saturation workload.
#
# design.md §alternating_saturation: 5-minute alternating-saturation workload, period 2s
# (= 2× budgeter tick at 1Hz). Pass criterion: output_throughput_inter
# ≥ output_throughput_off × 0.95 (no > 5% regression from over-firing).
#
# Two phases: off (budgeter=0) vs inter (budgeter=1, fires enabled).
# Each phase drives the same alternating workload via
# payload_adversarial.py (custom dispatcher; bench_serving doesn't support
# phase-switching prompt mixes).

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d8c_run}
GPU=${GPU:-3}
PORT=${PORT:-30077}
DURATION_S=${DURATION_S:-300}     # 5 min default
PHASE_S=${PHASE_S:-2.0}
RPS=${RPS:-100}                   # need higher to actually saturate
OUTPUT_LEN=${OUTPUT_LEN:-1024}    # match saturated_bubble: keep reqs alive long enough
                                  # to push usage_mamba_active past 0.50
KV_INPUT_LEN=${KV_INPUT_LEN:-2048}    # long KV phase
MAMBA_INPUT_LEN=${MAMBA_INPUT_LEN:-64}  # short mamba phase
mkdir -p "$OUT_DIR"

run_phase() {
    local label="$1"            # off / inter
    local enable_budgeter="$2"  # 0 / 1
    local log="$OUT_DIR/${label}.server.log"
    local bench="$OUT_DIR/${label}.bench.json"
    local benchlog="$OUT_DIR/${label}.driver.log"
    local budg="$OUT_DIR/${label}.budgeter.jsonl"
    rm -f "$log" "$bench" "$benchlog" "$budg"

    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 4

    echo "[alternating_saturation/$label] boot (budgeter=$enable_budgeter)"
    local env_budget=""
    if [ "$enable_budgeter" = "1" ]; then
        env_budget="SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_HIMA_LOG=$budg SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.50}"
    fi

    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_budget \
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
        sleep 10; waited=$((waited+10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[alternating_saturation/$label] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[alternating_saturation/$label] DIED at ${waited}s"; tail -25 "$log"; return 1
        fi
    done

    echo "[alternating_saturation/$label] driving adversarial workload for ${DURATION_S}s (phase=${PHASE_S}s, rps=${RPS}, out=${OUTPUT_LEN}, kv_in=${KV_INPUT_LEN}, mamba_in=${MAMBA_INPUT_LEN})"
    $VENV dev/interlayer/4_e2e/alternating_saturation/payload_adversarial.py \
        --host 127.0.0.1 --port $PORT \
        --duration $DURATION_S --phase_s $PHASE_S --rps $RPS \
        --output_len $OUTPUT_LEN \
        --kv_input_len $KV_INPUT_LEN \
        --mamba_input_len $MAMBA_INPUT_LEN \
        --out "$bench" > "$benchlog" 2>&1

    kill -9 $pid 2>/dev/null
    sleep 4
}

run_phase off   0 || { echo "[alternating_saturation] off phase failed"; exit 1; }
run_phase inter 1 || { echo "[alternating_saturation] inter phase failed"; exit 1; }

echo
echo "=== Validation ==="
$VENV dev/interlayer/4_e2e/alternating_saturation/validate_adversarial.py --out-dir "$OUT_DIR"
