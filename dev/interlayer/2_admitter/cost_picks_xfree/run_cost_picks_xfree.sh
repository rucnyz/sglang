#!/usr/bin/env bash
# cost_picks_xfree — Admitter picks cross-free when cheap (live workload).
#
# design.md §cost_picks_xfree (positive): under a workload where some arrivals see
# i_dst saturated AND i_src has FREE pages, the Admitter must pick
# `cross-free` for ≥80% of those decisions, and 0 must pick `defer`
# while i_src still has FREE pages.
#
# Single phase (Admitter enabled). The conjecture is decisional —
# validated from the per-arrival JSONL log, not from throughput.
#
# Setup:
#   - SGLANG_HIMA=1 (enables both Budgeter and Admitter with cross-fire)
#     + SGLANG_HIMA_ADMITTER_LOG=$ADMITTER_LOG (JSONL of every decision)
#   - Budgeter fires warm the c^xfer EWMA so the
#     cold-start cross-* gate clears
#   - Workload knobs (defaults below; the persisted PASS run at
#     run_2026-05-29/ used the README's tighter overrides):
#       MEM_FRACTION=0.55  (PASS run: 0.40)
#       MAMBA_CAP=256      (PASS run: 512)
#       MAX_RUNNING=256
#       INPUT_LEN=8192     (PASS run: 16384)
#       OUTPUT_LEN=512
#       RPS=16             (PASS run: 20)
#       WORKLOAD_S=120     (PASS run: 420)
#
# Validator parses the JSONL log and asserts:
#   §cost_picks_xfree (positive) :
#       (a) ≥100 cross-pool-feasible decisions seen (statistical floor)
#       (b) ≥80% of contentious arrivals (own_free=null) chose cross_free
#       (c) 0 decisions chose 'defer' while cross_free was finite
#   sweep-arm validation    :
#       cross_free / (cross_free + cross_evict) ≥ 0.95

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d6_run}
GPU=${GPU:-3}
TP=${TP:-1}
PORT=${PORT:-30077}
WORKLOAD_S=${WORKLOAD_S:-120}
# Workload knobs (KV-bound by default — Phase 5 Admitter is dst='kv' only).
MEM_FRACTION=${MEM_FRACTION:-0.55}
MAMBA_CAP=${MAMBA_CAP:-256}
MAX_RUNNING=${MAX_RUNNING:-256}
INPUT_LEN=${INPUT_LEN:-8192}
OUTPUT_LEN=${OUTPUT_LEN:-512}
RPS=${RPS:-16}
mkdir -p "$OUT_DIR"

LABEL=inter
LOG="$OUT_DIR/${LABEL}.server.log"
BENCH="$OUT_DIR/${LABEL}.bench.json"
BENCHLOG="$OUT_DIR/${LABEL}.bench.log"
BUDG="$OUT_DIR/${LABEL}.budgeter.jsonl"
ADM="$OUT_DIR/${LABEL}.admitter.jsonl"

rm -f "$LOG" "$BENCH" "$BENCHLOG" "$BUDG" "$ADM"

pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
sleep 3

echo "[cost_picks_xfree/$LABEL] boot (SGLANG_HIMA=1)"
env \
    SGLANG_HIMA=1 \
    SGLANG_HIMA_TICK_S=1.0 \
    SGLANG_HIMA_LOG="$BUDG" \
    SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.50} \
    SGLANG_HIMA_ADMITTER_LOG="$ADM" \
    SGLANG_XPOOL_QUEUE_WAIT_US=${SGLANG_XPOOL_QUEUE_WAIT_US:-125000} \
    CUDA_VISIBLE_DEVICES=$GPU \
    nohup $VENV -m sglang.launch_server \
        --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
        --tp $TP --mem-fraction-static "$MEM_FRACTION" \
        --max-running-requests "$MAX_RUNNING" \
        --max-mamba-cache-size "$MAMBA_CAP" \
        --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
        --log-level info > "$LOG" 2>&1 &
PID=$!

waited=0
while [ $waited -lt 1200 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
        echo "[cost_picks_xfree/$LABEL] ready after ${waited}s"; break
    fi
    if ! kill -0 $PID 2>/dev/null; then
        echo "[cost_picks_xfree/$LABEL] server died"; tail -25 "$LOG"; exit 1
    fi
done
[ $waited -ge 1200 ] && { echo "[cost_picks_xfree/$LABEL] TIMEOUT"; kill -9 $PID; exit 1; }

echo "[cost_picks_xfree/$LABEL] driving KV-bound workload for ${WORKLOAD_S}s (input=$INPUT_LEN output=$OUTPUT_LEN RPS=$RPS)"
$VENV -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --request-rate "$RPS" \
    --num-prompts $((WORKLOAD_S * RPS)) \
    --output-file "$BENCH" \
    > "$BENCHLOG" 2>&1 || echo "[cost_picks_xfree/$LABEL] bench rc=$?"

kill -9 $PID 2>/dev/null
sleep 4

echo
echo "=== §cost_picks_xfree validation ==="
$VENV dev/interlayer/2_admitter/cost_picks_xfree/validate_d6.py --admitter-log "$ADM"

echo
echo "=== sweep-arm validation ==="
$VENV dev/interlayer/2_admitter/cost_picks_xfree/validate_d3.py --admitter-log "$ADM"
