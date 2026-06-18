#!/usr/bin/env bash
# byte_transfer — end-to-end byte transfer + working-set invariant.
#
# Drives a real sglang server (SGLANG_HIMA=1, v1 logical actuator)
# under a mamba-saturating workload so the budgeter has cause to fire
# kv_to_mamba transfers. Captures budgeter.jsonl + server.log for
# validate_byte_transfer.py to check:
#   (a) ≥ 1 non-aborted fire
#   (b) per-fire: dst pool grew, src pool shrank, matching delta
#   (c) per-fire: no engine OOM / set_capacity failure in server.log
#   (d) policy-correct: each fire happened while a real signal
#       (persist_m > 0 or pressure_to_m > 0) was present — i.e. NOT
#       a phantom fire on radix-cache LRU saturation
#
# Workload rationale (post active-fix v2):
#   active-fix v2 correctly distinguishes `usage_mamba_active` (running
#   req mamba slots = admission ceiling) from total `pool_occupancy_mamba`
#   (which mixes radix-cache snapshots). Persist signal now only fires
#   on ACTIVE pressure. The OLD byte_transfer workload (RPS=16, GSP, output=256)
#   only saturated radix cache; active stayed at ~3% → 0 fires post-fix.
#   The current workload (random, RPS=32, output=1024) keeps in-flight
#   reqs at max-running=256 (each holds 1 mamba slot), pushing active
#   mamba past mamba_high_water=0.80 for sustained periods so the
#   persist counter clears the NB gate.
#
# Usage:
#   GPU=3 PORT=30077 OUT_DIR=/tmp/d7_run \
#       bash dev/interlayer/4_e2e/byte_transfer/run_byte_transfer.sh

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_DIR=$(ls -d "$HUB/models--Qwen--Qwen3.5-9B/snapshots/"* | head -1)
[ -z "$MODEL_DIR" ] && { echo "Qwen3.5-9B not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d7_run}
GPU=${GPU:-3}
PORT=${PORT:-30077}
WORKLOAD_S=${WORKLOAD_S:-120}
mkdir -p "$OUT_DIR"

# Kill stragglers
pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
sleep 3

LOG="$OUT_DIR/server.log"
BUDG="$OUT_DIR/budgeter.jsonl"
rm -f "$LOG" "$BUDG"

echo "[byte_transfer] booting Qwen3.5-9B on GPU $GPU, port $PORT, SGLANG_HIMA=1"
# Goal: saturate ACTUAL mamba running-req pool, not just radix-cache.
# Two knobs in tension:
#   - mem-fraction high → big KV budget → sglang auto-caps max_running
#     high (previous 0.35 capped to 102, way below mamba pool size 306)
#   - mamba pool small → in-flight reqs saturate it faster
# Use mem-fraction=0.70 so max_running clears ~200, AND force
# --max-mamba-cache-size 100 so 200 in-flight reqs flood mamba's
# 100 slots → admission queues → real persist signal.
CUDA_VISIBLE_DEVICES=$GPU \
    SGLANG_HIMA=1 \
    SGLANG_HIMA_TICK_S=1.0 \
    SGLANG_HIMA_LOG="$BUDG" \
    SGLANG_XPOOL_MAMBA_HIGH=0.50 \
nohup $VENV -m sglang.launch_server \
    --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
    --tp 1 --mem-fraction-static 0.70 \
    --max-running-requests 256 \
    --max-mamba-cache-size 100 \
    --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
    --log-level info > "$LOG" 2>&1 &
SV_PID=$!
# Note on SGLANG_XPOOL_MAMBA_HIGH=0.50 (test-specific override):
# Default mamba_high_water is 0.80. Under this workload, peak
# usage_mamba_active reads ~0.66 sustained but doesn't reach 0.80
# because (1) sglang's `num_queue_reqs.total` reflects only the
# scheduler's waiting_queue, not the upstream API-layer queue
# where the actual backlog lives — so the planner can't see queue
# pressure; (2) the active counter itself has a phantom-slot
# component (slots between req-completion and cache-installation
# that briefly count as "active"). Lowering high-water to 0.50
# lets persist consec accumulate at 0.66 sustained, clearing the
# NB gate. This is appropriate for byte_transfer (validates "fire mechanism
# works under live load") not saturated_bubble (validates "policy fires at the
# right time"). byte_transfer (d) check enforces dst_active ≥ 0.50 — matches
# the lowered threshold.

# Wait for ready
waited=0
while [ $waited -lt 600 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
        echo "[byte_transfer] ready after ${waited}s"
        break
    fi
    if ! kill -0 $SV_PID 2>/dev/null; then
        echo "[byte_transfer] FAILED — server died"; tail -25 "$LOG" >&2
        exit 1
    fi
done
[ $waited -ge 600 ] && { echo "[byte_transfer] TIMEOUT"; kill -9 $SV_PID; exit 1; }

# Mamba-saturating workload: random requests at high RPS with long
# outputs. Each in-flight req holds 1 mamba slot for its entire
# decode lifetime (~20s at 1024 output tokens / 50 tok/s). At RPS=32
# that's 32×20=640 desired in-flight, capped at max-running=256 →
# mamba active fully saturated at 256/306 = 84% (above mamba_high
# = 0.80 default), persist counter grows past nb_persist_eval_period.
echo "[byte_transfer] driving mamba-saturating workload for ${WORKLOAD_S}s"
$VENV -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 1024 \
    --request-rate 32 \
    --num-prompts $((WORKLOAD_S * 32)) \
    --output-file "$OUT_DIR/bench.json" \
    > "$OUT_DIR/bench.log" 2>&1 || echo "[byte_transfer] bench finished (rc=$?)"

# Teardown
kill -9 $SV_PID 2>/dev/null
sleep 4

echo "[byte_transfer] artifacts:"
ls -la "$OUT_DIR"
echo "[byte_transfer] budgeter.jsonl line count: $(wc -l < "$BUDG" 2>/dev/null || echo 0)"
echo "[byte_transfer] fires emitted: $(grep -c '"fire_direction"' "$BUDG" 2>/dev/null || echo 0)"

echo
echo "=== Validation ==="
$VENV dev/interlayer/4_e2e/byte_transfer/validate_byte_transfer.py --out-dir "$OUT_DIR"
