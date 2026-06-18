#!/usr/bin/env bash
# cc_traces_headline with Admitter — HEADLINE win on CC traces.
#
# design.md §cc_traces_headline: on real Claude Code agent traces, `inter+admitter`
# beats `off` by ≥ 1 of {mean TTFT -3%, p99 TTFT -3%, output_tps +3%,
# cache_hit +1pp} AND fire_count > 5.
#
# Prior cc_traces_headline (2026-05-29, Budgeter-only, no Admitter) was NEUTRAL — the
# cost-curve gate stays at L=0 without Admitter to seed per-arrival
# observations. With Phase 5/6/9 Admitter landed, the curve warms via
# Admitter's sync fires and the planner can engage.
#
# Two cells:
#   - off: INTRA=0 INTER=0 (baseline sglang)
#   - inter_admitter: INTER=1 + SGLANG_HIMA=1 +
#     SGLANG_XPOOL_QUEUE_WAIT_US (default 125000;
#     override the env to isolate the queue-pressure knob from the machinery)
#
# Each cell runs ~10 min of CC trace replay.

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/d10_run}
GPU=${GPU:-3}
TP=${TP:-1}
PORT=${PORT:-30077}
TRACES_FILE=${TRACES_FILE:-/scratch/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl}
NUM_CONCURRENCY=${NUM_CONCURRENCY:-14}
MAX_TIME_MIN=${MAX_TIME_MIN:-10}
MAX_TOKENS=${MAX_TOKENS:-1024}
# Mamba cache size — shrink (e.g. 64) to make the mamba pool the binding
# constraint so the OFF cell is forced to evict HOT snapshots (cache_hit
# drops below the high-reuse ceiling), exposing whether cross-fire helps.
MAX_MAMBA_CACHE=${MAX_MAMBA_CACHE:-256}

mkdir -p "$OUT_DIR/off" "$OUT_DIR/inter_admitter"

run_cell() {
    local cell="$1"          # off | inter_admitter
    local intra="$2"
    local inter="$3"
    local cell_out="$OUT_DIR/$cell"

    rm -f "$cell_out"/*.log "$cell_out"/*.json "$cell_out"/*.jsonl
    # Clean the per-request metrics subdir too: the server rotates
    # sglang-request-metrics-*.log per wall-clock hour, and validate_cc now
    # SUMS all logs in this dir — a stale log from a prior run would corrupt the
    # cell's cache-hit. (Cell-root *.log cleanup above does NOT touch the subdir.)
    rm -f "$cell_out"/metrics/sglang-request-metrics-*.log 2>/dev/null
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    # SGLANG_REQUEST_METRICS_SUFFIX → the server writes ONE named, non-rotated
    # per-request metrics JSONL (sglang-request-metrics-<suffix>.log) instead of
    # per-hour files. This is the authoritative per-request source (has real
    # cached_tokens from meta_info); the reader/validators read it directly.
    local env_str="SGLANG_REQUEST_METRICS_SUFFIX=$(basename "$OUT_DIR")_$cell"
    if [ "$inter" = "1" ]; then
        env_str="$env_str SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=2.0 \
                 SGLANG_HIMA_LOG=$cell_out/budgeter.jsonl \
                 SGLANG_HIMA_ADMITTER_LOG=$cell_out/admitter.jsonl \
                 SGLANG_XPOOL_QUEUE_WAIT_US=${SGLANG_XPOOL_QUEUE_WAIT_US:-125000}"
    fi

    echo "[cc_traces_headline/$cell] boot (intra=$intra inter=$inter)"
    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_str \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp $TP --mem-fraction-static 0.55 \
            --max-running-requests 256 \
            --max-mamba-cache-size $MAX_MAMBA_CACHE \
            --radix-eviction-policy "${EVICTION_POLICY:-lru}" \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --enable-metrics \
            --enable-request-time-stats-logging \
            --enable-mfu-metrics \
            --export-metrics-to-file \
            --export-metrics-to-file-dir "$cell_out/metrics" \
            --log-level info > "$cell_out/server.log" 2>&1 &
    local pid=$!

    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[cc_traces_headline/$cell] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[cc_traces_headline/$cell] server died"; tail -25 "$cell_out/server.log"; return 1
        fi
    done
    [ $waited -ge 1200 ] && { echo "[cc_traces_headline/$cell] TIMEOUT"; kill -9 $pid; return 1; }

    echo "[cc_traces_headline/$cell] CC-traj replay for ${MAX_TIME_MIN} min"
    $VENV dev/eval/main/cc_trace_replay.py \
        --api-base "http://127.0.0.1:$PORT" \
        --model "$MODEL_DIR" \
        --traces "$TRACES_FILE" \
        --num-concurrency "$NUM_CONCURRENCY" \
        --max-time-min "$MAX_TIME_MIN" \
        --max-sessions "${MAX_SESSIONS:-0}" \
        --max-tokens "$MAX_TOKENS" \
        --min-turns 15 --min-chars 30000 \
        --output-file "$cell_out/bench.json" \
        > "$cell_out/bench.log" 2>&1 || echo "[cc_traces_headline/$cell] bench rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4
}

# A/A control (SGLANG_CC_AA=1): run the SECOND cell with the Budgeter OFF too,
# so off-vs-"inter" measures only the harness's sequential-cell order bias (GPU
# clock/thermal drift between the first and second fresh-booted server) with the
# mechanism provably absent on both sides. Used to subtract that bias from the
# zero-downside read at mamba-slack caps where the mechanism is already dormant.
CELL2_INTER=1
[ "${SGLANG_CC_AA:-0}" = "1" ] && CELL2_INTER=0
run_cell off            0 0             || { echo "[cc_traces_headline] off failed"; exit 1; }
run_cell inter_admitter 0 "$CELL2_INTER" || { echo "[cc_traces_headline] inter_admitter failed"; exit 1; }

echo
echo "=== cc_traces_headline validation ==="
$VENV dev/interlayer/4_e2e/cc_traces_headline/validate_cc.py --out-dir "$OUT_DIR"
