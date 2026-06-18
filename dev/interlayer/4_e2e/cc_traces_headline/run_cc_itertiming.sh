#!/usr/bin/env bash
# Attribution probe: same cc A/B as run_cc.sh, but with the scheduler's
# per-iter phase-timing log (SGLANG_ITER_TIMING_LOG) + budgeter tick probe
# (SGLANG_HIMA_TICK_PROBE) enabled for BOTH cells, so we can attribute the
# inter_admitter wall-clock regression (recv/batch/run/proc/tick µs per iter).
#
# Why a separate script: editing the live run_cc.sh while a zero-downside
# 3-rep run is mid-flight would change the script rep3 re-reads → contaminate
# the running A/B. This copy keeps that script untouched.
#
# Run ALONE (no other GPU job on the host): the batch-~5 cc regime is
# scheduler-CPU-overhead-bound, so a concurrent server on another GPU steals
# host CPU and invalidates the timing attribution.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/itertiming_run}
GPU=${GPU:-7}
TP=${TP:-1}
PORT=${PORT:-30097}
TRACES_FILE=${TRACES_FILE:-/scratch/yuzhou/projects/vllm-songyang/dev/interlayer/cc_long_traces.jsonl}
NUM_CONCURRENCY=${NUM_CONCURRENCY:-22}
MAX_TIME_MIN=${MAX_TIME_MIN:-6}
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_SESSIONS=${MAX_SESSIONS:-10}
MAX_MAMBA_CACHE=${MAX_MAMBA_CACHE:-256}
# Knob held at stock default to match the isolation run (machinery-only).
export SGLANG_XPOOL_QUEUE_WAIT_US=${SGLANG_XPOOL_QUEUE_WAIT_US:-100}

mkdir -p "$OUT_DIR/off" "$OUT_DIR/arena_only" "$OUT_DIR/inter_admitter"

run_cell() {
    local cell="$1"          # off | arena_only | inter_admitter
    local cell_out="$OUT_DIR/$cell"

    rm -f "$cell_out"/*.log "$cell_out"/*.json "$cell_out"/*.jsonl
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    # Per-iter phase timing enabled for ALL cells (baseline + treatments), so
    # off's iter cost is the comparison floor.
    local env_str="\
                   SGLANG_ITER_TIMING_LOG=$cell_out/iter_timing.jsonl \
                   SGLANG_DECODE_ALLOC_PROBE=$cell_out/decode_alloc.jsonl"
    if [ "$cell" = "arena_only" ]; then
        # Decisive split: arena KV+mamba backend ON, but NO HiMA
        # decision loop. Isolates "arena memory backend per-step cost" from
        # "decision machinery cost". SGLANG_HIMA=1 would auto-promote
        # arena; here we promote arena directly and leave the loop off.
        env_str="$env_str SGLANG_ARENA_SHARED=1"
    elif [ "$cell" = "inter_admitter" ]; then
        env_str="$env_str SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=2.0 \
                 SGLANG_HIMA_LOG=$cell_out/budgeter.jsonl \
                 SGLANG_HIMA_TICK_PROBE=$cell_out/tick_probe.jsonl \
                 SGLANG_HIMA_ADMITTER_LOG=$cell_out/admitter.jsonl"
    fi

    echo "[itertiming/$cell] boot (env=$env_str) GPU=$GPU PORT=$PORT"
    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_str \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp $TP --mem-fraction-static 0.55 \
            --max-running-requests 256 \
            --max-mamba-cache-size $MAX_MAMBA_CACHE \
            --radix-eviction-policy "${EVICTION_POLICY:-lpb}" \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --enable-metrics \
            --log-level info > "$cell_out/server.log" 2>&1 &
    local pid=$!

    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[itertiming/$cell] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[itertiming/$cell] server died"; tail -25 "$cell_out/server.log"; return 1
        fi
    done
    [ $waited -ge 1200 ] && { echo "[itertiming/$cell] TIMEOUT"; kill -9 $pid; return 1; }

    echo "[itertiming/$cell] CC-traj replay for ${MAX_TIME_MIN} min"
    $VENV dev/eval/main/cc_trace_replay.py \
        --api-base "http://127.0.0.1:$PORT" \
        --model "$MODEL_DIR" \
        --traces "$TRACES_FILE" \
        --num-concurrency "$NUM_CONCURRENCY" \
        --max-time-min "$MAX_TIME_MIN" \
        --max-sessions "$MAX_SESSIONS" \
        --max-tokens "$MAX_TOKENS" \
        --min-turns 15 --min-chars 30000 \
        --output-file "$cell_out/bench.json" \
        > "$cell_out/bench.log" 2>&1 || echo "[itertiming/$cell] bench rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4
}

run_cell off            || { echo "[itertiming] off failed"; exit 1; }
run_cell arena_only     || { echo "[itertiming] arena_only failed"; exit 1; }
run_cell inter_admitter || { echo "[itertiming] inter_admitter failed"; exit 1; }

echo
echo "=== iter-timing attribution: median phase µs per iter (off vs arena_only vs inter) ==="
echo "    off=torch.zeros  arena_only=arena backend, NO machinery  inter=arena+budgeter+admitter"
$VENV - "$OUT_DIR" <<'PY'
import json, sys, statistics
out = sys.argv[1]
fields = ["iter_us", "recv_us", "batch_us", "run_us", "proc_us", "tick_us"]
for cell in ("off", "arena_only", "inter_admitter"):
    rows = []
    try:
        with open(f"{out}/{cell}/iter_timing.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        print(f"{cell}: no iter_timing.jsonl"); continue
    print(f"\n{cell}: n_iters={len(rows)}")
    for fld in fields:
        vals = [r[fld] for r in rows if fld in r]
        if vals:
            vals.sort()
            print(f"  {fld:10s} median={statistics.median(vals):8.1f}  "
                  f"mean={statistics.mean(vals):8.1f}  p95={vals[int(len(vals)*0.95)]:8.1f}  max={vals[-1]:8.1f}")
    # batch-size-conditioned: iter cost at small batch (the cc regime)
    small = [r for r in rows if r.get("bs", 0) and r["bs"] <= 6]
    if small:
        iu = sorted(r["iter_us"] for r in small)
        print(f"  [bs<=6] n={len(small)} iter_us median={statistics.median(iu):.1f}")
PY
