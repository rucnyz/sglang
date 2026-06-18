#!/bin/bash
#
# run_cc_traj_vllm.sh — Claude Code trajectory replay against a vLLM server.
# Mirrors run_cc_traj.sh but boots vLLM v0.20 instead of SGLang.
#
# Required env: MODEL TP GPU_LIST PORT OUT_DIR
# Optional:     MEM_FRAC NUM_CONCURRENCY MAX_TIME_MIN MAX_TOKENS
#               TRACES_FILE MIN_TURNS MIN_CHARS BOOT_TIMEOUT_S
#
# vLLM venv at /data/yuzhou/projects/vllm/.venv (same as run_vllm.sh).

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

VLLM_VENV=/data/yuzhou/projects/vllm/.venv
require_env() { [ -n "${!1:-}" ] || { echo "missing env: $1" >&2; exit 1; }; }
require_env MODEL; require_env TP; require_env GPU_LIST
require_env PORT; require_env OUT_DIR
mkdir -p "$OUT_DIR"

MEM_FRAC=${MEM_FRAC:-0.85}
BOOT_TIMEOUT_S=${BOOT_TIMEOUT_S:-1500}
MAX_TIME_MIN=${MAX_TIME_MIN:-10}
MAX_TOKENS=${MAX_TOKENS:-1024}
NUM_CONCURRENCY=${NUM_CONCURRENCY:-14}
TRACES_FILE=${TRACES_FILE:-dev/eval/datasets/cc_long_traces.jsonl}
MIN_TURNS=${MIN_TURNS:-15}
MIN_CHARS=${MIN_CHARS:-30000}

cell="vllm_cc_traj"
log="$OUT_DIR/server.log"

extra=""
case "$MODEL" in
    *Kimi*) extra="--trust-remote-code" ;;
esac

echo "[$cell] boot vLLM model=$MODEL tp=$TP gpus=$GPU_LIST port=$PORT mem_frac=$MEM_FRAC"

# Pre-boot cleanup: stale port + GPU stragglers (mirrors _common.sh logic)
pkill -9 -f "vllm.entrypoints.*--port $PORT" 2>/dev/null || true
pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null || true
if command -v fuser >/dev/null 2>&1; then
    sudo -n fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
if command -v lsof >/dev/null 2>&1; then
    stuck=$(sudo -n lsof -ti:"$PORT" 2>/dev/null || true)
    [ -n "$stuck" ] && sudo -n kill -9 $stuck 2>/dev/null || true
fi
for gpu in ${GPU_LIST//,/ }; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' \r\n,')
    for pid in $pids; do
        [ -z "$pid" ] && continue
        sudo -n kill -9 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    done
done
sleep 4

CUDA_VISIBLE_DEVICES=$GPU_LIST nohup \
    "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_FRAC" \
    --enable-prefix-caching \
    $extra \
    > "$log" 2>&1 &
SV_PID=$!
echo "[$cell] vllm pid=$SV_PID"

waited=0
ready=0
while [ $waited -lt $BOOT_TIMEOUT_S ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
        echo "[$cell] ready after ${waited}s"; ready=1; break
    fi
    if ! kill -0 $SV_PID 2>/dev/null; then
        echo "[$cell] server died — log tail:"; tail -25 "$log"; exit 1
    fi
done
if [ "$ready" -ne 1 ]; then
    echo "[$cell] FAILED to come up after ${waited}s"; tail -25 "$log"
    kill -9 $SV_PID 2>/dev/null || true
    exit 1
fi

echo "[$cell] CC-traj replay: conc=$NUM_CONCURRENCY traces=$TRACES_FILE max_time=${MAX_TIME_MIN}min max_tokens=$MAX_TOKENS"
.venv/bin/python dev/eval/main/cc_trace_replay.py \
    --api-base "http://127.0.0.1:$PORT" \
    --model "$MODEL" \
    --traces "$TRACES_FILE" \
    --num-concurrency "$NUM_CONCURRENCY" \
    --max-time-min "$MAX_TIME_MIN" \
    --max-tokens "$MAX_TOKENS" \
    --min-turns "$MIN_TURNS" \
    --min-chars "$MIN_CHARS" \
    --output-file "$OUT_DIR/bench.json" \
    > "$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed (see client.log)"

kill -9 $SV_PID 2>/dev/null || true
# vLLM TP=2 spawns Worker_TP0/Worker_TP1 as orphan-able children — the
# parent kill above doesn't catch them. SIGKILL by GPU PID lookup so the
# next run on these GPUs starts with free memory, not a zombie 125 GiB.
for gpu in ${GPU_LIST//,/ }; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' \r\n,')
    for pid in $pids; do
        [ -z "$pid" ] && continue
        sudo -n kill -9 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    done
done
sleep 4
echo "[$cell] vLLM CC-traj done -> $OUT_DIR"
