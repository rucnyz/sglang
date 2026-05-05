#!/bin/bash
# vLLM baseline cell (paper §sec:eval-vllm, vLLM column of tab:main-cross-model).
# Boots vLLM v0.20.x from /data/yuzhou/projects/vllm/.venv, runs the same
# workload client (multiturn for M1/M2, sglang.bench_serving phases for M3).
#
# Required env: MODEL TP GPU_LIST REGIME (m1|m2|m3) PORT OUT_DIR
# Optional:     MEM_FRAC

set -eu
cd /scratch/yuzhou/projects/sglang
export PYTHONPATH=${PYTHONPATH:-}

VLLM_VENV=/data/yuzhou/projects/vllm/.venv
require_env() { [ -n "${!1:-}" ] || { echo "missing env: $1" >&2; exit 1; }; }
require_env MODEL; require_env TP; require_env GPU_LIST
require_env REGIME; require_env PORT; require_env OUT_DIR
mkdir -p "$OUT_DIR"

MEM_FRAC=${MEM_FRAC:-0.85}
log="$OUT_DIR/server.log"
cell="vllm_v0_20_x"

echo "[$cell] boot vLLM model=$MODEL tp=$TP gpus=$GPU_LIST port=$PORT"
pkill -f "vllm.*--port $PORT" 2>/dev/null || true
sleep 4

CUDA_VISIBLE_DEVICES=$GPU_LIST nohup \
    "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_FRAC" \
    >"$log" 2>&1 &
SV_PID=$!
echo "[$cell] vllm pid=$SV_PID"

waited=0
while [ $waited -lt 600 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
        echo "[$cell] ready after ${waited}s"; break
    fi
    if ! kill -0 $SV_PID 2>/dev/null; then
        echo "[$cell] server died — log tail:"; tail -25 "$log"; exit 1
    fi
done
[ $waited -ge 600 ] && { echo "[$cell] FAILED to come up"; tail -25 "$log"; kill -9 $SV_PID 2>/dev/null||true; exit 1; }

# Drive the workload — same client as SGLang side, but pointed at vLLM.
case "$REGIME" in
    m1)
        NUM_CONCURRENCY=14 TURN_INPUT=4096 TURN_OUTPUT=4096 SESSION_CAP=60000 \
        MAX_TIME_S=480 STAGGER_S=0.0 MEASURE_AFTER_S=30 \
            /scratch/yuzhou/projects/sglang/.venv/bin/python \
            dev/eval/multiturn_client.py \
            --api-base "http://127.0.0.1:$PORT" --model "$MODEL" \
            --num-concurrency 14 --turn-input-tokens 4096 --turn-output-tokens 4096 \
            --session-cap-tokens 60000 --max-time-s 480 \
            --measure-after-s 30 --output-dir "$OUT_DIR" \
            >"$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed"
        ;;
    m2)
        /scratch/yuzhou/projects/sglang/.venv/bin/python \
            dev/eval/multiturn_client.py \
            --api-base "http://127.0.0.1:$PORT" --model "$MODEL" \
            --num-concurrency 800 --turn-input-tokens 256 --turn-output-tokens 256 \
            --session-cap-tokens 3000 --max-time-s 480 \
            --stagger-s 0.05 --measure-after-s 60 --output-dir "$OUT_DIR" \
            >"$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed"
        ;;
    m3)
        # Reuse SGLang's bench_serving against vLLM's OpenAI endpoint; phases A/B/C
        for phase in A B C; do
            case "$phase" in
                A) args="--dataset-name generated-shared-prefix --gsp-num-groups 16 --gsp-prompts-per-group 10 --gsp-system-prompt-len 6000 --gsp-question-len 64 --gsp-output-len 256 --request-rate 8 --num-prompts 1280" ;;
                B) args="--dataset-name random --random-input-len 8000 --random-output-len 64 --request-rate 4 --num-prompts 640" ;;
                C) args="--dataset-name random --random-input-len 4000 --random-output-len 64 --request-rate 8 --num-prompts 1280" ;;
            esac
            /scratch/yuzhou/projects/sglang/.venv/bin/python -m sglang.bench_serving \
                --backend vllm --host 127.0.0.1 --port "$PORT" \
                --model "$MODEL" --tokenizer "$MODEL" \
                --output-file "$OUT_DIR/phase${phase}_bench.json" \
                $args \
                >"$OUT_DIR/phase${phase}_bench.log" 2>&1 || echo "[$cell] phase $phase failed"
            sleep 5
        done
        [ -f "$OUT_DIR/phaseC_bench.json" ] && cp "$OUT_DIR/phaseC_bench.json" "$OUT_DIR/bench.json"
        ;;
    *) echo "unknown REGIME=$REGIME"; exit 1 ;;
esac

[ -f "$OUT_DIR/multiturn_summary.json" ] && cp "$OUT_DIR/multiturn_summary.json" "$OUT_DIR/bench.json"

kill -9 $SV_PID 2>/dev/null || true
sleep 4
echo "[$cell] vLLM regime=$REGIME done -> $OUT_DIR"
