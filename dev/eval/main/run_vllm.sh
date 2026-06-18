#!/bin/bash
# vLLM baseline cell (paper §sec:eval-vllm, vLLM column of tab:main-cross-model).
# Boots vLLM v0.20.x from /data/yuzhou/projects/vllm/.venv. M1/M2 driven by
# genai-bench (text-to-text-multi-turn task), M3 by sglang.bench_serving with
# --backend vllm against the OpenAI-compatible endpoint.
#
# Required env: MODEL TP GPU_LIST REGIME (m1|m2|m3) PORT OUT_DIR
# Optional:     MEM_FRAC BOOT_TIMEOUT_S MAX_TIME_MIN PHASE_DURATION_S

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=${PYTHONPATH:-}

VLLM_VENV=/data/yuzhou/projects/vllm/.venv
require_env() { [ -n "${!1:-}" ] || { echo "missing env: $1" >&2; exit 1; }; }
require_env MODEL; require_env TP; require_env GPU_LIST
require_env REGIME; require_env PORT; require_env OUT_DIR
mkdir -p "$OUT_DIR"

MEM_FRAC=${MEM_FRAC:-0.85}
BOOT_TIMEOUT_S=${BOOT_TIMEOUT_S:-1500}
MAX_TIME_MIN=${MAX_TIME_MIN:-10}
PHASE_DURATION_S=${PHASE_DURATION_S:-200}
log="$OUT_DIR/server.log"
cell="vllm_v0_20_x"

# Kimi-Linear's tokenizer + model both ship custom Python; vLLM needs the
# same flag SGLang uses.
extra=""
case "$MODEL" in
    *Kimi*) extra="--trust-remote-code" ;;
esac

echo "[$cell] boot vLLM model=$MODEL tp=$TP gpus=$GPU_LIST port=$PORT"

# Cleanup any stragglers (port stuck in TIME_WAIT, leftover CUDA processes
# on the GPUs we want, etc.) before launching. Mirrors _common.sh's
# cleanup_before_boot — duplicated here because run_vllm.sh stands alone
# (does not source _common.sh, since the SGLang-only knobs in apply_cell_env
# are not relevant to the vLLM cross-engine baseline).
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
    $extra \
    >"$log" 2>&1 &
SV_PID=$!
echo "[$cell] vllm pid=$SV_PID"

waited=0
while [ $waited -lt $BOOT_TIMEOUT_S ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
        echo "[$cell] ready after ${waited}s"; break
    fi
    if ! kill -0 $SV_PID 2>/dev/null; then
        echo "[$cell] server died — log tail:"; tail -25 "$log"; exit 1
    fi
done
[ $waited -ge $BOOT_TIMEOUT_S ] && { echo "[$cell] FAILED to come up"; tail -25 "$log"; kill -9 $SV_PID 2>/dev/null||true; exit 1; }

# Workload-specific client.
case "$REGIME" in
    m1)
        GENAI_BENCH_MT_SESSION_CAP_TOKENS=150000 \
        .venv/bin/python -m genai_bench.cli.cli benchmark \
            --api-backend openai \
            --api-base "http://127.0.0.1:$PORT" \
            --api-key dummy \
            --api-model-name "$MODEL" \
            --model-tokenizer "$MODEL" \
            --task text-to-text-multi-turn \
            --traffic-scenario "D(4096,4096)" \
            --num-concurrency 14 \
            --max-time-per-run $MAX_TIME_MIN \
            --max-requests-per-run 1000000 \
            --experiment-folder-name "$OUT_DIR/genai_results" \
            --server-engine vLLM \
            > "$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed"
        ;;
    m2)
        GENAI_BENCH_MT_SESSION_CAP_TOKENS=5000 \
        .venv/bin/python -m genai_bench.cli.cli benchmark \
            --api-backend openai \
            --api-base "http://127.0.0.1:$PORT" \
            --api-key dummy \
            --api-model-name "$MODEL" \
            --model-tokenizer "$MODEL" \
            --task text-to-text-multi-turn \
            --traffic-scenario "D(256,256)" \
            --num-concurrency 800 \
            --max-time-per-run $MAX_TIME_MIN \
            --max-requests-per-run 1000000 \
            --experiment-folder-name "$OUT_DIR/genai_results" \
            --server-engine vLLM \
            > "$OUT_DIR/client.log" 2>&1 || echo "[$cell] client failed"
        ;;
    m3)
        for phase in A B C; do
            case "$phase" in
                A) args="--dataset-name generated-shared-prefix --gsp-num-groups 16 --gsp-prompts-per-group 10 --gsp-system-prompt-len 6000 --gsp-question-len 64 --gsp-output-len 256 --request-rate 8 --num-prompts $((PHASE_DURATION_S * 8))" ;;
                B) args="--dataset-name random --random-input-len 8000 --random-output-len 64 --request-rate 4 --num-prompts $((PHASE_DURATION_S * 4))" ;;
                C) args="--dataset-name random --random-input-len 4000 --random-output-len 64 --request-rate 8 --num-prompts $((PHASE_DURATION_S * 8))" ;;
            esac
            .venv/bin/python -m sglang.bench_serving \
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

# Normalize genai-bench output for M1/M2 → bench.json (ms units)
SUMMARY=$(find "$OUT_DIR/genai_results" -maxdepth 2 -name "D*_text-to-text-multi-turn_*.json" 2>/dev/null | head -1)
if [ -n "$SUMMARY" ]; then
    .venv/bin/python -c "
import json, sys
d = json.load(open('$SUMMARY'))
m = d.get('aggregated_metrics', d)
s = m.get('stats', {})
ttft = s.get('ttft', {})
e2e = s.get('e2e_latency', {})
out = {
    'wall_s': m.get('run_duration', 0),
    'num_concurrency': m.get('num_concurrency'),
    'num_requests_total': m.get('num_requests'),
    'num_requests_valid': m.get('num_completed_requests'),
    'num_errors': m.get('num_error_requests', 0),
    'mean_ttft_ms': ttft.get('mean', 0) * 1000,
    'p50_ttft_ms': ttft.get('p50', 0) * 1000,
    'p99_ttft_ms': ttft.get('p99', 0) * 1000,
    'mean_e2e_ms': e2e.get('mean', 0) * 1000,
    'p50_e2e_ms': e2e.get('p50', 0) * 1000,
    'p99_e2e_ms': e2e.get('p99', 0) * 1000,
    'output_tps': m.get('mean_output_throughput_tokens_per_s', 0),
    'input_tps': m.get('mean_input_throughput_tokens_per_s', 0),
    'requests_per_second': m.get('requests_per_second', 0),
    'error_rate': m.get('error_rate', 0),
}
json.dump(out, open('$OUT_DIR/bench.json', 'w'), indent=2)
" >> "$OUT_DIR/client.log" 2>&1
fi

kill -9 $SV_PID 2>/dev/null || true
# vLLM TP=2 spawns Worker_TP0/Worker_TP1 as orphan-able children — the
# parent kill above doesn't catch them. SIGKILL by GPU PID lookup.
for gpu in ${GPU_LIST//,/ }; do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' \r\n,')
    for pid in $pids; do
        [ -z "$pid" ] && continue
        sudo -n kill -9 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    done
done
sleep 4
echo "[$cell] vLLM regime=$REGIME done -> $OUT_DIR"
