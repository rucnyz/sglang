# Shared boot / cell-env / teardown helpers for main-experiment runners.
# Sourced by run_m1.sh / run_m2.sh / run_m3.sh / run_static_best.sh.

# Required env: MODEL TP GPU_LIST INTRA INTER PORT OUT_DIR
# Optional: MEM_FRAC EXTRA_LAUNCH_FLAGS

require_env() {
    local var="$1"
    [ -n "${!var:-}" ] || { echo "missing required env: $var" >&2; exit 1; }
}

apply_cell_env() {
    # Clear all Fulcrum knobs to start clean.
    unset SGLANG_HPB_LRU SGLANG_HPB_WINDOW_S SGLANG_K_BIG SGLANG_K_BIG_AUTO_THRESHOLD
    unset SGLANG_ARENA_SHARED SGLANG_ARENA_FROM_BLOB SGLANG_ARENA_CHUNK_BYTES
    unset SGLANG_BUDGETER SGLANG_BUDGETER_TICK_S SGLANG_BUDGETER_LOG
    unset SGLANG_XPOOL_KV_HIGH SGLANG_XPOOL_KV_LOW
    unset SGLANG_XPOOL_MAMBA_HIGH SGLANG_XPOOL_MAMBA_LOW SGLANG_XPOOL_COOLDOWN
    unset SGLANG_XPOOL_EDGE_TRIGGER SGLANG_XPOOL_NON_BALANCED
    export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0

    if [ "$INTRA" = "1" ]; then
        export SGLANG_HPB_LRU=1
        export SGLANG_HPB_WINDOW_S=120.0
        export SGLANG_K_BIG=8192
        export SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85}
    fi

    if [ "$INTER" = "1" ]; then
        export SGLANG_ARENA_SHARED=1
        export SGLANG_ARENA_FROM_BLOB=1
        export SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))
        export SGLANG_BUDGETER=1
        export SGLANG_BUDGETER_TICK_S=2.0
        export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
        export SGLANG_XPOOL_KV_HIGH=${SGLANG_XPOOL_KV_HIGH:-0.5}
        export SGLANG_XPOOL_KV_LOW=${SGLANG_XPOOL_KV_LOW:-0.05}
        export SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.08}
        export SGLANG_XPOOL_MAMBA_LOW=${SGLANG_XPOOL_MAMBA_LOW:-0.03}
        export SGLANG_XPOOL_COOLDOWN=${SGLANG_XPOOL_COOLDOWN:-2}
        export SGLANG_XPOOL_EDGE_TRIGGER=1
        export SGLANG_XPOOL_NON_BALANCED=1
    fi
}

model_specific_flags() {
    # --schedule-policy lpm: longest-prefix-match scheduling. CC traj &
    # multi-turn shared-prefix workloads see ~+8% TPS / -37% P99 vs the
    # default fcfs (sanity Qwen3.5 single-trial: 963→1038 TPS, 5038→3164 ms).
    # Random-token workloads degenerate to fcfs (no prefix overlap) — flag
    # is a no-op, so safe as a global default.
    local common_flags="--schedule-policy lpm --enforce-piecewise-cuda-graph"
    case "$MODEL" in
        */Qwen3*|*Qwen3.5*)
            echo "--reasoning-parser qwen3 $common_flags"
            ;;
        *Kimi*)
            # Kimi-Linear's tokenizer + model both ship custom code on HF;
            # SGLang refuses to load without --trust-remote-code.
            echo "--trust-remote-code $common_flags"
            ;;
        *)
            echo "$common_flags"
            ;;
    esac
}

cell_label() {
    if [ -n "${CELL_LABEL_OVERRIDE:-}" ]; then
        echo "$CELL_LABEL_OVERRIDE"
    else
        echo "intra${INTRA}_inter${INTER}"
    fi
}

boot_sglang() {
    local cell
    cell=$(cell_label)
    local mem_frac=${MEM_FRAC:-0.8}
    [ "$INTER" = "1" ] && mem_frac=${MEM_FRAC:-0.7}
    local log="$OUT_DIR/server.log"
    local extra="$(model_specific_flags) ${EXTRA_LAUNCH_FLAGS:-}"

    echo "[$cell] boot model=$MODEL tp=$TP gpus=$GPU_LIST port=$PORT mem_frac=$mem_frac"

    cleanup_before_boot "$PORT" "$GPU_LIST"

    CUDA_VISIBLE_DEVICES=$GPU_LIST nohup \
        /scratch/yuzhou/projects/sglang/.venv/bin/python -m sglang.launch_server \
        --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --tp "$TP" \
        --mem-fraction-static "$mem_frac" --log-level info \
        $extra \
        > "$log" 2>&1 &
    SV_PID=$!
    echo "[$cell] server pid=$SV_PID"

    local boot_timeout=${BOOT_TIMEOUT_S:-1500}
    local waited=0
    while [ $waited -lt $boot_timeout ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[$cell] ready after ${waited}s"
            return 0
        fi
        if ! kill -0 $SV_PID 2>/dev/null; then
            echo "[$cell] server died — last 25 log lines:"
            tail -25 "$log" >&2
            return 1
        fi
    done
    echo "[$cell] FAILED to come up after ${waited}s — log tail:"; tail -25 "$log" >&2
    kill -9 $SV_PID 2>/dev/null || true
    return 1
}

teardown_sglang() {
    [ -n "${SV_PID:-}" ] && kill -9 $SV_PID 2>/dev/null || true
    sleep 4
}

# cleanup_before_boot: kill anything that would block a fresh server boot —
# stale processes on the requested PORT (TIME_WAIT-bound listeners or actual
# stragglers from a previous run that pkill missed) and any CUDA process
# still mapping memory on the requested GPUs. Uses sudo so we can also clean
# up other-user / orphaned processes when needed.
#
# Args:
#   $1 PORT      : TCP port the new server will bind
#   $2 GPU_LIST  : comma-separated GPU indices (e.g. "0" or "0,1")
cleanup_before_boot() {
    local port=$1 gpus=$2

    # 1. Same-user pkill of any sglang/vllm server bound to this port
    pkill -9 -f "launch_server.*--port $port" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints.*--port $port" 2>/dev/null || true

    # 2. Anything else still holding the TCP port (incl. cross-user stragglers)
    if command -v fuser >/dev/null 2>&1; then
        sudo -n fuser -k "${port}/tcp" 2>/dev/null || true
    fi
    if command -v lsof >/dev/null 2>&1; then
        local stuck_pids
        stuck_pids=$(sudo -n lsof -ti:"$port" 2>/dev/null || true)
        [ -n "$stuck_pids" ] && sudo -n kill -9 $stuck_pids 2>/dev/null || true
    fi

    # 3. Any CUDA process still mapping memory on the GPUs we're about to use
    local gpu pid pids
    for gpu in ${gpus//,/ }; do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' \r\n,')
        for pid in $pids; do
            [ -z "$pid" ] && continue
            sudo -n kill -9 "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        done
    done

    sleep 4
}

emit_xpool_summary() {
    local cell
    cell=$(cell_label)
    local jsonl="$OUT_DIR/budgeter.jsonl"
    if [ "$INTER" = "1" ] && [ -f "$jsonl" ]; then
        local total k2m m2k granted
        total=$(grep -c '"xpool_direction":' "$jsonl" 2>/dev/null || echo 0)
        k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" 2>/dev/null || echo 0)
        m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" 2>/dev/null || echo 0)
        granted=$(grep '"xpool_direction":' "$jsonl" 2>/dev/null \
            | python3 -c 'import json,sys; t=0
for ln in sys.stdin:
    try: t += json.loads(ln).get("xpool_granted_total",0)
    except: pass
print(t)' 2>/dev/null || echo 0)
        echo "[$cell] xpool: total=$total k2m=$k2m m2k=$m2k granted=$granted"
        echo "{\"fires_total\": $total, \"fires_k2m\": $k2m, \"fires_m2k\": $m2k, \"granted_total\": $granted}" \
            > "$OUT_DIR/xpool_summary.json"
    fi
}
