#!/bin/bash
# Top-level orchestrator for paper §sec:eval-main experiments.
# Schedules (model × regime × cell × trial) jobs across 8× H200, respecting
# per-model TP requirements.
#
# Usage:
#   bash dev/eval/main/orchestrator.sh                      # full run
#   SMOKE=1 bash dev/eval/main/orchestrator.sh              # 1-trial sanity
#   MODELS=qwen3.5 REGIMES=m2 bash ...                      # filter
#
# Env (with defaults):
#   SMOKE=0                    1 → 1-trial × 1 model only (sanity)
#   MODELS="qwen3.5 qwen3next kimi"
#   REGIMES="m1 m2 m3"
#   N_TRIALS_M1=5  N_TRIALS_M2=5  N_TRIALS_M3=3
#   INCLUDE_VLLM=1
#   INCLUDE_STATIC_BEST=1     (only on Qwen3.5)
#   PORT_BASE=33000
#   RUN_NAME=main-<timestamp>

set -euo pipefail
cd /scratch/yuzhou/projects/sglang

SMOKE=${SMOKE:-0}
BASELINE=${BASELINE:-0}    # 1 → only (0,0) stock cell
ABLATION=${ABLATION:-0}    # 1 → only (1,0), (0,1), (1,1) cells
PORT_BASE=${PORT_BASE:-33000}

# Long-form CLI flags map to the same env knobs.
for arg in "$@"; do
    case "$arg" in
        --baseline) BASELINE=1 ;;
        --ablation) ABLATION=1 ;;
        --smoke)    SMOKE=1 ;;
    esac
done

if [ "$BASELINE" = "1" ] && [ "$ABLATION" = "1" ]; then
    echo "[main-orch] error: --baseline and --ablation are mutually exclusive" >&2
    exit 2
fi

if [ "$SMOKE" = "1" ]; then
    # Smoke defaults — force-override the full-run defaults.
    MODELS=${MODELS:-"qwen3.5"}
    REGIMES=${REGIMES:-"m1 m2 m3"}
    N_TRIALS_M1=${N_TRIALS_M1:-1}
    N_TRIALS_M2=${N_TRIALS_M2:-1}
    N_TRIALS_M3=${N_TRIALS_M3:-1}
    INCLUDE_VLLM=${INCLUDE_VLLM:-0}
    INCLUDE_STATIC_BEST=${INCLUDE_STATIC_BEST:-0}
else
    MODELS=${MODELS:-"qwen3.5 qwen3next kimi"}
    REGIMES=${REGIMES:-"m1 m2 m3"}
    N_TRIALS_M1=${N_TRIALS_M1:-5}
    N_TRIALS_M2=${N_TRIALS_M2:-5}
    N_TRIALS_M3=${N_TRIALS_M3:-3}
    INCLUDE_VLLM=${INCLUDE_VLLM:-1}
    INCLUDE_STATIC_BEST=${INCLUDE_STATIC_BEST:-1}
fi

# --baseline forces only (0,0); skips static-best + vLLM by default
# (user can re-enable explicitly with INCLUDE_STATIC_BEST=1 etc).
if [ "$BASELINE" = "1" ]; then
    INCLUDE_VLLM=${INCLUDE_VLLM:-0}
    INCLUDE_STATIC_BEST=${INCLUDE_STATIC_BEST:-0}
    echo "[main-orch] BASELINE mode: only (0,0) stock cell will be run"
fi

TS=$(date +%Y%m%d-%H%M%S)
RUN_NAME=${RUN_NAME:-"main-$TS"}
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[main-orch] run=$RUN_NAME smoke=$SMOKE models='$MODELS' regimes='$REGIMES'"
echo "[main-orch] root=$ROOT"

# --- Model registry: hf-id, TP, GPUs-needed ---
get_model_hf() {
    case "$1" in
        qwen3.5)  echo "Qwen/Qwen3.5-35B-A3B" ;;
        qwen3next) echo "Qwen/Qwen3-Next-80B-A3B-Instruct" ;;
        kimi)     echo "moonshotai/Kimi-Linear-48B-A3B-Instruct" ;;
        *) echo ""; return 1 ;;
    esac
}
get_model_tp() {
    case "$1" in
        qwen3.5)  echo 1 ;;
        qwen3next) echo 2 ;;
        kimi)     echo 1 ;;
        *) echo 1 ;;
    esac
}

# Per-model run: launches up to (8/TP) cells in parallel.
run_model_regime() {
    local model_key="$1"; local regime="$2"
    local hf tp n_trials
    hf=$(get_model_hf "$model_key") || { echo "skip unknown $model_key"; return; }
    tp=$(get_model_tp "$model_key")
    case "$regime" in
        m1) n_trials=$N_TRIALS_M1 ;;
        m2) n_trials=$N_TRIALS_M2 ;;
        m3) n_trials=$N_TRIALS_M3 ;;
        *) echo "unknown regime $regime"; return ;;
    esac

    local CELLS=("0 0" "1 0" "0 1" "1 1")
    if [ "$BASELINE" = "1" ]; then
        CELLS=("0 0")
    elif [ "$ABLATION" = "1" ]; then
        CELLS=("1 0" "0 1" "1 1")
    fi
    local ncells=${#CELLS[@]}
    # `parallel` = how many cell-runs in flight at once. Capped by
    # available GPU groups (8/tp). NOT capped by ncells — multiple trials
    # of the same cell are independent and benefit from running in
    # parallel on different GPUs.
    local total_jobs=$((ncells * n_trials))
    local parallel=$((8 / tp))
    [ $parallel -gt $total_jobs ] && parallel=$total_jobs

    echo "[main-orch] model=$model_key ($hf, tp=$tp) regime=$regime trials=$n_trials cells=$ncells jobs=$total_jobs parallel=$parallel"

    local job_idx=0
    for trial in $(seq 1 $n_trials); do
        for pair in "${CELLS[@]}"; do
            set -- $pair; intra=$1; inter=$2
            cell="intra${intra}_inter${inter}"
            out_dir="$ROOT/${model_key}/${regime}/trial${trial}_${cell}"
            mkdir -p "$out_dir"
            local gpu_start=$(( (job_idx % parallel) * tp ))
            local gpu_list=""
            for ((g=0; g<tp; g++)); do
                [ -n "$gpu_list" ] && gpu_list+=","
                gpu_list+=$((gpu_start + g))
            done
            local port=$((PORT_BASE + job_idx))

            MODEL="$hf" TP="$tp" GPU_LIST="$gpu_list" \
                INTRA="$intra" INTER="$inter" PORT="$port" OUT_DIR="$out_dir" \
                bash "dev/eval/main/run_${regime}.sh" \
                > "$out_dir/runner.log" 2>&1 &

            job_idx=$((job_idx + 1))
            # Sync after every `parallel` jobs.
            if [ $((job_idx % parallel)) -eq 0 ]; then
                wait || true
            fi
        done
    done
    wait || true
    echo "[main-orch] $model_key/$regime done"
}

run_static_best_for_model() {
    local model_key="$1"
    [ "$model_key" != "qwen3.5" ] && return  # paper sweep is on Qwen3.5
    local hf tp; hf=$(get_model_hf "$model_key"); tp=$(get_model_tp "$model_key")
    for regime in $REGIMES; do
        local out_dir="$ROOT/${model_key}/${regime}/static_best"
        mkdir -p "$out_dir"
        local port=$((PORT_BASE + 100))
        MODEL="$hf" TP="$tp" GPU_LIST="0" REGIME="$regime" PORT="$port" \
            OUT_DIR="$out_dir" \
            bash dev/eval/main/run_static_best.sh \
            > "$out_dir/runner.log" 2>&1 || echo "[main-orch] static-best $regime failed"
    done
}

run_vllm_for_model() {
    local model_key="$1"
    local hf tp; hf=$(get_model_hf "$model_key"); tp=$(get_model_tp "$model_key")
    for regime in $REGIMES; do
        local out_dir="$ROOT/${model_key}/${regime}/vllm"
        mkdir -p "$out_dir"
        # Use GPU 0..(tp-1) for vLLM; serial across regimes
        local gpu_list=""
        for ((g=0; g<tp; g++)); do
            [ -n "$gpu_list" ] && gpu_list+=","
            gpu_list+=$g
        done
        local port=$((PORT_BASE + 200))
        MODEL="$hf" TP="$tp" GPU_LIST="$gpu_list" REGIME="$regime" \
            PORT="$port" OUT_DIR="$out_dir" \
            bash dev/eval/main/run_vllm.sh \
            > "$out_dir/runner.log" 2>&1 || echo "[main-orch] vllm $model_key/$regime failed"
    done
}

# Drive: model-by-model (each occupies 8 GPUs at most), regime-by-regime.
for model_key in $MODELS; do
    for regime in $REGIMES; do
        run_model_regime "$model_key" "$regime"
    done
    [ "$INCLUDE_STATIC_BEST" = "1" ] && run_static_best_for_model "$model_key"
    [ "$INCLUDE_VLLM" = "1" ] && run_vllm_for_model "$model_key"
done

echo "[main-orch] all done -> $ROOT"
echo "[main-orch] aggregate with: python3 dev/eval/main/aggregate.py $ROOT"
