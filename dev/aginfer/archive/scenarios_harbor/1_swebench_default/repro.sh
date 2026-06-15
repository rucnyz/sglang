#!/usr/bin/env bash
# T9 — 4-arm fairness matrix.
#
# Arms:
#   F'_now (LRU)            -> run_lru.sh
#   G_now  (ThunderAgent)   -> run_thunderagent.sh
#   H'_now (OURS inline)    -> already covered by previous H'_now matrix
#                              (run_H_prime_now_matrix_20260527_153233/)
#   OURS_full (daemon T11a) -> already covered by previous matrix ours
#                              (run_K_matrix_20260526_234639/ cycles 2/4/6)
#
# So this orchestrator only fires the two NEW arms × N=3 = 6 cycles.
# Alternate LRU / TA / LRU / TA / LRU / TA to neutralise drift.
#
# ETA: ~50 min/cycle × 6 = ~5 h.  Resumable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

MATRIX_TAG="${FOUR_ARM_TAG:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_ROOT="$AGINFER_RESULTS/run_4arm_matrix_${MATRIX_TAG}"
mkdir -p "$MATRIX_ROOT"
echo "[4arm] tag=$MATRIX_TAG root=$MATRIX_ROOT"
echo "[4arm] sglang HEAD: $(cd "$AGINFER_DIR/../.." && git rev-parse HEAD)"

CYCLES=(lru ta lru ta lru ta)

declare -A SCRIPT_OF=(
    [lru]=run_lru.sh
    [ta]=run_thunderagent.sh
)

declare -A DIRPREFIX_OF=(
    [lru]=run_LRU_now
    [ta]=run_TA_now
)

for i in "${!CYCLES[@]}"; do
    cycle_idx=$((i + 1))
    cfg="${CYCLES[$i]}"
    script="${SCRIPT_OF[$cfg]}"
    prefix="${DIRPREFIX_OF[$cfg]}"
    tag="matrix_${MATRIX_TAG}_cycle${cycle_idx}_${cfg}"
    cycle_dir="$AGINFER_RESULTS/${prefix}_${tag}"

    if [[ -d "$cycle_dir/harbor_jobs" ]] && \
       (find "$cycle_dir/harbor_jobs" -mindepth 3 -name 'result.json' 2>/dev/null | wc -l | grep -qE '^([3-9][0-9]|[1-9][0-9]{2,})$'); then
        echo "[4arm] cycle ${cycle_idx} (${cfg}) already complete, skipping"
        ln -sfn "$cycle_dir" "$MATRIX_ROOT/cycle${cycle_idx}_${cfg}"
        continue
    fi

    echo "[4arm] ==== cycle ${cycle_idx}/${#CYCLES[@]} (${cfg}) ===="
    echo "[4arm] $(date '+%Y-%m-%d %H:%M:%S') starting"

    RUN_K_RESULTS_TAG="$tag" \
    bash "$SCRIPT_DIR/$script" || rc=$?
    rc=${rc:-0}
    if [[ "$rc" -ne 0 ]]; then
        echo "[4arm] cycle ${cycle_idx} (${cfg}) rc=$rc; check $cycle_dir" >&2
    fi

    ln -sfn "$cycle_dir" "$MATRIX_ROOT/cycle${cycle_idx}_${cfg}"
    echo "[4arm] $(date '+%Y-%m-%d %H:%M:%S') cycle ${cycle_idx} done"
    echo ""

    # Active GPU drain wait
    IFS=',' read -ra _GPU_LIST <<< "${AGINFER_GPUS:-4,7}"
    for tick in $(seq 1 30); do
        all_clear=1
        for gpu in "${_GPU_LIST[@]}"; do
            used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            if (( ${used:-0} > 1024 )); then all_clear=0; fi
        done
        if (( all_clear )); then
            echo "[4arm] GPU drain confirmed @ tick $tick"
            break
        fi
        sleep 2
    done
done

echo "[4arm] all cycles done.  Aggregate via parse_matrix.py on $MATRIX_ROOT/"
