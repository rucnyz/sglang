#!/usr/bin/env bash
# T9 — N=3 H'_now matrix (no daemon, direct harbor → sglang).
#
# Mirrors the structure of run_matrix.sh: 3 cycles, resumable, each
# cycle gets its own RESULTS_DIR.  No alternation needed (single
# config).  Use this to get N=3 means for fair Welch comparison vs
# the matrix baseline 1389.3 ± 39.7 s.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

MATRIX_TAG="${HPRIME_MATRIX_TAG:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_ROOT="$AGINFER_RESULTS/run_H_prime_now_matrix_${MATRIX_TAG}"
mkdir -p "$MATRIX_ROOT"
echo "[h_prime_matrix] tag=$MATRIX_TAG root=$MATRIX_ROOT"
echo "[h_prime_matrix] sglang HEAD: $(cd "$AGINFER_DIR/../.." && git rev-parse HEAD)"

NUM_CYCLES="${HPRIME_NUM_CYCLES:-3}"

for cycle_idx in $(seq 1 "$NUM_CYCLES"); do
    tag="matrix_${MATRIX_TAG}_cycle${cycle_idx}"
    cycle_dir="$AGINFER_RESULTS/run_H_prime_now_${tag}"

    # Resumable: skip if harbor_jobs already has per-instance result.json.
    if [[ -d "$cycle_dir/harbor_jobs" ]] && \
       find "$cycle_dir/harbor_jobs" -mindepth 3 -name 'result.json' 2>/dev/null | head -32 | grep -q .; then
        n_done=$(find "$cycle_dir/harbor_jobs" -mindepth 3 -name 'result.json' 2>/dev/null | wc -l)
        if (( n_done >= 32 )); then
            echo "[h_prime_matrix] cycle ${cycle_idx} — already complete ($n_done trials), skipping"
            ln -sfn "$cycle_dir" "$MATRIX_ROOT/cycle${cycle_idx}_h_prime_now"
            continue
        fi
    fi

    echo "[h_prime_matrix] ==== cycle ${cycle_idx}/${NUM_CYCLES} ===="
    echo "[h_prime_matrix] $(date '+%Y-%m-%d %H:%M:%S') starting"

    RUN_K_RESULTS_TAG="$tag" \
    bash "$SCRIPT_DIR/run_h_prime.sh" || rc=$?
    rc=${rc:-0}

    if [[ "$rc" -ne 0 ]]; then
        echo "[h_prime_matrix] cycle ${cycle_idx} exited rc=$rc; check $cycle_dir" >&2
    fi

    ln -sfn "$cycle_dir" "$MATRIX_ROOT/cycle${cycle_idx}_h_prime_now"
    echo "[h_prime_matrix] $(date '+%Y-%m-%d %H:%M:%S') cycle ${cycle_idx} done"
    echo ""

    # Active GPU drain wait (same as run_matrix.sh).
    IFS=',' read -ra _GPU_LIST <<< "${AGINFER_GPUS:-4,7}"
    for tick in $(seq 1 30); do
        all_clear=1
        for gpu in "${_GPU_LIST[@]}"; do
            used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            if (( ${used:-0} > 1024 )); then all_clear=0; fi
        done
        if (( all_clear )); then
            echo "[h_prime_matrix] GPU drain confirmed @ tick $tick"
            break
        fi
        sleep 2
    done
done

echo "[h_prime_matrix] all cycles done.  Aggregate via parse_matrix.py on $MATRIX_ROOT/"
