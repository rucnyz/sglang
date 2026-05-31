#!/usr/bin/env bash
# T9 — chained A3 replication for N≥3 statistics.  Each cycle ~50 min;
# 2 cycles ≈ 1h40m total wall, on top of the already-completed v3+v4.
#
# After this completes, parse_daemon_events.py + parse_matrix.py
# can compute the A3-cycle across-cycle mean±std.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

REPEAT_TAG="${REPEAT_TAG:-a3_repeat_$(date +%Y%m%d_%H%M%S)}"
NUM="${A3_NUM_CYCLES:-2}"
IFS=',' read -ra _GPU_LIST <<< "${AGINFER_GPUS:-0,1}"

wait_for_gpu_drain() {
    local start; start=$(date +%s)
    while true; do
        local all_clear=1
        for gpu in "${_GPU_LIST[@]}"; do
            local used
            used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            if (( ${used:-0} > 1024 )); then all_clear=0; fi
        done
        if (( all_clear )); then
            echo "[a3_repeat] $(date '+%H:%M:%S') GPU drain OK"
            return 0
        fi
        if (( $(date +%s) - start > 600 )); then
            echo "[a3_repeat] $(date '+%H:%M:%S') GPU drain TIMEOUT — proceeding anyway" >&2
            return 1
        fi
        sleep 10
    done
}

for i in $(seq 1 "$NUM"); do
    [[ "$i" -gt 1 ]] && wait_for_gpu_drain
    tag="${REPEAT_TAG}_cycle$((i + 4))"   # v3=cycle 3, v4=cycle 4, this adds 5, 6
    echo "[a3_repeat] $(date '+%H:%M:%S') ==== firing A3 cycle ${tag} ===="
    RUN_K_RESULTS_TAG="$tag" \
        bash "$SCRIPT_DIR/run_k.sh" a3
    echo "[a3_repeat] $(date '+%H:%M:%S') cycle ${tag} done"
done

echo "[a3_repeat] done.  Aggregate via parse_daemon_events.py on each cycle."
