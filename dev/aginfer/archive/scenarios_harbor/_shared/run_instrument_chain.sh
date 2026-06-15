#!/usr/bin/env bash
# T9 — chained instrumented runs: K-a + Run J on the same GPU pair
# as the in-flight OURS_full cycle (0,1).  Waits for current GPU
# busy state to clear between cycles.
#
# Each variant's daemon emits aginfer_metric lines (commit 528962+);
# parse_daemon_events.py turns them into per-cycle counters.
#
# Sequence:
#   1. (already in flight) OURS_full
#   2. wait → K-a (kv_scheduler ON + admission OFF)
#   3. wait → Run J (full daemon + HiCache OFF)

set -uo pipefail   # no -e: we want each step to attempt even if prev quirks

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

VARIANTS=(ka J)
CHAIN_TAG="${CHAIN_TAG:-instrument_chain_$(date +%Y%m%d_%H%M%S)}"
IFS=',' read -ra _GPU_LIST <<< "${AGINFER_GPUS:-0,1}"

wait_for_gpu_drain() {
    # Wait up to 1 h for our GPU pair to drop below 1024 MiB.
    local tick start; start=$(date +%s)
    while true; do
        local all_clear=1
        for gpu in "${_GPU_LIST[@]}"; do
            local used
            used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            if (( ${used:-0} > 1024 )); then all_clear=0; fi
        done
        if (( all_clear )); then
            echo "[chain] $(date '+%H:%M:%S') GPU drain OK"
            return 0
        fi
        if (( $(date +%s) - start > 3600 )); then
            echo "[chain] $(date '+%H:%M:%S') GPU drain TIMEOUT" >&2
            return 1
        fi
        sleep 30
    done
}

for variant in "${VARIANTS[@]}"; do
    echo "[chain] $(date '+%H:%M:%S') waiting for GPU drain before $variant..."
    wait_for_gpu_drain || { echo "[chain] giving up on $variant"; continue; }
    tag="${CHAIN_TAG}_${variant}"
    echo "[chain] $(date '+%H:%M:%S') ==== firing $variant (tag=$tag) ===="
    RUN_K_RESULTS_TAG="$tag" \
        bash "$SCRIPT_DIR/run_k.sh" "$variant"
    rc=$?
    echo "[chain] $(date '+%H:%M:%S') $variant rc=$rc"
done

echo "[chain] $(date '+%H:%M:%S') chain done.  Aggregate with:"
echo "  python $SCRIPT_DIR/parse_daemon_events.py <each_cycle_dir>"
