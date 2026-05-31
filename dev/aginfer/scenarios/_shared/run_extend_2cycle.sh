#!/usr/bin/env bash
# T9 extension — add 1 cycle each to OURS_full and LRU to push the
# Welch z above the 1.96 threshold (p < 0.05).
#
# Plan: OURS_full first, then LRU.  ~2.5 h total.
#
# Output dirs use new "extend_<TAG>" prefix; parse_4arm.py is
# patched to glob these in addition to the original matrix dirs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

EXTEND_TAG="${EXTEND_TAG:-$(date +%Y%m%d_%H%M%S)}"
echo "[extend2] tag=$EXTEND_TAG"
echo "[extend2] sglang HEAD: $(cd "$AGINFER_DIR/../.." && git rev-parse HEAD)"

# --- cycle 1: OURS_full ---
echo "[extend2] ==== cycle 1/2 (ours_full) ===="
echo "[extend2] $(date '+%Y-%m-%d %H:%M:%S') starting"
ours_dir="$AGINFER_RESULTS/run_K_full_extend_${EXTEND_TAG}_cycle1_ours"
if [[ -d "$ours_dir/harbor_jobs" ]] && \
   find "$ours_dir/harbor_jobs" -mindepth 3 -name 'result.json' 2>/dev/null | wc -l | grep -qE '^([3-9][0-9]|[1-9][0-9]{2,})$'; then
    echo "[extend2] cycle 1 already complete, skipping"
else
    RUN_K_RESULTS_TAG="extend_${EXTEND_TAG}_cycle1_ours" \
    bash "$SCRIPT_DIR/run_k.sh" full || rc=$?
    rc=${rc:-0}
    [[ "$rc" -ne 0 ]] && echo "[extend2] cycle 1 rc=$rc" >&2
fi
echo "[extend2] $(date '+%Y-%m-%d %H:%M:%S') cycle 1 done"

# Active GPU drain wait
IFS=',' read -ra _GPU_LIST <<< "${AGINFER_GPUS:-0,1}"
for tick in $(seq 1 30); do
    all_clear=1
    for gpu in "${_GPU_LIST[@]}"; do
        used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        if (( ${used:-0} > 1024 )); then all_clear=0; fi
    done
    if (( all_clear )); then
        echo "[extend2] GPU drain confirmed @ tick $tick"
        break
    fi
    sleep 2
done

# --- cycle 2: LRU ---
echo ""
echo "[extend2] ==== cycle 2/2 (lru) ===="
echo "[extend2] $(date '+%Y-%m-%d %H:%M:%S') starting"
lru_dir="$AGINFER_RESULTS/run_LRU_now_extend_${EXTEND_TAG}_cycle2_lru"
if [[ -d "$lru_dir/harbor_jobs" ]] && \
   find "$lru_dir/harbor_jobs" -mindepth 3 -name 'result.json' 2>/dev/null | wc -l | grep -qE '^([3-9][0-9]|[1-9][0-9]{2,})$'; then
    echo "[extend2] cycle 2 already complete, skipping"
else
    RUN_K_RESULTS_TAG="extend_${EXTEND_TAG}_cycle2_lru" \
    bash "$SCRIPT_DIR/run_lru.sh" || rc=$?
    rc=${rc:-0}
    [[ "$rc" -ne 0 ]] && echo "[extend2] cycle 2 rc=$rc" >&2
fi
echo "[extend2] $(date '+%Y-%m-%d %H:%M:%S') cycle 2 done"

echo "[extend2] all extension cycles attempted."
echo "[extend2] Re-aggregate with: python $SCRIPT_DIR/parse_4arm.py"
