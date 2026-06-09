#!/usr/bin/env bash
# #208 — A3 (HBM-pressure) 5-arm FRESH campaign, N=3 each, same session +
# same GPU pair (no historical-comparison confound).  Arms:
#   lru          plain LRU eviction, no daemon
#   ta           ThunderAgent BFD-by-token proxy, no daemon
#   ours_inline  inline ours_greedy_score V_u, no daemon  (run_direct.sh)
#   ours_full    inline V_u + daemon (kv + admission)     (run_k.sh a3)
#   const_vu     ours_full but reuse signal neutralised   (AGINFER_CONST_VU=1)
#
# Decomposition this enables (per-trial mean ± std + Welch z):
#   ours_full − const_vu : value of the V_u RANKING (reuse prediction)
#   const_vu  − lru      : value of the multi-tier MACHINERY (reuse-blind)
#   ours_full − ours_inline : value of the DAEMON over the inline scorer
#   ours_full vs ta / lru   : vs the no-ML baselines
#
# ALL arms get the A3 workload: 256K KV pool (MAX_TOTAL_TOKENS) + 4k
# completion cap (A3_AK_CAP, plumbed into the baseline runners in #208).
# run_k.sh's `a3` variant sets both itself; the baseline runners need the
# two env vars exported (below).
#
# Each cycle ~50 min; 15 cycles ≈ 12.5 h wall on a dedicated GPU pair.
# Run in the background and poll the per-cycle result dirs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="$SCRIPT_DIR/../_shared"
INLINE_RUNNER="$SCRIPT_DIR/../4_ablation/daemon_overhead/run_direct.sh"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

N="${A3_CAMPAIGN_N:-3}"
STAMP="${A3_CAMPAIGN_STAMP:?set A3_CAMPAIGN_STAMP (e.g. date +%Y%m%d_%H%M%S) — passed in so resume reuses the same tag}"
IFS=',' read -ra _GPU_LIST <<< "${AGINFER_GPUS:-5,6}"

drain_gpus() {
    local start; start=$(date +%s)
    while true; do
        local clear=1
        for g in "${_GPU_LIST[@]}"; do
            local used
            used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
            (( ${used:-0} > 1024 )) && clear=0
        done
        (( clear )) && { echo "[a3camp] $(date '+%H:%M:%S') GPU drain OK"; return 0; }
        (( $(date +%s) - start > 600 )) && { echo "[a3camp] $(date '+%H:%M:%S') drain TIMEOUT — proceeding" >&2; return 1; }
        sleep 10
    done
}

# arm → runner command (as a function name).  Tag carries arm + cycle so
# each lands in a distinct results dir for parse_4arm aggregation.
run_arm_cycle() {
    local arm="$1" i="$2"
    local tag="a3camp_${STAMP}_${arm}_cycle${i}"
    echo "[a3camp] $(date '+%H:%M:%S') ==== ${arm} cycle ${i}/${N} (tag=${tag}) ===="
    case "$arm" in
        lru)
            MAX_TOTAL_TOKENS=262144 A3_AK_CAP=1 RUN_K_RESULTS_TAG="$tag" \
                bash "$SHARED/run_lru.sh" ;;
        ta)
            MAX_TOTAL_TOKENS=262144 A3_AK_CAP=1 RUN_K_RESULTS_TAG="$tag" \
                bash "$SHARED/run_thunderagent.sh" ;;
        ours_inline)
            MAX_TOTAL_TOKENS=262144 A3_AK_CAP=1 RUN_K_RESULTS_TAG="$tag" \
                bash "$INLINE_RUNNER" ;;
        ours_full)
            RUN_K_RESULTS_TAG="$tag" bash "$SHARED/run_k.sh" a3 ;;
        const_vu)
            AGINFER_CONST_VU=1 RUN_K_RESULTS_TAG="$tag" \
                bash "$SHARED/run_k.sh" a3 ;;
        *) echo "[a3camp] unknown arm: $arm" >&2; return 2 ;;
    esac
}

ARMS=(${A3_CAMPAIGN_ARMS:-lru ta ours_inline ours_full const_vu})

echo "[a3camp] $(date '+%H:%M:%S') START — arms=(${ARMS[*]}) N=$N GPUs=${AGINFER_GPUS:-5,6} stamp=$STAMP"
first=1
for arm in "${ARMS[@]}"; do
    for i in $(seq 1 "$N"); do
        [[ $first -eq 0 ]] && drain_gpus
        first=0
        run_arm_cycle "$arm" "$i" || echo "[a3camp] $(date '+%H:%M:%S') WARN ${arm} cycle ${i} returned $?"
        echo "[a3camp] $(date '+%H:%M:%S') ${arm} cycle ${i} done"
    done
done
echo "[a3camp] $(date '+%H:%M:%S') CAMPAIGN DONE — aggregate with parse_4arm (repoint globs to a3camp_${STAMP}_* dirs)"
