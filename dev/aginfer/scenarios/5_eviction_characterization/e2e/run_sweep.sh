#!/usr/bin/env bash
# Tier-2 e2e characterization sweep (#230) — grounds the Tier-1 trends on the
# real harbor/terminus-2 stack.  Each arm = one full run_k.sh on the chosen
# pressure regime with the inline scorer + hint freshness swapped via env.
#
# Needs GPUs free (4 for TP=4).  Run AFTER the do-no-harm campaign, or set
# AGINFER_GPUS to a free pair.  N cycles per arm; results land under
# results/run_K_a3_<tag>/ and are parsed by parse.py.
#
# Knobs the arms turn (all already wired):
#   SGLANG_KV_POLICY_MODULE  inline scorer  (#230 run_k respects pre-set)
#   AGINFER_HINT_DELAY_MS    hint staleness (#230 daemon knob)
#   MAX_TOTAL_TOKENS         pressure regime
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1   # dev/aginfer
export SGLANG_TP="${SGLANG_TP:-4}" SGLANG_EP="${SGLANG_EP:-4}" \
       AGINFER_GPUS="${AGINFER_GPUS:-4,5,6,7}"
L="$PWD/logs"
N="${N:-3}"
# Pressure regime: A3-tight by default; sweep by re-invoking with PRESSURE set.
PRESSURE="${MAX_TOTAL_TOKENS:-262144}"
ADAPT="baselines.sglang_adapter"

# arm = "tag|SGLANG_KV_POLICY_MODULE|AGINFER_HINT_DELAY_MS"
ARMS=(
  "lru|${ADAPT}:lru_score|0"                 # S1 baseline (no reuse steering)
  "constvu|${ADAPT}:const_v_u_score|0"       # S1 ablation (plumbing, no signal)
  "ours_fresh|${ADAPT}:ours_greedy_score|0"  # S1/S2 reference
  "ours_d100|${ADAPT}:ours_greedy_score|100" # S2 gradient
  "ours_d250|${ADAPT}:ours_greedy_score|250"
  "ours_d500|${ADAPT}:ours_greedy_score|500"
  "ours_d1000|${ADAPT}:ours_greedy_score|1000"
)

echo "[sweep] $(date '+%F %T') N=$N pressure(MAX_TOTAL_TOKENS)=$PRESSURE"
for arm in "${ARMS[@]}"; do
  IFS='|' read -r tag policy delay <<<"$arm"
  for i in $(seq 1 "$N"); do
    echo "[sweep] $(date '+%F %T') arm=$tag cycle=$i policy=$policy delay=${delay}ms START"
    SGLANG_KV_POLICY_MODULE="$policy" \
    AGINFER_HINT_DELAY_MS="$delay" \
    MAX_TOTAL_TOKENS="$PRESSURE" \
    RUN_K_RESULTS_TAG="char_${tag}_p${PRESSURE}_c${i}" \
      bash scenarios/_shared/run_k.sh a3 \
      > "$L/char_${tag}_c${i}.log" 2>&1
    echo "[sweep] $(date '+%F %T') arm=$tag cycle=$i DONE rc=$?"
    sleep 20
  done
done
echo "[sweep] $(date '+%F %T') ALL ARMS DONE — parse: python scenarios/5_eviction_characterization/e2e/parse.py"
