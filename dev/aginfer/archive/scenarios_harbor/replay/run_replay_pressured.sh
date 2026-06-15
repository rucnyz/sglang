#!/usr/bin/env bash
# Pressured trace-replay: ours (a3) vs B (a3_kvoff = HiCache+LRU, scheduling off),
# session mode with stretched tool gaps + a pressured KV pool so the FULL hierarchy
# drops (HiCache-DRAM overflows -> B re-prefills; ours' 4-tier + predictive promote
# retains). Metric = re-prefill tokens (parse_reprefill.py) + compare.py.
#
# Usage: run_replay_pressured.sh <trace.jsonl> [N] [GAP_SCALE] [POOL] [ARMS]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1     # dev/aginfer
TRACE="${1:?usage: run_replay_pressured.sh <trace.jsonl> [N] [GAP_SCALE] [POOL] [ARMS] [REPLICATE]}"
N="${2:-3}"
GAP="${3:-30}"
POOL="${4:-131072}"
ARMS="${5:-a3 a3_kvoff}"
REPLICATE="${6:-1}"
SCORER="${7:-}"          # e.g. lru_score | ours_greedy_score (sets SGLANG_KV_POLICY_MODULE)
MODE=session
if [[ -n "$SCORER" ]]; then
    if [[ "$SCORER" == *:* ]]; then export SGLANG_KV_POLICY_MODULE="$SCORER"
    else export SGLANG_KV_POLICY_MODULE="baselines.sglang_adapter:${SCORER}"; fi
fi
SCTAG="${SCORER:+_${SCORER//:/_}}${HICACHE_WRITE_POLICY:+_wp_${HICACHE_WRITE_POLICY}}"
[[ -f "$TRACE" ]] || { echo "trace not found: $TRACE" >&2; exit 1; }
DRIVER="$PWD/scenarios/replay/replay_driver.py"
BASE="$(basename "$TRACE" .jsonl)"
OUT_DIR="$PWD/scenarios/replay/results/${BASE}_pressured_g${GAP}_p${POOL}_x${REPLICATE}${SCTAG}"
mkdir -p "$OUT_DIR"
echo "[pressured] trace=$TRACE N=$N gap=$GAP pool=$POOL replicate=$REPLICATE arms='$ARMS' -> $OUT_DIR"
for arm in $ARMS; do
    for i in $(seq 1 "$N"); do
        label="${arm}_c${i}"
        out="$OUT_DIR/metrics_${label}.json"
        echo "[pressured] === arm=$arm trial=$i pool=$POOL gap=$GAP -> $out ==="
        MAX_TOTAL_TOKENS="$POOL" \
        RUN_K_WORKLOAD_CMD="python $DRIVER --trace $TRACE --base-url http://127.0.0.1:9100/v1 --mode $MODE --gap-scale $GAP --replicate $REPLICATE --label $label --out $out" \
        RUN_K_RESULTS_TAG="repl_${BASE}_${label}" \
            bash scenarios/_shared/run_k.sh "$arm"
        sleep 15
    done
done
echo "[pressured] ALL DONE -> $OUT_DIR"
python scenarios/replay/compare.py "$OUT_DIR" 2>/dev/null || true
echo "--- re-prefill tokens (the headline metric) ---"
python scenarios/replay/parse_reprefill.py "$OUT_DIR" 2>/dev/null || echo "(parse_reprefill needs the sglang logs in the results dirs)"
