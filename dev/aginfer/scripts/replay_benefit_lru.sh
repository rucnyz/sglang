#!/bin/bash
# #231 BENEFIT — ours (full daemon + ours_greedy_score inline) vs LRU
# (lru_score inline, no daemon).  The design's benefit binds when KV goes
# IDLE long enough to be evicted: the captured trace has ~0.2s tool gaps so
# the KV is never cold.  GAP_SCALE stretches the gaps to realistic slow-tool
# latency; under that idle pressure LRU evicts the reused prefix (re-prefill /
# reactive load) while ours protects it by value + promotes proactively ->
# fewer re-prefilled tokens / lower TTFT.
#
# Env knobs: MODE (session|arrival), GAP_SCALE, MAX_TOTAL_TOKENS, BENEFIT_N,
#            SLOWDOWN, AGINFER_GPUS, OUT_TAG.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
export SGLANG_TP=2 AGINFER_GPUS="${AGINFER_GPUS:-4,5}"
TRACE=scenarios/replay/traces/a3real.jsonl
DRIVER="$PWD/scenarios/replay/replay_driver.py"
MODE="${MODE:-session}"
GAP_SCALE="${GAP_SCALE:-30}"
POOL="${MAX_TOTAL_TOKENS:-98304}"
N="${BENEFIT_N:-2}"
SLOWDOWN="${SLOWDOWN:-1.0}"
TAG="${OUT_TAG:-benefit_lru_${MODE}_g${GAP_SCALE}_p${POOL}}"
OUT_DIR="$PWD/scenarios/replay/results/a3real_${TAG}"
mkdir -p "$OUT_DIR"

run_arm() {  # label  policy  variant
  local label="$1" policy="$2" variant="$3"
  for i in $(seq 1 "$N"); do
    echo "[benefit] === $label trial $i  mode=$MODE gap=$GAP_SCALE pool=$POOL  $(date '+%T') ==="
    SGLANG_KV_POLICY_MODULE="$policy" \
    MAX_TOTAL_TOKENS="$POOL" \
    RUN_K_WORKLOAD_CMD="python $DRIVER --trace $TRACE --base-url http://127.0.0.1:9100/v1 --mode $MODE --gap-scale $GAP_SCALE --slowdown $SLOWDOWN --label ${label}_c${i} --out $OUT_DIR/metrics_${label}_c${i}.json" \
    RUN_K_RESULTS_TAG="${TAG}_${label}_c${i}" \
      bash scenarios/_shared/run_k.sh "$variant"
    sleep 15
  done
}

echo "===== BENEFIT ours-vs-LRU  mode=$MODE gap=$GAP_SCALE pool=$POOL N=$N  $(date '+%F %T') ====="
run_arm a3       "baselines.sglang_adapter:ours_greedy_score" a3
run_arm a3_kvoff "baselines.sglang_adapter:lru_score"         a3_kvoff
echo "===== BENEFIT DONE $(date '+%F %T') -> $OUT_DIR ====="
python scenarios/replay/compare.py "$OUT_DIR" || true
python scenarios/replay/parse_reprefill.py "$OUT_DIR" || true
