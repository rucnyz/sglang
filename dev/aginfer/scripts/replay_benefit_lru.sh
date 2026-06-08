#!/bin/bash
# #231 BENEFIT — ours (full daemon + ours_greedy_score inline) vs LRU (weak
# baseline: lru_score inline, no daemon).  The do-no-harm compared ours-daemon
# vs ours-inline (marginal). The DESIGN's benefit is vs the LRU baseline it
# subsumes: under HBM pressure LRU evicts the reused prefix -> re-prefill;
# ours protects high-value reused units -> fewer re-prefilled tokens / lower
# TTFT (the reward's "prefill saved by hits" term).
#
# arrival mode (cache-hit / TTFT signal), N=3, TP=2 on GPUs 5,6, A3 pressure
# (MAX_TOTAL_TOKENS overridable).  Arm names kept a3(ours)/a3_kvoff(=LRU) so
# compare.py works; the cycle dirs keep each trial's sglang log for the
# re-prefill mechanism parse.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
export SGLANG_TP=2 AGINFER_GPUS=5,6
TRACE=scenarios/replay/traces/a3real.jsonl
DRIVER="$PWD/scenarios/replay/replay_driver.py"
OUT_DIR="$PWD/scenarios/replay/results/a3real_benefit_lru"
mkdir -p "$OUT_DIR"
N="${BENEFIT_N:-3}"
POOL="${MAX_TOTAL_TOKENS:-262144}"

run_arm() {  # label  policy  variant
  local label="$1" policy="$2" variant="$3"
  for i in $(seq 1 "$N"); do
    echo "[benefit] === $label trial $i  pool=$POOL  $(date '+%T') ==="
    SGLANG_KV_POLICY_MODULE="$policy" \
    MAX_TOTAL_TOKENS="$POOL" \
    RUN_K_WORKLOAD_CMD="python $DRIVER --trace $TRACE --base-url http://127.0.0.1:9100/v1 --mode arrival --label ${label}_c${i} --out $OUT_DIR/metrics_${label}_c${i}.json" \
    RUN_K_RESULTS_TAG="benefit_${label}_c${i}" \
      bash scenarios/_shared/run_k.sh "$variant"
    sleep 15
  done
}

echo "===== BENEFIT ours-vs-LRU (TP=2, pool=$POOL) N=$N $(date '+%F %T') ====="
run_arm a3       "baselines.sglang_adapter:ours_greedy_score" a3
run_arm a3_kvoff "baselines.sglang_adapter:lru_score"         a3_kvoff
echo "===== BENEFIT DONE $(date '+%F %T') ====="
python scenarios/replay/compare.py "$OUT_DIR" || true
python scenarios/replay/parse_reprefill.py "$OUT_DIR" || true
