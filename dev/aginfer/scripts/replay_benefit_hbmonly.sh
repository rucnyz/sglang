#!/bin/bash
# #231 BENEFIT (corrected baseline) — ours (full 4-tier: daemon + ours_greedy
# inline + HiCache) vs LRU HBM-ONLY (lru_score inline, HiCache OFF, no daemon).
#
# This is the paper's actual "LRU (literature)" baseline: residence restricted
# to {{HBM}, empty}, no DRAM/DISK.  The earlier ours-vs-LRU comparison gave LRU
# a DRAM tier via HiCache, so the design's core benefit (4-tier avoids re-
# prefill) was already in the baseline -> flat.  With HiCache OFF the LRU arm
# DROPS evicted prefix -> re-prefills on reuse; ours keeps it in DRAM ->
# fewer re-prefilled tokens / lower TTFT.  parse_reprefill.py measures it.
#
# Env: MODE GAP_SCALE MAX_TOTAL_TOKENS BENEFIT_N AGINFER_GPUS OUT_TAG SLOWDOWN.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
export SGLANG_TP=2 AGINFER_GPUS="${AGINFER_GPUS:-4,5}"
TRACE=scenarios/replay/traces/a3real.jsonl
DRIVER="$PWD/scenarios/replay/replay_driver.py"
MODE="${MODE:-session}"
GAP_SCALE="${GAP_SCALE:-30}"
POOL="${MAX_TOTAL_TOKENS:-98304}"
N="${BENEFIT_N:-3}"
SLOWDOWN="${SLOWDOWN:-1.0}"
TAG="${OUT_TAG:-benefit_hbmonly_${MODE}_g${GAP_SCALE}_p${POOL}}"
OUT_DIR="$PWD/scenarios/replay/results/a3real_${TAG}"
mkdir -p "$OUT_DIR"

run_arm() {  # label  policy  variant  hicache_off(0/1)
  local label="$1" policy="$2" variant="$3" hcoff="$4"
  for i in $(seq 1 "$N"); do
    echo "[benefit] === $label trial $i  mode=$MODE gap=$GAP_SCALE pool=$POOL hicache_off=$hcoff  $(date '+%T') ==="
    SGLANG_KV_POLICY_MODULE="$policy" \
    MAX_TOTAL_TOKENS="$POOL" \
    HICACHE_OFF="$hcoff" \
    RUN_K_WORKLOAD_CMD="python $DRIVER --trace $TRACE --base-url http://127.0.0.1:9100/v1 --mode $MODE --gap-scale $GAP_SCALE --slowdown $SLOWDOWN --label ${label}_c${i} --out $OUT_DIR/metrics_${label}_c${i}.json" \
    RUN_K_RESULTS_TAG="${TAG}_${label}_c${i}" \
      bash scenarios/_shared/run_k.sh "$variant"
    sleep 15
  done
}

echo "===== BENEFIT ours(4-tier) vs LRU(HBM-only)  mode=$MODE gap=$GAP_SCALE pool=$POOL N=$N  $(date '+%F %T') ====="
run_arm a3       "baselines.sglang_adapter:ours_greedy_score" a3       0   # ours: HiCache ON
run_arm a3_kvoff "baselines.sglang_adapter:lru_score"         a3_kvoff 1   # LRU: HiCache OFF (HBM-only)
echo "===== BENEFIT DONE $(date '+%F %T') -> $OUT_DIR ====="
python scenarios/replay/compare.py "$OUT_DIR" || true
python scenarios/replay/parse_reprefill.py "$OUT_DIR" || true
