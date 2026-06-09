#!/bin/bash
# HIGH-CONCURRENCY OVER-SUBSCRIPTION benefit — the ThunderAgent regime.
#
# The cache-hit / eviction-order line is a dead end vs fair baseline B
# (sglang+HiCache+LRU): with HiCache write-through on both arms the 4-tier
# residence is already in B, and value-aware eviction ORDER ties/loses LRU on
# recency-aligned agentic reuse.  The design's distinctive lever is
# acting-phase PAUSING + value-aware MIGRATION under genuine over-subscription
# (in-flight working set > HBM), which is exactly where ThunderAgent shows
# 1.4-3x.  At the trace's native ~30 concurrency HBM is full of EVICTABLE
# cached prefixes so inline eviction suffices and the daemon idles.  We
# replicate each program to REPLICATE salted concurrent agents (~30xK) so the
# live working set exceeds capacity -> thrash -> pausing/migration binds.
#
# Three-arm framing (all HiCache ON, per user: "至少在hicache场景里达到
# thunderagent的水平"): a3 = ours (daemon pause+migrate), a3_kvoff = B
# (sglang+HiCache+LRU, no daemon).  ThunderAgent arm added separately.
#
# Metric: makespan / throughput / goodput / cache-hit under IDENTICAL offered
# load (same replicated trace).  Env: REPLICATE GAP_SCALE MAX_TOTAL_TOKENS
# HICACHE_RATIO HICACHE_STORE_SIZE BENEFIT_N AGINFER_GPUS OUT_TAG MAXCONC.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
export SGLANG_TP=2 AGINFER_GPUS="${AGINFER_GPUS:-4,5}"
export HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
export HICACHE_STORE_SIZE="${HICACHE_STORE_SIZE:-200gb}"
TRACE=scenarios/replay/traces/a3real.jsonl
DRIVER="$PWD/scenarios/replay/replay_driver.py"
MODE=session
GAP_SCALE="${GAP_SCALE:-30}"
POOL="${MAX_TOTAL_TOKENS:-98304}"
REPLICATE="${REPLICATE:-4}"
MAXCONC="${MAXCONC:-4096}"
DEADLINE="${REQUEST_DEADLINE:-600}"
N="${BENEFIT_N:-2}"
TAG="${OUT_TAG:-overload_g${GAP_SCALE}_p${POOL}_x${REPLICATE}}"
OUT_DIR="$PWD/scenarios/replay/results/a3real_${TAG}"
mkdir -p "$OUT_DIR"

run_arm() {  # label  policy  variant
  local label="$1" policy="$2" variant="$3"
  for i in $(seq 1 "$N"); do
    echo "[overload] === $label trial $i  gap=$GAP_SCALE pool=$POOL x$REPLICATE  $(date '+%T') ==="
    SGLANG_KV_POLICY_MODULE="$policy" \
    MAX_TOTAL_TOKENS="$POOL" \
    RUN_K_WORKLOAD_CMD="python $DRIVER --trace $TRACE --base-url http://127.0.0.1:9100/v1 --mode $MODE --gap-scale $GAP_SCALE --replicate $REPLICATE --max-concurrency $MAXCONC --request-deadline $DEADLINE --label ${label}_c${i} --out $OUT_DIR/metrics_${label}_c${i}.json" \
    RUN_K_RESULTS_TAG="${TAG}_${label}_c${i}" \
      bash scenarios/_shared/run_k.sh "$variant"
    sleep 15
  done
}

echo "===== OVERLOAD ours-vs-B  gap=$GAP_SCALE pool=$POOL replicate=$REPLICATE ratio=$HICACHE_RATIO store=$HICACHE_STORE_SIZE N=$N  $(date '+%F %T') ====="
run_arm a3       "baselines.sglang_adapter:ours_greedy_score" a3
run_arm a3_kvoff "baselines.sglang_adapter:lru_score"         a3_kvoff
echo "===== OVERLOAD DONE $(date '+%F %T') -> $OUT_DIR ====="
python scenarios/replay/compare.py "$OUT_DIR" || true
python scenarios/replay/parse_reprefill.py "$OUT_DIR" || true
