#!/bin/bash
# OPEN-LOOP OVERLOAD — the regime where admission/pause can bite.
#
# Closed-loop session replay couldn't engage the daemon: sglang never
# preempts (the live decode set fits / queues; only the evictable cache
# overflows, which radix eviction handles), and a tool-gap program has no
# in-flight request to pause.  Open-loop arrival at a COMPRESSED timeline
# (--slowdown < 1) x REPLICATE copies drives the offered RATE past serving
# capacity, so the RUNNING set overflows HBM -> real memory-bound preemption.
# There, ours can ADMIT/PAUSE low-value load to protect goodput/p99 — a lever
# sglang+HiCache+LRU structurally lacks (LRU evicts cache; it cannot shed
# offered load).
#
# Metric: goodput (n_ok within deadline) / p99 e2e / throughput, IDENTICAL
# offered load.  Env: SLOWDOWN REPLICATE MAX_TOTAL_TOKENS HICACHE_RATIO
# HICACHE_STORE_SIZE BENEFIT_N AGINFER_GPUS OUT_TAG MAXCONC REQUEST_DEADLINE.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
export SGLANG_TP=2 AGINFER_GPUS="${AGINFER_GPUS:-4,5}"
export HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
export HICACHE_STORE_SIZE="${HICACHE_STORE_SIZE:-200gb}"
TRACE=scenarios/replay/traces/a3real.jsonl
DRIVER="$PWD/scenarios/replay/replay_driver.py"
SLOWDOWN="${SLOWDOWN:-0.3}"
POOL="${MAX_TOTAL_TOKENS:-98304}"
REPLICATE="${REPLICATE:-3}"
MAXCONC="${MAXCONC:-4096}"
DEADLINE="${REQUEST_DEADLINE:-120}"
N="${BENEFIT_N:-1}"
TAG="${OUT_TAG:-overload_open_s${SLOWDOWN}_p${POOL}_x${REPLICATE}}"
OUT_DIR="$PWD/scenarios/replay/results/a3real_${TAG}"
mkdir -p "$OUT_DIR"

run_arm() {  # label  policy  variant
  local label="$1" policy="$2" variant="$3"
  for i in $(seq 1 "$N"); do
    echo "[openload] === $label trial $i  slowdown=$SLOWDOWN pool=$POOL x$REPLICATE  $(date '+%T') ==="
    SGLANG_KV_POLICY_MODULE="$policy" \
    MAX_TOTAL_TOKENS="$POOL" \
    RUN_K_WORKLOAD_CMD="python $DRIVER --trace $TRACE --base-url http://127.0.0.1:9100/v1 --mode arrival --slowdown $SLOWDOWN --replicate $REPLICATE --max-concurrency $MAXCONC --request-deadline $DEADLINE --label ${label}_c${i} --out $OUT_DIR/metrics_${label}_c${i}.json" \
    RUN_K_RESULTS_TAG="${TAG}_${label}_c${i}" \
      bash scenarios/_shared/run_k.sh "$variant"
    sleep 15
  done
}

echo "===== OPEN-LOOP OVERLOAD ours-vs-B  slowdown=$SLOWDOWN pool=$POOL replicate=$REPLICATE ratio=$HICACHE_RATIO store=$HICACHE_STORE_SIZE N=$N  $(date '+%F %T') ====="
run_arm a3       "baselines.sglang_adapter:ours_greedy_score" a3
run_arm a3_kvoff "baselines.sglang_adapter:lru_score"         a3_kvoff
echo "===== OPEN-LOOP OVERLOAD DONE $(date '+%F %T') -> $OUT_DIR ====="
python scenarios/replay/compare.py "$OUT_DIR" || true
