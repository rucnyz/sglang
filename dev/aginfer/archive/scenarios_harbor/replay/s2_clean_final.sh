#!/bin/bash
# S2 CLEAN FINAL: airtight N=3, no overlapping jobs. Moderate flood (reliable tier).
# B/LRU vs OURS-evict-only (holder-count eviction lever, daemon migrate OFF so the
# net-counterproductive relief demote doesn't confound). Complete 4-bug holder-count
# fix in place. Metric: re-prefill (recompute of the 24K shared prefix).
set -x
cd /scratch/yuzhou/projects/sglang/dev/aginfer
T=/scratch/yuzhou/projects/sglang/reproduce/RQ1/scenarios/s2-shared-prefix-retention/traces/s2_churn.jsonl
POOL=65536
clean() {
  for p in $(nvidia-smi -i 5,6 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  fuser -k 9100/tcp 30000/tcp 50051/tcp 2>/dev/null; sleep 8
}
clean
echo "===== [B] LRU baseline, write_through, moderate ====="
HICACHE_WRITE_POLICY=write_through bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3_kvoff" 1 "lru_score"
clean
echo "===== [OURS] hint_v_u evict-only (DISABLE_MIGRATE), promote off, write_through ====="
AGINFER_DISABLE_MIGRATE=1 AGINFER_DISABLE_PROMOTE=1 HICACHE_WRITE_POLICY=write_through \
  bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3" 1 "aginfer:hint_v_u"
echo "CLEAN FINAL ALL DONE"
