#!/bin/bash
# S2 decisive re-measurement WITH the now-live host/DRAM tier (post-upstream-sync).
# Strong workload (LRU was 64/64 recompute, ours tied 64/64 in the dead-tier era).
# Hypothesis: a live DRAM tier lets ours DEMOTE the high-holder shared prefix to DRAM
# (freeing HBM for the active set) instead of dropping it -> host load-back, not recompute.
set -x
cd /scratch/yuzhou/projects/sglang/dev/aginfer
T=/scratch/yuzhou/projects/sglang/reproduce/RQ1/scenarios/s2-shared-prefix-retention/traces/s2_churn_strong.jsonl
POOL=49152

clean() {
  for p in $(nvidia-smi -i 5,6 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  fuser -k 9100/tcp 30000/tcp 50051/tcp 2>/dev/null; sleep 8
}

clean
echo "===== [B] LRU baseline, write_through, live tier ====="
HICACHE_WRITE_POLICY=write_through bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3_kvoff" 1 "lru_score"

clean
echo "===== [OURS] value-aware hint_v_u, promote off, write_through, live tier ====="
AGINFER_DISABLE_PROMOTE=1 HICACHE_WRITE_POLICY=write_through bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3" 1 "aginfer:hint_v_u"

echo "WRAPPER ALL DONE"
