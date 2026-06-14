#!/bin/bash
# S2 EVICTION-LEVER ISOLATION: ours with the daemon hint-push ON (so the eviction
# scorer gets n_holders=8) but the migrate/demote dispatch OFF (AGINFER_DISABLE_MIGRATE).
# This isolates the pure value-aware holder-count EVICTION lever — the design's S2
# retention mechanism — from the relief demote that is net-counterproductive under
# heavy churn (strong@65536: ours 40.5 > LRU 28.7). Moderate trace (reliable tier).
# Compare against the moderate LRU baseline from s2_mod_ab.sh (same session/settings).
set -x
cd /scratch/yuzhou/projects/sglang/dev/aginfer
T=/scratch/yuzhou/projects/sglang/reproduce/RQ1/scenarios/s2-shared-prefix-retention/traces/s2_churn.jsonl
POOL=65536
for p in $(nvidia-smi -i 5,6 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
fuser -k 9100/tcp 30000/tcp 50051/tcp 2>/dev/null; sleep 8
echo "===== [OURS-EVICT-ONLY] hint_v_u + DISABLE_MIGRATE, promote off, write_through ====="
AGINFER_DISABLE_MIGRATE=1 AGINFER_DISABLE_PROMOTE=1 HICACHE_WRITE_POLICY=write_through \
  RUN_K_RESULTS_TAG_SUFFIX=evictonly \
  bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3" 1 "aginfer:hint_v_u"
echo "EVICT-ISOLATION DONE"
