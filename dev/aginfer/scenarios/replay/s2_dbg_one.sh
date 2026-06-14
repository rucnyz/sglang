#!/bin/bash
# S2DBG: one ours cycle with AGINFER_S2DBG=1 to trace how the ~24K shared prefix
# is scored at eviction (n_holders applied? scored as a leaf at all?).
set -x
cd /scratch/yuzhou/projects/sglang/dev/aginfer
T=/scratch/yuzhou/projects/sglang/reproduce/RQ1/scenarios/s2-shared-prefix-retention/traces/s2_churn_strong.jsonl
for p in $(nvidia-smi -i 5,6 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
fuser -k 9100/tcp 30000/tcp 50051/tcp 2>/dev/null; sleep 8
AGINFER_S2DBG=1 AGINFER_DISABLE_PROMOTE=1 HICACHE_WRITE_POLICY=write_through \
  bash scenarios/replay/run_replay_pressured.sh "$T" 1 1 65536 "a3" 1 "aginfer:hint_v_u"
echo "S2DBG ONE DONE"
