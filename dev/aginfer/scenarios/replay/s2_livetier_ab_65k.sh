#!/bin/bash
# S2 decisive re-measurement at the LIVE-TIER pool (65536), strong-flood trace.
# At pool 49152 both arms tied 64/64 (HBM too tight to pin 24K + DRAM flooded, host_tok=0).
# At pool 65536: cache = 65536 + 1.5*65536 = 164K; strong bg flood = 240K/gap >> 164K
#   -> LRU loses the shared prefix from BOTH tiers every gap (high recompute),
#      BUT ours has room to PIN the 24K prefix in HBM (value rule, n_holders=8) -> device-hit.
# Hypothesis: clean separation (ours device-hits, LRU recomputes).
set -x
cd /scratch/yuzhou/projects/sglang/dev/aginfer
T=/scratch/yuzhou/projects/sglang/reproduce/RQ1/scenarios/s2-shared-prefix-retention/traces/s2_churn_strong.jsonl
POOL=65536

clean() {
  for p in $(nvidia-smi -i 5,6 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  fuser -k 9100/tcp 30000/tcp 50051/tcp 2>/dev/null; sleep 8
}

clean
echo "===== [B] LRU baseline, write_through, live tier, pool 65536 ====="
HICACHE_WRITE_POLICY=write_through bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3_kvoff" 1 "lru_score"

clean
echo "===== [OURS] value-aware hint_v_u, promote off, write_through, pool 65536 ====="
AGINFER_DISABLE_PROMOTE=1 HICACHE_WRITE_POLICY=write_through bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3" 1 "aginfer:hint_v_u"

echo "WRAPPER ALL DONE"
