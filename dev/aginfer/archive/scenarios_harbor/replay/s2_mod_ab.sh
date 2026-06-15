#!/bin/bash
# S2 moderate-flood A/B with the COMPLETE holder-count fix (all 4 bugs).
# Moderate trace: BG_PER_GAP=6 -> ~144K/gap < 164K cache (pool 65536), so the
# DRAM write-through can KEEP UP and the tier reliably populates — isolating the
# now-working holder-count eviction lever from the tier-flakiness confound that
# dominates the strong (240K/gap) regime. LRU drops the shared prefix by recency
# (~11/40 baseline); ours should retain it (HBM-pin or reliable DRAM-loadback).
set -x
cd /scratch/yuzhou/projects/sglang/dev/aginfer
T=/scratch/yuzhou/projects/sglang/reproduce/RQ1/scenarios/s2-shared-prefix-retention/traces/s2_churn.jsonl
POOL=65536
clean() {
  for p in $(nvidia-smi -i 5,6 --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  fuser -k 9100/tcp 30000/tcp 50051/tcp 2>/dev/null; sleep 8
}
clean
echo "===== [B] LRU, write_through, moderate flood ====="
HICACHE_WRITE_POLICY=write_through bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3_kvoff" 1 "lru_score"
clean
echo "===== [OURS] hint_v_u (FULL fix), promote off, write_through, moderate flood ====="
AGINFER_DISABLE_PROMOTE=1 HICACHE_WRITE_POLICY=write_through bash scenarios/replay/run_replay_pressured.sh "$T" 3 1 "$POOL" "a3" 1 "aginfer:hint_v_u"
echo "MOD WRAPPER ALL DONE"
