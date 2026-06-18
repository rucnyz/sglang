#!/bin/bash
# Three-arm ablation at Qwen native max context (262144), GPU 7. PLAN order:
#   base  = LRU, no budgeter            (baseline)
#   inter = LRU + cross-pool budgeter   (proves the inter-layer)
#   full  = LPB + cross-pool budgeter   (LPB adds further)
# Sequential (single GPU, one server at a time). LIM/N via env for fast
# validation (N=1) vs final (N=3). Each arm writes its own subdir.
set -u
cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen3p5_9b.jsonl
OUT=${OUT:-reproduce/RQ1/ablations/ar_262k_ab3}
LIM=${LIM:-400}
N=${N:-1}

mkdir -p "$OUT"

echo "[ab3] $(date) START base (LRU, no budgeter) LIM=$LIM N=$N"
bash reproduce/RQ1/run_arm.sh base "$TRACE" 0.02 64 "$LIM" "$N" "$OUT/base" 2>&1 | sed 's/^/[base] /'
echo "[ab3] $(date) START inter (LRU + budgeter)"
EVICT=lru bash reproduce/RQ1/run_arm.sh sys "$TRACE" 0.02 64 "$LIM" "$N" "$OUT/inter" 2>&1 | sed 's/^/[inter] /'
echo "[ab3] $(date) START full (LPB + budgeter)"
EVICT=lpb bash reproduce/RQ1/run_arm.sh sys "$TRACE" 0.02 64 "$LIM" "$N" "$OUT/full" 2>&1 | sed 's/^/[full] /'
echo "[ab3] $(date) ALL DONE -> $OUT"
