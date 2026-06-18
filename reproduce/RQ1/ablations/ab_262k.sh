#!/bin/bash
# Clean A/B at Qwen native max context (262144). Natural full trace, conc 64,
# N=3 per arm, GPU 7. base (lru) then sys (full win config). Sequential: single
# GPU, one server at a time. run_arm.sh handles boot/teardown/port-kill.
set -u
cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen3p5_9b.jsonl
OUT=reproduce/RQ1/ablations/ar_262k_ab
mkdir -p "$OUT"


echo "[ab] $(date) START base"
bash reproduce/RQ1/run_arm.sh base "$TRACE" 0.02 64 - 3 "$OUT" 2>&1 | sed 's/^/[base] /'
echo "[ab] $(date) START sys"
bash reproduce/RQ1/run_arm.sh sys  "$TRACE" 0.02 64 - 3 "$OUT" 2>&1 | sed 's/^/[sys] /'
echo "[ab] $(date) ALL DONE -> $OUT"
