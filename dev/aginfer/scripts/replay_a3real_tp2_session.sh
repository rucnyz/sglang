#!/bin/bash
# #231 benefit — closed-loop SESSION replay of the a3real trace at TP=2 on the
# free GPUs 5,6. Measures end-to-end makespan + per-session e2e: ours (a3,
# kv+admission ON) vs baseline (a3_kvoff). N=3 trials per arm, fresh stack each.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
export SGLANG_TP=2 AGINFER_GPUS=5,6
TRACE=scenarios/replay/traces/a3real.jsonl
echo "===== SESSION benefit (TP=2, GPUs 5,6) N=3 $(date '+%F %T') ====="
bash scenarios/replay/run_replay.sh "$TRACE" 3 session 1.0
echo "===== SESSION benefit DONE $(date '+%F %T') ====="
