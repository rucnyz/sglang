#!/bin/bash
# #231 — chained do-no-harm + benefit replay of the a3real trace.
# arrival mode (per-request latency / do-no-harm) then session mode
# (closed-loop end-to-end makespan), N=3 trials per arm, TP=4 on 4-7.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
export SGLANG_TP=4 AGINFER_GPUS=4,5,6,7
TRACE=scenarios/replay/traces/a3real.jsonl

echo "===== REPLAY arrival (do-no-harm) N=3 $(date '+%F %T') ====="
bash scenarios/replay/run_replay.sh "$TRACE" 3 arrival 1.0
echo "===== REPLAY session (end-to-end benefit) N=3 $(date '+%F %T') ====="
bash scenarios/replay/run_replay.sh "$TRACE" 3 session 1.0
echo "===== ALL REPLAY DONE $(date '+%F %T') ====="
