#!/bin/bash
# Clean B-vs-ours resume-TTFT head-to-head. Restart the stack between arms so
# neither arm rides the other's warmed cache (same per-program salts => same KV
# radix path => carryover). One cycle per arm per invocation; wrap in a loop for
# N>=3. Prints the resume-TTFT line for each arm.
cd "$(dirname "$0")/../.." || exit 1   # dev/aginfer
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
WT=/scratch/yuzhou/projects/sglang-sync
PROGRAMS="${PROGRAMS:-10}"; PREFIX="${PREFIX:-12000}"; GAP="${GAP:-20}"
ETA="${ETA:-20}"; OUT_TOK="${OUT_TOK:-1000}"; TURNS="${TURNS:-4}"; STAGGER="${STAGGER:-2.5}"
POOL="${POOL:-98304}"; MAXRUN="${MAXRUN:-6}"

restart() {
  # s1_stack_up.sh does its own PID-scoped, port-specific teardown (never a
  # broad pkill — it matches launch_server.*--port $SGLANG_PORT only).
  MAX_TOTAL_TOKENS=$POOL MAX_RUNNING_REQUESTS=$MAXRUN nohup bash \
    wherewewin/s1-toolcall-predictability/s1_stack_up.sh >/tmp/s1_ab_stack.log 2>&1 &
  for i in $(seq 1 70); do
    R=$(curl -s --max-time 20 http://127.0.0.1:30000/generate -H 'content-type: application/json' \
      -d '{"input_ids":[1,2,3],"sampling_params":{"max_new_tokens":2,"temperature":0},"program_id":"smoke"}' 2>/dev/null)
    echo "$R" | grep -q meta_info && return 0
    sleep 10
  done
  echo "[ab] stack failed to come up"; return 1
}

run_arm() {
  local arm="$1"
  PYTHONPATH="$WT/python:$WT/dev/aginfer" timeout 400 $PY \
    wherewewin/s1-toolcall-predictability/s1_driver.py --arm "$arm" \
    --programs $PROGRAMS --turns $TURNS --prefix-tokens $PREFIX --output-tokens $OUT_TOK \
    --gap-s $GAP --tool-eta-s $ETA --stagger-s $STAGGER 2>&1 | grep -E 'resume TTFT|resume cached'
}

for arm in ours b; do
  echo "[ab] === restart + arm=$arm ==="
  restart || exit 1
  echo "[ab] arm=$arm:"; run_arm "$arm"
done
echo "[ab] DONE"
