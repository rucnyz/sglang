#!/bin/bash
# Hardened S1 ours-run. Fixes the harness bugs that blocked clean measurement:
#  1) READY detection via stack_up's OWN "[s1-stack] READY" marker (not a
#     generate-poll that catches the OLD stack before teardown).
#  2) A real 12K prefill deadlock GUARD before the run (distinguishes
#     deadlock from not-loaded — ConnectionError=loading, ReadTimeout=hung).
#  3) python -u (unbuffered) so output survives a timeout-kill.
#  4) offset-bounded daemon counters captured BEFORE any further restart.
# Env: POOL, MAXRUN, PROGRAMS, PREFIX, OUT_TOK, TURNS, GAP, ETA, STAGGER.
set +e
WT=/scratch/yuzhou/projects/sglang-sync
cd "$WT/dev/aginfer" || exit 1
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
POOL="${POOL:-28672}"; MAXRUN="${MAXRUN:-1}"; PROGRAMS="${PROGRAMS:-2}"
PREFIX="${PREFIX:-12000}"; OUT_TOK="${OUT_TOK:-200}"; TURNS="${TURNS:-4}"
GAP="${GAP:-20}"; ETA="${ETA:-20}"; STAGGER="${STAGGER:-8}"
SLOG=/tmp/s1h_stack.log
: > "$SLOG"
echo "[h] restart POOL=$POOL MAXRUN=$MAXRUN"
MAX_TOTAL_TOKENS=$POOL MAX_RUNNING_REQUESTS=$MAXRUN nohup bash \
  wherewewin/s1-toolcall-predictability/s1_stack_up.sh >"$SLOG" 2>&1 &
# (1) wait for stack_up's own READY marker (up to ~9 min)
for i in $(seq 1 90); do grep -qa '\[s1-stack\] READY' "$SLOG" && break; sleep 6; done
grep -qa '\[s1-stack\] READY' "$SLOG" || { echo "[h] stack_up never signalled READY"; tail -4 "$SLOG"; exit 0; }
sleep 3
# (2) deadlock guard: a real PREFIX-sized prefill must return
GUARD=$($PY -u -c "
import requests,time
t=time.time()
try:
  r=requests.post('http://127.0.0.1:30000/generate',json={'input_ids':list(range(9,9+$PREFIX)),'sampling_params':{'max_new_tokens':3,'temperature':0},'program_id':'guard'},timeout=60)
  print('PREFILL_OK',r.json()['meta_info'].get('prompt_tokens'),round(time.time()-t,1))
except requests.exceptions.ReadTimeout: print('DEADLOCK_readtimeout',round(time.time()-t,1))
except Exception as e: print('PREFILL_ERR',type(e).__name__,round(time.time()-t,1))
")
echo "[h] guard: $GUARD"
echo "$GUARD" | grep -q PREFILL_OK || { echo "[h] not runnable at POOL=$POOL MAXRUN=$MAXRUN -> abort cleanly"; exit 0; }
DL=logs/s1_daemon.log; OFF=$(wc -l < "$DL" 2>/dev/null || echo 0)
echo "[h] === ours run (unbuffered) ==="
PYTHONPATH="$WT/python:$WT/dev/aginfer" timeout 300 $PY -u \
  wherewewin/s1-toolcall-predictability/s1_driver.py --arm ours \
  --programs $PROGRAMS --turns $TURNS --prefix-tokens $PREFIX --output-tokens $OUT_TOK \
  --gap-s $GAP --tool-eta-s $ETA --stagger-s $STAGGER 2>&1 | grep -E 'S1 arm|resume TTFT|resume prefills|resume cached'
T=$(tail -n +"$((OFF+1))" "$DL" 2>/dev/null)
echo "[h] tool_call_start=$(echo "$T"|grep -ac 'kind=tool_call_start') migrate_enq=$(echo "$T"|grep -ac migrate_enqueued) migrate_applied=$(echo "$T"|grep -ac migrate_applied) promote_disp=$(echo "$T"|grep -ac promote_dispatched)"
echo "$T" | grep -aoE 'kv_decide.*kind=tool_call_start.*outcome=[a-z_]+' | grep -oE 'outcome=[a-z_]+' | sort | uniq -c
echo "$T" | grep -ahoE 'remove_hbm_not_device_leaf:[a-z_0-9/=]+|add_already_present|promote_load_back_declined:[a-z_]+' | sort | uniq -c | head -5
echo "[h] DONE"
