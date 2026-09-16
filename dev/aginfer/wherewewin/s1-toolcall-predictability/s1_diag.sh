#!/bin/bash
# Diagnostic S1 run WITH live visibility: wait for stack, start driver in the
# background, and every 8s print HBM occupancy (from /aginfer/state) + the
# running daemon counters, so we can SEE whether occ crosses theta_hi and the
# demote fires (instead of running blind).
cd "$(dirname "$0")/../.." || exit 1   # dev/aginfer
PROGRAMS="${PROGRAMS:-10}"; PREFIX="${PREFIX:-12000}"; GAP="${GAP:-20}"
ETA="${ETA:-20}"; STAGGER="${STAGGER:-2.5}"; OUT_TOK="${OUT_TOK:-1000}"; TURNS="${TURNS:-4}"
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
DL=logs/s1_daemon.log
echo "[diag] waiting for sglang :30000 ..."
for i in $(seq 1 90); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break; sleep 6; done
curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "[diag] sglang not up"; exit 1; }
OFF=$(wc -l < "$DL" 2>/dev/null || echo 0)
echo "[diag] driver bg: programs=$PROGRAMS prefix=$PREFIX gap=$GAP eta=$ETA"
bash -c "source scripts/env.sh && python wherewewin/s1-toolcall-predictability/s1_driver.py --arm ours \
  --programs $PROGRAMS --turns $TURNS --prefix-tokens $PREFIX --output-tokens $OUT_TOK \
  --gap-s $GAP --tool-eta-s $ETA --stagger-s $STAGGER \
  --out wherewewin/s1-toolcall-predictability/s1_m1_resume_ours.jsonl" >/tmp/s1_diag_driver.log 2>&1 &
DRV=$!
occ() {
  curl -s --max-time 5 http://127.0.0.1:30000/aginfer/state 2>/dev/null | $PY -c "
import sys,json
try: d=json.load(sys.stdin)
except: print('?'); sys.exit()
r=d.get('per_rank',[d]); mx=0.0
for rk in r:
    for sp,v in rk.get('pool_usage',{}).get('HBM',{}).get('subpools',{}).items():
        c=v.get('cap_bytes') or 0; u=v.get('used_bytes') or 0
        if c: mx=max(mx, u/c)
print(f'{mx:.2f}')" 2>/dev/null
}
while kill -0 $DRV 2>/dev/null; do
  T=$(tail -n +"$((OFF+1))" "$DL" 2>/dev/null)
  printf '[diag] occ_HBM=%s  mem_pressure=%s migrate_enq=%s migrate_applied=%s promote_disp=%s admission_pause=%s\n' \
    "$(occ)" \
    "$(echo "$T" | grep -ac memory_pressure)" "$(echo "$T" | grep -ac migrate_enqueued)" \
    "$(echo "$T" | grep -ac migrate_applied)" "$(echo "$T" | grep -ac promote_dispatched)" \
    "$(echo "$T" | grep -ac admission_pause)"
  sleep 8
done
echo "[diag] === driver done ==="; tail -7 /tmp/s1_diag_driver.log | cut -c1-110
T=$(tail -n +"$((OFF+1))" "$DL" 2>/dev/null)
echo "[diag] final skip reasons:"; echo "$T" | grep -aoE 'reason=[a-z_]+(:[a-z_]+)?' | sed -E 's/=[0-9]+$//' | sort | uniq -c | sort -rn | head
echo "[diag] DONE"
