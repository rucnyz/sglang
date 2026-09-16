#!/bin/bash
# Wait for the S1 stack, run the calibrated driver (arm=ours), dump the verdict
# (daemon counters + skip reasons + resume TTFT). Args override the driver knobs.
cd "$(dirname "$0")/../.." || exit 1   # dev/aginfer
PROGRAMS="${PROGRAMS:-8}"; PREFIX="${PREFIX:-8000}"; GAP="${GAP:-15}"
ETA="${ETA:-15}"; STAGGER="${STAGGER:-2.5}"; OUT_TOK="${OUT_TOK:-1200}"; TURNS="${TURNS:-4}"
DL=logs/s1_daemon.log
echo "[s1-run] waiting for sglang :30000 ..."
for i in $(seq 1 80); do curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && break; sleep 6; done
curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "[s1-run] sglang not up"; exit 1; }
echo "[s1-run] stack up. pool=$(curl -s http://127.0.0.1:30000/get_server_info | python3 -c 'import sys,json;print(json.load(sys.stdin).get("max_total_num_tokens"))' 2>/dev/null)"
OFF=$(wc -l < "$DL" 2>/dev/null || echo 0)
echo "[s1-run] driver: programs=$PROGRAMS prefix=$PREFIX gap=$GAP eta=$ETA stagger=$STAGGER"
bash -c "source scripts/env.sh && timeout 600 python wherewewin/s1-toolcall-predictability/s1_driver.py --arm ours \
  --programs $PROGRAMS --turns $TURNS --prefix-tokens $PREFIX --output-tokens $OUT_TOK \
  --gap-s $GAP --tool-eta-s $ETA --stagger-s $STAGGER \
  --out wherewewin/s1-toolcall-predictability/s1_m1_resume_ours.jsonl" 2>&1 | tail -8
echo "[s1-run] === daemon counters (this run) ==="
TAIL=$(tail -n +"$((OFF+1))" "$DL" 2>/dev/null)
for k in memory_pressure migrate_enqueued migrate_applied promote_scheduled promote_dispatched promote_skipped admission_pause; do
  echo "  $k: $(echo "$TAIL" | grep -ac "$k")"
done
echo "[s1-run] --- skip reasons ---"; echo "$TAIL" | grep -aoE 'reason=[a-z_]+(:[a-z_]+)?' | sed -E 's/=[0-9]+$//' | sort | uniq -c | sort -rn | head
echo "[s1-run] DONE"
