#!/bin/bash
# S1 milestone-1: against an already-up a3 stack (small pool), run the S1 driver
# (staggered) and check whether the predictive promote SCHEDULES and FIRES.
# The fresh run_k stack already runs the updated daemon code, so no daemon
# restart here.
set -uo pipefail
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
source scripts/env.sh 2>/dev/null || true
HERE="$PWD/wherewewin/s1-toolcall-predictability"
DLOG="$AGINFER_RESULTS/run_K_a3_s1stack/daemon.log"
ARM="${ARM:-ours}"

echo "[m1] waiting for sglang :30000 + daemon :9100 ..."
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && \
  curl -sf http://127.0.0.1:9100/health >/dev/null 2>&1 && break
  sleep 5
done
curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "[m1] sglang not up"; exit 1; }
echo "[m1] stack UP. daemon log: $DLOG"

# mark a log offset so we only count THIS run's events
OFFSET=$(wc -l < "$DLOG" 2>/dev/null || echo 0)

echo "[m1] running S1 driver (arm=$ARM, staggered, small) ..."
python "$HERE/s1_driver.py" --arm "$ARM" \
  --programs 4 --turns 4 --prefix-tokens 8000 --output-tokens 1200 \
  --gap-s 7 --tool-eta-s 7 --stagger-s 2.0 \
  --out "$HERE/s1_m1_resume_${ARM}.jsonl" 2>&1 | tail -16

echo "[m1] === daemon promote/migrate evidence (this run) ==="
tail -n +"$((OFFSET+1))" "$DLOG" 2>/dev/null | grep -aE \
  'promote_scheduled|promote_dispatched|promote_skipped|migrate_enqueued|event=memory_pressure|admission_pause' | tail -30
echo "[m1] counts (this run):"
TAIL=$(tail -n +"$((OFFSET+1))" "$DLOG" 2>/dev/null)
for k in promote_scheduled promote_dispatched promote_skipped migrate_enqueued; do
  echo "  $k: $(echo "$TAIL" | grep -ac "$k")"
done
echo "  memory_pressure events: $(echo "$TAIL" | grep -ac 'event=memory_pressure')"
echo "[m1] sglang alive? $(pgrep -f sglang.launch_server >/dev/null && echo YES || echo NO-CRASHED)"
echo "[m1] done"
