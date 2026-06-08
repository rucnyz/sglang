#!/bin/bash
# e2e-sweep watchdog: kills stale broken-task verifier hangs (#222),
# logs GPU util + sweep progress every 5 min. Reads the live sweep
# pgid/log from /tmp pointers so it survives sweep relaunches.
OUT=/tmp/e2e_watchdog.out
: > "$OUT"
while true; do
  LOG=$(cat /tmp/e2e_log.txt 2>/dev/null)
  LOGABS="/scratch/yuzhou/projects/sglang/dev/aginfer/$LOG"
  echo "===== WD $(date '+%F %T') =====" >> "$OUT"
  pgrep -f 'run_sweep\.sh' >/dev/null 2>&1 || echo "sweep run_sweep.sh GONE/DONE" >> "$OUT"
  for pid in $(pgrep -f 'test\.sh|verifier|harbor' 2>/dev/null); do
    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$age" ] && continue
    if [ "$age" -gt 1500 ] && [ "$age" -lt 86400 ]; then
      echo "WATCHDOG kill stale $pid age=${age}s" >> "$OUT"
      kill -9 "$pid" 2>/dev/null
    fi
  done
  util=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '$1>=4 && $1<=7 {s+=$2} END{print s+0}')
  echo "gpu4-7 util_sum=${util}" >> "$OUT"
  tail -3 "$LOGABS" 2>/dev/null | sed 's/^/  /' >> "$OUT"
  sleep 300
done
