#!/bin/bash
# Self-sustaining 10-min monitor for the #231 replay run.
# Runs continuously (independent of any agent turn): every 600s it appends a
# status line, kills stale broken-task verifier hangs (#222), and once an
# arrival/session sweep has all 6 metrics files it runs compare.py and
# appends the verdict.  Exits when the replay run process is gone.
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
OUT=/tmp/replay_monitor.log
RESDIR=scenarios/replay/results
: > "$OUT"
say() { echo "[$(date '+%F %T')] $*" >> "$OUT"; }
say "monitor START (pid $$)"

declare -A COMPARED
while true; do
  alive=0; pgrep -f 'replay_a3real.sh|run_replay.sh' >/dev/null && alive=1
  nmetrics=$(find "$RESDIR" -name 'metrics_*.json' 2>/dev/null | wc -l)
  util=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
         | awk -F', ' '$1>=4 && $1<=7 {s+=$2} END{print s+0}')
  tailmsg=$(tail -1 logs/replay_run_a3real.log 2>/dev/null | cut -c1-80)
  say "alive=$alive metrics=$nmetrics gpu4-7util=$util | $tailmsg"

  # autonomous safety: kill stale verifier hangs
  for pid in $(pgrep -f 'test\.sh|verifier' 2>/dev/null); do
    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' '); [ -z "$age" ] && continue
    if [ "$age" -gt 1200 ] && [ "$age" -lt 86400 ]; then
      say "WATCHDOG kill stale verifier $pid age=${age}s"; kill -9 "$pid" 2>/dev/null
    fi
  done

  # auto-compare any sweep dir that now has both arms x3 trials and hasn't been compared
  for d in "$RESDIR"/a3real_*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -name 'metrics_*.json' 2>/dev/null | wc -l)
    if [ "$n" -ge 6 ] && [ -z "${COMPARED[$d]}" ]; then
      say "=== sweep $(basename "$d") has $n metrics — running compare.py ==="
      python scenarios/replay/compare.py "$d" >> "$OUT" 2>&1
      COMPARED[$d]=1
    fi
  done

  if [ "$alive" = 0 ]; then say "replay run DONE — monitor exiting"; break; fi
  sleep 600
done
