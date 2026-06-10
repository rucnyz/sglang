#!/usr/bin/env bash
# S1 HEADLINE: clean multi-program live A/B — Ours (daemon predictive-warm) vs B
# (default HiCache, no daemon events).  N concurrent programs, each does ONE
# establish->park->resume round (a burst of memory pressure, THEN idle gaps — the
# realistic agentic structure that leaves the GPU idle the warm needs).  Reports
# per-arm resume-TTFT mean+-std over N cycles.
#
# Requires the stack up (bash stack_up.sh -> "[s1-stack] READY").
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
WT_PY="${WT_PYTHONPATH:-${AGINFER_ROOT%/dev/aginfer}/python}"   # sglang python pkg
CYCLES="${CYCLES:-3}"
LOG=/tmp/rq1_s1_live_ab.log; : > "$LOG"

curl -sf --max-time 5 http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "sglang not up — run stack_up.sh first"; exit 1; }

for c in $(seq 1 "$CYCLES"); do
  for arm in b ours; do
    R=$(PYTHONPATH="$WT_PY" timeout 130 $PY -u "$HERE/live_clean.py" "$arm" 2>&1 \
        | grep -aoE '"arm": "[a-z]+"|"n": [0-9]+|"ttft_mean": [0-9.]+|"cached_mean": [0-9.]+')
    echo "cyc=$c arm=$arm $R" | tr '\n' ' '; echo "" | tee -a "$LOG"
    echo "cyc=$c arm=$arm $R" | tr '\n' ' ' >> "$LOG"; echo "" >> "$LOG"
    sleep 4
  done
done

echo "=== AGGREGATE (mean +- std over $CYCLES cycles) ==="
$PY - "$LOG" <<'PYEOF'
import re, sys, statistics
b_t,o_t,b_c,o_c=[],[],[],[]
for line in open(sys.argv[1]):
    if "cyc=" not in line: continue
    arm="ours" if "arm=ours" in line else "b"
    tm=re.search(r'"ttft_mean": ([0-9.]+)',line); cm=re.search(r'"cached_mean": ([0-9.]+)',line)
    if not tm: continue
    (o_t if arm=="ours" else b_t).append(float(tm.group(1)))
    if cm:(o_c if arm=="ours" else b_c).append(float(cm.group(1)))
def s(x): return f"{statistics.mean(x):.0f}+-{statistics.pstdev(x):.0f}" if x else "n/a"
print(f"B    resume TTFT: {s(b_t)} ms   cached: {s(b_c)}")
print(f"Ours resume TTFT: {s(o_t)} ms   cached: {s(o_c)}")
if b_t and o_t:
    w=100*(statistics.mean(b_t)-statistics.mean(o_t))/statistics.mean(b_t)
    print(f"=> live win = {w:.0f}% faster TTFT (N={len(o_t)} cycles)")
PYEOF
