#!/usr/bin/env bash
# S1 controlled magnitude: the DAEMON warms a fully-evicted 50K victim (action-
# timeline promote + prefill-only warm) vs B (recompute on resume). N=3.
# Shows the clean win magnitude (91%) with full eviction + reliable warm timing.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT_PY="${WT_PYTHONPATH:-${AGINFER_ROOT%/dev/aginfer}/python}"
curl -sf --max-time 5 http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "sglang not up"; exit 1; }
PYTHONPATH="$WT_PY" ${PY:-python} -u "$HERE/auto_victim_warm.py"
