#!/usr/bin/env bash
# S1 per-resume microbench (no daemon needed): flood a victim prefix out of
# HBM+DRAM, then time its resume TTFT for B (direct, pays recompute/DISK-load on
# the critical path) vs a pre-staged access (the promote, off the critical path).
# Args: VICTIM_TOKENS FLOOD_N FLOOD_LEN TRIALS  (default 50000 8 50000 3)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WT_PY="${WT_PYTHONPATH:-${AGINFER_ROOT%/dev/aginfer}/python}"
curl -sf --max-time 5 http://127.0.0.1:30000/health >/dev/null 2>&1 || { echo "sglang not up"; exit 1; }
PYTHONPATH="$WT_PY" ${PY:-python} -u "$HERE/s1_disk_microbench.py" "$@"
