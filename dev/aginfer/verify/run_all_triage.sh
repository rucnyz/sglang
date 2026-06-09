#!/bin/bash
# Triage runner for the verify/ suite on the rebased aginfer-synced branch.
# Runs every verify/*/verify.py with the rebased sglang on PYTHONPATH, captures
# exit code + last error line. NOT a pass/fail gate -- a landscape sweep so we
# can fix rebase-induced breakage one-by-one.
WT=/scratch/yuzhou/projects/sglang-sync
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
DA="$WT/dev/aginfer"
export AGINFER_ROOT="$DA"
export AGINFER_LOGS="$DA/logs"
export AGINFER_RESULTS="$DA/results"
export AGINFER_MOONCAKE_SSD=/scratch/yuzhou/mooncake_ssd
export PYTHONPATH="$WT/python:$DA"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export CUDA_HOME=/usr/local/cuda-13.2
cd "$DA"
OUT="$DA/verify/_triage_$(ls verify/_triage_*.txt 2>/dev/null | wc -l).txt"
: > "$OUT"
TIMEOUT="${TIMEOUT:-150}"
for d in $(ls -d verify/*/ | sort); do
  v="$d/verify.py"
  [ -f "$v" ] || continue
  name=$(basename "$d")
  start=$SECONDS
  log=$(timeout "$TIMEOUT" "$PY" "$v" 2>&1)
  rc=$?
  dur=$((SECONDS-start))
  # extract the most informative tail line (error type or PASS marker)
  tail_line=$(echo "$log" | grep -aE 'PASSED|FAILED|Error|Traceback|Exception|assert|Refused|Connection|404|Timeout|ImportError|AttributeError|ModuleNotFound' | tail -1 | cut -c1-110)
  [ -z "$tail_line" ] && tail_line=$(echo "$log" | tail -1 | cut -c1-110)
  printf '%-26s rc=%-3s %3ss | %s\n' "$name" "$rc" "$dur" "$tail_line" | tee -a "$OUT"
done
echo "=== triage written to $OUT ==="
