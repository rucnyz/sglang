#!/bin/bash
# Run the 3 settle-fixed e2e verifies, each on a FRESH sglang (clean tree).
# t17 has a slow 10k-unit stress stage -> 1500s budget. session needs chunked-64.
WT=/scratch/yuzhou/projects/sglang-sync
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
DA="$WT/dev/aginfer"
cd "$DA"
export PYTHONPATH="$WT/python:$DA" AGINFER_VERIFY_BASE=http://127.0.0.1:30000 \
       AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 PYTHONUNBUFFERED=1

run_one () {
  local t="$1"; local budget="$2"; local chunked="$3"
  echo "[e2e] === $t (fresh sglang, budget=${budget}s, CHUNKED=${chunked:-default}) ==="
  CHUNKED="$chunked" bash verify/_sglang_t17profile.sh >/tmp/sglang_${t}.boot 2>&1
  if ! curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; then
     echo "[e2e] $t: sglang FAILED to boot"; tail -6 /tmp/sglang_${t}.boot; return
  fi
  timeout "$budget" $PY verify/$t/verify.py > /tmp/ext_$t.log 2>&1
  local rc=$?
  echo "[e2e] $t rc=$rc :: $(grep -aE 'PASSED|FAILED' /tmp/ext_$t.log | tail -1 | cut -c1-90)"
}

run_one t20 600 ""
run_one session_id_passthrough 600 64
run_one t17 1500 ""
echo "[e2e] ALL DONE"
