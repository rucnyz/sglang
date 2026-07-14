#!/bin/bash
set -eu
OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/profile/async_e2e_v2
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR/sys_boot1"
cd /scratch/yuzhou/projects/sglang
PORT=30097 GPU=7 bash reproduce/RQ1/run_arm.sh sys \
  /scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6_v2.jsonl \
  0.5 256 - 1 "$OUTDIR/sys_boot1"

LOG="$OUTDIR/sys_boot1/server_sys.log"
echo ""
echo "=== RESULTS ==="
grep -oE '"throughput_tok_s": [0-9.]+|"cache_hit": [0-9.]+|"n_error": [0-9]+' "$OUTDIR/sys_boot1/sys_r1.json" 2>/dev/null
echo ""
echo "=== ASYNC FIRE CHECKS ==="
echo "fires: $(grep -c 'XPoolFirePlanner.build' "$LOG" 2>/dev/null || echo 0)"
echo "applies: $(grep -c 'apply_pending' "$LOG" 2>/dev/null || echo 0)"
echo "deferred: $(grep -c 'deferred to apply_pending' "$LOG" 2>/dev/null || echo 0)"
echo "cuda_errors: $(grep -ci 'cuda error\|illegal memory\|device-side assert' "$LOG" 2>/dev/null || echo 0)"
echo ""
echo "=== EXECUTE_ASYNC TIMING (first 3 + last 3) ==="
grep "execute_async.*DONE" "$LOG" 2>/dev/null | head -3
echo "..."
grep "execute_async.*DONE" "$LOG" 2>/dev/null | tail -3
echo ""
echo "=== DEFERRED APPLY (first 5) ==="
grep "apply_pending" "$LOG" 2>/dev/null | head -5
