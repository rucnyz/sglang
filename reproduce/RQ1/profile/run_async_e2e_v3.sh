#!/bin/bash
set -eu
OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/profile/async_e2e_v3
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
echo "=== ADMISSION-CAP ==="
grep "admission-cap" "$LOG" 2>/dev/null | head -3
echo "..."
grep "admission-cap" "$LOG" 2>/dev/null | tail -3
echo "total cap messages: $(grep -c 'admission-cap' "$LOG" 2>/dev/null)"
echo ""
echo "=== FIRE + ASYNC ==="
echo "fires: $(grep -c 'XPoolFirePlanner.build' "$LOG" 2>/dev/null || echo 0)"
echo "applies: $(grep -c 'apply_pending' "$LOG" 2>/dev/null || echo 0)"
echo "cuda_errors: $(grep -ci 'cuda error\|illegal memory\|device-side assert' "$LOG" 2>/dev/null || echo 0)"
echo ""
echo "=== MAX_RUNNING check ==="
grep "max_running_requests" "$LOG" 2>/dev/null | tail -3
echo ""
echo "=== RUNNING-REQ PEAK ==="
grep -oE "#running-req: [0-9]+" "$LOG" 2>/dev/null | sed 's/#running-req: //' | sort -n | tail -3
echo ""
echo "=== COMPARISON ==="
echo "Before fix (sync, conc256): tps=471.3, cache_hit=0.6139"
echo "After async (no cap fix):   tps=483.8, cache_hit=0.6292"
echo "LPB-only (no HIMA):         tps=558.8, cache_hit=0.8648"
echo "This run (async + cap fix): see above"
