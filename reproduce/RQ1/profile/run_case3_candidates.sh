#!/bin/bash
# Try different workloads for case3 tps win. Quick 1-rep probes.
set -eu
SCRIPT=/scratch/yuzhou/projects/sglang/reproduce/RQ1/run_arm.sh
T6=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6_v2.jsonl
T12=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t12.jsonl
PORT=30097; GPU=7

run_probe() {
  local LABEL=$1 TRACE=$2 STAGGER=$3 CONC=$4
  local OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/profile/case3_${LABEL}
  mkdir -p "$OUTDIR/base_boot1" "$OUTDIR/sys_boot1"
  echo "[$(date +%H:%M:%S)] $LABEL: base"
  PORT=$PORT GPU=$GPU bash "$SCRIPT" base "$TRACE" $STAGGER $CONC - 1 "$OUTDIR/base_boot1"
  sleep 5
  echo "[$(date +%H:%M:%S)] $LABEL: sys"
  PORT=$PORT GPU=$GPU bash "$SCRIPT" sys "$TRACE" $STAGGER $CONC - 1 "$OUTDIR/sys_boot1"
  sleep 5

  python3 -c "
import json
b = json.load(open('$OUTDIR/base_boot1/base_r1.json'))
s = json.load(open('$OUTDIR/sys_boot1/sys_r1.json'))
dt = (s['throughput_tok_s'] - b['throughput_tok_s']) / b['throughput_tok_s'] * 100
dh = s['cache_hit'] - b['cache_hit']
bt = b.get('ttft_ms',{}).get('mean',0); st = s.get('ttft_ms',{}).get('mean',0)
dttft = (st-bt)/bt*100 if bt else 0
print(f'$LABEL: base_tps={b[\"throughput_tok_s\"]:.1f} sys_tps={s[\"throughput_tok_s\"]:.1f} delta={dt:+.1f}% hit_delta={dh:+.4f} ttft_delta={dttft:+.1f}%')
print(f'  n_err: base={b[\"n_error\"]} sys={s[\"n_error\"]}')
" 2>/dev/null
  echo ""
}

# Candidate A: t6 + conc=128 (higher concurrency than case1's 64)
run_probe "t6_conc128" "$T6" 0.5 128

# Candidate B: t6 + stagger=0.3 (faster arrival, more burst)
run_probe "t6_stag03" "$T6" 0.3 64

# Candidate C: t12 + stagger=0.3 (case2's trace but faster arrival)
run_probe "t12_stag03" "$T12" 0.3 64

echo "=== ALL PROBES DONE ==="
