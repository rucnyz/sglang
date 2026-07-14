#!/bin/bash
# Case2+3 probe: 1 rep each, full trace, sequential on GPU 7.
# Case2: t12 full trace, default config (no MAMBA_CAP) — many-session KV pressure
# Case3: t6 full trace, MEMFRAC=0.75 — tight memory amplifies m2k benefit
set -eu
SCRIPT=/scratch/yuzhou/projects/sglang/reproduce/RQ1/run_arm.sh
T12=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t12.jsonl
T6=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6_v2.jsonl
PORT=30097; GPU=7

echo "=== CASE2: t12 full, default config ==="
OUTB=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case2/runs/t12_default_fp/base_boot1
OUTS=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case2/runs/t12_default_fp/sys_boot1
mkdir -p "$OUTB" "$OUTS"
echo "[$(date +%H:%M:%S)] case2 base start"
PORT=$PORT GPU=$GPU bash "$SCRIPT" base "$T12" 0.5 64 - 1 "$OUTB"
echo "[$(date +%H:%M:%S)] case2 base done"
sleep 5
echo "[$(date +%H:%M:%S)] case2 sys start"
PORT=$PORT GPU=$GPU bash "$SCRIPT" sys "$T12" 0.5 64 - 1 "$OUTS"
echo "[$(date +%H:%M:%S)] case2 sys done"
sleep 5

echo "=== CASE3: t6 full, MEMFRAC=0.75 ==="
OUTB3=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case3/runs/t6_mf75/base_boot1
OUTS3=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case3/runs/t6_mf75/sys_boot1
mkdir -p "$OUTB3" "$OUTS3"
echo "[$(date +%H:%M:%S)] case3 base start"
MEMFRAC=0.75 PORT=$PORT GPU=$GPU bash "$SCRIPT" base "$T6" 0.5 64 - 1 "$OUTB3"
echo "[$(date +%H:%M:%S)] case3 base done"
sleep 5
echo "[$(date +%H:%M:%S)] case3 sys start"
MEMFRAC=0.75 PORT=$PORT GPU=$GPU bash "$SCRIPT" sys "$T6" 0.5 64 - 1 "$OUTS3"
echo "[$(date +%H:%M:%S)] case3 sys done"

echo "=== ALL DONE ==="
echo "Case2 results:"
grep -oE '"throughput_tok_s": [0-9.]+|"cache_hit": [0-9.]+' "$OUTB/base_r1.json" 2>/dev/null
grep -oE '"throughput_tok_s": [0-9.]+|"cache_hit": [0-9.]+' "$OUTS/sys_r1.json" 2>/dev/null
echo "Case3 results:"
grep -oE '"throughput_tok_s": [0-9.]+|"cache_hit": [0-9.]+' "$OUTB3/base_r1.json" 2>/dev/null
grep -oE '"throughput_tok_s": [0-9.]+|"cache_hit": [0-9.]+' "$OUTS3/sys_r1.json" 2>/dev/null
