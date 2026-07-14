#!/bin/bash
# 35B RQ1 campaign (serial, GPU7). run_arm now auto-selects the calibrated 35B
# csigma for MODEL=*35B* (fixes the 9B-csigma oscillation/OOM). Robust: no set -e.
set -u
RQ1=/scratch/yuzhou/projects/sglang/reproduce/RQ1
OUT=$RQ1/campaign
T6=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6_v2.jsonl
log(){ echo "[$(date +%H:%M:%S)] CAMPAIGN35B: $*"; }

log "=== P1: 35B Case1/2/3 base+sys N=3 (calibrated csigma) ==="
MODEL=Qwen/Qwen3.5-35B-A3B RUNTAG=official_35b GPU=7 PORT=30097 \
  bash "$RQ1/run_official_case123.sh" 2>&1 || log "P1 FAILED"

log "=== P2: 35B vLLM t6_v2@64 ==="
GPU=7 PORT=30098 MODEL=Qwen/Qwen3.5-35B-A3B MAXLEN=110000 \
  bash "$RQ1/run_vllm_arm.sh" "$T6" 0.5 64 - "$OUT/35b_vllm" 2>&1 || log "P2 FAILED"

log "=== P3: 35B static-best base RATIO=0.8 ==="
RATIO=0.8 MODEL=Qwen/Qwen3.5-35B-A3B GPU=7 PORT=30097 \
  bash "$RQ1/run_arm.sh" base "$T6" 0.5 64 - 1 "$OUT/35b_static_r0.8" 2>&1 || log "P3 FAILED"

log "=== CAMPAIGN35B COMPLETE ==="
