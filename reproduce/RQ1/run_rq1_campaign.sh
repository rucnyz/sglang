#!/bin/bash
# RQ1 full measurement campaign (serial, GPU 7 only). Fills the remaining RQ1
# cells beyond the already-reproduced 9B Case1/2/3 base+sys:
#   P1  9B  vLLM (t6_v2@64)            -> main-table vLLM cell, 9B
#   P2  9B  static-best base RATIO 0.8 -> static-best cell, 9B
#   P3  35B Case1 base+sys N=1 SMOKE   -> de-risk before the long N=3
#   P4  35B Case1/2/3 base+sys N=3     -> 35B default+sys rows
#   P5  35B vLLM (t6_v2@64)            -> main-table vLLM cell, 35B
#   P6  35B static-best base RATIO 0.8 -> static-best cell, 35B
# Robust: no set -e; each phase logged; a failure logs + continues.
set -u
RQ1=/scratch/yuzhou/projects/sglang/reproduce/RQ1
OUT=$RQ1/campaign
mkdir -p "$OUT"
T6=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6_v2.jsonl
log(){ echo "[$(date +%H:%M:%S)] CAMPAIGN: $*"; }

log "=== P1: 9B vLLM t6_v2@64 ==="
GPU=7 PORT=30098 MODEL=Qwen/Qwen3.5-9B MAXLEN=110000 \
  bash "$RQ1/run_vllm_arm.sh" "$T6" 0.5 64 - "$OUT/9b_vllm" 2>&1 || log "P1 9B vLLM FAILED"

log "=== P2: 9B static-best base RATIO=0.8 t6_v2@64 ==="
RATIO=0.8 GPU=7 PORT=30097 MODEL=Qwen/Qwen3.5-9B \
  bash "$RQ1/run_arm.sh" base "$T6" 0.5 64 - 1 "$OUT/9b_static_r0.8" 2>&1 || log "P2 9B static FAILED"

log "=== P3: 35B Case1 base+sys N=1 SMOKE ==="
SMOKE_OK=1
MODEL=Qwen/Qwen3.5-35B-A3B GPU=7 PORT=30097 \
  bash "$RQ1/run_arm.sh" base "$T6" 0.5 64 - 1 "$OUT/35b_smoke_base" 2>&1 || { log "P3 35B base smoke FAILED"; SMOKE_OK=0; }
MODEL=Qwen/Qwen3.5-35B-A3B GPU=7 PORT=30097 \
  bash "$RQ1/run_arm.sh" sys  "$T6" 0.5 64 - 1 "$OUT/35b_smoke_sys"  2>&1 || { log "P3 35B sys smoke FAILED"; SMOKE_OK=0; }

if [ "$SMOKE_OK" = "1" ]; then
  log "=== P4: 35B Case1/2/3 base+sys N=3 (RUNTAG=official_35b) ==="
  MODEL=Qwen/Qwen3.5-35B-A3B RUNTAG=official_35b GPU=7 PORT=30097 \
    bash "$RQ1/run_official_case123.sh" 2>&1 || log "P4 35B official FAILED"
else
  log "=== P4 SKIPPED (35B smoke failed) ==="
fi

log "=== P5: 35B vLLM t6_v2@64 ==="
GPU=7 PORT=30098 MODEL=Qwen/Qwen3.5-35B-A3B MAXLEN=110000 \
  bash "$RQ1/run_vllm_arm.sh" "$T6" 0.5 64 - "$OUT/35b_vllm" 2>&1 || log "P5 35B vLLM FAILED"

log "=== P6: 35B static-best base RATIO=0.8 ==="
RATIO=0.8 MODEL=Qwen/Qwen3.5-35B-A3B GPU=7 PORT=30097 \
  bash "$RQ1/run_arm.sh" base "$T6" 0.5 64 - 1 "$OUT/35b_static_r0.8" 2>&1 || log "P6 35B static FAILED"

log "=== CAMPAIGN COMPLETE ==="
