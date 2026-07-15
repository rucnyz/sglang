#!/bin/bash
# Cross-model RQ1 campaign on the canonical corpus traces (option A: all models
# on the SAME frozen-corpus traces, program-identical across tokenizers).
#
# SERIAL by design: a prior run put 5 heavy jobs (incl. a 120B TP4) on one box
# in parallel and the throughput cross-contaminated (per-rep tps collapsed) AND
# the driver wedged (NVML "Insufficient Permissions", CUDA device_count=0). So
# here Nemotron owns GPUs 4-7 and the small models run ONE-AT-A-TIME on GPU 0;
# never more than the 120B + one small model at once.
#
# Traces (agentreplay/data/traces, built by build_canonical from
# data/corpus/data/contrib): cc_{qwen,nemotron}_t6 (Case1@64 / Case3@128),
# cc_{qwen,nemotron}_t12 (Case2@64). qwen serves 9B AND 35B (shared tokenizer).
set -u
TD=/scratch/yuzhou/projects/agentreplay/data/traces
OUT=${OUT:-/tmp/campaign_out}; mkdir -p "$OUT"
RA=reproduce/RQ1/run_arm.sh

run_pair() {  # model tag trace stagger conc gpus tp port extra_env
  local MODEL=$1 TAG=$2 TR=$3 STAG=$4 CONC=$5 GPUS=$6 TP=$7 PORT=$8; shift 8
  local O="$OUT/$TAG"; mkdir -p "$O"
  echo ">>> $TAG: $MODEL $TR @conc$CONC gpus=$GPUS tp=$TP"
  env "$@" GPUS="$GPUS" TP="$TP" PORT="$PORT" MODEL="$MODEL" \
    bash "$RA" base "$TR" "$STAG" "$CONC" - 3 "$O/base"
  env "$@" GPUS="$GPUS" TP="$TP" PORT="$PORT" MODEL="$MODEL" \
    bash "$RA" sys  "$TR" "$STAG" "$CONC" - 3 "$O/sys"
}

NEMO=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
NEMO_ENV=(REASONING=none MEMFRAC=0.85 MAMBA_CAP=256 MAMBA_STRAT=no_buffer \
          CUDA_GRAPH_DECODE=full CUDA_GRAPH_PREFILL=disabled)

# --- Nemotron (owns GPUs 4-7, TP4), Case1/2/3 serially ---
run_pair "$NEMO" nemo_case1 "$TD/cc_nemotron_t6.jsonl"  0.5 64  4,5,6,7 4 30098 "${NEMO_ENV[@]}"
run_pair "$NEMO" nemo_case2 "$TD/cc_nemotron_t12.jsonl" 0.5 64  4,5,6,7 4 30098 "${NEMO_ENV[@]}"
run_pair "$NEMO" nemo_case3 "$TD/cc_nemotron_t6.jsonl"  0.5 128 4,5,6,7 4 30098 "${NEMO_ENV[@]}"

# --- 9B (GPU 0), Case1/2/3 serially ---
run_pair Qwen/Qwen3.5-9B 9b_case1 "$TD/cc_qwen_t6.jsonl"  0.5 64  0 1 30099
run_pair Qwen/Qwen3.5-9B 9b_case2 "$TD/cc_qwen_t12.jsonl" 0.5 64  0 1 30099
run_pair Qwen/Qwen3.5-9B 9b_case3 "$TD/cc_qwen_t6.jsonl"  0.5 128 0 1 30099

# --- 35B (GPU 0), Case1 only (Case2/3 known to crash: c_M=0 wrong for 35B
# mamba-bound regimes, open issue #276) ---
run_pair Qwen/Qwen3.5-35B-A3B 35b_case1 "$TD/cc_qwen_t6.jsonl" 0.5 64 0 1 30099 MEMFRAC=0.85

echo "=== CAMPAIGN_DONE -> $OUT ==="
