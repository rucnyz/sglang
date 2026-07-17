#!/usr/bin/env bash
# One-shot κ_i calibration (#118): sweep prefill length with
# `sglang.bench_one_batch` (pure GPU-forward wall, no HTTP/decode in the
# timed window), then fit c_KV(L) = kv_α·L² + kv_β·L + kv_γ and emit the
# `export SGLANG_CSIGMA_*` lines on STDOUT.
#
# Usage:
#   eval "$(bash dev/eval/cost_model/calibrate.sh <model_path> <device_label>)"
# e.g.
#   eval "$(bash dev/eval/cost_model/calibrate.sh Qwen/Qwen3.5-9B H200)"
#
# Env overrides: REPEATS (default 3, median per L denoises small-L),
# MEM_FRACTION (default 0.85), CUDA_VISIBLE_DEVICES (default 0),
# EXTRA_FLAGS (default empty) appended verbatim to bench_one_batch — for
# multi-GPU models set e.g. EXTRA_FLAGS="--tp 4 --max-mamba-cache-size 16
# --trust-remote-code --cuda-graph-backend-decode full" (big mamba states
# OOM the profiler at the default cap; a tiny cap is enough since the
# batch-1 prefill sweep only ever uses one mamba slot).
#
# The sweep runs from the floor plateau (L≤256, pure per-forward
# kernel-launch cost) up into the multi-chunk regime where attention's
# L² emerges. The top lengths may OOM on smaller GPUs — that is fine:
# bench stops there and the fitter aligns by throughput, so a truncated
# sweep still calibrates from whatever completed.
set -euo pipefail

MODEL="${1:?usage: calibrate.sh <model_path> <device_label>}"
DEVICE="${2:?usage: calibrate.sh <model_path> <device_label>}"
REPEATS="${REPEATS:-3}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="$REPO/.venv/bin/python"
LENGTHS=(32 64 128 256 512 768 1024 1536 2048 2560 3072 4096 6144 8192 \
         16384 24576 32768 49152 65536 98304 131072)
LOGDIR="$(mktemp -d)"
trap 'rm -rf "$LOGDIR"' EXIT

echo ">> κ_i sweep: $MODEL on $DEVICE, ${REPEATS} repeat(s), lengths ${LENGTHS[*]}" >&2
for run in $(seq 1 "$REPEATS"); do
    echo ">> bench run $run/$REPEATS ..." >&2
    # bench may exit non-zero on an end-of-sweep OOM; the prefill lines
    # are already flushed, so keep the log and continue.
    "$PY" -m sglang.bench_one_batch \
        --model-path "$MODEL" --batch-size 1 \
        --input-len "${LENGTHS[@]}" --output-len 1 \
        --mem-fraction-static "$MEM_FRACTION" $EXTRA_FLAGS \
        > "$LOGDIR/run${run}.log" 2>&1 || true
done

# Parser prints `export ...` to STDOUT, diagnostics + plot to STDERR.
"$PY" "$HERE/calibrate_kappa.py" "$LOGDIR"/run*.log \
    --model "$MODEL" --device "$DEVICE"
