#!/usr/bin/env bash
# #231 capture — record ONE real request trace through the daemon proxy.
#
# Runs a pressured A3 harbor workload a single time with the trace recorder
# armed (AGINFER_TRACE_CAPTURE).  The daemon proxy writes one JSONL line per
# request (arrival offset, program_id, messages, generated output_len) to
# scenarios/replay/traces/<tag>.jsonl.  That trace is then replayed
# byte-identically against ours/baseline by run_replay.sh.
#
# Capture arm is a3_kvoff: the proxy still records + emits events, but the
# daemon does NO kv-scheduling, so the captured request stream is not
# perturbed by the thing we're about to measure.  (Replay forces output
# length regardless, so the capture arm does not affect fairness.)
#
# Usage:  bash capture_trace.sh [tag]
#   CAP_N_TASKS / CAP_N_CONCURRENT / CAP_MAX_TURNS override the workload size.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1   # dev/aginfer

TAG="${1:-a3pressure}"
TRACE_DIR="$PWD/scenarios/replay/traces"
mkdir -p "$TRACE_DIR"
TRACE="$TRACE_DIR/${TAG}.jsonl"
: > "$TRACE"

echo "[capture] tag=$TAG -> $TRACE"
echo "[capture] workload: tasks=${CAP_N_TASKS:-32} concurrent=${CAP_N_CONCURRENT:-32} max_turns=${CAP_MAX_TURNS:-50}"

AGINFER_TRACE_CAPTURE="$TRACE" \
SMOKE_N_TASKS="${CAP_N_TASKS:-32}" \
SMOKE_N_CONCURRENT="${CAP_N_CONCURRENT:-32}" \
SMOKE_MAX_TURNS="${CAP_MAX_TURNS:-50}" \
RUN_K_RESULTS_TAG="replaycap_${TAG}" \
  bash scenarios/_shared/run_k.sh a3_kvoff

LINES=$(wc -l < "$TRACE" 2>/dev/null || echo 0)
echo "[capture] DONE — $LINES requests captured in $TRACE"
if [[ "$LINES" -eq 0 ]]; then
    echo "[capture] WARNING: empty trace (did the daemon start? check AGINFER_TRACE_CAPTURE plumbing)" >&2
    exit 1
fi
