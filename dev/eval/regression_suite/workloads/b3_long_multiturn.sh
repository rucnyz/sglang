#!/bin/bash
# B3 — long-multiturn pool-binding shift (genuine α/β/α/β workload).
# Phase α: 24 concurrent × short → mamba slot bottleneck.
# Phase β:  8 concurrent × 96K-token input → KV token bottleneck.
# 4×90s = 360 s, no drains between phases.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

PORT=$PORT MODEL=$MODEL OUT_DIR=$OUT_DIR METRICS_PATH=$METRICS_PATH \
  .venv/bin/python "$SCRIPT_DIR/dispatcher_b3.py" \
  >"$OUT_DIR/dispatcher.log" 2>&1 || echo "[suite-runner] dispatcher errored"

teardown_server

if [ ! -f "$METRICS_PATH" ]; then
  echo "[suite-runner] FAIL: dispatcher did not write $METRICS_PATH"
  echo "{\"error\": \"dispatcher missing metrics\"}" > "$METRICS_PATH"
fi
echo "[suite-runner] B3 metrics: $(cat "$METRICS_PATH")"
