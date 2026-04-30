#!/bin/bash
# B1 — phase-shift cyclic (mamba-heavy ↔ KV-heavy, continuous traffic, no drains).
# Uses a custom Python dispatcher that issues requests back-to-back from a
# rotating prompt source so that phase transitions happen mid-flight.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
export SGLANG_BUDGETER_LOG="$OUT_DIR/budgeter.jsonl"
boot_server

# Phase plan: 90s mamba-heavy, 90s KV-heavy, 90s mamba-heavy, 90s KV-heavy.
# Continuous concurrent requests; phase boundary just changes the prompt
# source — incoming RPS stays constant.
PORT=$PORT MODEL=$MODEL OUT_DIR=$OUT_DIR \
  .venv/bin/python "$SCRIPT_DIR/dispatcher_b1.py" \
  >"$OUT_DIR/dispatcher.log" 2>&1 || echo "[suite-runner] dispatcher errored"

teardown_server

# The dispatcher writes its own metrics.json directly.
if [ ! -f "$METRICS_PATH" ]; then
  echo "[suite-runner] FAIL: dispatcher did not write $METRICS_PATH"
  echo "{\"error\": \"dispatcher missing metrics\"}" > "$METRICS_PATH"
fi
echo "[suite-runner] B1 metrics: $(cat "$METRICS_PATH")"
