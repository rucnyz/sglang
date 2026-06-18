#!/bin/bash
# Command 2 of 2: replay one trace file against a running baseline server via the
# agentreplay token-exact harness.
# Usage: replay.sh <port> <datafile> <conc> <gapscale> <outdir> <label>
#   gapscale 0 = back-to-back (ignore think gaps), keeps the bound pool saturated.
set -u
PORT=$1; DATA=$2; CONC=$3; GAPSCALE=$4; OUT=$5; LABEL=$6
AR=/scratch/yuzhou/projects/agentreplay
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
mkdir -p "$OUT"

TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay \
  --trace "$DATA" --mode session --stagger 0.02 --gap-scale "$GAPSCALE" \
  --max-concurrency "$CONC" --flush \
  --url "http://127.0.0.1:$PORT/generate" --label "$LABEL" \
  --out "$OUT/${LABEL}_replay.json"
echo "[replay] $LABEL done -> $OUT/${LABEL}_replay.json"
