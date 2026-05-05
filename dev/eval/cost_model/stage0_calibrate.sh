#!/bin/bash
# Stage-0 calibration wrapper — runs the cost-curve probe and emits env vars
# for the budgeter on stdout. Source it before launching sglang.launch_server:
#
#   eval "$(CUDA_VISIBLE_DEVICES=2 bash dev/eval/cost_model/stage0_calibrate.sh)"
#   .venv/bin/python -m sglang.launch_server --model-path ...
#
# Or to refresh the cached calibration:
#   eval "$(CUDA_VISIBLE_DEVICES=2 bash dev/eval/cost_model/stage0_calibrate.sh --force)"
#
# All progress / human-readable output goes to stderr; stdout is clean
# `export FOO=bar` lines suitable for `eval`.

set -eu
cd "$(dirname "$0")/../../.."  # → sglang repo root
exec .venv/bin/python -u dev/eval/cost_model/stage0_calibrate.py --print-env "$@"
