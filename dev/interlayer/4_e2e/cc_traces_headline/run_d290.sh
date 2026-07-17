#!/usr/bin/env bash
# #290 — scale the cross-fire DRAIN win, with the #276 fix + REAL calibration.
#
# This is the load-bearing run for #276: it SOURCES the real H200 κ profile
# (κ_M=0.0 — the post-calibration state), so PRE-#276-fix the drain gate would
# have failed closed (κ_M==0) and the inter cell would equal off. POST-fix the
# gate keys on the non-degenerate κ_KV, so the cross-fire drain fires and the
# win appears. LPB is required by the gate. Scaled knobs push toward the
# static envelope (#290): bigger fires, shorter cooldown, lower NB margin.
set -uo pipefail
cd "$(dirname "$0")"

source /scratch/yuzhou/projects/sglang/dev/eval/cost_model/profiles/Qwen_Qwen3.5-9B_H200.sh

# #290 scaling knobs (vs defaults PAGES_PER_FIRE=4, COOLDOWN=16, NB_MARGIN=1.5)
export SGLANG_BUDGETER_PAGES_PER_FIRE=${SGLANG_BUDGETER_PAGES_PER_FIRE:-8}
# cooldown/amortize in WALL SECONDS (#302 τ-invariance): old COOLDOWN=6
# ticks at TICK_S=2 = 12 s. cooldown_min_s == amortize_horizon_s here.
export SGLANG_XPOOL_COOLDOWN_S=${SGLANG_XPOOL_COOLDOWN_S:-12}
export SGLANG_XPOOL_NB_MARGIN=${SGLANG_XPOOL_NB_MARGIN:-1.2}
export EVICTION_POLICY=lpb

GPU=0 PORT=30090 \
  OUT_DIR=${OUT_DIR:-/tmp/d290_run} \
  NUM_CONCURRENCY=${NUM_CONCURRENCY:-22} \
  MAX_TIME_MIN=${MAX_TIME_MIN:-10} \
  MAX_MAMBA_CACHE=${MAX_MAMBA_CACHE:-256} \
  bash /scratch/yuzhou/projects/sglang/dev/interlayer/4_e2e/cc_traces_headline/run_cc.sh
