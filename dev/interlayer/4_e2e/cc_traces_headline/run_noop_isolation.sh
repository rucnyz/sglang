#!/usr/bin/env bash
# #278 isolation: what costs the −14% out_tps on the starve regime?
# Two parallel starve runs with the SAME (bounded-R) code:
#   real : real fires (cuMemUnmap/Map + pool resize + cap-barrier + predict)
#   noop : SGLANG_XPOOL_WORKER_NOOP=1 — fires DECIDE + cap-barrier + the
#          per-tick predict_evict_cost walks, but the worker rolls back the
#          cap-barrier and SKIPS cuMemUnmap/Map + pool resize.
# Compare each cell's inter-vs-off delta:
#   noop inter≈off  AND real inter<off  → cost is cuMem remap / CUDA-graph
#                                          re-capture (removable: fewer/bigger fires)
#   noop inter<off (≈ real)             → cost is scheduler-thread overhead
#                                          (cap-barrier / predict walks)
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

rm -rf /tmp/d10_noop_real /tmp/d10_noop_noop
MAX_MAMBA_CACHE=64 OUT_DIR=/tmp/d10_noop_real GPU=4 PORT=30094 \
    bash dev/interlayer/4_e2e/cc_traces_headline/run_cc.sh \
    > /tmp/d10_noop_real.log 2>&1 &
pA=$!
SGLANG_XPOOL_WORKER_NOOP=1 MAX_MAMBA_CACHE=64 OUT_DIR=/tmp/d10_noop_noop GPU=5 PORT=30095 \
    bash dev/interlayer/4_e2e/cc_traces_headline/run_cc.sh \
    > /tmp/d10_noop_noop.log 2>&1 &
pB=$!
wait $pA || true
wait $pB || true

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
echo "=== REAL-fire run (cuMem remap + resize) ==="
$VENV dev/interlayer/4_e2e/cc_traces_headline/validate_cc.py --out-dir /tmp/d10_noop_real
echo
echo "=== NOOP-fire run (no cuMem, no resize; decide+cap-barrier+predict only) ==="
$VENV dev/interlayer/4_e2e/cc_traces_headline/validate_cc.py --out-dir /tmp/d10_noop_noop
echo "[noop-iso] done"
