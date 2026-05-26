#!/usr/bin/env bash
# T9 — N=3 matrix of baseline ↔ ours, alternating order to neutralise
# time-of-day / GPU-warmth drift.  See verify/t9/methodology.md.
#
# Config matrix:
#   baseline = run_k.sh kv_off   (kv_scheduler OFF + admission OFF;
#                                  inline scorer + daemon proxy only)
#   ours     = run_k.sh full     (kv_scheduler ON  + admission ON;
#                                  uses today's daemon code = T11a
#                                  program-alive rule + admission)
#
# 6 cycles total: B-O-B-O-B-O.  Each ~50 min wall.  Total ~5 h.
#
# Resumable: each cycle gets its own RESULTS_DIR tagged with
# matrix_cycle{i}_{config}; existing dirs are skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGINFER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$AGINFER_DIR/scripts/env.sh"

MATRIX_TAG="${MATRIX_TAG:-$(date +%Y%m%d_%H%M%S)}"
MATRIX_ROOT="$AGINFER_RESULTS/run_K_matrix_${MATRIX_TAG}"
mkdir -p "$MATRIX_ROOT"
echo "[matrix] tag=$MATRIX_TAG root=$MATRIX_ROOT"
echo "[matrix] sglang HEAD: $(cd "$AGINFER_DIR/../.." && git rev-parse HEAD)"

# Cycle order: alternate baseline / ours.
CYCLES=(baseline ours baseline ours baseline ours)

# Map cycle name → run_k.sh variant.
declare -A VARIANT_OF=(
    [baseline]=kv_off
    [ours]=full
)

for i in "${!CYCLES[@]}"; do
    cycle_idx=$((i + 1))
    cfg="${CYCLES[$i]}"
    variant="${VARIANT_OF[$cfg]}"
    tag="matrix_${MATRIX_TAG}_cycle${cycle_idx}_${cfg}"
    cycle_dir="$AGINFER_RESULTS/run_K_${variant}_${tag}"

    # Resumable: skip if harbor results already present.
    if [[ -d "$cycle_dir/harbor_jobs" ]] && \
       find "$cycle_dir/harbor_jobs" -mindepth 2 -maxdepth 2 -name 'result.json' 2>/dev/null | grep -q .; then
        echo "[matrix] cycle ${cycle_idx} (${cfg}) — already complete at $cycle_dir, skipping"
        # Symlink into matrix root for convenience.
        ln -sfn "$cycle_dir" "$MATRIX_ROOT/cycle${cycle_idx}_${cfg}"
        continue
    fi

    echo "[matrix] ==== cycle ${cycle_idx}/6 (${cfg} = variant ${variant}) ===="
    echo "[matrix] $(date '+%Y-%m-%d %H:%M:%S') starting"

    RUN_K_RESULTS_TAG="$tag" \
    bash "$SCRIPT_DIR/run_k.sh" "$variant" || rc=$?
    rc=${rc:-0}

    if [[ "$rc" -ne 0 ]]; then
        echo "[matrix] cycle ${cycle_idx} (${cfg}) exited with rc=$rc; check $cycle_dir" >&2
        # Don't bail — let the next cycle try (some failures are
        # cleanup-phase only, harbor results may still be valid).
    fi

    # Symlink for convenience.
    ln -sfn "$cycle_dir" "$MATRIX_ROOT/cycle${cycle_idx}_${cfg}"
    echo "[matrix] $(date '+%Y-%m-%d %H:%M:%S') cycle ${cycle_idx} done"
    echo ""

    # Cool-off between cycles: give the kernel + nvidia-smi time to
    # actually reap the zombies the cleanup tried to drain.
    sleep 5
done

echo "[matrix] all cycles attempted.  Results under $MATRIX_ROOT/cycleN_<cfg>/"
echo "[matrix] aggregate with: python $SCRIPT_DIR/parse_matrix.py $MATRIX_ROOT"
