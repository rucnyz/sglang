#!/usr/bin/env bash
# #231 replay — replay a captured trace against ours (a3) and baseline
# (a3_kvoff), N trials each, on a freshly-launched stack per trial.
#
# Both arms replay the SAME trace with output length FORCED, so they do
# byte-identical work; the only difference is the daemon's kv-scheduling.
# Per trial we collect TTFT / TPOT / e2e / throughput into a metrics JSON.
#
# Uses run_k.sh's RUN_K_WORKLOAD_CMD hook: run_k launches the full stack
# (mooncake + daemon + sglang) + checks invariants, then runs the replay
# driver against :9100 instead of harbor, then tears the stack down.
#
# Usage:  bash run_replay.sh <trace.jsonl> [N=3] [mode=arrival] [slowdown=1.0]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1   # dev/aginfer

TRACE="${1:?usage: run_replay.sh <trace.jsonl> [N] [mode] [slowdown]}"
N="${2:-3}"
MODE="${3:-arrival}"
SLOWDOWN="${4:-1.0}"
[[ -f "$TRACE" ]] || { echo "trace not found: $TRACE" >&2; exit 1; }

DRIVER="$PWD/scenarios/replay/replay_driver.py"
BASENAME="$(basename "$TRACE" .jsonl)"
OUT_DIR="$PWD/scenarios/replay/results/${BASENAME}_${MODE}"
mkdir -p "$OUT_DIR"
echo "[replay] trace=$TRACE N=$N mode=$MODE slowdown=$SLOWDOWN -> $OUT_DIR"

# ARMS: ours first, baseline second.  Interleaving by arm (not trial) keeps
# each arm's trials close in time; the per-trial fresh stack removes
# cross-trial cache carryover.
for arm in a3 a3_kvoff; do
    for i in $(seq 1 "$N"); do
        label="${arm}_c${i}"
        out="$OUT_DIR/metrics_${label}.json"
        echo "[replay] === arm=$arm trial=$i -> $out ==="
        RUN_K_WORKLOAD_CMD="python $DRIVER --trace $TRACE --base-url http://127.0.0.1:9100/v1 --mode $MODE --slowdown $SLOWDOWN --label $label --out $out" \
        RUN_K_RESULTS_TAG="replay_${BASENAME}_${label}" \
            bash scenarios/_shared/run_k.sh "$arm"
        sleep 15
    done
done

echo "[replay] ALL DONE -> $OUT_DIR"
python scenarios/replay/compare.py "$OUT_DIR" || true
