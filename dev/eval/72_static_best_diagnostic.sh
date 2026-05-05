#!/bin/bash
# Diagnostic: does varying mamba_full_memory_ratio at boot move TPS on
# fan-out / longhorizon? If yes → there IS a static optimum L2 should be
# able to reach, so L1+L2's failure to win is an actuator-handoff bug.
# If no → the workload at this configuration isn't actually slot-pool-
# bound and L2 has nothing to win.
#
# 8 cells: 4 ratios × 2 workloads, all baseline (L1=0, L2=0) so we
# isolate the static effect from any L1/L2 contribution. 1 trial each
# for speed (this is diagnostic, not paper-grade).
#
# Reuses the same 39/40 drivers + their extended NUM_PROMPTS.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30900}"
RUN_NAME="${RUN_NAME:-static-best-diag-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[static-best-diag] root=$ROOT"

GPUS=(0 1 3 4 5 6 7)
N_GPUS=${#GPUS[@]}
RATIOS=("0.3" "0.5" "0.7" "0.9")
WORKLOADS=("fanout" "longhorizon")

JOBS=()
for wl in "${WORKLOADS[@]}"; do
  for r in "${RATIOS[@]}"; do
    JOBS+=("$wl $r")
  done
done
N_JOBS=${#JOBS[@]}
echo "[static-best-diag] $N_JOBS jobs across $N_GPUS GPUs"

driver_for_workload() {
  case "$1" in
    fanout)      echo "dev/eval/39_fanout_demo.sh" ;;
    longhorizon) echo "dev/eval/40_longhorizon_demo.sh" ;;
  esac
}

workload_num_prompts() {
  case "$1" in
    fanout)      echo "6000" ;;
    longhorizon) echo "360" ;;
  esac
}

launch_wave() {
  local start=$1 end=$2
  local pids=()
  local i=$start
  while [ $i -lt $end ] && [ $i -lt $N_JOBS ]; do
    set -- ${JOBS[$i]}
    local wl=$1 ratio=$2
    local cell="L10_L20"  # baseline only, no L1 no L2
    local out_dir="$ROOT/${wl}_ratio${ratio}_${cell}"
    local gpu=${GPUS[$((i % N_GPUS))]}
    local port=$((PORT_BASE + i))
    local driver=$(driver_for_workload "$wl")
    local nprompts=$(workload_num_prompts "$wl")
    mkdir -p "$out_dir"
    echo "  [job $i] wl=$wl ratio=$ratio gpu=$gpu port=$port"
    ONLY_L1=0 ONLY_L2=0 \
      CUDA_VISIBLE_DEVICES=$gpu PORT=$port OUT_DIR="$out_dir" \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      MAMBA_RATIO=$ratio \
      NUM_PROMPTS=$nprompts \
      bash "$driver" \
      > "$out_dir/runner.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
  done
  echo "  waiting on ${#pids[@]} pids"
  for p in "${pids[@]}"; do wait $p || echo "  pid $p exited non-zero"; done
}

i=0; wave=1
while [ $i -lt $N_JOBS ]; do
  echo "==== WAVE $wave ===="
  launch_wave $i $((i + N_GPUS))
  i=$((i + N_GPUS))
  wave=$((wave + 1))
done

echo
echo "==== STATIC RATIO SWEEP SUMMARY ($ROOT) ===="
python3 - <<PY
import json, os
root = "$ROOT"
ratios = ["0.3", "0.5", "0.7", "0.9"]
workloads = ["fanout", "longhorizon"]
print(f"{'workload':<14}{'ratio':>8}{'TPS':>10}{'mean_TTFT':>14}{'P99_TTFT':>14}{'median_E2E':>14}")
for wl in workloads:
    for r in ratios:
        out_dir = os.path.join(root, f"{wl}_ratio{r}_L10_L20")
        bench = os.path.join(out_dir, "L10_L20_bench.json")
        if not os.path.exists(bench):
            print(f"  {wl:<12}{r:>8}  no data"); continue
        try:
            with open(bench) as f:
                lines = [l for l in f if l.strip()]
            d = json.loads(lines[-1])
        except Exception as e:
            print(f"  {wl:<12}{r:>8}  parse error: {e}"); continue
        print(f"  {wl:<12}{r:>8}{d.get('input_throughput',0):>10.0f}{d.get('mean_ttft_ms',0):>12.0f}ms{d.get('p99_ttft_ms',0):>12.0f}ms{d.get('median_e2e_latency_ms',0):>12.0f}ms")
PY
echo "[static-best-diag] done"
