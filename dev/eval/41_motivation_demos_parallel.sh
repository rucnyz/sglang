#!/bin/bash
# Parallel 4-cell × 3-trial × 2-workload run: fan-out demo + long-horizon
# demo, the motivation-faithful workloads from paper §motivation §76.
#
# Total: 24 cell-runs distributed across 7 GPUs in 4 waves. Each cell
# ~12 minutes. Wall-clock ~50 minutes.
#
# Goal: prove L2 useful end-to-end. (1,1) > (1,0) > (0,0) by margin
# greater than combined std on at least one of TPS / mean_TTFT / P99
# on EACH workload (fan-out should show k2m fire benefit; long-horizon
# should show m2k fire benefit).

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30300}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-motivation-demos-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[motivation-demos] root=$ROOT"

GPUS=(0 1 3 4 5 6 7)
N_GPUS=${#GPUS[@]}

CELLS=("0 0" "1 0" "0 1" "1 1")
WORKLOADS=("fanout" "longhorizon")

# Build job list: (workload, trial, L1, L2) tuples.
JOBS=()
for wl in "${WORKLOADS[@]}"; do
  for trial in $(seq 1 $N_TRIALS); do
    for pair in "${CELLS[@]}"; do
      set -- $pair
      JOBS+=("$wl $trial $1 $2")
    done
  done
done
N_JOBS=${#JOBS[@]}
echo "[motivation-demos] $N_JOBS jobs across $N_GPUS GPUs"

driver_for_workload() {
  case "$1" in
    fanout)      echo "dev/eval/39_fanout_demo.sh" ;;
    longhorizon) echo "dev/eval/40_longhorizon_demo.sh" ;;
    *) echo "UNKNOWN workload: $1" >&2; exit 1 ;;
  esac
}

launch_wave() {
  local start=$1 end=$2
  local pids=()
  local i=$start
  while [ $i -lt $end ] && [ $i -lt $N_JOBS ]; do
    set -- ${JOBS[$i]}
    local wl=$1 trial=$2 L1=$3 L2=$4
    local cell="L1${L1}_L2${L2}"
    local out_dir="$ROOT/${wl}_trial${trial}_${cell}"
    local gpu=${GPUS[$((i % N_GPUS))]}
    local port=$((PORT_BASE + i))
    local driver=$(driver_for_workload "$wl")
    mkdir -p "$out_dir"
    echo "  [job $i] wl=$wl trial=$trial cell=$cell gpu=$gpu port=$port"
    ONLY_L1=$L1 ONLY_L2=$L2 \
      CUDA_VISIBLE_DEVICES=$gpu PORT=$port OUT_DIR="$out_dir" \
      SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
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
  echo "==== WAVE $wave (jobs $i..$((i+N_GPUS-1))) ===="
  launch_wave $i $((i + N_GPUS))
  i=$((i + N_GPUS))
  wave=$((wave + 1))
done

echo
echo "==== SUMMARY ($ROOT) ===="
python3 - <<PY
import json, os, statistics
root = "$ROOT"
n_trials = $N_TRIALS
cells = [(0,0),(1,0),(0,1),(1,1)]
workloads = ["fanout", "longhorizon"]
metrics = ["input_throughput", "mean_ttft_ms", "p99_ttft_ms", "median_e2e_latency_ms"]

def load(wl, trial, cell):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"{wl}_trial{trial}_{cstr}", f"{cstr}_bench.json")
    if not os.path.exists(fp): return None
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

def fires(wl, trial, cell):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"{wl}_trial{trial}_{cstr}", f"{cstr}_budgeter.jsonl")
    if not os.path.exists(fp): return (0, 0, 0)
    k2m = m2k = mov = 0
    with open(fp) as f:
        for L in f:
            try: j = json.loads(L)
            except: continue
            d = j.get("xpool_direction")
            if d == "kv_to_mamba": k2m += 1
            elif d == "mamba_to_kv": m2k += 1
            else: continue
            if j.get("xpool_unmapped_total", 0) > 0: mov += 1
    return (k2m, m2k, mov)

for wl in workloads:
    print()
    print(f"---- {wl} workload ----")
    print(f"{'cell':<8}{'TPS':>14}{'mean_TTFT':>16}{'P99_TTFT':>16}{'median_E2E':>16}{'fires(k/m/mv)':>18}")
    for cell in cells:
        cstr = f"({cell[0]},{cell[1]})"
        runs = [load(wl, t, cell) for t in range(1, n_trials+1)]
        runs = [r for r in runs if r is not None]
        if not runs:
            print(f"{cstr:<8}  (no data)"); continue
        means = {m: statistics.mean(r.get(m,0) for r in runs) for m in metrics}
        stds  = {m: (statistics.stdev(r.get(m,0) for r in runs) if len(runs)>1 else 0) for m in metrics}
        ftots = [fires(wl, t, cell) for t in range(1, n_trials+1)]
        k2m_avg = statistics.mean(f[0] for f in ftots)
        m2k_avg = statistics.mean(f[1] for f in ftots)
        mv_avg  = statistics.mean(f[2] for f in ftots)
        def fmt(m): return f"{means[m]:.0f}±{stds[m]:.0f}"
        print(f"{cstr:<8}{fmt('input_throughput'):>14}{fmt('mean_ttft_ms')+'ms':>16}{fmt('p99_ttft_ms')+'ms':>16}{fmt('median_e2e_latency_ms')+'ms':>16}{f'{k2m_avg:.1f}/{m2k_avg:.1f}/{mv_avg:.1f}':>18}")
PY

echo "[motivation-demos] done"
