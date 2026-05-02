#!/bin/bash
# Parallel 4-cell × 3-trial variance run on the multi-turn long-horizon
# agent benchmark (paper §motivation §76).
#
# Distributes 12 cell-runs across 7 GPUs in 2 waves. Each cell ~7 min
# (110s warmup + 300s bench + cleanup). Wall-clock ~16 min total.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30400}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-multiturn-variance-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[multiturn-variance] root=$ROOT"

GPUS=(0 1 3 4 5 6 7)
N_GPUS=${#GPUS[@]}

CELLS=("0 0" "1 0" "0 1" "1 1")

JOBS=()
for trial in $(seq 1 $N_TRIALS); do
  for pair in "${CELLS[@]}"; do
    set -- $pair
    JOBS+=("$trial $1 $2")
  done
done
N_JOBS=${#JOBS[@]}
echo "[multiturn-variance] $N_JOBS jobs across $N_GPUS GPUs"

launch_wave() {
  local start=$1 end=$2
  local pids=()
  local i=$start
  while [ $i -lt $end ] && [ $i -lt $N_JOBS ]; do
    set -- ${JOBS[$i]}
    local trial=$1 L1=$2 L2=$3
    local cell="L1${L1}_L2${L2}"
    local out_dir="$ROOT/trial${trial}_${cell}"
    local gpu=${GPUS[$((i % N_GPUS))]}
    local port=$((PORT_BASE + i))
    mkdir -p "$out_dir"
    echo "  [job $i] trial=$trial cell=$cell gpu=$gpu port=$port"
    ONLY_L1=$L1 ONLY_L2=$L2 \
      CUDA_VISIBLE_DEVICES=$gpu PORT=$port OUT_DIR="$out_dir" \
      NUM_CONCURRENCY=${NUM_CONCURRENCY:-16} \
      TURN_INPUT=${TURN_INPUT:-4096} TURN_OUTPUT=${TURN_OUTPUT:-4096} \
      SESSION_CAP=${SESSION_CAP:-60000} \
      MAX_TIME_S=${MAX_TIME_S:-300} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      SGLANG_XPOOL_UNIT=${SGLANG_XPOOL_UNIT:-1} \
      bash dev/eval/42_multiturn_per_cell.sh \
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
echo "==== SUMMARY ($ROOT) ===="
python3 - <<PY
import json, os, statistics
root = "$ROOT"
n_trials = $N_TRIALS
cells = [(0,0),(1,0),(0,1),(1,1)]
metrics = ["mean_ttft_ms", "p99_ttft_ms", "mean_e2e_ms", "p99_e2e_ms",
           "input_tps", "output_tps", "num_requests_valid", "num_errors",
           "max_session_tokens_observed"]

def load(trial, cell):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"trial{trial}_{cstr}", "multiturn_summary.json")
    if not os.path.exists(fp): return None
    with open(fp) as f:
        try: return json.load(f)
        except: return None

def fires(trial, cell):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"trial{trial}_{cstr}", f"{cstr}_budgeter.jsonl")
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

print(f"{'cell':<8}{'reqs':>10}{'TTFT_mean':>14}{'TTFT_p99':>14}{'E2E_mean':>14}{'in_TPS':>10}{'out_TPS':>10}{'fires(k/m/mv)':>18}")
for cell in cells:
    cstr = f"({cell[0]},{cell[1]})"
    runs = [load(t, cell) for t in range(1, n_trials+1)]
    runs = [r for r in runs if r is not None]
    if not runs:
        print(f"{cstr:<8}  (no data)"); continue
    means = {m: statistics.mean(r.get(m,0) for r in runs) for m in metrics}
    stds  = {m: (statistics.stdev(r.get(m,0) for r in runs) if len(runs)>1 else 0) for m in metrics}
    ftots = [fires(t, cell) for t in range(1, n_trials+1)]
    k2m_avg = statistics.mean(f[0] for f in ftots)
    m2k_avg = statistics.mean(f[1] for f in ftots)
    mv_avg  = statistics.mean(f[2] for f in ftots)
    def fmt(m, scale=1):
        return f"{means[m]/scale:.0f}±{stds[m]/scale:.0f}"
    print(f"{cstr:<8}{fmt('num_requests_valid'):>10}{fmt('mean_ttft_ms')+'ms':>14}{fmt('p99_ttft_ms')+'ms':>14}{fmt('mean_e2e_ms')+'ms':>14}{fmt('input_tps'):>10}{fmt('output_tps'):>10}{f'{k2m_avg:.1f}/{m2k_avg:.1f}/{mv_avg:.1f}':>18}")
PY

echo "[multiturn-variance] done"
