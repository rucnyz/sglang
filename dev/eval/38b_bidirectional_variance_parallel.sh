#!/bin/bash
# Parallel 4-cell × 3-trial variance run on the bidirectional-fire-
# friendly workload (38_bidirectional_pool_binding.sh).
#
# Distributes 12 cell-runs across 7 GPUs in 2 waves (~14 min total).
# Apples-to-apples calibration: mem_frac=0.8 across all cells,
# ARENA_CHUNK_BYTES=256 MiB, K_BIG_AUTO_THRESHOLD=0.85, KV_HIGH=0.4,
# MAMBA_HIGH=0.5.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30200}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-bidirectional-variance-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[bidirectional-variance] root=$ROOT"

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
echo "[bidirectional-variance] $N_JOBS jobs across $N_GPUS GPUs"

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
      SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      SGLANG_XPOOL_KV_HIGH=${SGLANG_XPOOL_KV_HIGH:-0.4} \
      SGLANG_XPOOL_MAMBA_HIGH=${SGLANG_XPOOL_MAMBA_HIGH:-0.5} \
      bash dev/eval/38_bidirectional_pool_binding.sh \
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
metrics = ["input_throughput", "mean_ttft_ms", "p99_ttft_ms", "median_e2e_latency_ms"]

def load(trial, cell):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"trial{trial}_{cstr}", f"{cstr}_bench.json")
    if not os.path.exists(fp): return None
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

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

print(f"{'cell':<8}{'TPS':>14}{'mean_TTFT':>16}{'P99_TTFT':>16}{'median_E2E':>16}{'fires(k2m/m2k/mv)':>22}")
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
    def fmt(m): return f"{means[m]:.0f}±{stds[m]:.0f}"
    print(f"{cstr:<8}{fmt('input_throughput'):>14}{fmt('mean_ttft_ms')+'ms':>16}{fmt('p99_ttft_ms')+'ms':>16}{fmt('median_e2e_latency_ms')+'ms':>16}{f'{k2m_avg:.1f}/{m2k_avg:.1f}/{mv_avg:.1f}':>22}")
PY

echo "[bidirectional-variance] done"
