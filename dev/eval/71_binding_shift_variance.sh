#!/bin/bash
# Binding-shift variance: paper §motivation §76's two complementary
# regimes run as separate cells × workloads to demonstrate L1+L2's
# mutual-enabling story.
#
#   - Phase A (fan-out agent, mamba-bound): many short-prompt sub-task
#     calls; mamba slot pool saturates while paged KV is idle.
#     L2 should fire kv_to_mamba (steal idle KV chunks for mamba slots).
#
#   - Phase B (long-horizon agent, KV-bound): few very-long-context
#     sessions; paged KV saturates while mamba slots idle.
#     L2 should fire mamba_to_kv (steal idle mamba chunks for KV).
#
# A single static partition cannot serve both regimes; L1+L2 should
# beat L1-only on EACH regime by virtue of L2 reallocating chunks toward
# the binding pool. This is the workload where L2 is designed to win,
# in contrast to the steady multi-turn workload where L2 ≈ L1-only is
# the expected (paper invariant 1) outcome.
#
# Wires Stage-0 cost calibration + NB direction-aware gate (default ON
# in cross_pool_planner.py) + saturation guard (refusing to shrink a
# source above its high-water).
#
# Total: 24 jobs (2 workloads × 4 cells × 3 trials), 2 GPU waves of 7,
# wall-clock ~50 minutes.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30800}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-binding-shift-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[binding-shift] root=$ROOT"

# Stage-0 calibration (single GPU; cached by gpu_name × model so all
# subsequent cells reuse the JSON without re-running the bench).
echo "[binding-shift] Stage-0 calibration (GPU 0)..."
eval "$(CUDA_VISIBLE_DEVICES=0 bash dev/eval/cost_model/stage0_calibrate.sh 2>/dev/null)"
echo "  CSIGMA_KV_GAMMA=${SGLANG_CSIGMA_KV_GAMMA:-?} CSIGMA_M_BETA=${SGLANG_CSIGMA_M_BETA:-?} L*=${SGLANG_CSIGMA_LSTAR:-?}"

GPUS=(0 1 3 4 5 6 7)
N_GPUS=${#GPUS[@]}
CELLS=("0 0" "1 0" "0 1" "1 1")
WORKLOADS=("fanout" "longhorizon")

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
echo "[binding-shift] $N_JOBS jobs across $N_GPUS GPUs"

driver_for_workload() {
  case "$1" in
    fanout)      echo "dev/eval/39_fanout_demo.sh" ;;
    longhorizon) echo "dev/eval/40_longhorizon_demo.sh" ;;
    *) echo "UNKNOWN workload: $1" >&2; exit 1 ;;
  esac
}

workload_num_prompts() {
  case "$1" in
    fanout)      echo "${NUM_PROMPTS_FANOUT:-6000}" ;;       # ~80-100s runtime at TPS=20K
    longhorizon) echo "${NUM_PROMPTS_LONGHORIZON:-360}" ;;    # ~3x baseline → ~3min runtime
    *) echo "1200" ;;
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
    local driver num_prompts
    driver=$(driver_for_workload "$wl")
    num_prompts=$(workload_num_prompts "$wl")
    mkdir -p "$out_dir"
    echo "  [job $i] wl=$wl trial=$trial cell=$cell gpu=$gpu port=$port"
    ONLY_L1=$L1 ONLY_L2=$L2 \
      CUDA_VISIBLE_DEVICES=$gpu PORT=$port OUT_DIR="$out_dir" \
      SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      SGLANG_CSIGMA_KV_ALPHA="${SGLANG_CSIGMA_KV_ALPHA:-}" \
      SGLANG_CSIGMA_KV_BETA="${SGLANG_CSIGMA_KV_BETA:-}" \
      SGLANG_CSIGMA_KV_GAMMA="${SGLANG_CSIGMA_KV_GAMMA:-}" \
      SGLANG_CSIGMA_M_ALPHA="${SGLANG_CSIGMA_M_ALPHA:-}" \
      SGLANG_CSIGMA_M_BETA="${SGLANG_CSIGMA_M_BETA:-}" \
      SGLANG_CSIGMA_LSTAR="${SGLANG_CSIGMA_LSTAR:-}" \
      SGLANG_CSIGMA_JSON="${SGLANG_CSIGMA_JSON:-}" \
      SGLANG_XPOOL_NB_DIRECTION_AWARE=1 \
      SGLANG_XPOOL_COOLDOWN=${SGLANG_XPOOL_COOLDOWN:-8} \
      NUM_PROMPTS=$num_prompts \
      SGLANG_XPOOL_DEFAULT_L=${SGLANG_XPOOL_DEFAULT_L:-4096} \
      SGLANG_XPOOL_COST_LOG="$out_dir/${cell}_xpool_cost.jsonl" \
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
metrics = ["input_throughput", "mean_ttft_ms", "p99_ttft_ms",
           "median_e2e_latency_ms"]

def load_bench(wl, trial, cell):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"{wl}_trial{trial}_{cstr}", f"{cstr}_bench.json")
    if not os.path.exists(fp): return None
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1]) if lines else None

def cost_log_stats(wl, trial, cell):
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    fp = os.path.join(root, f"{wl}_trial{trial}_{cstr}", f"{cstr}_xpool_cost.jsonl")
    if not os.path.exists(fp): return None
    fires = 0; fires_by_dir = {}
    L_sum = 0; L_n = 0; total = 0
    for L in open(fp):
        try: r = json.loads(L)
        except: continue
        total += 1
        d = r.get("direction")
        if d:
            fires += 1
            fires_by_dir[d] = fires_by_dir.get(d, 0) + 1
        kvL = r.get("mean_recovery_len_kv")
        if kvL: L_sum += kvL; L_n += 1
    return dict(ticks=total, fires=fires, fires_by_dir=fires_by_dir,
                mean_kv_L=L_sum/max(L_n,1))

for wl in workloads:
    print()
    print(f"---- {wl} workload ----")
    print(f"{'cell':<8}{'TPS':>14}{'mean_TTFT':>16}{'P99_TTFT':>16}{'median_E2E':>16}{'fires k/m':>12}{'mean_L':>9}")
    for cell in cells:
        cstr = f"({cell[0]},{cell[1]})"
        runs = [load_bench(wl, t, cell) for t in range(1, n_trials+1)]
        runs = [r for r in runs if r is not None]
        if not runs:
            print(f"{cstr:<8}  (no data)"); continue
        means = {m: statistics.mean(r.get(m,0) for r in runs) for m in metrics}
        stds  = {m: (statistics.stdev(r.get(m,0) for r in runs) if len(runs)>1 else 0) for m in metrics}
        cs_list = [cost_log_stats(wl, t, cell) for t in range(1, n_trials+1)]
        cs_list = [c for c in cs_list if c is not None]
        if cs_list:
            k2m = statistics.mean(c['fires_by_dir'].get('kv_to_mamba', 0) for c in cs_list)
            m2k = statistics.mean(c['fires_by_dir'].get('mamba_to_kv', 0) for c in cs_list)
            L_avg = statistics.mean(c['mean_kv_L'] for c in cs_list)
            fire_str = f"{k2m:.1f}/{m2k:.1f}"
        else:
            L_avg = 0
            fire_str = "-/-"
        def fmt(m): return f"{means[m]:.0f}±{stds[m]:.0f}"
        print(f"{cstr:<8}{fmt('input_throughput'):>14}{fmt('mean_ttft_ms')+'ms':>16}{fmt('p99_ttft_ms')+'ms':>16}{fmt('median_e2e_latency_ms')+'ms':>16}{fire_str:>12}{L_avg:>9.0f}")
PY

echo "[binding-shift] done"
echo "Per-cell cost logs: $ROOT/<workload>_trial*/L*_xpool_cost.jsonl"
