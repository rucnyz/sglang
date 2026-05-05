#!/bin/bash
# Parallel 4-cell × N-trial fan-out variance run on genai-bench.
# Distributes 12-20 cell-runs across 7 GPUs, ~2 waves wall-clock.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30600}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-fanout-genaibench-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[fanout-genaibench] root=$ROOT"

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
echo "[fanout-genaibench] $N_JOBS jobs across $N_GPUS GPUs"

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
      NUM_CONCURRENCY=${NUM_CONCURRENCY:-400} \
      TRAFFIC_SCENARIO="${TRAFFIC_SCENARIO:-D(256,32)}" \
      MAX_TIME_MIN=${MAX_TIME_MIN:-5} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0.85} \
      SGLANG_XPOOL_UNIT=${SGLANG_XPOOL_UNIT:-1} \
      SGLANG_XPOOL_MAMBA_FLUSH_CAP=${SGLANG_XPOOL_MAMBA_FLUSH_CAP:-256} \
      bash dev/eval/44_fanout_genaibench_per_cell.sh \
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
import glob, json, os, statistics
root = "$ROOT"
n_trials = $N_TRIALS
cells = [(0,0),(1,0),(0,1),(1,1)]

def find_genai_summary(out_dir):
    pat = os.path.join(out_dir, "genai_results", "*_text-to-text_*.json")
    matches = [m for m in glob.glob(pat)
               if os.path.basename(m) != "experiment_metadata.json"]
    matches = [m for m in matches if "summary" not in os.path.basename(m).lower()]
    return matches[0] if matches else None

def fires(out_dir, cell):
    fp = os.path.join(out_dir, f"{cell}_budgeter.jsonl")
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

print(f"{'cell':<8}{'n':>5}{'reqs':>10}{'TTFT_mean':>14}{'TTFT_p99':>14}{'E2E_mean':>14}{'out_TPS':>10}{'fires(k/m/mv)':>18}")
for cell in cells:
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    label = f"({cell[0]},{cell[1]})"
    runs = []
    fire_tots = []
    for t in range(1, n_trials + 1):
        out_dir = os.path.join(root, f"trial{t}_{cstr}")
        if not os.path.isdir(out_dir): continue
        sum_path = find_genai_summary(out_dir)
        if not sum_path: continue
        try:
            with open(sum_path) as f:
                d = json.load(f)
            agg = d.get("aggregated_metrics", d)
            runs.append(agg)
            fire_tots.append(fires(out_dir, cstr))
        except Exception:
            continue
    if not runs:
        print(f"  {label}  no data"); continue
    def fmt(L):
        if not L: return "—"
        m = statistics.mean(L)
        s = statistics.stdev(L) if len(L) > 1 else 0
        return f"{m:.0f}±{s:.0f}"
    ttft_mean = []
    for r in runs:
        v = r.get("mean_ttft", r.get("mean_ttft_ms"))
        if v is None: continue
        # genai-bench reports ttft in seconds when key is "mean_ttft" (no _ms)
        if v < 100: v = v * 1000
        ttft_mean.append(v)
    ttft_p99 = []
    for r in runs:
        v = r.get("p99_ttft", r.get("p99_ttft_ms"))
        if v is None: continue
        if v < 100: v = v * 1000
        ttft_p99.append(v)
    e2e_mean = []
    for r in runs:
        v = r.get("mean_e2e_latency", r.get("mean_e2e_latency_ms"))
        if v is None: continue
        if v < 1000: v = v * 1000
        e2e_mean.append(v)
    out_tps = [r.get("mean_output_throughput_tokens_per_s", 0) for r in runs]
    reqs = [r.get("num_completed_requests", 0) for r in runs]
    k2m_avg = statistics.mean(f[0] for f in fire_tots) if fire_tots else 0
    m2k_avg = statistics.mean(f[1] for f in fire_tots) if fire_tots else 0
    mv_avg  = statistics.mean(f[2] for f in fire_tots) if fire_tots else 0
    print(f"  {label:<6}{len(runs):>5}{fmt(reqs):>10}{fmt(ttft_mean)+'ms':>14}{fmt(ttft_p99)+'ms':>14}{fmt(e2e_mean)+'ms':>14}{fmt(out_tps):>10}{f'{k2m_avg:.1f}/{m2k_avg:.1f}/{mv_avg:.1f}':>18}")
PY

echo "[fanout-genaibench] done"
