#!/bin/bash
# Cost-aware-gate variance: parallel 4-cell × N-trial run on the multi-turn
# long-horizon genai-bench scenario, with Stage-0 calibration env vars
# pinned and SGLANG_XPOOL_COST_LOG enabled per cell.
#
# What's new vs 44b_fanout_genaibench_variance_parallel.sh:
#   1. Runs `dev/eval/cost_model/stage0_calibrate.sh` once per GPU to
#      generate ~/.cache/sglang/cost_calibration/<gpu>__<model>.json and
#      pin SGLANG_CSIGMA_* env. Subsequent cell launches inherit.
#   2. Sets SGLANG_XPOOL_COST_LOG=$out_dir/${cell}_xpool_cost.jsonl so
#      every gate decision (fire and no-fire) is logged in JSONL form
#      for post-hoc analysis.
#   3. Uses 43_multiturn_genaibench_per_cell.sh as the per-cell driver
#      (steady-state multi-turn, paper §motivation §76 long-horizon).
#
# Outputs per cell:
#   - genai_results/         (genai-bench summary)
#   - <cell>_server.log      (sglang stdout)
#   - <cell>_client.log      (genai-bench stdout)
#   - <cell>_budgeter.jsonl  (existing budgeter log)
#   - <cell>_xpool_cost.jsonl  (new — cost-aware gate decisions)

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-30700}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-cost-aware-variance-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[cost-aware-variance] root=$ROOT"

GPUS=(0 1 3 4 5 6 7)
N_GPUS=${#GPUS[@]}
CELLS=("0 0" "1 0" "0 1" "1 1")  # (L1, L2) baseline / L1-only / L2-only / L1+L2

# Step 0: calibrate once per GPU. Stage-0 caches by (gpu, model) so repeats
# are instant. Using GPU 0 to seed the cache; the JSON path is independent
# of GPU index since it keys on gpu_name. (All 7 GPUs are H200s on this host.)
echo "[cost-aware-variance] Stage-0 calibration (GPU 0)..."
eval "$(CUDA_VISIBLE_DEVICES=0 bash dev/eval/cost_model/stage0_calibrate.sh 2>/dev/null)"
echo "  CSIGMA_KV_GAMMA=${SGLANG_CSIGMA_KV_GAMMA:-?} CSIGMA_M_BETA=${SGLANG_CSIGMA_M_BETA:-?} L*=${SGLANG_CSIGMA_LSTAR:-?}"

JOBS=()
for trial in $(seq 1 $N_TRIALS); do
  for pair in "${CELLS[@]}"; do
    set -- $pair
    JOBS+=("$trial $1 $2")
  done
done
N_JOBS=${#JOBS[@]}
echo "[cost-aware-variance] $N_JOBS jobs across $N_GPUS GPUs"

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
      NUM_CONCURRENCY=${NUM_CONCURRENCY:-14} \
      TRAFFIC_SCENARIO=${TRAFFIC_SCENARIO:-"D(4096,4096)"} \
      SESSION_CAP=${SESSION_CAP:-60000} \
      MAX_TIME_MIN=${MAX_TIME_MIN:-8} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      SGLANG_K_BIG_AUTO_THRESHOLD=${SGLANG_K_BIG_AUTO_THRESHOLD:-0} \
      SGLANG_XPOOL_UNIT=${SGLANG_XPOOL_UNIT:-1} \
      SGLANG_XPOOL_MAMBA_FLUSH_CAP=${SGLANG_XPOOL_MAMBA_FLUSH_CAP:-256} \
      SGLANG_CSIGMA_KV_ALPHA="${SGLANG_CSIGMA_KV_ALPHA:-}" \
      SGLANG_CSIGMA_KV_BETA="${SGLANG_CSIGMA_KV_BETA:-}" \
      SGLANG_CSIGMA_KV_GAMMA="${SGLANG_CSIGMA_KV_GAMMA:-}" \
      SGLANG_CSIGMA_M_ALPHA="${SGLANG_CSIGMA_M_ALPHA:-}" \
      SGLANG_CSIGMA_M_BETA="${SGLANG_CSIGMA_M_BETA:-}" \
      SGLANG_CSIGMA_LSTAR="${SGLANG_CSIGMA_LSTAR:-}" \
      SGLANG_CSIGMA_JSON="${SGLANG_CSIGMA_JSON:-}" \
      SGLANG_CSIGMA_MODEL="${SGLANG_CSIGMA_MODEL:-}" \
      SGLANG_CSIGMA_DEVICE="${SGLANG_CSIGMA_DEVICE:-}" \
      SGLANG_XPOOL_COOLDOWN="${SGLANG_XPOOL_COOLDOWN:-30}" \
      SGLANG_XPOOL_COST_LOG="$out_dir/${cell}_xpool_cost.jsonl" \
      bash dev/eval/43_multiturn_genaibench_per_cell.sh \
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
    pat = os.path.join(out_dir, "genai_results", "*_text-to-text*_*.json")
    matches = [m for m in glob.glob(pat)
               if os.path.basename(m) != "experiment_metadata.json"
               and "summary" not in os.path.basename(m).lower()]
    return matches[0] if matches else None

def cost_log_stats(out_dir, cell):
    fp = os.path.join(out_dir, f"{cell}_xpool_cost.jsonl")
    if not os.path.exists(fp):
        return dict(ticks=0, fires=0, m_exp=0, kv_exp=0, mean_kv_L=0)
    ticks = fires = m_exp = kv_exp = 0
    L_sum = 0; L_n = 0
    with open(fp) as f:
        for L in f:
            try: r = json.loads(L)
            except: continue
            ticks += 1
            if r.get("direction"): fires += 1
            reg = r.get("regime")
            if reg == "M-expensive": m_exp += 1
            elif reg == "KV-expensive": kv_exp += 1
            kvL = r.get("mean_recovery_len_kv")
            if kvL: L_sum += kvL; L_n += 1
    return dict(ticks=ticks, fires=fires,
                m_exp=m_exp, kv_exp=kv_exp,
                mean_kv_L=(L_sum/L_n if L_n else 0))

print(f"{'cell':<8}{'n':>4}{'reqs':>9}{'TTFT_mean':>13}{'TTFT_p99':>13}{'E2E_mean':>13}{'out_TPS':>10}{'fires':>8}{'M_exp%':>9}{'mean_L':>9}")
for cell in cells:
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    label = f"({cell[0]},{cell[1]})"
    runs = []
    cost_stats = []
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
            cost_stats.append(cost_log_stats(out_dir, cstr))
        except Exception:
            continue
    if not runs:
        print(f"  {label}  no data"); continue
    def fmt(L):
        if not L: return "-"
        m = statistics.mean(L); s = statistics.stdev(L) if len(L) > 1 else 0
        return f"{m:.0f}±{s:.0f}"
    def get_ms(r, *keys):
        for k in keys:
            v = r.get(k)
            if v is not None:
                return v * 1000 if v < 100 else v
        return None
    ttft_mean = [v for r in runs if (v := get_ms(r, "mean_ttft", "mean_ttft_ms")) is not None]
    ttft_p99 = [v for r in runs if (v := get_ms(r, "p99_ttft", "p99_ttft_ms")) is not None]
    e2e_mean = [v for r in runs if (v := get_ms(r, "mean_e2e_latency", "mean_e2e_latency_ms")) is not None]
    out_tps = [r.get("mean_output_throughput_tokens_per_s", 0) for r in runs]
    reqs = [r.get("num_completed_requests", 0) for r in runs]
    fires_avg = statistics.mean(c["fires"] for c in cost_stats) if cost_stats else 0
    m_exp_avg = statistics.mean(c["m_exp"] for c in cost_stats) if cost_stats else 0
    ticks_avg = max(1, statistics.mean(c["ticks"] for c in cost_stats) if cost_stats else 1)
    L_avg = statistics.mean(c["mean_kv_L"] for c in cost_stats) if cost_stats else 0
    print(f"  {label:<6}{len(runs):>4}{fmt(reqs):>9}{fmt(ttft_mean)+'ms':>13}"
          f"{fmt(ttft_p99)+'ms':>13}{fmt(e2e_mean)+'ms':>13}{fmt(out_tps):>10}"
          f"{fires_avg:>8.1f}{100*m_exp_avg/ticks_avg:>8.0f}%{L_avg:>9.0f}")
PY

echo "[cost-aware-variance] done"
echo "Per-cell cost logs: $ROOT/trial*/L*_xpool_cost.jsonl"
