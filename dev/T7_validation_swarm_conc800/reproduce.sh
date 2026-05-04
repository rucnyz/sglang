#!/bin/bash
# T7 reproduce: 4-cell × 5-trial swarm at conc=800 with the FULL T1+T3+T4+T6
# stack on for L2 cells. Distributes 20 jobs across 8 GPUs in 3 waves.
#
# Output: dev/T7_validation_swarm_conc800/results/

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
PORT_BASE="${PORT_BASE:-31800}"
N_TRIALS="${N_TRIALS:-5}"
RUN_NAME="${RUN_NAME:-t7-swarm-conc800-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/T7_validation_swarm_conc800/results/$RUN_NAME"
mkdir -p "$ROOT"
echo "[t7] root=$ROOT"

GPUS=(0 1 2 3 4 5 6 7)
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
echo "[t7] $N_JOBS jobs across $N_GPUS GPUs"

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
      NUM_CONCURRENCY=${NUM_CONCURRENCY:-800} \
      TRAFFIC_SCENARIO="${TRAFFIC_SCENARIO:-D(256,256)}" \
      SESSION_CAP=${SESSION_CAP:-3000} \
      MAX_TIME_MIN=${MAX_TIME_MIN:-8} \
      MEM_FRAC=${MEM_FRAC:-0.8} \
      bash dev/T7_validation_swarm_conc800/test/per_cell.sh \
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
import glob, json, os, statistics, re
root = "$ROOT"
n_trials = $N_TRIALS
cells = [(0,0),(1,0),(0,1),(1,1)]

def find_summary(out_dir):
    pat = os.path.join(out_dir, "genai_results", "*text-to-text-multi-turn*.json")
    matches = [m for m in glob.glob(pat) if os.path.basename(m) != "experiment_metadata.json"]
    return matches[0] if matches else None

def fires(out_dir, cell):
    fp = os.path.join(out_dir, f"{cell}_budgeter.jsonl")
    if not os.path.exists(fp): return (0, 0, 0)
    k=m=mv=0
    with open(fp) as f:
        for L in f:
            try: j=json.loads(L)
            except: continue
            d=j.get("xpool_direction")
            if d=="kv_to_mamba": k+=1
            elif d=="mamba_to_kv": m+=1
            else: continue
            if j.get("xpool_unmapped_total",0)>0: mv+=1
    return (k,m,mv)

def grep_count(path, pattern):
    if not os.path.exists(path): return 0
    n = 0
    with open(path) as f:
        for L in f:
            if re.search(pattern, L):
                n += 1
    return n

print(f"{'cell':<6}{'n':>3}{'reqs':>10}{'TTFTm(ms)':>14}{'TTFTp99(ms)':>14}{'E2Em(ms)':>14}{'outTPS':>10}{'fires(k/m/mv)':>18}{'T3/T4/T6':>12}")
results = {}
for cell in cells:
    cstr=f"L1{cell[0]}_L2{cell[1]}"; lab=f"({cell[0]},{cell[1]})"
    ttft_m=[]; ttft_p99=[]; e2e_m=[]; tps=[]; reqs=[]; ftots=[]
    t3=t4=t6=0
    for t in range(1,n_trials+1):
        od=os.path.join(root,f"trial{t}_{cstr}")
        sp=find_summary(od)
        if sp:
            with open(sp) as f: d=json.load(f)
            a=d.get("aggregated_metrics",d); s=a.get("stats",{})
            if "ttft" in s and s["ttft"].get("mean") is not None:
                ttft_m.append(s["ttft"]["mean"]*1000)
                ttft_p99.append(s["ttft"]["p99"]*1000)
                e2e_m.append(s["e2e_latency"]["mean"]*1000)
                tps.append(a.get("mean_output_throughput_tokens_per_s",0))
                reqs.append(a.get("num_completed_requests",0))
                ftots.append(fires(od,cstr))
        srv = os.path.join(od, f"{cstr}_server.log")
        t3 += grep_count(srv, r"T3 smart over-cap selection")
        t4 += grep_count(srv, r"T4 atomic migration")
        t6 += grep_count(srv, r"T6 admission-time fire")
    n=len(ttft_m)
    if n==0:
        print(f"  {lab:<6}  no data"); continue
    def fmt(L):
        m=statistics.mean(L); s=statistics.stdev(L) if len(L)>1 else 0
        return f"{m:.0f}±{s:.0f}"
    k_avg=statistics.mean(f[0] for f in ftots) if ftots else 0
    m_avg=statistics.mean(f[1] for f in ftots) if ftots else 0
    mv_avg=statistics.mean(f[2] for f in ftots) if ftots else 0
    print(f"  {lab:<4}{n:>3}{fmt(reqs):>10}{fmt(ttft_m):>14}{fmt(ttft_p99):>14}{fmt(e2e_m):>14}{fmt(tps):>10}{f'{k_avg:.1f}/{m_avg:.1f}/{mv_avg:.1f}':>18}{f'{t3}/{t4}/{t6}':>12}")
    results[cell]=(ttft_m,ttft_p99,e2e_m,tps,reqs)

print()
print("=== Δ vs (0,0) baseline ===")
b=results.get((0,0))
if b:
    b_ttft=statistics.mean(b[0]); b_p99=statistics.mean(b[1]); b_e2e=statistics.mean(b[2]); b_tps=statistics.mean(b[3]); b_reqs=statistics.mean(b[4])
    for cell,(t,p,e,tps,r) in results.items():
        if cell==(0,0): continue
        dt=(statistics.mean(t)-b_ttft)/b_ttft*100
        dp=(statistics.mean(p)-b_p99)/b_p99*100
        de=(statistics.mean(e)-b_e2e)/b_e2e*100
        dtps=(statistics.mean(tps)-b_tps)/b_tps*100
        dreqs=(statistics.mean(r)-b_reqs)/b_reqs*100
        print(f"  {cell}  TTFTm {dt:+.1f}%  P99 {dp:+.1f}%  E2Em {de:+.1f}%  outTPS {dtps:+.1f}%  reqs {dreqs:+.1f}%")
PY
echo "[t7] done"
