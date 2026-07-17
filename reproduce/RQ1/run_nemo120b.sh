#!/bin/bash
# Nemotron-3-Super-120B-A12B RQ1 Case 1/2/3: base (LRU) vs sys (LPB+HiMA).
# N=3 reps per arm (same server boot), fresh boot between arms.
# Uses the corpus-built cc_nemotron_t{6,12} traces.
set -eu
SCRIPT=/scratch/yuzhou/projects/sglang/reproduce/RQ1/run_arm.sh
T=/scratch/yuzhou/projects/agentreplay/data/traces
T6=$T/cc_nemotron_t6.jsonl
T12=$T/cc_nemotron_t12.jsonl

MODEL=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
GPUS=${GPUS:-3,4,5,7}
TP=4
PORT=${PORT:-30098}
STAGGER=0.5
OUTDIR=${OUTDIR:-/data/yuzhou/projects/hybrid-inference/figures/data/nemo_120b}
CASES="${CASES:-case1 case2 case3}"

export REASONING=none MEMFRAC=0.85 MAMBA_CAP=256 MAMBA_STRAT=no_buffer
export CUDA_GRAPH_DECODE=full CUDA_GRAPH_PREFILL=disabled

run_case() {
  local CASE=$1 TRACE=$2 CONC=$3
  local D="$OUTDIR/$CASE"
  mkdir -p "$D"
  echo "[$(date +%H:%M:%S)] === $CASE START (conc=$CONC, TP=$TP on GPUs $GPUS) ==="

  for ARM in base sys; do
    echo "[$(date +%H:%M:%S)] $CASE $ARM"
    GPUS=$GPUS TP=$TP PORT=$PORT MODEL=$MODEL \
      bash "$SCRIPT" "$ARM" "$TRACE" $STAGGER $CONC - 3 "$D/$ARM"
    sleep 10
  done

  echo "[$(date +%H:%M:%S)] === $CASE RESULTS ==="
  for ARM in base sys; do
    /scratch/yuzhou/projects/sglang/.venv/bin/python -c "
import json, statistics
tps_list, hit_list = [], []
for rep in range(1, 4):
    f = '$D/${ARM}/${ARM}_r' + str(rep) + '.json'
    try:
        d = json.load(open(f))
        tps_list.append(d['throughput_tok_s'])
        hit_list.append(d['cache_hit'])
        err = d.get('n_error', 0)
        if err > 0:
            print(f'  WARNING: $ARM r{rep} has {err} errors!')
    except: pass
if tps_list:
    avg_tps = statistics.mean(tps_list)
    std_tps = statistics.stdev(tps_list) if len(tps_list) > 1 else 0
    avg_hit = statistics.mean(hit_list)
    print(f'$ARM: tps={avg_tps:.1f}±{std_tps:.1f} hit={avg_hit:.4f} (N={len(tps_list)}, reps={[round(t,1) for t in tps_list]})')
" 2>/dev/null
  done
  echo ""
}

case " $CASES " in *" case1 "*) run_case "case1" "$T6"  64  ;; esac
case " $CASES " in *" case2 "*) run_case "case2" "$T12" 64  ;; esac
case " $CASES " in *" case3 "*) run_case "case3" "$T6"  128 ;; esac

echo "=== NEMOTRON-120B ALL CASES COMPLETE ==="
