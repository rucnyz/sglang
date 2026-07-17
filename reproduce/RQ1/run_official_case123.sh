#!/bin/bash
# Official RQ1 Case 1/2/3: base (LRU) vs sys (LPB+HiMA), N=3 per arm, fresh boot/rep.
# Token-exact agentreplay traces, built from the frozen corpus per REPRODUCE.md
# (`convert --projects data/corpus/data/contrib`, program-identical across tokenizers):
#   Case1: t6  @ conc 64   (long-horizon agent, KV-bound)
#   Case2: t12 @ conc 64   (agent swarm, high eviction volume)
#   Case3: t6  @ conc 128  (Case1 trace at 2x concurrency = dynamic/high pressure)
# stagger=0.5, default config; only workload/concurrency differ.
# CASES env selects a subset, e.g. CASES="case2 case3" bash run_official_case123.sh
#
# The pre-2026-07 numbers in FINDINGS.md were measured on the SUPERSEDED
# direct-from-projects traces (cc_qwen_t6_v2, 1200 req). Those are not comparable
# to a corpus-built run: same base-vs-sys question, different workload mix.
set -eu
SCRIPT=/scratch/yuzhou/projects/sglang/reproduce/RQ1/run_arm.sh
T6=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6.jsonl
T12=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t12.jsonl
PORT=${PORT:-30097}; GPU=${GPU:-7}
STAGGER=0.5
CASES="${CASES:-case1 case2 case3}"
# RUNTAG names the per-model output subdir so different MODELs don't collide
# (9B -> runs/official, 35B -> runs/official_35b). MODEL is passed through to
# run_arm.sh via the environment.
RUNTAG="${RUNTAG:-official}"

run_case() {
  local CASE=$1 TRACE=$2 CONC=$3
  local OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/$CASE/runs/$RUNTAG
  mkdir -p "$OUTDIR"
  echo "[$(date +%H:%M:%S)] === $CASE START (conc=$CONC) ==="

  for ARM in base sys; do
    for REP in 1 2 3; do
      local BOOTDIR="$OUTDIR/${ARM}_boot${REP}"
      mkdir -p "$BOOTDIR"
      echo "[$(date +%H:%M:%S)] $CASE $ARM rep$REP"
      PORT=$PORT GPU=$GPU bash "$SCRIPT" "$ARM" "$TRACE" $STAGGER $CONC - 1 "$BOOTDIR"
      cp "$BOOTDIR/${ARM}_r1.json" "$OUTDIR/${ARM}_r${REP}.json" 2>/dev/null
      sleep 5
    done
  done

  echo "[$(date +%H:%M:%S)] === $CASE RESULTS ==="
  for ARM in base sys; do
    /scratch/yuzhou/projects/sglang/.venv/bin/python -c "
import json, statistics
tps_list, hit_list = [], []
for rep in range(1, 4):
    f = '$OUTDIR/${ARM}_r' + str(rep) + '.json'
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

echo "=== ALL OFFICIAL RUNS COMPLETE ==="
