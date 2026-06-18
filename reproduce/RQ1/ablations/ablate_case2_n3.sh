#!/bin/bash
# N=3 root-cause: is SGLANG_ADMISSION_MAX_FACTOR (dynamic-cap mode) a real
# regression or N=1 noise? 3 decisive arms, N=3 each, mean+-std at the end.
#   base          : default split, no machinery (floor)
#   budadm_lru    : budgeter+arena+admitter, factor OFF (1.0) -> isolates machinery
#   budadmfac_lru : + SGLANG_ADMISSION_MAX_FACTOR=4 -> isolates the dynamic-cap pool
# Same 6000-swarm, default split, GPU 7 / port 30097, fresh boot per rep.
set -u
cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case2/data/cc_qwen_case2_swarm.jsonl
OUT=reproduce/RQ1/ablations/ablate_case2_n3; mkdir -p "$OUT"
PORT=30097; GPU=7; MODEL=Qwen/Qwen3.5-9B
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
AR=/scratch/yuzhou/projects/agentreplay
COMMON="--model-path $MODEL --host 127.0.0.1 --port $PORT --tp 1 --reasoning-parser qwen3 \
 --enforce-piecewise-cuda-graph --mem-fraction-static 0.45 --enable-cache-report \
 --context-length 262144 --log-level info"
CSIGMA="SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 SGLANG_CSIGMA_KV_BETA=0.024570739655696554 SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0"
BUD="SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_LPB_WINDOW_S=120.0 $CSIGMA"
ADM="SGLANG_HIMA=1 SGLANG_XPOOL_QUEUE_WAIT_US=125000 SGLANG_XPOOL_COOLDOWN_S=1.0"

run_rep() {  # <name> <rep> <evict> <env...>
  local name=$1 rep=$2 evict=$3; shift 3; local env="$*"
  ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1 | xargs -r kill -9 2>/dev/null
  for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done; sleep 2
  env $env CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $VENV -m sglang.launch_server $COMMON --radix-eviction-policy $evict \
    > "$OUT/server_${name}_r${rep}.log" 2>&1 &
  local svpid=$!
  for i in $(seq 1 200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
    kill -0 $svpid 2>/dev/null || { echo "[$name r$rep] DIED"; return 1; }; sleep 5
  done
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay --trace "$TRACE" \
    --stagger 0.02 --gap-scale 0 --max-concurrency 256 --flush \
    --url http://127.0.0.1:$PORT/generate --label "${name}_r${rep}" --out "$OUT/${name}_r${rep}.json" \
    > "$OUT/${name}_r${rep}.replay.log" 2>&1
  echo "[$name r$rep] $(grep -oE '\"throughput_tok_s\": [0-9.]+' "$OUT/${name}_r${rep}.json")"
  kill -9 $svpid 2>/dev/null; sleep 2
}

for rep in 1 2 3; do
  run_rep base          $rep lru
  run_rep budadm_lru    $rep lru $BUD $ADM
  run_rep budadmfac_lru $rep lru $BUD $ADM SGLANG_ADMISSION_MAX_FACTOR=4.0
done
echo "=== ABLATION N=3 DONE; aggregate ==="
$VENV - <<PY
import json,glob,statistics as st
for n in ("base","budadm_lru","budadmfac_lru"):
    v=[json.load(open(f))["throughput_tok_s"] for f in sorted(glob.glob("$OUT/%s_r*.json"%n))]
    if v: print("%-14s tps mean=%.1f std=%.1f  reps=%s"%(n,st.mean(v),st.pstdev(v) if len(v)>1 else 0,[round(x) for x in v]))
PY
