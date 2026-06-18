#!/bin/bash
# Isolate Budgeter-tick vs arena-per-step in the ~3.7% machinery regression.
# budadm_lru at TICK_S=10 (10x fewer ticks) vs TICK_S=1 (=6306) vs base (=6547).
# recover->tick is the cost; flat->arena per-step is the cost. N=3.
set -u; cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case2/data/cc_qwen_case2_swarm.jsonl; OUT=reproduce/RQ1/ablations/ablate_tick; mkdir -p "$OUT"
PORT=30097; GPU=7; VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python; AR=/scratch/yuzhou/projects/agentreplay
COMMON="--model-path Qwen/Qwen3.5-9B --host 127.0.0.1 --port $PORT --tp 1 --reasoning-parser qwen3 --enforce-piecewise-cuda-graph --mem-fraction-static 0.45 --enable-cache-report --context-length 262144 --log-level info"
CSIGMA="SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 SGLANG_CSIGMA_KV_BETA=0.024570739655696554 SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0"
ENV="SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=10.0 SGLANG_LPB_WINDOW_S=120.0 $CSIGMA SGLANG_HIMA=1 SGLANG_XPOOL_QUEUE_WAIT_US=125000 SGLANG_XPOOL_COOLDOWN_S=1.0"
for rep in 1 2 3; do
  ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1 | xargs -r kill -9 2>/dev/null
  for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done; sleep 2
  env $ENV CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $VENV -m sglang.launch_server $COMMON --radix-eviction-policy lru > "$OUT/tick10_r${rep}.server.log" 2>&1 &
  sv=$!; for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break; kill -0 $sv 2>/dev/null || break; sleep 5; done
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay --trace "$TRACE" --stagger 0.02 --gap-scale 0 --max-concurrency 256 --flush --url http://127.0.0.1:$PORT/generate --label tick10_r$rep --out "$OUT/tick10_r${rep}.json" > "$OUT/tick10_r${rep}.replay.log" 2>&1
  echo "[tick10 r$rep] $(grep -oE '\"throughput_tok_s\": [0-9.]+' "$OUT/tick10_r${rep}.json")"; kill -9 $sv 2>/dev/null; sleep 2
done
$VENV -c "import json,glob,statistics as st; v=[json.load(open(f))['throughput_tok_s'] for f in sorted(glob.glob('$OUT/tick10_r*.json'))]; print('tick10 mean=%.1f std=%.1f reps=%s (vs budadm 6306, base 6547)'%(st.mean(v),st.pstdev(v) if len(v)>1 else 0,[round(x) for x in v]))"
