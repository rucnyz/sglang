#!/bin/bash
# Root-cause the case2 default-split regression: sys is ~6% slower than base with
# 0 cross-pool fires (same work), so it is pure machinery overhead. Additively
# toggle ONE component per arm on the SAME swarm, measure tps/makespan, attribute
# the 6%. Default split (NO RATIO). GPU 7 / port 30097, fresh boot per arm.
set -u
cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case2/data/cc_qwen_case2_swarm.jsonl
OUT=reproduce/RQ1/ablations/ablate_case2; mkdir -p "$OUT"
PORT=30097; GPU=7; MODEL=Qwen/Qwen3.5-9B
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
AR=/scratch/yuzhou/projects/agentreplay
COMMON="--model-path $MODEL --host 127.0.0.1 --port $PORT --tp 1 --reasoning-parser qwen3 \
 --enforce-piecewise-cuda-graph --mem-fraction-static 0.45 --enable-cache-report \
 --context-length 262144 --log-level info"
CSIGMA="SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 SGLANG_CSIGMA_KV_BETA=0.024570739655696554 SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0"
BUD="SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_LPB_WINDOW_S=120.0 $CSIGMA"
ADM="SGLANG_HIMA=1 SGLANG_XPOOL_QUEUE_WAIT_US=125000 SGLANG_XPOOL_COOLDOWN_S=1.0"

run_arm() {  # <name> <evict> <env...>
  local name=$1 evict=$2; shift 2; local env="$*"
  ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1 | xargs -r kill -9 2>/dev/null
  for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done; sleep 2
  echo "[$name] boot evict=$evict env=[$env]"
  env $env CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $VENV -m sglang.launch_server $COMMON --radix-eviction-policy $evict \
    > "$OUT/server_${name}.log" 2>&1 &
  local svpid=$!
  local ready=0
  for i in $(seq 1 200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && { ready=1; break; }
    kill -0 $svpid 2>/dev/null || { echo "[$name] DIED"; tail -15 "$OUT/server_${name}.log"; return 1; }
    sleep 5
  done
  [ "$ready" = 1 ] || { echo "[$name] BOOT TIMEOUT"; kill -9 $svpid 2>/dev/null; return 2; }
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay --trace "$TRACE" \
    --stagger 0.02 --gap-scale 0 --max-concurrency 256 --flush \
    --url http://127.0.0.1:$PORT/generate --label "$name" --out "$OUT/${name}.json" \
    > "$OUT/${name}.replay.log" 2>&1
  echo "[$name] $(grep -oE '\"throughput_tok_s\": [0-9.]+|\"makespan_s\": [0-9.]+|\"cache_hit\": [0-9.]+|\"n_error\": [0-9]+' "$OUT/${name}.json" | tr '\n' ' ')"
  kill -9 $svpid 2>/dev/null; sleep 2
}

run_arm base          lru
run_arm lpb_only      lpb
run_arm bud_lru       lru $BUD
run_arm budadm_lru    lru $BUD $ADM
run_arm budadmfac_lru lru $BUD $ADM SGLANG_ADMISSION_MAX_FACTOR=4.0
run_arm full          lpb $BUD $ADM SGLANG_ADMISSION_MAX_FACTOR=4.0
echo "=== ABLATION DONE -> $OUT ==="
