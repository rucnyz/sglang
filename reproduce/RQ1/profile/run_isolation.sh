#!/bin/bash
set -eu
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
AR=/scratch/yuzhou/projects/agentreplay
TRACE=$AR/data/traces/cc_qwen_t6_v2.jsonl
MODEL=Qwen/Qwen3.5-9B
PORT=30097; GPU=7

run_arm() {
  local LABEL=$1 EVICT=$2 HIMA=$3
  local OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/profile/${LABEL}
  mkdir -p "$OUTDIR"

  pkill -9 -f "sglang.launch_server.*$PORT" 2>/dev/null || true
  sleep 3

  local EXTRA=""
  if [ "$HIMA" = "1" ]; then
    export SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_LPB_WINDOW_S=120.0 \
           SGLANG_XPOOL_QUEUE_WAIT_US=100 SGLANG_XPOOL_COOLDOWN_S=1.0 \
           SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 \
           SGLANG_CSIGMA_KV_BETA=0.024570739655696554 \
           SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 \
           SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0
  else
    unset SGLANG_HIMA SGLANG_HIMA_TICK_S SGLANG_LPB_WINDOW_S \
          SGLANG_XPOOL_QUEUE_WAIT_US SGLANG_XPOOL_COOLDOWN_S \
          SGLANG_CSIGMA_KV_ALPHA SGLANG_CSIGMA_KV_BETA SGLANG_CSIGMA_KV_GAMMA \
          SGLANG_CSIGMA_M_ALPHA SGLANG_CSIGMA_M_BETA SGLANG_CSIGMA_L_STAR 2>/dev/null || true
  fi

  echo "[$LABEL] Starting server (evict=$EVICT, hima=$HIMA)"
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $VENV -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port $PORT \
    --reasoning-parser qwen3 --mamba-scheduler-strategy extra_buffer \
    --enable-cache-report --log-level info \
    --radix-eviction-policy $EVICT \
    > "$OUTDIR/server.log" 2>&1 &
  local SVPID=$!

  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    [ "$code" = "200" ] && break
    kill -0 $SVPID 2>/dev/null || { echo "[$LABEL] DIED"; tail -5 "$OUTDIR/server.log"; return 1; }
    sleep 5
  done

  echo "[$LABEL] Replaying (conc=256, full trace)"
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay \
    --trace "$TRACE" --stagger 0.5 --max-concurrency 256 --flush \
    --url "http://127.0.0.1:$PORT/generate" --label "$LABEL" \
    --out "$OUTDIR/result.json" > "$OUTDIR/replay.log" 2>&1

  echo "[$LABEL] $(grep -oE '"throughput_tok_s": [0-9.]+|"cache_hit": [0-9.]+' "$OUTDIR/result.json" | tr '\n' ' ')"
  kill -9 $SVPID 2>/dev/null || true
  sleep 3
}

# A: LRU, no HIMA (baseline)
run_arm "A_lru_nohima" "lru" "0"

# B: LPB, no HIMA (isolate LPB overhead)
run_arm "B_lpb_nohima" "lpb" "0"

# C: LRU, HIMA on (isolate HIMA overhead)
run_arm "C_lru_hima" "lru" "1"

# D: LPB + HIMA (full sys)
run_arm "D_lpb_hima" "lpb" "1"

echo ""
echo "=== ISOLATION MATRIX ==="
for d in A_lru_nohima B_lpb_nohima C_lru_hima D_lpb_hima; do
  echo -n "$d: "
  grep -oE '"throughput_tok_s": [0-9.]+|"cache_hit": [0-9.]+' \
    /scratch/yuzhou/projects/sglang/reproduce/RQ1/profile/$d/result.json 2>/dev/null | tr '\n' ' '
  echo
done
