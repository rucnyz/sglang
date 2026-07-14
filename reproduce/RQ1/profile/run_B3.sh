#!/bin/bash
set -eu
OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/profile/B3_lpb_nohima_csigma
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"

pkill -9 -f "sglang.launch_server.*30097" 2>/dev/null || true; sleep 3

# LPB needs cost curves for eviction_priority(). Set CSIGMA but NOT HIMA.
export SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 \
       SGLANG_CSIGMA_KV_BETA=0.024570739655696554 \
       SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 \
       SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
AR=/scratch/yuzhou/projects/agentreplay

cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  $VENV -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B --host 127.0.0.1 --port 30097 \
  --reasoning-parser qwen3 --mamba-scheduler-strategy extra_buffer \
  --enable-cache-report --log-level info \
  --radix-eviction-policy lpb \
  > "$OUTDIR/server.log" 2>&1 &
SVPID=$!

for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:30097/health" 2>/dev/null || true)
  [ "$code" = "200" ] && break
  kill -0 $SVPID 2>/dev/null || { echo "DIED"; tail -10 "$OUTDIR/server.log"; exit 1; }
  sleep 5
done

TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay \
  --trace $AR/data/traces/cc_qwen_t6_v2.jsonl \
  --stagger 0.5 --max-concurrency 256 --flush \
  --url "http://127.0.0.1:30097/generate" --label "B3" \
  --out "$OUTDIR/result.json" > "$OUTDIR/replay.log" 2>&1

kill -9 $SVPID 2>/dev/null

python3 -c "
import json; d=json.load(open('$OUTDIR/result.json'))
print(f'B3 (LPB, no HIMA, +CSIGMA):')
print(f'  tps={d[\"throughput_tok_s\"]:.1f} hit={d[\"cache_hit\"]:.4f}')
print(f'  prompt={d[\"total_prompt_tokens\"]:,} out={d[\"total_out_tokens\"]:,}')
print(f'  n_ok={d[\"n_ok\"]} n_err={d[\"n_error\"]}')
print()
print('=== FULL COMPARISON ===')
print('A  (LRU, no HIMA):        tps=458.2 hit=0.5947 prompt=37,015,694')
print('B3 (LPB, no HIMA+CSIGMA): see above')
print('C  (LRU, HIMA):           tps=471.3 hit=0.6141 prompt=37,015,694')
print('D  (LPB, HIMA, 120s):     tps=471.3 hit=0.6139 prompt=37,015,694')
print('E  (LPB, HIMA, 60s):      tps=484.6 hit=0.6302 prompt=37,015,694')
" 2>/dev/null
