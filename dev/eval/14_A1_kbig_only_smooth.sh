#!/bin/bash
# Ablation A1 — Layer 1 sub-features on smooth GSP workload.
#
# We already have on smooth GSP:
#   no-Layer-1 baseline      : Q3.D recency arm (351.7 ms mean TTFT)
#   HPB-only Layer 1         : Q3.D hpb arm (282.2 ms, -19.77%)
#   full Layer 1 (HPB+K_big) : Setting 3.A layer1 arm (328.8 ms, +16% vs baseline)
#   A2 K_big sweep at 8192   : 321 ms (+14% vs no K_big)
#
# Missing piece: K_big-only Layer 1 = K_big=8192 + recency LRU. This
# script measures it. Together with the above the A1 "Layer 1 sub-
# features" decomposition is complete.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/a1_kbig_only_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

# K_big=8192 + recency LRU (no HPB).
extra_env="SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
log="$OUT_DIR/kbig_only_server.log"
bench_out="$OUT_DIR/kbig_only_bench.json"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

nohup env $extra_env \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
pid=$!
echo "[kbig_only] pid=$pid"

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[kbig_only] ready after ${waited}s"
    break
  fi
done

echo "[kbig_only] running GSP bench..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 8 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 2 \
  --output-file "$bench_out" \
  >"$OUT_DIR/kbig_only_bench.log" 2>&1 || echo "kbig_only bench failed"
echo "[kbig_only] bench done"

hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
total=$(grep -c "Prefill batch" "$log" || true)
echo "[kbig_only] prefill batches: $total, with cached-token > 0: $hit"

kill -9 $pid 2>/dev/null || true
sleep 5

.venv/bin/python <<PY
import json
out = "$OUT_DIR"
p = f"{out}/kbig_only_bench.json"
with open(p) as f:
    lines = [l for l in f if l.strip()]
d = json.loads(lines[-1]) if lines else {}
print(f"\n=== A1 K_big-only smooth GSP ===")
print(f"  input TPS    : {d.get('input_throughput',0):.1f}")
print(f"  mean TTFT    : {d.get('mean_ttft_ms',0):.1f} ms")
print(f"  P99 TTFT     : {d.get('p99_ttft_ms',0):.1f} ms")
print(f"  median E2E   : {d.get('median_e2e_latency_ms',0):.1f} ms")
PY
