#!/bin/bash
# Q3.A 4th arm — engine default + host-DRAM tier (HiMambaRadixCache).
#
# Setting 3.A v2 measured 3 cache configs on GSP:
#   default       — MambaRadixCache, no host tier
#   extra_buffer  — page_size=8192
#   layer1        — HPB + K_big=8192
#
# This script adds the missing 4th arm: default MambaRadixCache WITH the
# hierarchical host-DRAM tier (HiMambaRadixCache) enabled. The paper §6.3
# Q3.A claims the host-tier-cost-dominated slope is the failure mode that
# Layer 1 must beat. We need this comparison data.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
OUT_DIR=/tmp/q3a_host_tier_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

log="$OUT_DIR/host_tier_server.log"
bench_out="$OUT_DIR/host_tier_bench.json"

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

# HiMambaRadixCache enabled via --enable-hierarchical-cache.
nohup .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.8 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    >"$log" 2>&1 &
pid=$!
echo "[host_tier] pid=$pid"

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[host_tier] ready after ${waited}s"
    break
  fi
done

echo "[host_tier] running GSP bench..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 8 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 2 \
  --output-file "$bench_out" \
  >"$OUT_DIR/host_tier_bench.log" 2>&1 || echo "[host_tier] bench failed"
echo "[host_tier] bench done"

hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
total=$(grep -c "Prefill batch" "$log" || true)
echo "[host_tier] prefill batches: $total, with cached-token > 0: $hit"

kill -9 $pid 2>/dev/null || true
sleep 5

.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
p = f"{out}/host_tier_bench.json"
if not os.path.exists(p):
    print("FAILED — no bench output")
else:
    with open(p) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1]) if lines else {}
    print(f"\n=== Q3.A host-tier-on (HiMambaRadixCache) ===")
    print(f"  input TPS    : {d.get('input_throughput',0):.1f}")
    print(f"  mean TTFT    : {d.get('mean_ttft_ms',0):.1f} ms")
    print(f"  P99 TTFT     : {d.get('p99_ttft_ms',0):.1f} ms")
    print(f"  median E2E   : {d.get('median_e2e_latency_ms',0):.1f} ms")
PY
