#!/bin/bash
# Focused L10_L21-only smoke: minimum viable test for mobile-soft + MemPool path.
# Spins up ONE server with KV mobile_soft=1, mamba=0, FROM_BLOB=0,
# runs Phase A (mamba-bound) for ~5 min, checks budgeter for granted>0.
#
# ~7 min total (110s warmup + 5min bench + cleanup). Beats 4-cell smoke.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT="${PORT:-31000}"
OUT_DIR="dev/eval/runs/l2-mobile-soft-focused-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"
echo "[l2-mobile-soft-focused] out=$OUT_DIR gpu=$GPU"

extra_env=""
extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_K_BIG_AUTO_THRESHOLD=0.5 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=0 SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))"
extra_env="$extra_env SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0"
extra_env="$extra_env SGLANG_BUDGETER_LOG=$OUT_DIR/budgeter.jsonl"
extra_env="$extra_env SGLANG_XPOOL_KV_HIGH=0.5 SGLANG_XPOOL_KV_LOW=0.05 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.03 SGLANG_XPOOL_COOLDOWN=2"
extra_env="$extra_env SGLANG_XPOOL_EDGE_TRIGGER=1"
extra_env="$extra_env SGLANG_ARENA_KV_MOBILE_SOFT_CHUNKS=1 SGLANG_ARENA_MAMBA_MOBILE_SOFT_CHUNKS=0"

log="$OUT_DIR/server.log"
pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 4

CUDA_VISIBLE_DEVICES=$GPU nohup env $extra_env \
  .venv/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B --host 127.0.0.1 --port $PORT \
    --mem-fraction-static 0.7 --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    >"$log" 2>&1 &
pid=$!
echo "pid=$pid"

waited=0
while [ $waited -lt 240 ]; do
  sleep 10; waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "ready after ${waited}s"; break
  fi
done
if [ $waited -ge 240 ]; then
  echo "FAILED to come up — log tail:"
  tail -25 "$log"
  kill -9 $pid 2>/dev/null || true
  exit 1
fi

# Phase A only — mamba-bound GSP, the workload most likely to trigger
# kv_to_mamba fires.
echo "Phase A (mamba-bound GSP) ~5 min..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 16 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 8 \
  --output-file "$OUT_DIR/bench.json" \
  >"$OUT_DIR/bench.log" 2>&1 || echo "bench failed (non-fatal for this smoke)"

echo "===== budgeter summary:"
python3 - <<PY
import json, os
fp = "$OUT_DIR/budgeter.jsonl"
fires, k2m, m2k, granted, unmapped = 0, 0, 0, 0, 0
fires_with_movement = 0
sample = None
with open(fp) as f:
    for ln in f:
        try: d = json.loads(ln)
        except: continue
        dirn = d.get("xpool_direction")
        if dirn in ("kv_to_mamba", "mamba_to_kv"):
            fires += 1
            if dirn == "kv_to_mamba": k2m += 1
            else: m2k += 1
            g = d.get("xpool_granted_total", 0)
            u = d.get("xpool_unmapped_total", 0)
            granted += g
            unmapped += u
            if g > 0 or u > 0:
                fires_with_movement += 1
                if sample is None:
                    sample = d
print(f"Total fires: {fires} (k2m={k2m}, m2k={m2k})")
print(f"Total chunks granted: {granted}")
print(f"Total chunks unmapped: {unmapped}")
print(f"Fires with non-zero movement: {fires_with_movement} / {fires}")
if sample:
    print()
    print("FIRST FIRE WITH REAL MOVEMENT:")
    print(f"  direction: {sample['xpool_direction']}")
    print(f"  unmapped_total: {sample.get('xpool_unmapped_total')}")
    print(f"  granted_total: {sample.get('xpool_granted_total')}")
    print(f"  kv_capacity_tokens: {sample.get('xpool_kv_capacity_tokens')}")
    print(f"  mamba_capacity_tokens: {sample.get('xpool_mamba_capacity_tokens')}")
    print()
    print("BREAKTHROUGH: L2 actuator now physically moves chunks.")
elif fires > 0:
    print()
    print("NEGATIVE: fires happened but all unmapped/granted=0. Mobile-soft not effective.")
else:
    print()
    print("NO L2 FIRES at all. Gate suppressed all decisions.")
PY

kill -9 $pid 2>/dev/null || true
sleep 4
echo "done"
