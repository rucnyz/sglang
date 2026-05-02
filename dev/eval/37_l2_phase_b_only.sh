#!/bin/bash
# Phase B only: long-horizon agent / KV-bound. Goal is to trigger
# mamba_to_kv. Workload: high concurrency × long context, so KV
# saturates while each req holds only one mamba slot.
#
# KV pool ~879K pages, mamba ~384 slots on Qwen3.5-35B-A3B / H200.
# Target: ~80 concurrent reqs × ~6K tokens = 480K KV tokens (>54%
# of KV pool, clears KV_HIGH=0.5) while mamba uses 80/384 = 21%
# (well below MAMBA_HIGH=0.5).

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT="${PORT:-31000}"
OUT_DIR="dev/eval/runs/l2-phase-b-only-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"
echo "[l2-phase-b-only] out=$OUT_DIR gpu=$GPU"

extra_env=""
extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_K_BIG_AUTO_THRESHOLD=0.5 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=0 SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))"
extra_env="$extra_env SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0"
extra_env="$extra_env SGLANG_BUDGETER_LOG=$OUT_DIR/budgeter.jsonl"
# Asymmetric thresholds: KV_HIGH lower (0.4) so the long-horizon
# workload's actual peak KV usage triggers; MAMBA_HIGH high (0.5)
# so the few-mamba-slots-in-use state stays in_band.
extra_env="$extra_env SGLANG_XPOOL_KV_HIGH=0.4 SGLANG_XPOOL_KV_LOW=0.05 SGLANG_XPOOL_MAMBA_HIGH=0.5 SGLANG_XPOOL_MAMBA_LOW=0.05 SGLANG_XPOOL_COOLDOWN=2"

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

# Long-horizon agent regime: 4K input + 4K output per req, RPS=8,
# 100 prompts. At RPS=8 with output ~30s decode, peak concurrency
# reaches ~80 reqs × 8K total tokens each = ~640K KV tokens.
echo "Phase B (long-horizon / KV-bound) ~6 min..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
  --dataset-name random \
  --random-input-len 4096 --random-output-len 4096 \
  --random-range-ratio 1.0 \
  --num-prompts 100 \
  --request-rate 8 \
  --output-file "$OUT_DIR/bench.json" \
  >"$OUT_DIR/bench.log" 2>&1 || echo "bench failed (non-fatal)"

echo "===== budgeter summary:"
python3 - <<PY
import json
fp = "$OUT_DIR/budgeter.jsonl"
fires_k2m, fires_m2k = [], []
peak_kv = peak_m = 0
with open(fp) as f:
    for ln in f:
        try: d = json.loads(ln)
        except: continue
        peak_kv = max(peak_kv, d.get("xpool_plan_usage_kv_inst", 0))
        peak_m  = max(peak_m,  d.get("xpool_plan_usage_mamba_inst", 0))
        dirn = d.get("xpool_direction")
        if dirn == "kv_to_mamba": fires_k2m.append(d)
        elif dirn == "mamba_to_kv": fires_m2k.append(d)

def summary(label, fires):
    if not fires:
        print(f"  {label}: 0 fires"); return 0
    movers = [f for f in fires if f.get("xpool_unmapped_total",0) > 0
              or f.get("xpool_granted_total",0) > 0]
    print(f"  {label}: {len(fires)} fires ({len(movers)} with movement)")
    for f in fires[:5]:
        u = f.get("xpool_unmapped_total",0); g = f.get("xpool_granted_total",0)
        s = f.get("xpool_skipped","-"); e = f.get("xpool_edge_active",False)
        r = f.get("xpool_plan_reason","")[:130]
        print(f"    tick={f['tick']} edge={e} unmapped={u} granted={g} skipped={s}")
        print(f"      reason={r}")
    return len(movers)

print(f"  peak usage: kv={peak_kv:.2f}, mamba={peak_m:.2f}")
m_k2m = summary("kv_to_mamba", fires_k2m)
m_m2k = summary("mamba_to_kv", fires_m2k)

print()
if m_m2k > 0:
    print("MAMBA_TO_KV VERIFIED: at least one fire moved chunks in mamba→kv direction.")
    print("Combined with prior smoke v6 (kv_to_mamba w/ movement),")
    print("paper §74 'fires bidirectionally' is supported by ≥1 fire each direction.")
elif fires_m2k:
    print(f"PARTIAL: {len(fires_m2k)} mamba_to_kv fires triggered but all drain_pending.")
    print("Workload is correctly KV-bound but high-concurrency phase has no")
    print("free mamba slots above new cap (opportunistic drain working as designed).")
else:
    print("NEGATIVE: no mamba_to_kv fires.")
    print(f"  KV peak {peak_kv:.2f} (need >0.4 to cross KV_HIGH).")
    print(f"  Mamba peak {peak_m:.2f} (need <0.5 to keep in_band).")
PY

kill -9 $pid 2>/dev/null || true
sleep 4
echo "done"
