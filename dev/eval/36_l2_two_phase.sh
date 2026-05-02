#!/bin/bash
# Two-phase smoke: fan-out (mamba-bound) → idle gap → long-horizon (KV-bound).
# Verifies BOTH directions of cross-pool transfer: kv_to_mamba in Phase A and
# mamba_to_kv in Phase B. ~12 min total (110s warmup + ~5min Phase A + 30s
# idle + ~5min Phase B + cleanup).
#
# Paper §74: 'fires bidirectionally as the binding pool flips between
# phases'. This smoke is the minimum reproducer.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT="${PORT:-31000}"
OUT_DIR="dev/eval/runs/l2-two-phase-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"
echo "[l2-two-phase] out=$OUT_DIR gpu=$GPU"

extra_env=""
extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_K_BIG_AUTO_THRESHOLD=0.5 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=0 SGLANG_ARENA_CHUNK_BYTES=$((256*1024*1024))"
extra_env="$extra_env SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0"
extra_env="$extra_env SGLANG_BUDGETER_LOG=$OUT_DIR/budgeter.jsonl"
extra_env="$extra_env SGLANG_XPOOL_KV_HIGH=0.7 SGLANG_XPOOL_KV_LOW=0.1 SGLANG_XPOOL_MAMBA_HIGH=0.5 SGLANG_XPOOL_MAMBA_LOW=0.1 SGLANG_XPOOL_COOLDOWN=2"

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

# ---- Phase A: fan-out agent regime (mamba-bound) ----
# Many concurrent short-context calls — saturates DeltaNet recurrent-slot
# pool while KV stays cold. Triggers kv_to_mamba.
echo "Phase A (fan-out / mamba-bound GSP) ~5 min..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 16 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 8 \
  --output-file "$OUT_DIR/bench_phaseA.json" \
  >"$OUT_DIR/bench_phaseA.log" 2>&1 || echo "Phase A bench failed (non-fatal)"

# ---- Idle gap: let in-flight requests drain so peak-tracker decays ----
echo "Idle gap 30s..."
sleep 30

# ---- Phase B: long-horizon agent regime (KV-bound) ----
# Few concurrent 8K-token prompts — one mamba slot per req but each holds
# many KV tokens. KV pool fills, mamba pool stays low. Triggers
# mamba_to_kv.
echo "Phase B (long-horizon / KV-bound random 8K) ~5 min..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 64 \
  --random-range-ratio 1.0 \
  --num-prompts 80 \
  --request-rate 4 \
  --output-file "$OUT_DIR/bench_phaseB.json" \
  >"$OUT_DIR/bench_phaseB.log" 2>&1 || echo "Phase B bench failed (non-fatal)"

echo "===== budgeter summary:"
python3 - <<PY
import json
fp = "$OUT_DIR/budgeter.jsonl"
phase_starts = []
fires_k2m, fires_m2k = [], []
with open(fp) as f:
    for ln in f:
        try: d = json.loads(ln)
        except: continue
        dirn = d.get("xpool_direction")
        if dirn == "kv_to_mamba":
            fires_k2m.append(d)
        elif dirn == "mamba_to_kv":
            fires_m2k.append(d)

def summary(label, fires):
    if not fires:
        print(f"  {label}: 0 fires")
        return 0
    movers = [f for f in fires if f.get("xpool_unmapped_total",0) > 0
              or f.get("xpool_granted_total",0) > 0]
    print(f"  {label}: {len(fires)} fires ({len(movers)} with movement)")
    for f in fires[:5]:
        u = f.get("xpool_unmapped_total",0); g = f.get("xpool_granted_total",0)
        s = f.get("xpool_skipped","-"); e = f.get("xpool_edge_active",False)
        r = f.get("xpool_plan_reason","")[:120]
        print(f"    tick={f['tick']} edge={e} unmapped={u} granted={g} skipped={s}")
        print(f"      reason={r}")
    return len(movers)

m_k2m = summary("kv_to_mamba", fires_k2m)
m_m2k = summary("mamba_to_kv", fires_m2k)

print()
if m_k2m > 0 and m_m2k > 0:
    print("BIDIRECTIONAL FIRE VERIFIED: both kv_to_mamba and mamba_to_kv")
    print("commit non-zero physical movement under live traffic.")
    print("Paper §74 'fires bidirectionally as the binding pool flips' supported.")
elif m_k2m > 0:
    print("PARTIAL: kv_to_mamba fires moved chunks; mamba_to_kv did not.")
    print("Phase B may not have built enough KV pressure to trigger m2k —")
    print("consider --random-input-len higher or --request-rate higher.")
elif m_m2k > 0:
    print("PARTIAL: mamba_to_kv fires moved chunks; kv_to_mamba did not.")
else:
    print("NEGATIVE: no fires moved chunks. Investigate why.")
PY

kill -9 $pid 2>/dev/null || true
sleep 4
echo "done"
