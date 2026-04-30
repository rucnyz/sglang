#!/bin/bash
# Ablation A3: Layer 2 hysteresis sweep (Δ_hyst).
#
# Paper §6.7: "Sweep Δ_hyst ∈ {0%, 1%, 5%, 10%, 20%}. Expectation:
# 0% thrashes, 20% lags; ours uses 5% as default."
#
# Mechanism: when the LagrangePlanner emits a target capacity that
# differs from current by < Δ_hyst, suppress the change. Without this,
# the planner thrashes when pressure signals oscillate near the
# threshold.
#
# Workload: re-runs the headline trace (Phase 1+2+3 from
# 25_xpool_planner_trace.sh) for each Δ_hyst value. Reports number of
# transfers fired and number of "thrash reversals" (transfer in one
# direction immediately followed by the opposite direction).

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
OUT_DIR=/tmp/a3_hyst_$$
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR"

# Hysteresis is implemented via SGLANG_XPOOL_KV_HIGH/_LOW gap. We
# parameterize it as a fraction "watermark_gap" — Δ_hyst = (high-low)/2.
# kv_high = (default 0.04) * (1 + Δ_hyst); kv_low = (default 0.04) * (1 - Δ_hyst).

run_point() {
  local hyst="$1"
  local label="hyst${hyst}"
  local log="$OUT_DIR/${label}_server.log"
  local jsonl="$OUT_DIR/${label}_budgeter.jsonl"
  echo "=== Δ_hyst=$hyst ==="

  # Compute high/low gap.
  local kv_high=$(.venv/bin/python -c "print(round(0.04 * (1 + $hyst), 4))")
  local kv_low=$(.venv/bin/python -c "print(round(0.04 * (1 - $hyst), 4))")
  local mamba_high=$(.venv/bin/python -c "print(round(0.08 * (1 + $hyst), 4))")
  local mamba_low=$(.venv/bin/python -c "print(round(0.08 * (1 - $hyst), 4))")
  echo "[hyst=$hyst] kv: $kv_low ↔ $kv_high, mamba: $mamba_low ↔ $mamba_high"

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env \
    SGLANG_ARENA_SHARED=1 \
    SGLANG_ARENA_FROM_BLOB=1 \
    SGLANG_BUDGETER=1 \
    SGLANG_BUDGETER_XPOOL_PLANNER=1 \
    SGLANG_BUDGETER_XPOOL_COORDINATED=1 \
    SGLANG_BUDGETER_TICK_S=0.5 \
    SGLANG_BUDGETER_LOG="$jsonl" \
    SGLANG_XPOOL_KV_HIGH=$kv_high \
    SGLANG_XPOOL_KV_LOW=$kv_low \
    SGLANG_XPOOL_MAMBA_HIGH=$mamba_high \
    SGLANG_XPOOL_MAMBA_LOW=$mamba_low \
    SGLANG_XPOOL_COOLDOWN=2 \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[hyst=$hyst] ready after ${waited}s"
      break
    fi
  done

  # Run a smaller version of the headline trace.
  .venv/bin/python <<PY
import json, urllib.request, time, threading
LONG_BASE = "Compute step by step. " * 250
def fire(prompt):
    data = json.dumps({'model': '$MODEL', 'prompt': prompt, 'max_tokens': 64, 'temperature': 0}).encode()
    req = urllib.request.Request('http://127.0.0.1:$PORT/v1/completions',
        data=data, headers={'Content-Type': 'application/json'})
    try: urllib.request.urlopen(req, timeout=300).read()
    except: pass

# Phase 1: 30 concurrent long-context.
threads = []
for i in range(30):
    t = threading.Thread(target=fire, args=(LONG_BASE + f' Q{i}: name a fruit:',), daemon=True)
    t.start(); threads.append(t)
    time.sleep(0.05)
for t in threads: t.join(timeout=300)

time.sleep(8)
# Phase 2: 40 concurrent short.
threads = []
for i in range(40):
    t = threading.Thread(target=fire, args=(f'Q{i}: name a color:',), daemon=True)
    t.start(); threads.append(t)
    time.sleep(0.03)
for t in threads: t.join(timeout=180)

time.sleep(8)
# Phase 3: 4 sequential 32K.
LONG2 = "Compute step by step. " * 5500
for i in range(4):
    fire(LONG2 + f' Q{i}: name a fruit:')
    time.sleep(1)
PY

  sleep 6
  kill -9 $pid 2>/dev/null || true
  sleep 5

  # Analyze: count transfers and thrash reversals.
  local transfers=$(grep -c '"xpool_direction":' "$jsonl" 2>/dev/null || echo 0)
  local k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$jsonl" 2>/dev/null || echo 0)
  local m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$jsonl" 2>/dev/null || echo 0)
  echo "[hyst=$hyst] transfers: total=$transfers, kv_to_mamba=$k2m, mamba_to_kv=$m2k"
}

for hyst in 0 0.01 0.05 0.10 0.20; do
  run_point "$hyst" || echo "[hyst=$hyst] FAILED, continuing"
done

echo
echo "=== A3 hysteresis summary ==="
.venv/bin/python <<PY
import json, glob, os
out = "$OUT_DIR"
print(f"\n{'Δ_hyst':>8} {'total':>7} {'kv→mamba':>10} {'mamba→kv':>10} {'reversals':>11}")
print('-' * 55)
for hyst in ('0', '0.01', '0.05', '0.10', '0.20'):
    j = f"{out}/hyst{hyst}_budgeter.jsonl"
    if not os.path.exists(j):
        print(f"{hyst:>8} {'N/A':>7} {'N/A':>10} {'N/A':>10} {'N/A':>11}")
        continue
    transfers = []
    with open(j) as f:
        for line in f:
            try: d = json.loads(line)
            except: continue
            d_dir = d.get('xpool_direction')
            if d_dir in ('kv_to_mamba', 'mamba_to_kv'):
                transfers.append(d_dir)
    k2m = transfers.count('kv_to_mamba')
    m2k = transfers.count('mamba_to_kv')
    # Reversal: a transfer immediately followed by opposite direction.
    reversals = sum(1 for i in range(1, len(transfers)) if transfers[i] != transfers[i-1])
    print(f"{hyst:>8} {len(transfers):>7} {k2m:>10} {m2k:>10} {reversals:>11}")
print()
print("Expected: 0 → many thrash reversals; 20% → very few transfers (lag).")
PY
