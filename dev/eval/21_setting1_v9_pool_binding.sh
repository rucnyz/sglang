#!/bin/bash
# Setting 1 v9 — phase-binding trace redesigned to actually bind on
# different pools.
#
# v6/v7/v8 used three short-prompt phases that all bound on the mamba
# pool, producing null cell-vs-cell differentiation. v9 designs phases
# that genuinely bind on different pools within a single Qwen3.5-35B-A3B
# server:
#
#   Phase A (mamba-bound) : GSP shared-prefix, 16 groups × 10 prompts,
#                           12K system prompt, RPS=8 → many mamba
#                           snapshots; KV stays modest.
#   Phase B (KV-bound)    : random 8192-token prompts, RPS=4 → fills
#                           KV pool; one mamba snapshot per prompt.
#   Phase C (mixed)       : random 4096-token prompts, RPS=8 → moderate
#                           pressure on both pools.
#
# Run as 4-cell ablation: (L1, L2) ∈ {(0,0), (1,0), (0,1), (1,1)}.
# Single instance of the script handles ONE cell at a time; outer
# driver fans 4 cells across GPUs.
#
# Use:
#   ONLY_L1=$x ONLY_L2=$y CUDA_VISIBLE_DEVICES=$g PORT=$p OUT_DIR=$d \
#     dev/eval/21_setting1_v9_pool_binding.sh

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-300}
ONLY_L1=${ONLY_L1:?missing}
ONLY_L2=${ONLY_L2:?missing}
OUT_DIR=${OUT_DIR:?missing}
mkdir -p "$OUT_DIR"

cell="L1${ONLY_L1}_L2${ONLY_L2}"
extra_env=""
if [ "$ONLY_L1" = "1" ]; then
  extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
  # K_BIG_AUTO_THRESHOLD: when set, K_BIG only activates if mamba_usage
  # >= threshold. Fixes the Phase A regression observed in v9 first run
  # (K_BIG hurts on prefix-friendly mamba-bound workloads where mamba
  # isn't saturated).
  if [ -n "${SGLANG_K_BIG_AUTO_THRESHOLD:-}" ]; then
    extra_env="$extra_env SGLANG_K_BIG_AUTO_THRESHOLD=$SGLANG_K_BIG_AUTO_THRESHOLD"
  fi
fi
if [ "$ONLY_L2" = "1" ]; then
  # Chunk size: 256 MiB default. 1 GiB was too coarse — at the
  # paper's per-pool budget per sub-pool (~780 MB on Qwen3.5-35B-A3B
  # KV at mem-fraction-static=0.7), 1 GiB chunks make init_chunks=
  # static_min=1 per sub-pool and the actuator has no shrink headroom
  # (every fire skipped src_at_static_min). 256 MiB gives each KV
  # sub-pool ~3 init chunks, enough for the budgeter to actually
  # shrink under the drain-protocol invariants.
  chunk_bytes=${SGLANG_ARENA_CHUNK_BYTES:-$((256*1024*1024))}
  # Threshold defaults match the original v9 binding-shift trace; can be
  # overridden from the wrapper's environment (e.g., to test the
  # paper-faithful symmetric KV_HIGH=MAMBA_HIGH=0.5 regime).
  kv_hi=${SGLANG_XPOOL_KV_HIGH:-0.04}
  kv_lo=${SGLANG_XPOOL_KV_LOW:-0.015}
  m_hi=${SGLANG_XPOOL_MAMBA_HIGH:-0.08}
  m_lo=${SGLANG_XPOOL_MAMBA_LOW:-0.03}
  extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=$chunk_bytes SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0 SGLANG_BUDGETER_LOG=$OUT_DIR/${cell}_budgeter.jsonl SGLANG_XPOOL_KV_HIGH=$kv_hi SGLANG_XPOOL_KV_LOW=$kv_lo SGLANG_XPOOL_MAMBA_HIGH=$m_hi SGLANG_XPOOL_MAMBA_LOW=$m_lo SGLANG_XPOOL_COOLDOWN=2"
  # Paper §design-l2: when SGLANG_ARENA_SHARED=1, the engine boots
  # both pools at full init capacity and sets static_min=1 chunk per
  # sub-pool (the actuator floor below which drain protocol won't
  # shrink). No MOBILE_SOFT env vars needed — earlier "donate at boot"
  # workaround was replaced by drain protocol (sglang commit landing
  # in this same series).
fi
log="$OUT_DIR/${cell}_server.log"
echo "=== cell=$cell ($extra_env) ==="

pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 6

# Apples-to-apples: keep mem-fraction-static identical across cells
# so the L1-only / L1+L2 comparison isn't confounded by a 14 GB
# difference in KV pool budget. Under the new opportunistic-drain
# path the arena reserves VA only (no extra HBM), so L2 cells run
# fine at 0.8 too. Override with MEM_FRAC if needed.
mem_frac="${MEM_FRAC:-0.8}"

nohup env $extra_env \
  .venv/bin/python -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
    --mem-fraction-static $mem_frac --log-level info \
    --enforce-piecewise-cuda-graph \
    --reasoning-parser qwen3 \
    ${EXTRA_FLAGS:-} \
    >"$log" 2>&1 &
pid=$!
echo "[$cell] pid=$pid"

waited=0
while [ $waited -lt $WARMUP_S ]; do
  sleep 10
  waited=$((waited + 10))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
    echo "[$cell] ready after ${waited}s"
    break
  fi
done

# Phase A: mamba-bound (GSP shared-prefix, many small snapshots).
echo "[$cell] Phase A (mamba-bound GSP)..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 16 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 12000 --gsp-question-len 64 \
  --gsp-output-len 256 \
  --request-rate 8 \
  --output-file "$OUT_DIR/${cell}_phase_A_bench.json" \
  >"$OUT_DIR/${cell}_phase_A_bench.log" 2>&1 || echo "[$cell] phase A failed"
echo "[$cell] Phase A done"
sleep 30

# Phase B: KV-bound (random LONG prompts, deep KV but shallow mamba).
echo "[$cell] Phase B (KV-bound random 8K)..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 200 \
  --random-input-len 8192 --random-output-len 64 \
  --request-rate 4 \
  --output-file "$OUT_DIR/${cell}_phase_B_bench.json" \
  >"$OUT_DIR/${cell}_phase_B_bench.log" 2>&1 || echo "[$cell] phase B failed"
echo "[$cell] Phase B done"
sleep 30

# Phase C: mixed (random 4K prompts, moderate pressure on both pools).
echo "[$cell] Phase C (mixed random 4K)..."
.venv/bin/python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --tokenizer "$MODEL" \
  --dataset-name random --num-prompts 400 \
  --random-input-len 4096 --random-output-len 128 \
  --request-rate 8 \
  --output-file "$OUT_DIR/${cell}_phase_C_bench.json" \
  >"$OUT_DIR/${cell}_phase_C_bench.log" 2>&1 || echo "[$cell] phase C failed"
echo "[$cell] Phase C done"

# Cell-end stats from the budgeter log.
if [ "$ONLY_L2" = "1" ]; then
  total=$(grep -c '"xpool_direction":' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
  echo "[$cell] xpool transfers: total=$total kv→mamba=$k2m mamba→kv=$m2k"
fi
hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
pftot=$(grep -c "Prefill batch" "$log" || true)
echo "[$cell] prefill batches: $pftot, with cached-token > 0: $hit"

kill -9 $pid 2>/dev/null || true
sleep 6
