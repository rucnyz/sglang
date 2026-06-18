#!/bin/bash
# Run baseline (no L1, no L2) on the long-horizon and fan-out workloads.
# A sidecar samples /metrics at 1Hz to capture sglang:token_usage and
# sglang:mamba_usage over time, producing two CSVs we plot afterward.
#
# Output:
#   dev/figures/data/baseline_longhorizon.csv
#   dev/figures/data/baseline_swarm.csv
# (each: t,token_usage,mamba_usage,num_running)
#
# Then run plot_bubble_real.py to render the figure.

set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

PORT=${PORT:-30950}
GPU=${GPU:-2}
MEM_FRAC=${MEM_FRAC:-0.8}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
OUT_DIR=dev/figures/data
mkdir -p "$OUT_DIR"

start_server() {
  local extra="$1"
  local budgeter_log="$2"  # path for budgeter snapshot JSONL
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 4
  # Truncate budgeter log so each run starts fresh — JSONL is otherwise
  # append-only across runs and concatenates stale state from prior benches.
  if [ -n "$budgeter_log" ]; then
    : > "$budgeter_log"
  fi
  # SGLANG_HIMA=1 enables the per-tick snapshot agent. It computes
  # usage_kv_inst / usage_mamba_inst directly as (pool.size - available) /
  # pool.size, INCLUDING radix-tree-cached prefix/snapshots — what we
  # actually want for the "real pool fill" bubble figure. The Prometheus
  # `sglang:full_token_usage` and `sglang:mamba_usage` metrics both
  # SUBTRACT the evictable size from the numerator (admission-pressure
  # framing), so they hide the cache-fill component.
  # We do NOT enable XPOOL_PLANNER or arena — pure observation.
  SGLANG_HIMA=1 SGLANG_HIMA_LOG="$budgeter_log" \
  SGLANG_HIMA_TICK_S=1.0 \
  CUDA_VISIBLE_DEVICES=$GPU \
    nohup .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static $MEM_FRAC --log-level info \
      --enforce-piecewise-cuda-graph --reasoning-parser qwen3 \
      --enable-metrics \
      $extra > /tmp/server_baseline_$PORT.log 2>&1 &
  local pid=$!
  echo "$pid" > /tmp/server_baseline_$PORT.pid
  echo "[bench-baseline] server pid=$pid waiting for /health..."
  local waited=0
  while [ $waited -lt 240 ]; do
    sleep 5; waited=$((waited+5))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ]; then
      echo "[bench-baseline] ready after ${waited}s"; return 0
    fi
  done
  echo "[bench-baseline] server failed to come up; tail:"
  tail -30 /tmp/server_baseline_$PORT.log
  return 1
}

stop_server() {
  local pid=$(cat /tmp/server_baseline_$PORT.pid 2>/dev/null || echo "")
  if [ -n "$pid" ]; then kill -9 "$pid" 2>/dev/null || true; fi
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 4
}

start_sampler() {
  # Background sidecar: 1Hz poll of /metrics, parse token_usage +
  # mamba_usage + num_running_reqs, append to CSV with monotonic-time stamp.
  local out_csv="$1"
  echo "t,token_usage,mamba_usage,num_running" > "$out_csv"
  (
    local t0=$(date +%s.%N)
    while true; do
      local now=$(date +%s.%N)
      local dt=$(echo "$now - $t0" | bc -l 2>/dev/null || python3 -c "print($now - $t0)")
      local m=$(curl -s --max-time 1 http://127.0.0.1:$PORT/metrics 2>/dev/null || true)
      if [ -z "$m" ]; then sleep 1; continue; fi
      # full_token_usage = KV-only (paged-attention pool); token_usage is
      # the max across pools (binding-pool indicator) — wrong for our
      # bubble figure since it would equal mamba_usage when mamba binds.
      local tok=$(echo "$m" | grep -E '^sglang:full_token_usage{' | head -1 | awk '{print $NF}')
      local mam=$(echo "$m" | grep -E '^sglang:mamba_usage{' | head -1 | awk '{print $NF}')
      local nr=$(echo "$m" | grep -E '^sglang:num_running_reqs{' | head -1 | awk '{print $NF}')
      tok=${tok:-0}; mam=${mam:-0}; nr=${nr:-0}
      echo "$dt,$tok,$mam,$nr" >> "$out_csv"
      sleep 1
    done
  ) &
  echo "$!" > /tmp/sampler_$PORT.pid
}

stop_sampler() {
  local pid=$(cat /tmp/sampler_$PORT.pid 2>/dev/null || echo "")
  if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi
}

# ---------- Long-horizon: continuous multi-turn agent via genai-bench ----------
# Each Locust user maintains conversation history across turns; their
# context grows monotonically until SESSION_CAP, simulating a real
# long-horizon agent session. With NUM_CONCURRENCY=14 users at steady
# state, total KV use ≈ 14 × accumulated_context, which keeps KV
# pinned high persistently — NOT the sawtooth pattern of independent
# fresh-prompt benches.
echo "============================="
echo "Long-horizon baseline (multi-turn agent)"
echo "============================="
start_server "" "$OUT_DIR/baseline_longhorizon_budgeter.jsonl"
start_sampler "$OUT_DIR/baseline_longhorizon.csv"
GENAI_BENCH_MT_SESSION_CAP_TOKENS=${SESSION_CAP:-60000} \
  .venv/bin/python -m genai_bench.cli.cli benchmark \
  --api-backend sglang \
  --api-base "http://127.0.0.1:$PORT" \
  --api-key dummy \
  --api-model-name "$MODEL" \
  --model-tokenizer "$MODEL" \
  --task text-to-text-multi-turn \
  --traffic-scenario "${TRAFFIC_SCENARIO:-D(4096,4096)}" \
  --num-concurrency ${NUM_CONCURRENCY:-24} \
  --max-time-per-run ${MAX_TIME_MIN:-8} \
  --max-requests-per-run 1000000 \
  --experiment-folder-name "$OUT_DIR/longhorizon_genai_results" \
  --server-engine SGLang \
  >"$OUT_DIR/longhorizon_bench.log" 2>&1 || echo "[longhorizon] bench failed"
stop_sampler
stop_server

# ---------- Swarm: short multi-turn sub-agents (genai-bench) ----------
# Real agent swarm: an orchestrator dispatches many sub-agents, each
# carrying a short ongoing conversation (2-5 turns of <512 tokens each)
# rather than independent single-shot prompts. With high concurrency
# (each sub-agent occupies one mamba slot), recurrent pool fills; KV
# stays low because per-turn context never grows past a few thousand
# tokens before the session resets.
echo "============================="
echo "Swarm baseline (short multi-turn sub-agents)"
echo "============================="
start_server "" "$OUT_DIR/baseline_swarm_budgeter.jsonl"
start_sampler "$OUT_DIR/baseline_swarm.csv"
GENAI_BENCH_MT_SESSION_CAP_TOKENS=${SWARM_SESSION_CAP:-3000} \
  .venv/bin/python -m genai_bench.cli.cli benchmark \
  --api-backend sglang \
  --api-base "http://127.0.0.1:$PORT" \
  --api-key dummy \
  --api-model-name "$MODEL" \
  --model-tokenizer "$MODEL" \
  --task text-to-text-multi-turn \
  --traffic-scenario "${SWARM_SCENARIO:-D(256,256)}" \
  --num-concurrency ${SWARM_CONCURRENCY:-120} \
  --max-time-per-run ${SWARM_MAX_TIME_MIN:-5} \
  --max-requests-per-run 1000000 \
  --experiment-folder-name "$OUT_DIR/swarm_genai_results" \
  --server-engine SGLang \
  >"$OUT_DIR/swarm_bench.log" 2>&1 || echo "[swarm] bench failed"
stop_sampler
stop_server

echo "============================="
echo "Done. CSVs:"
ls -la "$OUT_DIR"/baseline_*.csv
echo "Now plot via: .venv/bin/python dev/figures/plot_bubble_real.py"
