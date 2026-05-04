#!/bin/bash
# T2 reproduce: A/B placement-bias on vs off, simple throughput parity check.
#
# Limitation noted: this reproduce can only show "T2 doesn't degrade
# serving"; it can't directly observe live-block clustering at pool head
# (no /dump_state endpoint, and per-page free mask isn't exposed until T3).
# The 64-prompt sustained load is too light to push pool fill into the
# regime where placement bias would actually matter; both arms produce
# similar tail/throughput by construction.
#
# Output: dev/T2_allocator_placement_bias/results/ab_summary.txt

set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

ROOT=dev/T2_allocator_placement_bias/results
mkdir -p "$ROOT"
SUMMARY="$ROOT/ab_summary.txt"
: > "$SUMMARY"

GPU=${GPU:-2}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}

run_one() {
  local label="$1"
  local placement="$2"   # 0 or 1
  local port=$((31400 + RANDOM % 100))
  local log="$ROOT/${label}_server.log"

  echo "============================="
  echo "[$label] SGLANG_ALLOCATOR_PLACEMENT_BIAS=$placement"
  echo "============================="
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 4

  SGLANG_ARENA_SHARED=1 \
  SGLANG_ARENA_FROM_BLOB=1 \
  SGLANG_ALLOCATOR_PLACEMENT_BIAS=$placement \
  SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
  CUDA_VISIBLE_DEVICES=$GPU \
    nohup .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $port \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph --reasoning-parser qwen3 \
      > "$log" 2>&1 &
  local pid=$!

  local waited=0
  while [ $waited -lt 600 ]; do
    sleep 5; waited=$((waited+5))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            http://127.0.0.1:$port/health 2>/dev/null)" = "200" ]; then
      break
    fi
  done
  if [ $waited -ge 600 ]; then
    echo "[$label] BOOT FAILED" | tee -a "$SUMMARY"
    tail -25 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  # 32-prompt sequential serve (not parallel — we want per-prompt latency).
  # 256 output tokens each.
  local t0=$(date +%s.%N)
  local successes=0
  for i in $(seq 1 32); do
    local r=$(curl -s --max-time 30 -X POST http://127.0.0.1:$port/generate \
      -H "Content-Type: application/json" \
      -d "{\"text\":\"What is fact $i about LLMs?\",\"sampling_params\":{\"max_new_tokens\":128,\"temperature\":0}}" \
      2>/dev/null)
    if echo "$r" | grep -q '"text"'; then
      successes=$((successes+1))
    fi
  done
  local t1=$(date +%s.%N)
  local wall=$(python3 -c "print(round($t1 - $t0, 2))")

  local placement_log_count=$(grep -c "T2 placement bias active" "$log" 2>/dev/null || echo 0)

  echo "[$label] wall=${wall}s successes=${successes}/32 placement_log=${placement_log_count}" | tee -a "$SUMMARY"

  kill -9 $pid 2>/dev/null || true
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 4
}

echo "T2 A/B reproduce: placement-bias on vs off (arena+page-grain underneath)" | tee -a "$SUMMARY"
echo "Model=$MODEL GPU=$GPU" | tee -a "$SUMMARY"
echo "32 sequential prompts × 128 output tokens" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_one "bias_on"  1
run_one "bias_off" 0

echo "" | tee -a "$SUMMARY"
echo "Verdict: throughput parity expected (small workload, bias only matters under pool-pressure)" | tee -a "$SUMMARY"
echo "Real bias verification deferred to T3 (per-page free mask query)" | tee -a "$SUMMARY"
