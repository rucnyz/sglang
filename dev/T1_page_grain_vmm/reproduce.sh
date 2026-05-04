#!/bin/bash
# T1 reproduce: A/B boot + 5-prompt smoke at 2 MiB (default after T1) vs
# 64 MiB (legacy). Captures boot-time delta and verifies serving works.
#
# Output: dev/T1_page_grain_vmm/results/ab_smoke.txt + per-config logs.

set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

ROOT=dev/T1_page_grain_vmm/results
mkdir -p "$ROOT"
SUMMARY="$ROOT/ab_smoke.txt"
: > "$SUMMARY"

GPU=${GPU:-2}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT_BASE=${PORT_BASE:-31100}

run_one() {
  local label="$1"
  local chunk_bytes="$2"
  local port=$((PORT_BASE + RANDOM % 1000))
  local log="$ROOT/${label}_server.log"
  local smoke_log="$ROOT/${label}_smoke.log"

  echo "============================="
  echo "[$label] chunk_bytes=$chunk_bytes (= $((chunk_bytes / (1024*1024))) MiB)"
  echo "============================="
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 4

  SGLANG_ARENA_SHARED=1 \
  SGLANG_ARENA_FROM_BLOB=1 \
  SGLANG_ARENA_CHUNK_BYTES=$chunk_bytes \
  SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
  CUDA_VISIBLE_DEVICES=$GPU \
    nohup .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $port \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph --reasoning-parser qwen3 \
      > "$log" 2>&1 &
  local pid=$!

  local t0=$(date +%s)
  local waited=0
  while [ $waited -lt 600 ]; do
    sleep 5; waited=$((waited+5))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            http://127.0.0.1:$port/health 2>/dev/null)" = "200" ]; then
      break
    fi
  done
  local boot_s=$(( $(date +%s) - t0 ))
  if [ $waited -ge 600 ]; then
    echo "[$label] BOOT FAILED — log tail:"
    tail -25 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi
  echo "[$label] boot=${boot_s}s"

  # 5-prompt smoke; hit pool a few times.
  local smoke_t0=$(date +%s.%N)
  for i in 1 2 3 4 5; do
    curl -s --max-time 60 -X POST http://127.0.0.1:$port/generate \
      -H "Content-Type: application/json" \
      -d '{"text":"Tell me a one-sentence fun fact about hybrid LLMs.","sampling_params":{"max_new_tokens":48,"temperature":0}}' \
      >> "$smoke_log" 2>&1
    echo >> "$smoke_log"
  done
  local smoke_s=$(python3 -c "print(round($(date +%s.%N) - $smoke_t0, 2))")

  # cuMemMap call count via grep on init logs.
  local map_count=$(grep -c "cuMemMap" "$log" 2>/dev/null || echo 0)
  local arena_init=$(grep -c "MultiTensorArena\|MambaPool arena\|KV pool arena" "$log" 2>/dev/null || echo 0)

  printf '[%s] boot=%ss smoke(5 prompts)=%ss arena-init-lines=%s\n' \
    "$label" "$boot_s" "$smoke_s" "$arena_init" | tee -a "$SUMMARY"

  kill -9 $pid 2>/dev/null || true
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 4
}

echo "T1 A/B reproduce: page-grain (2 MiB) vs chunk-grain (64 MiB)" | tee -a "$SUMMARY"
echo "Model=$MODEL GPU=$GPU" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

run_one "page2MiB"  $((2 * 1024 * 1024))
run_one "chunk64MiB" $((64 * 1024 * 1024))

echo "" | tee -a "$SUMMARY"
echo "Summary above; full logs in $ROOT/" | tee -a "$SUMMARY"
