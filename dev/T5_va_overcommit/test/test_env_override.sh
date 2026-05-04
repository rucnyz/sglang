#!/bin/bash
# T5 env-override test: verify the precedence
#   BYTES set         → bytes / chunk_size chunks
#   BYTES=0 + CHUNKS  → legacy 4-chunk default (or whatever CHUNKS says)
#   neither set       → 80 GiB / chunk_size chunks (T5 default)
#
# Boots three quick configs, greps the arena init log for max_tokens.
# Only need the boot to print the init line — kill server immediately
# after readiness.

set -euo pipefail
cd "$(dirname "$0")/../../.."
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH

GPU=${GPU:-2}
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
ROOT=dev/T5_va_overcommit/results
SUMMARY="$ROOT/env_override.txt"
: > "$SUMMARY"

run_one() {
  local label="$1"
  local kv_bytes="$2"
  local kv_chunks="$3"
  local port=$((31700 + RANDOM % 100))
  local log="$ROOT/${label}_server.log"

  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 3

  local extra_env=""
  [ -n "$kv_bytes" ] && extra_env="${extra_env} SGLANG_ARENA_KV_HEADROOM_BYTES=$kv_bytes"
  [ -n "$kv_chunks" ] && extra_env="${extra_env} SGLANG_ARENA_KV_HEADROOM_CHUNKS=$kv_chunks"

  echo "===== [$label] ${extra_env} =====" | tee -a "$SUMMARY"
  env $extra_env \
  SGLANG_ARENA_SHARED=1 \
  SGLANG_ARENA_FROM_BLOB=1 \
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

  local kv_line=$(grep "MultiTensorArena initialized" "$log" | grep "n_kinds=2" | head -1)
  local max_tokens=$(echo "$kv_line" | grep -oE "max_tokens=[0-9]+" | cut -d= -f2)
  local init_tokens=$(echo "$kv_line" | grep -oE "init_tokens=[0-9]+" | cut -d= -f2)
  local headroom=$((max_tokens - init_tokens))

  printf '[%s] init=%s max=%s headroom=%s\n' "$label" "$init_tokens" "$max_tokens" "$headroom" \
    | tee -a "$SUMMARY"

  kill -9 $pid 2>/dev/null || true
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 4
}

# Default: BYTES unset, CHUNKS unset → 80 GiB headroom (= 40 960 chunks at
# 2 MiB; KV tokens_per_chunk=2048 → 80 GiB headroom = 83 886 080 tokens).
run_one "default"   "" ""
# Legacy: BYTES unset, CHUNKS=4 → 4 chunks × 2048 = 8192 tokens of headroom.
run_one "chunks_4"  "" "4"
# Explicit override: BYTES=1 GiB → 512 chunks × 2048 = 1 048 576 tokens of headroom.
run_one "bytes_1g"  "1073741824" ""
# Edge: BYTES=0 (explicit "0 bytes" of headroom) → 0 chunks × 2048 = 0 headroom.
# (precedence: BYTES set wins, even if 0).
run_one "bytes_0"   "0" "4"

echo ""                                                              | tee -a "$SUMMARY"
echo "Expected (KV @ 2 MiB chunk, 2048 tok/chunk):"                  | tee -a "$SUMMARY"
echo "  default:   headroom = 80 GiB / 2 MiB × 2048 = 83 886 080"   | tee -a "$SUMMARY"
echo "  chunks_4:  headroom = 4 × 2048 = 8 192"                     | tee -a "$SUMMARY"
echo "  bytes_1g:  headroom = 1 GiB / 2 MiB × 2048 = 1 048 576"     | tee -a "$SUMMARY"
echo "  bytes_0:   headroom = 0 (BYTES takes precedence over CHUNKS)" | tee -a "$SUMMARY"
echo ""                                                              | tee -a "$SUMMARY"
echo "Summary saved to $SUMMARY"
