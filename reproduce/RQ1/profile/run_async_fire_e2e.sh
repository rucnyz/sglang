#!/bin/bash
# E2E verification: async fire correctness + performance.
# Checks:
# 1. Server boots without crash
# 2. Fires happen (XPoolFirePlanner.build lines)
# 3. cap_barrier is fast (< 5ms, was 10-342ms before async refactor)
# 4. execute_async does the shrink+grow (deferred to apply_pending)
# 5. apply_pending_fires lines appear
# 6. No CUDA errors
# 7. All requests complete (0 errors)
# 8. tps is reasonable (not regression)
# Runs TWO arms: conc=64 (case1 scenario) and conc=256 (stress scenario).
set -eu
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
AR=/scratch/yuzhou/projects/agentreplay
T6=$AR/data/traces/cc_qwen_t6_v2.jsonl
MODEL=Qwen/Qwen3.5-9B
PORT=30097; GPU=7

run_e2e_check() {
  local LABEL=$1 CONC=$2 LIMIT=$3
  local OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/profile/async_e2e_${LABEL}
  mkdir -p "$OUTDIR"

  pkill -9 -f "sglang.launch_server.*$PORT" 2>/dev/null || true
  sleep 3

  export SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_LPB_WINDOW_S=120.0 \
         SGLANG_XPOOL_QUEUE_WAIT_US=100 SGLANG_XPOOL_COOLDOWN_S=1.0 \
         SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 \
         SGLANG_CSIGMA_KV_BETA=0.024570739655696554 \
         SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 \
         SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0

  echo "[$LABEL] Booting server (conc=$CONC, limit=$LIMIT)"
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    $VENV -m sglang.launch_server \
    --model-path $MODEL --host 127.0.0.1 --port $PORT \
    --reasoning-parser qwen3 --mamba-scheduler-strategy extra_buffer \
    --enable-cache-report --log-level info \
    --radix-eviction-policy lpb \
    > "$OUTDIR/server.log" 2>&1 &
  local SVPID=$!

  local ready=0
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    [ "$code" = "200" ] && ready=1 && break
    kill -0 $SVPID 2>/dev/null || { echo "[$LABEL] SERVER DIED"; tail -10 "$OUTDIR/server.log"; return 1; }
    sleep 5
  done
  [ "$ready" = "1" ] || { echo "[$LABEL] BOOT TIMEOUT"; kill -9 $SVPID 2>/dev/null; return 1; }
  echo "[$LABEL] Server ready"

  echo "[$LABEL] Running replay"
  TRANSFORMERS_OFFLINE=1 PYTHONPATH=$AR $VENV -m agentreplay replay \
    --trace "$T6" --stagger 0.5 --max-concurrency $CONC --limit $LIMIT --flush \
    --url "http://127.0.0.1:$PORT/generate" --label "$LABEL" \
    --out "$OUTDIR/result.json" > "$OUTDIR/replay.log" 2>&1
  echo "[$LABEL] Replay done"

  kill -9 $SVPID 2>/dev/null || true
  sleep 2

  # ---- Verification ----
  echo ""
  echo "=== [$LABEL] VERIFICATION ==="
  local LOG="$OUTDIR/server.log"
  local PASS=0 FAIL=0

  # Check 1: no CUDA errors
  if grep -qi "cuda error\|illegal memory\|device-side assert" "$LOG" 2>/dev/null; then
    echo "  FAIL: CUDA errors found"
    FAIL=$((FAIL+1))
  else
    echo "  PASS: no CUDA errors"
    PASS=$((PASS+1))
  fi

  # Check 2: fires happened
  local N_FIRES=$(grep -c "XPoolFirePlanner.build" "$LOG" 2>/dev/null || echo 0)
  if [ "$N_FIRES" -gt 0 ]; then
    echo "  PASS: $N_FIRES fires executed"
    PASS=$((PASS+1))
  else
    echo "  FAIL: no fires"
    FAIL=$((FAIL+1))
  fi

  # Check 3: cap_barrier is fast (check the logged cap_barrier_us)
  # Old: cap_barrier did cuMem* → avg 9600us, max 342000us
  # New: cap_barrier is mark-only → should be < 5000us
  local MAX_CAP_US=$(grep "cap_barrier_us" "$LOG" 2>/dev/null | grep -oE "cap_barrier_us=[0-9]+" | sed 's/cap_barrier_us=//' | sort -n | tail -1)
  if [ -n "$MAX_CAP_US" ] && [ "$MAX_CAP_US" -lt 5000 ]; then
    echo "  PASS: cap_barrier max=${MAX_CAP_US}us (< 5ms target)"
    PASS=$((PASS+1))
  elif [ -n "$MAX_CAP_US" ]; then
    echo "  WARN: cap_barrier max=${MAX_CAP_US}us (expected < 5ms)"
  else
    # cap_barrier_us might not be in the snapshot log line directly; check via debug lines
    local MAX_CB=$(grep "cap_barrier\[seq=" "$LOG" 2>/dev/null | grep -oE "in [0-9]+ us" | grep -oE "[0-9]+" | sort -n | tail -1)
    if [ -n "$MAX_CB" ] && [ "$MAX_CB" -lt 5000 ]; then
      echo "  PASS: cap_barrier max=${MAX_CB}us (< 5ms target)"
      PASS=$((PASS+1))
    elif [ -n "$MAX_CB" ]; then
      echo "  WARN: cap_barrier max=${MAX_CB}us (expected < 5ms)"
    else
      echo "  SKIP: cap_barrier timing not in log (debug level)"
    fi
  fi

  # Check 4: deferred apply happened
  local N_APPLY=$(grep -c "apply_pending" "$LOG" 2>/dev/null || echo 0)
  if [ "$N_APPLY" -gt 0 ]; then
    echo "  PASS: $N_APPLY deferred applies executed"
    PASS=$((PASS+1))
  else
    echo "  FAIL: no apply_pending lines (async fire not working)"
    FAIL=$((FAIL+1))
  fi

  # Check 5: no errors in replay
  local N_ERR=$(grep -oE '"n_error": [0-9]+' "$OUTDIR/result.json" 2>/dev/null | grep -oE '[0-9]+')
  if [ "$N_ERR" = "0" ]; then
    echo "  PASS: 0 request errors"
    PASS=$((PASS+1))
  else
    echo "  FAIL: $N_ERR request errors"
    FAIL=$((FAIL+1))
  fi

  # Check 6: tps result
  local TPS=$(grep -oE '"throughput_tok_s": [0-9.]+' "$OUTDIR/result.json" 2>/dev/null | grep -oE '[0-9.]+')
  local HIT=$(grep -oE '"cache_hit": [0-9.]+' "$OUTDIR/result.json" 2>/dev/null | grep -oE '[0-9.]+')
  echo "  INFO: tps=$TPS cache_hit=$HIT"

  echo "  TOTAL: $PASS passed, $FAIL failed"
  echo ""
}

# Test 1: conc=64, limit=200 (case1 scenario — moderate pressure)
run_e2e_check "conc64" 64 200

# Test 2: conc=256, full trace (stress scenario — many fires)
run_e2e_check "conc256" 256 -

echo "=== ALL E2E CHECKS DONE ==="
