#!/bin/bash
# C1 (preemption / retract-resume) + C2 (over-subscribed throughput) driver.
# Launch plain sglang (override is in-code, no flag) with a CONSTRAINED pool so
#   - the KV token pool is small enough that a concurrent burst forces retract/resume
#     (C1), and
#   - max-running-requests < test concurrency so requests queue (C2 over-subscribed).
# No HiCache / daemon needed: these test the sglang-side forcing path under pressure.
# Env: GPUS (4,5), PORT (30000).
set -uo pipefail
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
source scripts/env.sh 2>/dev/null || true
HERE="$PWD/wherewewin/harness/teacher_forcing"
GPUS="${GPUS:-4,5}"
PORT="${PORT:-30000}"
MAX_TOTAL="${MAX_TOTAL:-16384}"          # small token pool → preemption under burst
MAX_RUN="${MAX_RUN:-64}"                 # < test concurrency → over-subscribed queueing
LOG="$AGINFER_LOGS/tf_c1c2_sglang.log"
RES="$HERE/RESULTS_partC1C2.txt"
mkdir -p "$AGINFER_LOGS"

echo "[c1c2] launching sglang GPUs=$GPUS max-total=$MAX_TOTAL max-running=$MAX_RUN ..." | tee "$RES"
CUDA_VISIBLE_DEVICES="$GPUS" python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.1 \
  --mem-fraction-static 0.85 --context-length 65536 \
  --max-total-tokens "$MAX_TOTAL" --max-running-requests "$MAX_RUN" \
  --enable-cache-report \
  --reasoning-parser deepseek-r1 --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
SGLANG_PID=$!
echo "[c1c2] sglang pid=$SGLANG_PID, waiting for readiness (up to 600s) ..." | tee -a "$RES"
cleanup() { kill -9 "$SGLANG_PID" 2>/dev/null; }
trap cleanup EXIT

up=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "[c1c2] sglang died — tail log:" | tee -a "$RES"; tail -n 30 "$LOG" | tee -a "$RES"; exit 1
  fi
  sleep 5
done
[ "$up" = 1 ] || { echo "[c1c2] sglang not ready in time" | tee -a "$RES"; exit 1; }

echo "" | tee -a "$RES"
echo "=== C1: preemption / retract-resume fidelity ===" | tee -a "$RES"
python "$HERE/test_partC1.py" --base-url "http://127.0.0.1:$PORT" 2>&1 | tee -a "$RES"
c1=${PIPESTATUS[0]}
echo "[c1c2] retract evidence in sglang log:" | tee -a "$RES"
grep -aiE 'retract|preempt' "$LOG" 2>/dev/null | tail -4 | tee -a "$RES"

echo "" | tee -a "$RES"
echo "=== C2: over-subscribed throughput no-op (conc > max-running=$MAX_RUN) ===" | tee -a "$RES"
python "$HERE/test_partC.py" --base-url "http://127.0.0.1:$PORT" \
  --concurrency "96,128,192" --rounds 5 2>&1 | tee -a "$RES"
c2=${PIPESTATUS[0]}

echo "[c1c2] done (C1 rc=$c1, C2 rc=$c2). teardown." | tee -a "$RES"
[ "$c1" = 0 ] && [ "$c2" = 0 ]; exit $?
