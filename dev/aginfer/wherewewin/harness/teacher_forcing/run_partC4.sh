#!/bin/bash
# Part C4 driver: launch sglang (override is in-code, no flag) with cache report
# so meta_info carries cached_tokens, run the multi-turn continuation test.
# Env: GPUS (4,5), PORT (30000).
set -uo pipefail
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
source scripts/env.sh 2>/dev/null || true
HERE="$PWD/wherewewin/harness/teacher_forcing"
GPUS="${GPUS:-4,5}"
PORT="${PORT:-30000}"
LOG="$AGINFER_LOGS/tf_partC4_sglang.log"
RES="$HERE/RESULTS_partC4.txt"
mkdir -p "$AGINFER_LOGS"

echo "[partC4] launching sglang on GPUs $GPUS ..." | tee "$RES"
CUDA_VISIBLE_DEVICES="$GPUS" python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.1 \
  --mem-fraction-static 0.85 --context-length 65536 \
  --enable-cache-report \
  --reasoning-parser deepseek-r1 --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
SGLANG_PID=$!
echo "[partC4] sglang pid=$SGLANG_PID, waiting for readiness (up to 600s) ..." | tee -a "$RES"
cleanup() { kill -9 "$SGLANG_PID" 2>/dev/null; }
trap cleanup EXIT

up=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "[partC4] sglang died — tail log:" | tee -a "$RES"; tail -n 30 "$LOG" | tee -a "$RES"; exit 1
  fi
  sleep 5
done
[ "$up" = 1 ] || { echo "[partC4] sglang not ready in time" | tee -a "$RES"; exit 1; }
echo "[partC4] sglang up. Running test ..." | tee -a "$RES"

python "$HERE/test_partC4.py" --base-url "http://127.0.0.1:$PORT" 2>&1 | tee -a "$RES"
rc=${PIPESTATUS[0]}
echo "[partC4] done (rc=$rc). teardown." | tee -a "$RES"
exit $rc
