#!/bin/bash
# Part A driver: launch sglang (DeepSeek-V4-Flash, TP=2) with the custom logit
# processor enabled, run the teacher-forcing no-op test, tear down.
# Env: GPUS (default 4,5), PORT (30000), REPS (7), OUT_LEN (256).
set -uo pipefail
cd /scratch/yuzhou/projects/sglang/dev/aginfer || exit 1
source scripts/env.sh 2>/dev/null || true   # conda agsched + paths
HERE="$PWD/wherewewin/harness/teacher_forcing"
GPUS="${GPUS:-4,5}"
PORT="${PORT:-30000}"
REPS="${REPS:-7}"
OUT_LEN="${OUT_LEN:-256}"
LOG="$AGINFER_LOGS/tf_partA_sglang.log"
RES="$HERE/RESULTS_partA.txt"
mkdir -p "$AGINFER_LOGS"

echo "[partA] launching sglang on GPUs $GPUS ..." | tee "$RES"
CUDA_VISIBLE_DEVICES="$GPUS" python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.1 \
  --mem-fraction-static 0.85 --context-length 65536 \
  \
  --reasoning-parser deepseek-r1 --trust-remote-code \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
SGLANG_PID=$!
echo "[partA] sglang pid=$SGLANG_PID, waiting for readiness (up to 600s) ..." | tee -a "$RES"

cleanup() { kill -9 "$SGLANG_PID" 2>/dev/null; }
trap cleanup EXIT

up=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "[partA] sglang died during startup — tail of log:" | tee -a "$RES"
    tail -n 30 "$LOG" | tee -a "$RES"; exit 1
  fi
  sleep 5
done
[ "$up" = 1 ] || { echo "[partA] sglang not ready in time" | tee -a "$RES"; exit 1; }
echo "[partA] sglang up. Running test (reps=$REPS out_len=$OUT_LEN) ..." | tee -a "$RES"

python "$HERE/test_partA.py" --base-url "http://127.0.0.1:$PORT" \
  --reps "$REPS" --out-len "$OUT_LEN" 2>&1 | tee -a "$RES"
rc=${PIPESTATUS[0]}
echo "[partA] done (rc=$rc). teardown." | tee -a "$RES"
exit $rc
