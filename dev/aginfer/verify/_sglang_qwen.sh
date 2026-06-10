#!/bin/bash
# Parametrized Qwen3-0.6B launcher for verify e2e. Env knobs:
#   GPU (default 7)  PORT (30000)  WRITE_POLICY (write_through)  CHUNKED (unset)
WT=/scratch/yuzhou/projects/sglang
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
GPU="${GPU:-7}"; PORT="${PORT:-30000}"; WP="${WRITE_POLICY:-write_through}"
for p in $(pgrep -f "launch_server.*--port $PORT"); do kill -9 "$p" 2>/dev/null; done
sleep 3
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$WT/python" \
nohup $PY -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 --port $PORT \
  --tp 1 --mem-fraction-static 0.15 --max-total-tokens 65536 \
  --trust-remote-code --attention-backend flashinfer \
  --enable-hierarchical-cache --hicache-ratio 1.5 \
  --hicache-write-policy "$WP" \
  ${CHUNKED:+--chunked-prefill-size $CHUNKED} \
  --enable-cache-report \
  > /tmp/qwen_${PORT}.log 2>&1 &
echo "sglang pid $! on GPU$GPU :$PORT policy=$WP chunked=${CHUNKED:-default}"
for i in $(seq 1 80); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { echo "UP"; exit 0; }
  sleep 3
done
echo "TIMEOUT"; tail -12 /tmp/qwen_${PORT}.log
