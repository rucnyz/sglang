#!/bin/bash
# Single sglang on :30000 with the t17 README canonical profile (write_through
# forces unit population; flashinfer attn; small KV pool; chunked64 for
# session_id_passthrough). No daemon -- these tests hit sglang directly.
WT=/scratch/yuzhou/projects/sglang
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
for p in $(pgrep -f "launch_server.*--port 30000"); do kill -9 "$p" 2>/dev/null; done
sleep 3
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$WT/python" \
nohup $PY -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B --host 127.0.0.1 --port 30000 \
  --tp 1 --mem-fraction-static 0.15 --max-total-tokens 65536 \
  --trust-remote-code --attention-backend flashinfer \
  --enable-hierarchical-cache \
  ${CHUNKED:+--chunked-prefill-size $CHUNKED} \
  --enable-cache-report \
  > /tmp/qwen_t17profile.log 2>&1 &
echo "sglang pid $!"
for i in $(seq 1 80); do
  curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo "UP"; exit 0; }
  sleep 3
done
echo "TIMEOUT"; tail -15 /tmp/qwen_t17profile.log
