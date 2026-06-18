#!/bin/bash
# Bring up a Qwen3-0.6B aginfer stack for the external-stack verify tests
# (t17, t20, session_id_passthrough) which expect BASE=http://127.0.0.1:30001
# (daemon proxy) forwarding to sglang on :30000.  HiCache HBM<->DRAM only (no
# mooncake L3 needed for these). GPU 7 so it can run alongside integration_stress.
set -uo pipefail
WT=/scratch/yuzhou/projects/sglang
PY=/scratch/yuzhou/miniconda3/envs/agsched-rebase/bin/python
DA="$WT/dev/aginfer"
GPU="${STACK_GPU:-7}"
SGLANG_PORT=30000
DAEMON_PORT=30001
export PYTHONPATH="$WT/python:$DA"
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_HOME=/usr/local/cuda-13.2
SGL_LOG=/tmp/qwen_stack_sglang.log
DMN_LOG=/tmp/qwen_stack_daemon.log

# PID-scoped teardown of a previous stack on these ports.
for p in $(pgrep -f "launch_server.*Qwen3-0.6B.*port $SGLANG_PORT" ; pgrep -f "daemon.main.*--port $DAEMON_PORT"); do
  kill -9 "$p" 2>/dev/null; done
sleep 2

# DAEMON FIRST: sglang does a bootstrap-threshold fetch from the daemon at
# startup (#165 G9) and HALTS if the daemon isn't up yet.
echo "[stack] launching daemon proxy on :$DAEMON_PORT"
cd "$DA"
PYTHONPATH="$WT/python:$DA" nohup $PY -m daemon.main \
  --sglang-base-url http://127.0.0.1:$SGLANG_PORT \
  --host 127.0.0.1 --port $DAEMON_PORT \
  --kv-scheduler enabled --admission-controller enabled \
  --observability-summary-every-n 50 \
  >"$DMN_LOG" 2>&1 &
DMN=$!
for i in $(seq 1 40); do
  curl -sf http://127.0.0.1:$DAEMON_PORT/health >/dev/null 2>&1 && { echo "[stack] daemon UP"; break; }
  kill -0 $DMN 2>/dev/null || { echo "[stack] daemon DIED"; tail -20 "$DMN_LOG"; exit 1; }
  sleep 2
done
curl -sf http://127.0.0.1:$DAEMON_PORT/health >/dev/null 2>&1 || { echo "[stack] daemon never came up"; tail -20 "$DMN_LOG"; exit 1; }

echo "[stack] launching Qwen sglang on GPU$GPU :$SGLANG_PORT"
CUDA_VISIBLE_DEVICES=$GPU nohup $PY -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port $SGLANG_PORT --tp 1 \
  --mem-fraction-static 0.10 \
  --enable-hierarchical-cache \
  --trust-remote-code \
  --enable-metrics --enable-cache-report \
  --aginfer-notify-url http://127.0.0.1:$DAEMON_PORT/aginfer/event \
  >"$SGL_LOG" 2>&1 &
SGL=$!
for i in $(seq 1 100); do
  curl -sf http://127.0.0.1:$SGLANG_PORT/health >/dev/null 2>&1 && { echo "[stack] sglang UP"; break; }
  kill -0 $SGL 2>/dev/null || { echo "[stack] sglang DIED"; tail -20 "$SGL_LOG"; exit 1; }
  sleep 3
done
echo "[stack] READY  sglang_pid=$SGL daemon_pid=$DMN  BASE=http://127.0.0.1:$DAEMON_PORT"
echo "$SGL $DMN" > /tmp/qwen_stack.pids
