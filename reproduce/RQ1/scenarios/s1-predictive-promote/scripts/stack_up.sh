#!/usr/bin/env bash
# Bring up the S1 "ours" stack (a3 config) on the REBASED build and HOLD it so
# s1_driver can run against it.  Order: mooncake -> daemon (kv+admission ON) ->
# sglang (V4-Flash flashinfer_mxfp4 + HiCache + mooncake, 256K KV pool).
# Env (agsched-rebase + sglang-sync) comes from scripts/env.sh.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1   # dev/aginfer
source scripts/env.sh

# tp2 on GPUs 5,6 = the historical a3 config; its deep_gemm JIT shapes are
# already warm from the prior campaign (fast boot) vs a cold tp4 recompile.
export AGINFER_GPUS="5,6"
export SGLANG_TP="${SGLANG_TP:-2}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"   # 256K -> HBM pressure
export SGLANG_KV_POLICY_MODULE="${SGLANG_KV_POLICY_MODULE:-baselines.sglang_adapter:ours_greedy_score}"
SGLANG_PORT=30000 DAEMON_PORT=9100
ML=$AGINFER_LOGS/s1_mooncake.log
DL=$AGINFER_LOGS/s1_daemon.log
SL=$AGINFER_LOGS/s1_sglang.log
ST=$AGINFER_LOGS/s1_stack.status
: > "$ST"

log(){ echo "[s1-stack] $*" | tee -a "$ST"; }

# 0. teardown any prior S1 stack (PID-scoped on our ports; never broad pkill).
for p in $(pgrep -f "launch_server.*--port $SGLANG_PORT"; pgrep -f "daemon.main.*--port=$DAEMON_PORT"; pgrep -f 'mooncake_master'); do
  kill -9 "$p" 2>/dev/null; done
sleep 3

# 1. mooncake master (DRAM + SSD-offload DISK tier).
log "starting mooncake_master ..."
bash scripts/start_mooncake_master.sh >"$ML" 2>&1 &
sleep 4
pgrep -f mooncake_master >/dev/null || { log "mooncake FAILED"; tail -5 "$ML"; exit 1; }

# 2. daemon FIRST (sglang bootstrap-fetches thresholds from it).
log "starting daemon (kv=enabled admission=enabled) on :$DAEMON_PORT ..."
PYTHONPATH="$AGINFER_ROOT:${PYTHONPATH:-}" nohup python -m daemon.main \
  --sglang-base-url=http://127.0.0.1:$SGLANG_PORT \
  --port=$DAEMON_PORT --kv-scheduler=enabled --admission-controller=enabled \
  >"$DL" 2>&1 &
for i in $(seq 1 30); do grep -q "Uvicorn running" "$DL" 2>/dev/null && break; sleep 1; done
grep -q "Uvicorn running" "$DL" || { log "daemon FAILED"; tail -8 "$DL"; exit 1; }
log "daemon UP"

# 3. sglang (V4-Flash + HiCache + mooncake; flashinfer_mxfp4 per launch script).
log "starting sglang (tp=$SGLANG_TP gpus=$AGINFER_GPUS pool=$MAX_TOTAL_TOKENS) ..."
AGINFER_NOTIFY_URL="http://127.0.0.1:$DAEMON_PORT" \
  bash scripts/launch_sglang_v4flash.sh >"$SL" 2>&1 &
for i in $(seq 1 400); do
  curl -sf http://127.0.0.1:$SGLANG_PORT/health >/dev/null 2>&1 && { log "sglang UP after ${i}x6s"; break; }
  sleep 6
done
curl -sf http://127.0.0.1:$SGLANG_PORT/health >/dev/null 2>&1 || { log "sglang never came up"; tail -15 "$SL"; exit 1; }

log "READY  sglang=:$SGLANG_PORT daemon=:$DAEMON_PORT  (logs: $SL $DL $ML)"
