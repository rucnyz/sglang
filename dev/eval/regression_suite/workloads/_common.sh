#!/bin/bash
# Shared boot/health/teardown for regression-suite workload runners.
# Caller sets:
#   PORT, OUT_DIR, METRICS_PATH, MEM_FRACTION
#   MODEL (default Qwen/Qwen3.5-35B-A3B unless overridden by workload)
#   EXTRA_FLAGS (default empty)
#   All SGLANG_* env vars are inherited from the parent.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
WARMUP_S=${WARMUP_S:-300}
SERVER_LOG="$OUT_DIR/server.log"

# Qwen3.5-mamba-only flags (piecewise CUDA graph + qwen3 reasoning parser).
# Workloads using non-mamba models (e.g. R3 LoRA on Qwen3-4B) override
# MAMBA_FLAGS="" so these stay off — they break the LoRA Triton dispatch.
# Use ${VAR-default} (no colon) so an explicitly-set empty string disables
# the flags. ${VAR:-default} would treat empty as "unset" and re-apply default.
MAMBA_FLAGS=${MAMBA_FLAGS---enforce-piecewise-cuda-graph --reasoning-parser qwen3}

boot_server() {
  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 4
  nohup .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static $MEM_FRACTION --log-level info \
      $MAMBA_FLAGS \
      ${EXTRA_FLAGS:-} \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "[suite-runner] launched pid=$SERVER_PID port=$PORT model=$MODEL"
  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[suite-runner] ready after ${waited}s"
      return 0
    fi
  done
  echo "[suite-runner] FAIL: server not ready after ${WARMUP_S}s"
  tail -30 "$SERVER_LOG"
  return 1
}

teardown_server() {
  kill -9 ${SERVER_PID:-0} 2>/dev/null || true
  sleep 3
}

# Extract metrics from a bench_serving JSON file (last line is the summary).
# $1 = bench json path, $2 = output metrics.json path
emit_metrics_from_bench() {
  local bench_json="$1"
  local metrics_out="$2"
  local jsonl_log="${SGLANG_BUDGETER_LOG:-}"
  python3 - "$bench_json" "$metrics_out" "$jsonl_log" <<'PY'
import json, sys, os
bench, out, jsonl = sys.argv[1], sys.argv[2], sys.argv[3]
m = {}
try:
    with open(bench) as f:
        lines = [l for l in f if l.strip()]
    if lines:
        d = json.loads(lines[-1])
        m["input_tps"] = d.get("input_throughput", 0)
        m["mean_ttft_ms"] = d.get("mean_ttft_ms", 0)
        m["p99_ttft_ms"] = d.get("p99_ttft_ms", 0)
        m["median_e2e_ms"] = d.get("median_e2e_latency_ms", 0)
        m["mean_e2e_ms"] = d.get("mean_e2e_latency_ms", 0)
except Exception as e:
    m["bench_error"] = str(e)
xfers = 0
if jsonl and os.path.exists(jsonl):
    try:
        with open(jsonl) as f:
            for l in f:
                if '"xpool_direction":' in l and '"none"' not in l:
                    xfers += 1
    except Exception:
        pass
m["xpool_transfers"] = xfers
with open(out, "w") as f:
    json.dump(m, f, indent=2)
PY
}
