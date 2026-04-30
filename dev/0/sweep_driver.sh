#!/usr/bin/env bash
# sweep_driver.sh — Phase 0 config sweep driver.
#
# For a given (model, knob, knob_values, extra_flags, workload), launch the
# sglang server once per knob value, run a fixed workload, scrape metrics,
# and write per-point JSON+text plus a consolidated results.csv.
#
# Usage:
#   ./sweep_driver.sh <sweep_name> <model_path> <knob_flag> <knob_values_csv> <extra_flags> <bench_args>
#
# Example (Sweep 1):
#   ./sweep_driver.sh sweep1_kv_vs_ssm Qwen/Qwen3.5-35B-A3B \
#     --mamba-full-memory-ratio "0.1,0.3,0.5,0.7,0.9" \
#     "--mem-fraction-static 0.85 --enable-metrics" \
#     "--dataset-name random --num-prompts 1000 --random-input 512 --random-output 64 --request-rate 32"
#
# Outputs: dev/0/<sweep_name>/<knob_value>.{bench.json,metrics.txt,server_info.json}
#          dev/0/<sweep_name>/results.csv
#
# Assumptions:
# - Run from /scratch/yuzhou/projects/sglang (i.e. project root).
# - .venv exists and is set up per dev/0.md.
# - .env (CUDA_VISIBLE_DEVICES, HF_HOME) sourced by caller, OR we source it.

set -euo pipefail

SWEEP_NAME="${1:?sweep name required, e.g. sweep1_kv_vs_ssm}"
MODEL_PATH="${2:?model path required}"
KNOB_FLAG="${3:?knob flag required, e.g. --mamba-full-memory-ratio}"
KNOB_VALUES_CSV="${4:?knob values CSV required, e.g. 0.1,0.3,0.5,0.7,0.9}"
EXTRA_FLAGS="${5:-}"
BENCH_ARGS="${6:-}"

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yuzhou/projects/sglang}"
OUT_DIR="$PROJECT_ROOT/dev/0/$SWEEP_NAME"
HOST="127.0.0.1"
PORT="${PORT:-30000}"
BOOT_TIMEOUT_SEC=600     # 35B model can take 4-5 min cold

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

# Source .env if not already sourced (CUDA_VISIBLE_DEVICES, HF_HOME, etc.)
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# Make sure .venv/bin is first on PATH (ninja, cmake, python all there)
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"

PY="$PROJECT_ROOT/.venv/bin/python"

log() { echo "[$(date -u +%FT%TZ) sweep:$SWEEP_NAME] $*"; }

# CSV header
RESULTS_CSV="$OUT_DIR/results.csv"
echo "knob_value,throughput_input_tps,throughput_output_tps,mean_ttft_ms,p99_ttft_ms,token_usage_peak,token_usage_mean,cache_hit_rate,full_token_usage_peak,swa_token_usage_peak,mamba_usage_peak,duration_s" > "$RESULTS_CSV"

IFS=',' read -ra KNOB_VALUES <<< "$KNOB_VALUES_CSV"

for kv in "${KNOB_VALUES[@]}"; do
  log "=== knob $KNOB_FLAG=$kv ==="
  TAG="${kv//\//_}"   # safe filename
  SRV_LOG="$OUT_DIR/${TAG}.server.log"
  BENCH_JSON="$OUT_DIR/${TAG}.bench.json"
  METRICS_TXT="$OUT_DIR/${TAG}.metrics.txt"
  INFO_JSON="$OUT_DIR/${TAG}.server_info.json"

  # Launch server
  log "launching server..."
  $PY -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --enable-metrics \
    --log-level warning \
    $EXTRA_FLAGS \
    "$KNOB_FLAG" "$kv" \
    > "$SRV_LOG" 2>&1 &
  SRV_PID=$!
  log "server pid=$SRV_PID, waiting up to ${BOOT_TIMEOUT_SEC}s for /health..."

  # Wait for server up
  T0=$(date +%s)
  while true; do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      log "SERVER DIED. tail of log:"
      tail -40 "$SRV_LOG" >&2 || true
      log "skipping this knob value"
      continue 2
    fi
    if (( $(date +%s) - T0 > BOOT_TIMEOUT_SEC )); then
      log "SERVER BOOT TIMEOUT. killing."
      kill -9 "$SRV_PID" 2>/dev/null || true
      continue 2
    fi
    sleep 5
  done
  BOOT_S=$(( $(date +%s) - T0 ))
  log "server up in ${BOOT_S}s"

  # Snapshot config
  curl -s "http://$HOST:$PORT/get_server_info" > "$INFO_JSON"

  # Start a background metrics collector. Pool gauges are point-in-time; sample
  # every 2 s during the workload and aggregate (peak/mean) afterwards.
  METRICS_SAMPLES="$OUT_DIR/${TAG}.metrics_samples.jsonl"
  : > "$METRICS_SAMPLES"
  (
    while true; do
      ts=$(date +%s.%3N)
      curl -s "http://$HOST:$PORT/metrics" 2>/dev/null \
        | awk -v ts="$ts" '
            BEGIN {
              wanted["token_usage"]=1; wanted["cache_hit_rate"]=1;
              wanted["full_token_usage"]=1; wanted["swa_token_usage"]=1;
              wanted["mamba_usage"]=1; wanted["gen_throughput"]=1;
              wanted["num_running_reqs"]=1; wanted["num_queue_reqs"]=1;
              printf "{\"ts\":%s", ts
            }
            # Match "sglang:<name>{...labels...} <value>"
            /^sglang:[a-z_]+{/ {
              # split first whitespace-separated field on "{"
              n = split($1, a, "{")
              name = a[1]
              gsub(/^sglang:/, "", name)
              if (name in wanted) {
                printf ",\"%s\":%s", name, $NF
              }
            }
            END { print "}" }' >> "$METRICS_SAMPLES" 2>/dev/null || true
      sleep 2
    done
  ) &
  COLLECTOR_PID=$!

  # Run bench_serving
  log "running bench_serving..."
  BENCH_START=$(date +%s)
  $PY -m sglang.bench_serving \
    --backend sglang \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL_PATH" \
    --output-file "$BENCH_JSON" \
    $BENCH_ARGS \
    > "${BENCH_JSON}.stdout" 2>&1 || log "bench failed (continuing to scrape metrics)"
  BENCH_S=$(( $(date +%s) - BENCH_START ))
  log "bench done in ${BENCH_S}s"

  # Stop collector
  kill "$COLLECTOR_PID" 2>/dev/null || true
  wait "$COLLECTOR_PID" 2>/dev/null || true

  # Snapshot final metrics too (cumulative counters like prefix-cache hits)
  curl -s "http://$HOST:$PORT/metrics" > "$METRICS_TXT" || true

  # Extract row for results.csv: peak/mean of pool gauges across mid-run samples;
  # cumulative counters (cache_hit_rate) from final snapshot.
  $PY - <<EOF
import json, re, csv, os, sys, statistics
bench_path = "$BENCH_JSON"
metrics_path = "$METRICS_TXT"
samples_path = "$METRICS_SAMPLES"
out_csv = "$RESULTS_CSV"
kv_str = "$kv"
duration = $BENCH_S

# bench_serving writes JSONL (one line per run); take last (most complete)
bench = {}
if os.path.exists(bench_path):
    try:
        with open(bench_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    bench = json.loads(line)
    except Exception as e:
        print(f"warn: couldn't parse bench json: {e}", file=sys.stderr)

# Aggregate gauge samples (peak + mean across mid-run snapshots)
samples = []
if os.path.exists(samples_path):
    for line in open(samples_path):
        line = line.strip()
        if not line: continue
        try: samples.append(json.loads(line))
        except: pass

def agg(metric, fn=max):
    vals = [s[metric] for s in samples if metric in s]
    return fn(vals) if vals else float('nan')

# Final cumulative gauges (e.g. cache_hit_rate is a ratio of cumulative counters)
def final_gauge(name):
    pat = re.compile(rf'^{re.escape(name)}\\b[^\\s]*\\s+([0-9.eE+-]+)$', re.M)
    m = pat.search(open(metrics_path).read()) if os.path.exists(metrics_path) else None
    return float(m.group(1)) if m else float('nan')

row = {
    'knob_value':            kv_str,
    'throughput_input_tps':  bench.get('input_throughput',  bench.get('total_input_throughput',  '')),
    'throughput_output_tps': bench.get('output_throughput', bench.get('total_output_throughput', '')),
    'mean_ttft_ms':          bench.get('mean_ttft_ms', ''),
    'p99_ttft_ms':           bench.get('p99_ttft_ms', ''),
    'token_usage_peak':      agg('token_usage', max),
    'token_usage_mean':      agg('token_usage', statistics.mean) if any('token_usage' in s for s in samples) else float('nan'),
    'cache_hit_rate':        final_gauge('sglang:cache_hit_rate'),
    'full_token_usage_peak': agg('full_token_usage', max),
    'swa_token_usage_peak':  agg('swa_token_usage', max),
    'mamba_usage_peak':      agg('mamba_usage', max),
    'duration_s':            duration,
}
with open(out_csv, 'a', newline='') as f:
    w = csv.writer(f)
    w.writerow([row[k] for k in [
        'knob_value','throughput_input_tps','throughput_output_tps',
        'mean_ttft_ms','p99_ttft_ms',
        'token_usage_peak','token_usage_mean','cache_hit_rate',
        'full_token_usage_peak','swa_token_usage_peak','mamba_usage_peak',
        'duration_s']])
print("appended row:", row)
EOF

  # Kill server, wait for GPU to clear
  log "killing server pid=$SRV_PID"
  kill "$SRV_PID" 2>/dev/null || true
  for i in $(seq 1 30); do
    if ! kill -0 "$SRV_PID" 2>/dev/null; then break; fi
    sleep 2
  done
  kill -9 "$SRV_PID" 2>/dev/null || true
  sleep 5    # let CUDA context tear down
  log "knob $kv done"
done

log "=== sweep complete; results: $RESULTS_CSV ==="
cat "$RESULTS_CSV"
