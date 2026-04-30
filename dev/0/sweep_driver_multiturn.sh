#!/usr/bin/env bash
# Multiturn variant of sweep_driver.sh — uses benchmark/hicache/bench_multiturn.py
# instead of sglang.bench_serving so that prefix-cache temporal locality actually
# fires (each round depends on prior round's KV). Same arg shape as sweep_driver.sh.

set -euo pipefail

SWEEP_NAME="${1:?sweep name required}"
MODEL_PATH="${2:?model path required}"
KNOB_FLAG="${3:?knob flag required}"
KNOB_VALUES_CSV="${4:?knob values CSV required}"
EXTRA_FLAGS="${5:-}"
BENCH_ARGS="${6:-}"   # passed to bench_multiturn.py (excluding --host/--port/--model-path)

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yuzhou/projects/sglang}"
OUT_DIR="$PROJECT_ROOT/dev/0/$SWEEP_NAME"
HOST="127.0.0.1"; PORT="${PORT:-30000}"
BOOT_TIMEOUT_SEC=600

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
PY="$PROJECT_ROOT/.venv/bin/python"

log() { echo "[$(date -u +%FT%TZ) sweep:$SWEEP_NAME] $*"; }

RESULTS_CSV="$OUT_DIR/results.csv"
echo "knob_value,total_throughput_input_tps,total_throughput_output_tps,mean_ttft_ms,p99_ttft_ms,token_usage_peak,token_usage_mean,cache_hit_rate,full_token_usage_peak,swa_token_usage_peak,mamba_usage_peak,duration_s" > "$RESULTS_CSV"

IFS=',' read -ra KNOB_VALUES <<< "$KNOB_VALUES_CSV"

for kv in "${KNOB_VALUES[@]}"; do
  log "=== knob $KNOB_FLAG=$kv ==="
  TAG="${kv//\//_}"
  SRV_LOG="$OUT_DIR/${TAG}.server.log"
  BENCH_JSON="$OUT_DIR/${TAG}.bench.jsonl"
  METRICS_TXT="$OUT_DIR/${TAG}.metrics.txt"
  INFO_JSON="$OUT_DIR/${TAG}.server_info.json"
  METRICS_SAMPLES="$OUT_DIR/${TAG}.metrics_samples.jsonl"

  # Launch server
  $PY -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" --port "$PORT" \
    --enable-metrics --log-level warning \
    $EXTRA_FLAGS \
    "$KNOB_FLAG" "$kv" \
    > "$SRV_LOG" 2>&1 &
  SRV_PID=$!
  log "server pid=$SRV_PID, waiting for /health..."

  T0=$(date +%s)
  while true; do
    curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && break
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      log "SERVER DIED. tail:"; tail -40 "$SRV_LOG" >&2 || true
      continue 2
    fi
    if (( $(date +%s) - T0 > BOOT_TIMEOUT_SEC )); then
      log "BOOT TIMEOUT"; kill -9 "$SRV_PID" 2>/dev/null || true; continue 2
    fi
    sleep 5
  done
  log "server up in $(( $(date +%s) - T0 ))s"
  curl -s "http://$HOST:$PORT/get_server_info" > "$INFO_JSON"

  # Background metrics collector
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
            /^sglang:[a-z_]+{/ {
              n = split($1, a, "{"); name = a[1]; gsub(/^sglang:/, "", name)
              if (name in wanted) printf ",\"%s\":%s", name, $NF
            }
            END { print "}" }' >> "$METRICS_SAMPLES" 2>/dev/null || true
      sleep 2
    done
  ) &
  COLLECTOR_PID=$!

  # Run bench_multiturn (CWD into the script's dir so its log paths work)
  log "running bench_multiturn..."
  BENCH_START=$(date +%s)
  $PY benchmark/hicache/bench_multiturn.py \
    --host "$HOST" --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --log-file "$BENCH_JSON" \
    $BENCH_ARGS \
    > "${BENCH_JSON}.stdout" 2>&1 || log "bench failed (still scraping metrics)"
  BENCH_S=$(( $(date +%s) - BENCH_START ))
  log "bench done in ${BENCH_S}s"

  kill "$COLLECTOR_PID" 2>/dev/null || true
  wait "$COLLECTOR_PID" 2>/dev/null || true
  curl -s "http://$HOST:$PORT/metrics" > "$METRICS_TXT" || true

  # Aggregate -> CSV row
  $PY - <<EOF
import json, re, csv, os, statistics
bench_path="$BENCH_JSON"; metrics_path="$METRICS_TXT"; samples_path="$METRICS_SAMPLES"
out_csv="$RESULTS_CSV"; kv_str="$kv"; duration=$BENCH_S

# bench_multiturn writes JSONL with per-round / aggregate stats
agg = {}
if os.path.exists(bench_path):
    for line in open(bench_path):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except: continue
        # last entry usually has the aggregate; or merge progressively
        agg.update({k:v for k,v in d.items() if isinstance(v,(int,float,str)) or v is None})

# Pull useful numerics from agg with a few possible key spellings
def pick(*keys):
    for k in keys:
        if k in agg: return agg[k]
    return ''
input_tps  = pick('total_throughput_input', 'input_throughput', 'total_input_throughput')
output_tps = pick('total_throughput_output','output_throughput','total_output_throughput')
ttft_mean  = pick('mean_ttft_ms','avg_ttft_ms','ttft_ms_mean')
ttft_p99   = pick('p99_ttft_ms','ttft_ms_p99')

# Aggregate samples
samples=[]
if os.path.exists(samples_path):
    for line in open(samples_path):
        line=line.strip()
        if not line: continue
        try: samples.append(json.loads(line))
        except: pass
def agg_metric(name, fn=max):
    vals=[s[name] for s in samples if name in s]
    return fn(vals) if vals else float('nan')
def final_gauge(name):
    pat=re.compile(rf'^{re.escape(name)}\\b[^\\s]*\\s+([0-9.eE+-]+)$', re.M)
    m = pat.search(open(metrics_path).read()) if os.path.exists(metrics_path) else None
    return float(m.group(1)) if m else float('nan')

row=[kv_str, input_tps, output_tps, ttft_mean, ttft_p99,
     agg_metric('token_usage', max),
     statistics.mean([s['token_usage'] for s in samples if 'token_usage' in s]) if any('token_usage' in s for s in samples) else float('nan'),
     final_gauge('sglang:cache_hit_rate'),
     agg_metric('full_token_usage', max),
     agg_metric('swa_token_usage', max),
     agg_metric('mamba_usage', max),
     duration]
with open(out_csv,'a',newline='') as f:
    csv.writer(f).writerow(row)
print('appended:', row)
EOF

  log "killing server"
  kill "$SRV_PID" 2>/dev/null || true
  for i in $(seq 1 30); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$SRV_PID" 2>/dev/null || true
  sleep 5
done

log "=== sweep complete ==="
cat "$RESULTS_CSV"
