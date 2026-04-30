#!/usr/bin/env bash
# Parallel sweep driver: assign each knob value to a separate idle GPU and
# port, run all servers + benches concurrently. Total wall clock ≈ single-point
# time instead of N × single-point.
#
# Usage:
#   ./sweep_driver_parallel.sh <sweep_name> <model_path> <knob_flag> \
#       <knob_values_csv> <extra_flags> <bench_args> [bench_tool]
#
# Optional 7th arg `bench_tool` ∈ {serving, multiturn} (default: serving).
#
# GPU selection: defaults to auto — picks idle GPUs (memory.used==0 AND
# util==0) up to N = number of knob values. Override with PARALLEL_GPUS=0,3,5
# env var.

set -euo pipefail

SWEEP_NAME="${1:?sweep name required}"
MODEL_PATH="${2:?model path required}"
KNOB_FLAG="${3:?knob flag required}"
KNOB_VALUES_CSV="${4:?knob values CSV required}"
EXTRA_FLAGS="${5:-}"
BENCH_ARGS="${6:-}"
BENCH_TOOL="${7:-serving}"   # serving | multiturn

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yuzhou/projects/sglang}"
OUT_DIR="$PROJECT_ROOT/dev/0/$SWEEP_NAME"
HOST="127.0.0.1"
BOOT_TIMEOUT_SEC=600
BASE_PORT="${BASE_PORT:-31000}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
PY="$PROJECT_ROOT/.venv/bin/python"

log() { echo "[$(date -u +%FT%TZ) sweep:$SWEEP_NAME] $*"; }

IFS=',' read -ra KNOB_VALUES <<< "$KNOB_VALUES_CSV"
NKNOBS=${#KNOB_VALUES[@]}

# Pick GPUs: explicit override, or auto-detect idle ones
if [[ -n "${PARALLEL_GPUS:-}" ]]; then
  IFS=',' read -ra GPUS <<< "$PARALLEL_GPUS"
else
  log "auto-selecting $NKNOBS idle GPUs..."
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F', ' '$2 == 0 && $3 == 0 { print $1 }' \
      | head -n "$NKNOBS"
  )
fi
if (( ${#GPUS[@]} < NKNOBS )); then
  log "ERROR: need $NKNOBS GPUs, found ${#GPUS[@]} idle: ${GPUS[*]:-(none)}"
  log "       try setting PARALLEL_GPUS=g1,g2,... or wait for GPUs to clear"
  exit 1
fi
log "GPU plan: ${KNOB_VALUES[*]} -> GPUs ${GPUS[*]:0:$NKNOBS}"

RESULTS_CSV="$OUT_DIR/results.csv"
echo "knob_value,throughput_input_tps,throughput_output_tps,mean_ttft_ms,p99_ttft_ms,token_usage_peak,token_usage_mean,cache_hit_rate,full_token_usage_peak,swa_token_usage_peak,mamba_usage_peak,duration_s" > "$RESULTS_CSV"

# Per-point arrays
declare -a PORTS GPU_LIST PIDS COLLECTOR_PIDS
declare -a SRV_LOGS BENCH_OUTS METRICS_TXTS METRICS_SAMPLES INFO_JSONS

for i in "${!KNOB_VALUES[@]}"; do
  kv="${KNOB_VALUES[$i]}"
  TAG="${kv//\//_}"
  PORTS[$i]=$((BASE_PORT + i))
  GPU_LIST[$i]="${GPUS[$i]}"
  SRV_LOGS[$i]="$OUT_DIR/${TAG}.server.log"
  if [[ "$BENCH_TOOL" == "multiturn" ]]; then
    BENCH_OUTS[$i]="$OUT_DIR/${TAG}.bench.jsonl"
  else
    BENCH_OUTS[$i]="$OUT_DIR/${TAG}.bench.json"
  fi
  METRICS_TXTS[$i]="$OUT_DIR/${TAG}.metrics.txt"
  METRICS_SAMPLES[$i]="$OUT_DIR/${TAG}.metrics_samples.jsonl"
  INFO_JSONS[$i]="$OUT_DIR/${TAG}.server_info.json"
done

# ---- 1. Launch all servers in parallel ----
for i in "${!KNOB_VALUES[@]}"; do
  kv="${KNOB_VALUES[$i]}"
  port="${PORTS[$i]}"
  gpu="${GPU_LIST[$i]}"
  log "launch knob=$kv on GPU $gpu, port $port"
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" --port "$port" \
    --enable-metrics --log-level warning \
    $EXTRA_FLAGS \
    "$KNOB_FLAG" "$kv" \
    > "${SRV_LOGS[$i]}" 2>&1 &
  PIDS[$i]=$!
done

# ---- 2. Wait for all servers up ----
log "waiting for ${NKNOBS} servers..."
T0=$(date +%s)
for i in "${!KNOB_VALUES[@]}"; do
  port="${PORTS[$i]}"
  while true; do
    if curl -sf "http://$HOST:$port/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      log "ERROR: server for knob ${KNOB_VALUES[$i]} (port $port) DIED"
      tail -30 "${SRV_LOGS[$i]}" >&2
      # Kill all servers and bail
      for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
      exit 1
    fi
    if (( $(date +%s) - T0 > BOOT_TIMEOUT_SEC )); then
      log "ERROR: boot timeout"
      for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
      exit 1
    fi
    sleep 5
  done
  curl -s "http://$HOST:$port/get_server_info" > "${INFO_JSONS[$i]}"
done
log "all servers up in $(( $(date +%s) - T0 ))s"

# ---- 3. Start metrics collectors per server ----
for i in "${!KNOB_VALUES[@]}"; do
  port="${PORTS[$i]}"
  samples="${METRICS_SAMPLES[$i]}"
  : > "$samples"
  (
    while true; do
      ts=$(date +%s.%3N)
      curl -s "http://$HOST:$port/metrics" 2>/dev/null \
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
            END { print "}" }' >> "$samples" 2>/dev/null || true
      sleep 2
    done
  ) &
  COLLECTOR_PIDS[$i]=$!
done

# ---- 4. Run benches in parallel ----
declare -a BENCH_PIDS
BENCH_START_GLOBAL=$(date +%s)
for i in "${!KNOB_VALUES[@]}"; do
  port="${PORTS[$i]}"
  out="${BENCH_OUTS[$i]}"
  (
    if [[ "$BENCH_TOOL" == "multiturn" ]]; then
      $PY benchmark/hicache/bench_multiturn.py \
        --host "$HOST" --port "$port" \
        --model-path "$MODEL_PATH" \
        --log-file "$out" \
        $BENCH_ARGS \
        > "${out}.stdout" 2>&1 || true
    else
      $PY -m sglang.bench_serving --backend sglang \
        --host "$HOST" --port "$port" \
        --model "$MODEL_PATH" \
        --output-file "$out" \
        $BENCH_ARGS \
        > "${out}.stdout" 2>&1 || true
    fi
  ) &
  BENCH_PIDS[$i]=$!
done

log "${NKNOBS} benches launched in parallel; waiting for completion..."
for i in "${!KNOB_VALUES[@]}"; do
  wait "${BENCH_PIDS[$i]}" || true
  log "  knob=${KNOB_VALUES[$i]} bench done"
done
DUR=$(( $(date +%s) - BENCH_START_GLOBAL ))
log "all benches done in ${DUR}s wall clock"

# ---- 5. Stop collectors, snapshot final metrics ----
for i in "${!KNOB_VALUES[@]}"; do
  kill "${COLLECTOR_PIDS[$i]}" 2>/dev/null || true
  wait "${COLLECTOR_PIDS[$i]}" 2>/dev/null || true
  curl -s "http://$HOST:${PORTS[$i]}/metrics" > "${METRICS_TXTS[$i]}" || true
done

# ---- 6. Aggregate -> CSV ----
NKNOBS="$NKNOBS" RESULTS_CSV="$RESULTS_CSV" OUT_DIR="$OUT_DIR" BENCH_TOOL="$BENCH_TOOL" \
  KNOB_VALUES_CSV="$KNOB_VALUES_CSV" DUR="$DUR" $PY - <<'EOF'
import json, re, csv, os, statistics
results_csv = os.environ['RESULTS_CSV']
out_dir = os.environ['OUT_DIR']
bench_tool = os.environ['BENCH_TOOL']
knob_values = os.environ['KNOB_VALUES_CSV'].split(',')
duration = int(os.environ['DUR'])

def gauge_from_text(path, name):
    if not os.path.exists(path): return float('nan')
    pat = re.compile(rf'^{re.escape(name)}\b[^\s]*\s+([0-9.eE+-]+)$', re.M)
    m = pat.search(open(path).read())
    return float(m.group(1)) if m else float('nan')

def agg(samples, name, fn=max):
    vals = [s[name] for s in samples if name in s]
    return fn(vals) if vals else float('nan')

with open(results_csv, 'a', newline='') as f:
    w = csv.writer(f)
    for kv in knob_values:
        tag = kv.replace('/', '_')
        bench_path = f"{out_dir}/{tag}.bench.{'jsonl' if bench_tool=='multiturn' else 'json'}"
        metrics_path = f"{out_dir}/{tag}.metrics.txt"
        samples_path = f"{out_dir}/{tag}.metrics_samples.jsonl"
        bench = {}
        if os.path.exists(bench_path):
            for line in open(bench_path):
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        if isinstance(d, dict): bench.update(d)
                    except: pass
        samples = []
        if os.path.exists(samples_path):
            for line in open(samples_path):
                line = line.strip()
                if line:
                    try: samples.append(json.loads(line))
                    except: pass
        in_tps = bench.get('input_throughput', bench.get('total_input_throughput', bench.get('total_throughput_input', '')))
        out_tps = bench.get('output_throughput', bench.get('total_output_throughput', bench.get('total_throughput_output', '')))
        ttft_mean = bench.get('mean_ttft_ms', bench.get('avg_ttft_ms', ''))
        ttft_p99 = bench.get('p99_ttft_ms', '')
        row = [kv, in_tps, out_tps, ttft_mean, ttft_p99,
               agg(samples, 'token_usage', max),
               statistics.mean([s['token_usage'] for s in samples if 'token_usage' in s]) if any('token_usage' in s for s in samples) else float('nan'),
               gauge_from_text(metrics_path, 'sglang:cache_hit_rate'),
               agg(samples, 'full_token_usage', max),
               agg(samples, 'swa_token_usage', max),
               agg(samples, 'mamba_usage', max),
               duration]
        w.writerow(row)
        print('row:', row)
EOF

# ---- 7. Tear down all servers ----
log "killing all servers"
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
sleep 5
for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done

log "=== sweep complete ==="
cat "$RESULTS_CSV"
