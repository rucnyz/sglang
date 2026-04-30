#!/usr/bin/env bash
# Phase 2 no-regression check: random-uniform traffic, 2 arms (OFF / ON),
# repeated N times each, averaged. The tightened policy should produce
# 0 decisions on this workload, so OFF and ON should be statistically
# indistinguishable.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yuzhou/projects/sglang}"
GPU="${GPU:-1}"
PORT="${PORT:-30003}"
TRIALS="${TRIALS:-2}"
OUT_DIR="$PROJECT_ROOT/dev/2/norep"

cd "$PROJECT_ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
PY="$PROJECT_ROOT/.venv/bin/python"

mkdir -p "$OUT_DIR"
RESULTS="$OUT_DIR/results.csv"
echo "arm,trial,success,mean_ttft_ms,p99_ttft_ms,mean_e2e_ms,p99_e2e_ms,input_tps,output_tps,duration_s,decisions" > "$RESULTS"

run_arm() {
  local arm="$1"          # off|on
  local trial="$2"
  local extra_env="$3"
  local bench_log="$OUT_DIR/${arm}_t${trial}.bench.log"
  local srv_log="$OUT_DIR/${arm}_t${trial}.server.log"
  local jsonl="$OUT_DIR/${arm}_t${trial}.budgeter.jsonl"

  echo "[$(date -u +%H:%M:%S)] arm=$arm trial=$trial launching..."
  env CUDA_VISIBLE_DEVICES="$GPU" \
      SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
      SGLANG_BUDGETER_LOG="$jsonl" \
      $extra_env \
      "$PY" -m sglang.launch_server \
        --model-path Qwen/Qwen3-4B \
        --port "$PORT" --host 127.0.0.1 \
        --mem-fraction-static 0.15 \
        --enable-metrics --log-level warning \
        > "$srv_log" 2>&1 &
  local SP=$!
  for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    if ! kill -0 "$SP" 2>/dev/null; then
      echo "  SERVER DIED"
      tail -20 "$srv_log" >&2
      return 1
    fi
    sleep 5
  done
  echo "[$(date -u +%H:%M:%S)] up; running bench..."

  $PY -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port "$PORT" \
    --model Qwen/Qwen3-4B \
    --dataset-name random --num-prompts 1000 \
    --random-input-len 1024 --random-output-len 256 \
    --request-rate 64 \
    > "$bench_log" 2>&1 || true

  echo "[$(date -u +%H:%M:%S)] arm=$arm trial=$trial bench done; cleanup..."
  kill "$SP" 2>/dev/null || true
  for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  kill -9 "$SP" 2>/dev/null || true
  sleep 5

  # Extract metrics
  $PY - <<EOF >> "$RESULTS"
import re, json, csv, sys, os
arm = "$arm"
trial = $trial
log = open("$bench_log").read()

def f(pat, default=""):
    m = re.search(pat, log)
    return m.group(1) if m else default

success      = f(r"Successful requests:\s+(\d+)")
mean_ttft    = f(r"Mean TTFT \(ms\):\s+([0-9.]+)")
p99_ttft     = f(r"P99 TTFT \(ms\):\s+([0-9.]+)")
mean_e2e     = f(r"Mean E2E Latency \(ms\):\s+([0-9.]+)")
p99_e2e      = f(r"P99 E2E Latency \(ms\):\s+([0-9.]+)")
input_tps    = f(r"Input token throughput \(tok/s\):\s+([0-9.]+)")
output_tps   = f(r"Output token throughput \(tok/s\):\s+([0-9.]+)")
duration     = f(r"Benchmark duration \(s\):\s+([0-9.]+)")

# decisions from budgeter jsonl, if present
jpath = "$jsonl"
decisions = 0
if os.path.exists(jpath):
    for line in open(jpath):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except: continue
        r = d.get("budgeter_decision","")
        if r and r not in ("noop","cooldown",""):
            decisions += 1
print(f"{arm},{trial},{success},{mean_ttft},{p99_ttft},{mean_e2e},{p99_e2e},{input_tps},{output_tps},{duration},{decisions}")
EOF
}

for trial in $(seq 1 $TRIALS); do
  run_arm off "$trial" ""
  run_arm on  "$trial" "SGLANG_BUDGETER=1 SGLANG_BUDGETER_ACTUATE=1 SGLANG_BUDGETER_TICK_S=1.0"
done

echo
echo "=== results ==="
cat "$RESULTS" | column -t -s,
echo
echo "=== summary ==="
RESULTS_PATH="$RESULTS" $PY - <<'EOF'
import csv, statistics, os
rows = list(csv.DictReader(open(os.environ["RESULTS_PATH"])))
def stats(arm, key):
    v = [float(r[key]) for r in rows if r["arm"] == arm and r[key]]
    return statistics.mean(v) if v else float('nan')
print("arm  mean_ttft  p99_ttft  mean_e2e  p99_e2e  in_tps   out_tps  decisions")
for arm in ("off", "on"):
    print(f"{arm}  {stats(arm,'mean_ttft_ms'):8.1f}  {stats(arm,'p99_ttft_ms'):8.1f}  "
          f"{stats(arm,'mean_e2e_ms'):8.1f}  {stats(arm,'p99_e2e_ms'):8.1f}  "
          f"{stats(arm,'input_tps'):7.1f}  {stats(arm,'output_tps'):7.1f}  "
          f"{stats(arm,'decisions'):.1f}")
def delta(key):
    off = stats('off', key)
    on  = stats('on',  key)
    return (on - off) / off * 100 if off else 0
print()
print(f"delta (on vs off):  ttft={delta('mean_ttft_ms'):+.1f}%  "
      f"p99_ttft={delta('p99_ttft_ms'):+.1f}%  "
      f"e2e={delta('mean_e2e_ms'):+.1f}%  "
      f"in_tps={delta('input_tps'):+.1f}%")
EOF
