#!/usr/bin/env bash
# Phase 2d: workload-shift A/B.
#
# Each arm runs:
#   - 60 s of multi-turn shared-prefix traffic (fills the prefix cache)
#   - immediately followed by 60 s of random un-shared traffic
#     (the prefix cache from phase A is now dead weight)
# Expected: budgeter ON recovers throughput / TTFT faster in phase B
# because retraction pressure triggers pre-eviction of the stale cache.
#
# Metrics are sampled by sample_metrics.py at 1 Hz throughout.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yuzhou/projects/sglang}"
GPU="${GPU:-1}"
PORT="${PORT:-30003}"
PHASE_SEC="${PHASE_SEC:-60}"
OUT_DIR="$PROJECT_ROOT/dev/2/phase2d"

cd "$PROJECT_ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
PY="$PROJECT_ROOT/.venv/bin/python"

mkdir -p "$OUT_DIR"

run_arm() {
  local arm="$1"          # off|on
  local extra_env="$2"
  local srv_log="$OUT_DIR/${arm}.server.log"
  local jsonl="$OUT_DIR/${arm}.budgeter.jsonl"
  local metrics="$OUT_DIR/${arm}.metrics.jsonl"
  local phaseA_log="$OUT_DIR/${arm}.phaseA.jsonl"
  local phaseB_log="$OUT_DIR/${arm}.phaseB.json"
  local sampler_log="$OUT_DIR/${arm}.sampler.log"

  echo "[$(date -u +%H:%M:%S)] === arm=$arm ==="

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
    if ! kill -0 "$SP" 2>/dev/null; then echo "  SERVER DIED"; tail -10 "$srv_log" >&2; return 1; fi
    sleep 5
  done
  echo "[$(date -u +%H:%M:%S)] $arm: server up"

  # Sidecar metrics sampler
  : > "$metrics"
  "$PY" dev/1/sample_metrics.py --host 127.0.0.1 --port "$PORT" \
    --interval 1.0 --out "$metrics" \
    > "$sampler_log" 2>&1 &
  local SAMP=$!
  sleep 3

  # Phase A: multi-turn shared prefix (fills the prefix cache)
  echo "[$(date -u +%H:%M:%S)] $arm: PHASE A (multi-turn, ${PHASE_SEC}s)"
  PHASEA_START_TS=$(date +%s.%3N)
  echo "$PHASEA_START_TS" > "$OUT_DIR/${arm}.phaseA.start_ts"
  "$PY" benchmark/hicache/bench_multiturn.py \
    --host 127.0.0.1 --port "$PORT" \
    --model-path Qwen/Qwen3-4B \
    --num-clients 32 --max-parallel 16 --num-rounds 6 \
    --request-length 512 --output-length 96 --request-rate 8 \
    --distribution poisson \
    --log-file "$phaseA_log" \
    > "$OUT_DIR/${arm}.phaseA.stdout" 2>&1 || echo "  phase A returned nonzero"

  # Phase B: random un-shared (cache becomes dead weight)
  echo "[$(date -u +%H:%M:%S)] $arm: PHASE B (random, ${PHASE_SEC}s-ish)"
  PHASEB_START_TS=$(date +%s.%3N)
  echo "$PHASEB_START_TS" > "$OUT_DIR/${arm}.phaseB.start_ts"
  "$PY" -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port "$PORT" \
    --model Qwen/Qwen3-4B \
    --dataset-name random --num-prompts 600 \
    --random-input-len 1024 --random-output-len 256 \
    --request-rate 32 \
    --output-file "$phaseB_log" \
    > "$OUT_DIR/${arm}.phaseB.stdout" 2>&1 || echo "  phase B returned nonzero"
  echo "[$(date -u +%H:%M:%S)] $arm: phase B bench done"

  # Stop sampler & server
  kill -INT "$SAMP" 2>/dev/null || true
  sleep 2
  kill "$SP" 2>/dev/null || true
  for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  kill -9 "$SP" 2>/dev/null || true
  sleep 5
  echo "[$(date -u +%H:%M:%S)] $arm: cleanup done"
}

run_arm off ""
run_arm on  "SGLANG_BUDGETER=1 SGLANG_BUDGETER_ACTUATE=1 SGLANG_BUDGETER_TICK_S=1.0"

echo
echo "[$(date -u +%H:%M:%S)] === both arms done; analyzing ==="
"$PY" - <<'PYEOF'
import json, os, sys
out_dir = "/scratch/yuzhou/projects/sglang/dev/2/phase2d"

def load_metrics(path):
    if not os.path.exists(path): return []
    rows = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except: pass
    return rows

def load_phaseB(path):
    if not os.path.exists(path): return {}
    try:
        with open(path) as f: return json.loads(f.read())
    except: return {}

def get_start_ts(arm, phase):
    p = f"{out_dir}/{arm}.phase{phase}.start_ts"
    return float(open(p).read().strip()) if os.path.exists(p) else None

print(f"{'metric':<35} {'OFF':>10} {'ON':>10} {'Δ':>10}")
print("-" * 70)

for arm in ("off", "on"):
    pb = load_phaseB(f"{out_dir}/{arm}.phaseB.json")
    print(f"  arm={arm} phase B: TTFT_mean={pb.get('mean_ttft_ms','?'):.1f} TTFT_p99={pb.get('p99_ttft_ms','?'):.1f} "
          f"in_tps={pb.get('input_throughput',0):.0f} out_tps={pb.get('output_throughput',0):.0f}")

# Per-phase decisions
for arm in ("off", "on"):
    j = f"{out_dir}/{arm}.budgeter.jsonl"
    if not os.path.exists(j): continue
    rows = load_metrics(j)
    decisions = [r for r in rows if r.get("budgeter_decision") and r["budgeter_decision"] not in ("noop","cooldown")]
    tsA = get_start_ts(arm, "A")
    tsB = get_start_ts(arm, "B")
    decA = sum(1 for d in decisions if tsA and tsB and tsA <= d["ts"] < tsB)
    decB = sum(1 for d in decisions if tsB and d["ts"] >= tsB)
    evA = sum(d.get("budgeter_actually_evicted_kv",0) for d in decisions if tsA and tsB and tsA <= d["ts"] < tsB)
    evB = sum(d.get("budgeter_actually_evicted_kv",0) for d in decisions if tsB and d["ts"] >= tsB)
    print(f"  arm={arm} decisions: phaseA={decA} (evicted {evA} tok) phaseB={decB} (evicted {evB} tok)")

PYEOF
