#!/usr/bin/env bash
# Phase 2d single-arm worker.
#
# Reads from environment (set by parallel_gpu.sh):
#   ARG    -- the arm name: "off" | "on"
#   GPU    -- GPU id to bind (CUDA_VISIBLE_DEVICES)
#   PORT   -- TCP port for this arm's SGLang server
#
# Runs the standard phase2d workflow: boot server → phase A (multi-turn)
# → phase B (random un-shared) → kill. Outputs go under dev/2/phase2d/<arm>.*
#
# Driver invocation:
#   dev/parallel_gpu.sh dev/2/run_phase2d_arm.sh off on

set -euo pipefail

ARM="${ARG:?ARG (off|on) required}"
GPU="${GPU:?GPU required}"
PORT="${PORT:?PORT required}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yuzhou/projects/sglang}"
PHASE_SEC="${PHASE_SEC:-60}"
MEM_FRACTION="${MEM_FRACTION:-0.15}"
PHASEA_CLIENTS="${PHASEA_CLIENTS:-32}"
PHASEA_PARALLEL="${PHASEA_PARALLEL:-16}"
PHASEA_RATE="${PHASEA_RATE:-8}"
PHASEB_PROMPTS="${PHASEB_PROMPTS:-600}"
PHASEB_OUT_LEN="${PHASEB_OUT_LEN:-256}"
PHASEB_RATE="${PHASEB_RATE:-32}"
OUT_DIR="${PHASE2D_OUT_DIR:-$PROJECT_ROOT/dev/2/phase2d}"

cd "$PROJECT_ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
PY="$PROJECT_ROOT/.venv/bin/python"

mkdir -p "$OUT_DIR"
ts() { date -u +%H:%M:%S; }

case "$ARM" in
  off) EXTRA_ENV=() ;;
  on)  EXTRA_ENV=(SGLANG_BUDGETER=1 SGLANG_BUDGETER_ACTUATE=1 SGLANG_BUDGETER_TICK_S=1.0) ;;
  *)   echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

SRV_LOG="$OUT_DIR/${ARM}.server.log"
JSONL="$OUT_DIR/${ARM}.budgeter.jsonl"
METRICS="$OUT_DIR/${ARM}.metrics.jsonl"
PHASEA_LOG="$OUT_DIR/${ARM}.phaseA.jsonl"
PHASEB_LOG="$OUT_DIR/${ARM}.phaseB.json"
SAMPLER_LOG="$OUT_DIR/${ARM}.sampler.log"

echo "[$(ts)] arm=$ARM GPU=$GPU PORT=$PORT launching server..."
env CUDA_VISIBLE_DEVICES="$GPU" \
    SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
    SGLANG_BUDGETER_LOG="$JSONL" \
    "${EXTRA_ENV[@]}" \
    "$PY" -m sglang.launch_server \
      --model-path Qwen/Qwen3-4B \
      --port "$PORT" --host 127.0.0.1 \
      --mem-fraction-static "$MEM_FRACTION" \
      --enable-metrics --log-level warning \
      > "$SRV_LOG" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null || true; kill -9 $SP 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  if ! kill -0 "$SP" 2>/dev/null; then echo "[$(ts)] arm=$ARM SERVER DIED"; tail -10 "$SRV_LOG" >&2; exit 1; fi
  sleep 5
done
echo "[$(ts)] arm=$ARM server up"

: > "$METRICS"
"$PY" dev/1/sample_metrics.py --host 127.0.0.1 --port "$PORT" \
  --interval 1.0 --out "$METRICS" \
  > "$SAMPLER_LOG" 2>&1 &
SAMP=$!
sleep 3

echo "[$(ts)] arm=$ARM PHASE A (multi-turn)"
PHASEA_START=$(date +%s.%3N)
echo "$PHASEA_START" > "$OUT_DIR/${ARM}.phaseA.start_ts"
"$PY" benchmark/hicache/bench_multiturn.py \
  --host 127.0.0.1 --port "$PORT" \
  --model-path Qwen/Qwen3-4B \
  --num-clients "$PHASEA_CLIENTS" --max-parallel "$PHASEA_PARALLEL" --num-rounds 6 \
  --request-length 512 --output-length 96 --request-rate "$PHASEA_RATE" \
  --distribution poisson \
  --log-file "$PHASEA_LOG" \
  > "$OUT_DIR/${ARM}.phaseA.stdout" 2>&1 || echo "[$(ts)] arm=$ARM phase A returned nonzero"

echo "[$(ts)] arm=$ARM PHASE B (random)"
PHASEB_START=$(date +%s.%3N)
echo "$PHASEB_START" > "$OUT_DIR/${ARM}.phaseB.start_ts"
"$PY" -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port "$PORT" \
  --model Qwen/Qwen3-4B \
  --dataset-name random --num-prompts "$PHASEB_PROMPTS" \
  --random-input-len 1024 --random-output-len "$PHASEB_OUT_LEN" \
  --request-rate "$PHASEB_RATE" \
  --output-file "$PHASEB_LOG" \
  > "$OUT_DIR/${ARM}.phaseB.stdout" 2>&1 || echo "[$(ts)] arm=$ARM phase B returned nonzero"
echo "[$(ts)] arm=$ARM phase B bench done"

kill -INT "$SAMP" 2>/dev/null || true
sleep 2
kill "$SP" 2>/dev/null || true
trap - EXIT
for i in $(seq 1 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
kill -9 "$SP" 2>/dev/null || true
sleep 5
echo "[$(ts)] arm=$ARM cleanup done"
