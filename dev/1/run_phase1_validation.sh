#!/usr/bin/env bash
# Orchestrates the Phase 1 validation:
#   1. boots SGLang with Qwen3-4B + 8 LoRA adapters
#   2. starts sample_metrics.py as a sidecar (1 Hz scrape)
#   3. runs run_validation.py 3-phase mixed workload
#   4. stops everything
#   5. renders the dashboard PNG
#
# Outputs land under dev/1/. Pick a free GPU via PARALLEL_GPUS=N (default GPU 1).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/yuzhou/projects/sglang}"
GPU="${GPU:-1}"
PORT="${PORT:-30001}"
PHASE_SEC="${PHASE_SEC:-60}"
RATE_A="${RATE_A:-8}"
RATE_B="${RATE_B:-8}"
RATE_C="${RATE_C:-4}"   # phase C requests are heavier (1k-token prefix), so slow it down

cd "$PROJECT_ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
PY="$PROJECT_ROOT/.venv/bin/python"

OUT_DIR="$PROJECT_ROOT/dev/1"
SAMPLES="$OUT_DIR/validation.metrics.jsonl"
TIMELINE="$OUT_DIR/validation.timeline.jsonl"
PLOT="$OUT_DIR/validation.dashboard.png"
SRV_LOG="$OUT_DIR/validation.server.log"

# Build LoRA paths array
LORA_NAMES=()
LORA_PATH_ARGS=()
for i in 0 1 2 3 4 5 6 7; do
  LORA_NAMES+=("lora_$i")
  LORA_PATH_ARGS+=("lora_$i=/scratch/yuzhou/.cache/synthetic_loras/qwen3-4b-r16/lora_$i")
done

echo "=== launching SGLang on GPU $GPU port $PORT ==="
CUDA_VISIBLE_DEVICES="$GPU" $PY -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --host 127.0.0.1 --port "$PORT" \
  --enable-lora --max-loaded-loras 8 --max-loras-per-batch 4 \
  --lora-paths "${LORA_PATH_ARGS[@]}" \
  --mem-fraction-static 0.4 \
  --enable-metrics --log-level warning \
  > "$SRV_LOG" 2>&1 &
SRV_PID=$!
echo "server pid=$SRV_PID"

trap 'kill $SRV_PID 2>/dev/null || true; kill $SAMPLER_PID 2>/dev/null || true' EXIT

# Wait for /health
echo "=== waiting for server ==="
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "SERVER DIED"; tail -30 "$SRV_LOG"; exit 1; fi
  sleep 5
done
echo "server up"

echo "=== launching sample_metrics sidecar ==="
$PY dev/1/sample_metrics.py \
  --host 127.0.0.1 --port "$PORT" \
  --interval 1.0 \
  --out "$SAMPLES" \
  > "$OUT_DIR/sampler.log" 2>&1 &
SAMPLER_PID=$!
echo "sampler pid=$SAMPLER_PID"
sleep 3   # let it grab a baseline

echo "=== running 3-phase validation workload ==="
$PY dev/1/run_validation.py \
  --host 127.0.0.1 --port "$PORT" \
  --lora-names "${LORA_NAMES[@]}" \
  --phase-seconds "$PHASE_SEC" \
  --rate-a "$RATE_A" --rate-b "$RATE_B" --rate-c "$RATE_C" \
  --out "$TIMELINE"

echo "=== stopping sampler & server ==="
kill -INT "$SAMPLER_PID" 2>/dev/null || true
sleep 2
kill "$SRV_PID" 2>/dev/null || true
trap - EXIT
sleep 5

echo "=== rendering dashboard ==="
PHA=0
PHB=$PHASE_SEC
PHC=$((PHASE_SEC * 2))
PHEND=$((PHASE_SEC * 3))
$PY dev/1/dashboard.py "$SAMPLES" \
  --out "$PLOT" \
  --phase-marks "$PHB=B,$PHC=C" \
  --title "Phase 1 validation: 3-phase mixed workload (Qwen3-4B + 8 LoRAs)"

echo "=== summary ==="
echo "  metrics samples: $(wc -l < "$SAMPLES") rows"
echo "  timeline events: $(wc -l < "$TIMELINE") rows"
echo "  dashboard:       $PLOT"
ls -lh "$SAMPLES" "$TIMELINE" "$PLOT"
