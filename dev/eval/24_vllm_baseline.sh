#!/bin/bash
# vLLM baseline on the same v9-auto workload (3 phases) for paper comparison.
# Runs vLLM server (separate venv) on a chosen GPU, fires same 3-phase
# bench harness as sglang's 21_setting1_v9_pool_binding.sh.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
GPU="${GPU:-3}"
PORT="${PORT:-32099}"
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
RUN_NAME="${RUN_NAME:-vllm-baseline-$(date +%Y%m%d-%H%M%S)}"
OUT_DIR=${OUT_DIR:-dev/eval/runs/$RUN_NAME}
mkdir -p "$OUT_DIR"
echo "[vllm-baseline] out=$OUT_DIR gpu=$GPU port=$PORT"

VLLM_VENV=/scratch/yuzhou/projects/vllm-baseline/.venv
log="$OUT_DIR/vllm_server.log"
pkill -f "vllm.*--port $PORT" 2>/dev/null || true
sleep 4

CUDA_VISIBLE_DEVICES=$GPU PATH=$VLLM_VENV/bin:$PATH \
  nohup $VLLM_VENV/bin/vllm serve "$MODEL" \
    --host 127.0.0.1 --port $PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --enforce-eager \
    >"$log" 2>&1 &
pid=$!
echo "[vllm-baseline] pid=$pid"

waited=0
# vLLM with mamba+MoE on Hopper does heavy JIT compile (FlashInfer fused_moe
# kernels, GDN linear-attn) on first launch — observed ~9 min total. After
# the first launch the JIT cache lives at ~/.cache/flashinfer and launches
# are fast. Allow up to 1500 s on cold launch.
while [ $waited -lt 1500 ]; do
  sleep 30; waited=$((waited + 30))
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" = "200" ]; then
    echo "[vllm-baseline] ready after ${waited}s"; break
  fi
done

if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
        "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" != "200" ]; then
  echo "[vllm-baseline] FAIL: server not ready after ${waited}s"
  tail -30 "$log"
  kill -9 $pid 2>/dev/null || true
  exit 1
fi

# 3 phases matching 21_setting1_v9_pool_binding.sh
for phase_def in \
  "A:--dataset-name generated-shared-prefix --gsp-num-groups 16 --gsp-prompts-per-group 10 --gsp-system-prompt-len 12000 --gsp-question-len 64 --gsp-output-len 256 --request-rate 8" \
  "B:--dataset-name random --num-prompts 200 --random-input-len 8192 --random-output-len 64 --request-rate 4" \
  "C:--dataset-name random --num-prompts 400 --random-input-len 4096 --random-output-len 128 --request-rate 8"
do
  phase=${phase_def%%:*}
  args=${phase_def#*:}
  echo "[vllm-baseline] Phase $phase ..."
  .venv/bin/python -m sglang.bench_serving \
    --backend vllm --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --tokenizer "$MODEL" \
    $args \
    --output-file "$OUT_DIR/vllm_phase_${phase}_bench.json" \
    >"$OUT_DIR/vllm_phase_${phase}_bench.log" 2>&1 || echo "[vllm-baseline] phase $phase failed"
  sleep 30
done

kill -9 $pid 2>/dev/null || true
sleep 6

echo
echo "=========================================================="
echo "[vllm-baseline] SUMMARY"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$OUT_DIR"
for ph in ["A","B","C"]:
    fp = f"{root}/vllm_phase_{ph}_bench.json"
    if not os.path.exists(fp):
        print(f"phase {ph}: missing"); continue
    with open(fp) as f:
        lines = [l for l in f if l.strip()]
    d = json.loads(lines[-1])
    print(f"phase {ph}: TPS {d.get('input_throughput',0):.0f}, "
          f"mean TTFT {d.get('mean_ttft_ms',0):.1f}ms, "
          f"P99 {d.get('p99_ttft_ms',0):.1f}ms, "
          f"med E2E {d.get('median_e2e_latency_ms',0):.1f}ms")
PY
