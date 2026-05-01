#!/bin/bash
# Static-best partition baseline for Setting 1 v9-auto:
# sweep mamba_full_memory_ratio ∈ {0.3, 0.5, 0.7, 0.9} on baseline
# (L1=0, L2=0) and run all three v9 phases. The per-phase BEST static
# config is a tighter baseline than the engine default (= 0.9). The
# paper's dynamic-allocation claim is "dynamic ≥ per-phase best static"
# rather than just "dynamic ≥ default static".
#
# 4 ratios × 1 cell × 3 phases = 4 runs × ~12 min = ~50 min wall.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-4}"
PORT_BASE="${PORT_BASE:-31000}"
RUN_NAME="${RUN_NAME:-static-best-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[static-best] root=$ROOT gpu=$GPU"

RATIOS=(0.3 0.5 0.7 0.9)
idx=0
for r in "${RATIOS[@]}"; do
  cell="ratio${r}"
  out_dir="$ROOT/$cell"
  mkdir -p "$out_dir"
  port=$((PORT_BASE + idx))
  idx=$((idx + 1))
  log="$out_dir/server.log"

  echo
  echo "=========================================================="
  echo "[static-best] $cell port=$port"
  echo "=========================================================="
  pkill -f "launch_server.*--port $port" 2>/dev/null || true
  sleep 4

  CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -m sglang.launch_server \
      --model-path Qwen/Qwen3.5-35B-A3B --host 127.0.0.1 --port $port \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      --mamba-full-memory-ratio $r \
      >"$log" 2>&1 &
  pid=$!
  echo "[$cell] pid=$pid"
  waited=0
  while [ $waited -lt 300 ]; do
    sleep 10; waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$port/health" 2>/dev/null)" = "200" ]; then
      echo "[$cell] ready after ${waited}s"; break
    fi
  done

  for phase in A B C; do
    bench_args=()
    case $phase in
      A) bench_args=(--dataset-name generated-shared-prefix
                     --gsp-num-groups 16 --gsp-prompts-per-group 10
                     --gsp-system-prompt-len 12000 --gsp-question-len 64
                     --gsp-output-len 256 --request-rate 8) ;;
      B) bench_args=(--dataset-name random --num-prompts 200
                     --random-input-len 8192 --random-output-len 64
                     --request-rate 4) ;;
      C) bench_args=(--dataset-name random --num-prompts 400
                     --random-input-len 4096 --random-output-len 128
                     --request-rate 8) ;;
    esac
    .venv/bin/python -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port $port \
      --model Qwen/Qwen3.5-35B-A3B --tokenizer Qwen/Qwen3.5-35B-A3B \
      "${bench_args[@]}" \
      --output-file "$out_dir/phase_${phase}_bench.json" \
      >"$out_dir/phase_${phase}_bench.log" 2>&1 || echo "[$cell] $phase failed"
    sleep 30
  done
  kill -9 $pid 2>/dev/null || true
  sleep 6
done

echo
echo "=========================================================="
echo "[static-best] SUMMARY"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$ROOT"
ratios = [0.3, 0.5, 0.7, 0.9]
phases = ["A", "B", "C"]
print(f"\n{'ratio':<8}{'phase':<8}{'mean_ttft':>11}{'p99_ttft':>11}{'tps':>10}")
print("-"*48)
best = {p: (None, float('inf')) for p in phases}
for r in ratios:
    for p in phases:
        fp = f"{root}/ratio{r}/phase_{p}_bench.json"
        if not os.path.exists(fp): continue
        with open(fp) as f:
            lines = [l for l in f if l.strip()]
        d = json.loads(lines[-1])
        m = d["mean_ttft_ms"]
        print(f"{r:<8}{p:<8}{m:>11.1f}{d['p99_ttft_ms']:>11.1f}{d['input_throughput']:>10.0f}")
        if m < best[p][1]:
            best[p] = (r, m)

print(f"\nPer-phase best-static ratio: {[(p, best[p]) for p in phases]}")
PY
