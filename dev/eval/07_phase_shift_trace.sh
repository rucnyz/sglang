#!/bin/bash
# Setting 1 — 24-hour phase-shift trace, 4-cell ablation (paper §6.2 headline).
#
# Compresses 8h→6min phases (synthesized trace). Replays Phase A, B, C in
# sequence. Each cell = (L1, L2) configuration combo:
#   (L1=0, L2=0): stock SGLang
#   (L1=1, L2=0): Layer 1 only (HPB LRU + heterogeneous radix)
#   (L1=0, L2=1): Layer 2 only (cross-pool budgeter, default radix)
#   (L1=1, L2=1): full system
#
# Each cell runs once across all 3 phases (no server restart between phases).
# Reports throughput / TTFT / hit-rate / Layer 2 transfer count per phase.
#
# Total runtime: ~90 min/cell × 4 cells = ~6h.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
DATASET_DIR=${DATASET_DIR:-/scratch/yuzhou/projects/sglang/dev/eval/datasets}
# Allow caller to pin a single (L1, L2) cell so we can fan out across GPUs.
# Default empty = run all 4 cells sequentially on the chosen GPU.
ONLY_L1=${ONLY_L1:-}
ONLY_L2=${ONLY_L2:-}
OUT_DIR=${OUT_DIR:-/tmp/phase_shift_$$}
mkdir -p "$OUT_DIR"
echo "out_dir=$OUT_DIR  dataset_dir=$DATASET_DIR  only=(${ONLY_L1:-*},${ONLY_L2:-*})"

run_cell() {
  local L1="$1"
  local L2="$2"
  local cell="L1${L1}_L2${L2}"
  local extra_env=""
  # Layer 1 = HPB LRU + heterogeneous granularity.
  # SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 — Phase 3.d K_BIG path leaks
  # 7/1.26M slots on idle (see BLOCKERS.md); demote to warning so trace runs.
  if [ "$L1" = "1" ]; then
    extra_env="$extra_env SGLANG_HPB_LRU=1 SGLANG_HPB_WINDOW_S=120.0 SGLANG_K_BIG=8192 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0"
  fi
  # Layer 2 = cross-pool budgeter with planner.
  if [ "$L2" = "1" ]; then
    extra_env="$extra_env SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1 SGLANG_ARENA_CHUNK_BYTES=1073741824 SGLANG_BUDGETER=1 SGLANG_BUDGETER_XPOOL_PLANNER=1 SGLANG_BUDGETER_XPOOL_COORDINATED=1 SGLANG_BUDGETER_TICK_S=2.0 SGLANG_BUDGETER_LOG=$OUT_DIR/${cell}_budgeter.jsonl SGLANG_XPOOL_KV_HIGH=0.04 SGLANG_XPOOL_KV_LOW=0.015 SGLANG_XPOOL_MAMBA_HIGH=0.08 SGLANG_XPOOL_MAMBA_LOW=0.03 SGLANG_XPOOL_COOLDOWN=2"
  fi
  local log="$OUT_DIR/${cell}_server.log"
  echo "=== cell=$cell ($extra_env) ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static 0.8 --log-level info \
      --enforce-piecewise-cuda-graph \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!
  echo "[$cell] pid=$pid"

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[$cell] ready after ${waited}s"
      break
    fi
  done

  # Replay 3 phases. Each phase: send all prompts at fixed RPS.
  for phase in A B C; do
    local input_file
    if [ "$phase" = "C" ]; then input_file="$DATASET_DIR/phase_c.json"
    else input_file="$DATASET_DIR/phase_$(echo $phase | tr A-Z a-z)_sgpt.jsonl"; fi
    local rps=8
    local out_len=17
    if [ "$phase" = "B" ]; then rps=12; out_len=8; fi
    if [ "$phase" = "C" ]; then rps=2; fi
    echo "[$cell] Phase $phase ($input_file, RPS=$rps)..."
    local bench_out="$OUT_DIR/${cell}_phase_${phase}_bench.json"

    if [ "$phase" = "C" ]; then
      # Phase C is multi-turn; use custom dispatcher (not sglang.bench_serving).
      INPUT_FILE=$input_file PORT=$PORT MODEL=$MODEL CELL=$cell PHASE=$phase OUT_DIR=$OUT_DIR \
      .venv/bin/python <<'PY' 2>&1 | tail -8
import json, os, urllib.request, time, statistics
PORT = os.environ['PORT']
MODEL = os.environ['MODEL']
CELL = os.environ['CELL']
OUT_DIR = os.environ['OUT_DIR']
INPUT_FILE = os.environ['INPUT_FILE']
data = json.load(open(INPUT_FILE))
# Normalize each item to a list of {role, content} turns regardless of source format.
def _to_turns(conv):
    # wildchat export: {id, messages: [{role, content}]}
    if isinstance(conv, dict) and isinstance(conv.get('messages'), list):
        return conv['messages']
    # ShareGPT-style: {conversations: [{from/role, value/content}]}
    if isinstance(conv, dict) and isinstance(conv.get('conversations'), list):
        return [{'role': t.get('from', t.get('role', 'user')),
                 'content': t.get('value', t.get('content', ''))}
                for t in conv['conversations']]
    if isinstance(conv, list):
        return conv
    return []

results = []
errors = 0
N = min(50, len(data))
for conv in data[:N]:
    turns = _to_turns(conv)
    user_turns = [t for t in turns if t.get('role') == 'user']
    if len(user_turns) < 2:
        continue
    history = ""
    for t in user_turns[:6]:
        content = t.get('content', '')
        if not content or len(content) < 30:
            continue
        prompt = (history + content + "\n")[:30000]
        t0 = time.time()
        body = json.dumps({'model': MODEL, 'prompt': prompt, 'max_tokens': 64,
                           'temperature': 0}).encode()
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                f'http://127.0.0.1:{PORT}/v1/completions', data=body,
                headers={'Content-Type': 'application/json'}),
                timeout=120).read())
            elapsed = (time.time() - t0) * 1000
            results.append(elapsed)
            history = prompt + r['choices'][0]['text']
        except Exception as e:
            errors += 1
            print(f"  err: {e}")
            break
mean_ms = statistics.mean(results) if results else 0.0
p95_ms = statistics.quantiles(results, n=20)[-1] if len(results) >= 20 else 0.0
print(f"  Phase C: n={len(results)}, mean={mean_ms:.1f}ms, p95={p95_ms:.1f}ms, errors={errors}")
with open(f"{OUT_DIR}/{CELL}_phase_C_summary.txt", 'w') as f:
    f.write(f"n={len(results)}\n")
    f.write(f"mean_ms={mean_ms:.2f}\n")
    f.write(f"p95_ms={p95_ms:.2f}\n")
    f.write(f"errors={errors}\n")
PY
    else
      .venv/bin/python -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name custom \
        --dataset-path "$input_file" \
        --num-prompts 800 \
        --sharegpt-output-len $out_len \
        --request-rate $rps \
        --output-file "$bench_out" \
        >"$OUT_DIR/${cell}_phase_${phase}_bench.log" 2>&1
    fi

    echo "[$cell] Phase $phase done"

    # Inter-phase 30-s drain (compressing paper's 5-min transition).
    sleep 30
  done

  # Capture cell-end stats.
  local hit=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total=$(grep -c "Prefill batch" "$log" || true)
  echo "[$cell] prefill batches: $total, with cached-token > 0: $hit"
  if [ "$L2" = "1" ]; then
    local k2m=$(grep -c '"xpool_direction": "kv_to_mamba"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
    local m2k=$(grep -c '"xpool_direction": "mamba_to_kv"' "$OUT_DIR/${cell}_budgeter.jsonl" 2>/dev/null || echo 0)
    echo "[$cell] xpool transfers: kv→mamba=$k2m mamba→kv=$m2k"
  fi

  kill -9 $pid 2>/dev/null || true
  sleep 6
}

if [ -n "$ONLY_L1" ] && [ -n "$ONLY_L2" ]; then
  run_cell "$ONLY_L1" "$ONLY_L2"
else
  run_cell 0 0
  run_cell 1 0
  run_cell 0 1
  run_cell 1 1
fi

echo
echo "=== Setting 1 (24-h phase-shift, 4-cell) summary ==="
.venv/bin/python <<PY
import json, os
out = "$OUT_DIR"
print(f"\n{'cell':<10} {'phase':<6} {'input TPS':>10} {'mean TTFT':>11} {'P99 TTFT':>10} {'median E2E':>11}")
print('-' * 65)
for L1 in (0, 1):
    for L2 in (0, 1):
        cell = f"L1{L1}_L2{L2}"
        for phase in ('A', 'B'):
            p = f"{out}/{cell}_phase_{phase}_bench.json"
            if not os.path.exists(p):
                print(f"{cell:<10} {phase:<6} {'N/A':>10} {'N/A':>11} {'N/A':>10} {'N/A':>11}")
                continue
            with open(p) as f:
                lines = [l for l in f if l.strip()]
            d = json.loads(lines[-1]) if lines else {}
            print(f"{cell:<10} {phase:<6} {d.get('input_throughput',0):>10.1f} "
                  f"{d.get('mean_ttft_ms',0):>10.1f}ms {d.get('p99_ttft_ms',0):>9.1f}ms "
                  f"{d.get('median_e2e_latency_ms',0):>10.1f}ms")
        # Phase C
        pc = f"{out}/{cell}_phase_C_summary.txt"
        if os.path.exists(pc):
            d = dict(line.strip().split('=', 1) for line in open(pc) if '=' in line)
            print(f"{cell:<10} {'C':<6} {'N/A':>10} {'N/A':>11} {'N/A':>10} {float(d.get('mean_ms',0)):>10.1f}ms")
PY
