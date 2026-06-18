#!/usr/bin/env bash
# Drill into the +3.5% TTFT cost arena_only carries (per
# `bisect_3phase.sh`). The arena has two tensor-backing paths
# (multi_tensor_arena.py:234-301):
#
#   MemPool path (SGLANG_ARENA_FROM_BLOB=0, default):
#     CUDAPluggableAllocator + torch.cuda.MemPool. Per the code
#     comment, "Pays a +6-7% TTFT regression because PyTorch silently
#     disables expandable_segments when a user MemPool is active"
#     (pytorch issue 165419).
#
#   from_blob path (SGLANG_ARENA_FROM_BLOB=1):
#     at::from_blob over cuMemMap-backed VA. No MemPool registered.
#     Per the same comment, this "eliminates the MemPool path's
#     overhead and recovers the regression."
#
# `mempool_penalty_demo` (N=10 synthetic kernel) refuted the
# MemPool penalty on isolated allocation kernels — but missed real
# attention + CUDA graph capture. This bisect tests the claim under
# real serving (the idle_no_regression workload).
#
# 3 configs, N=3 paired per rep:
#   off            : no arena, no budgeter  (baseline)
#   arena_mempool  : SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=0
#   arena_fromblob : SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1
#
# Decision:
#   - arena_fromblob ≈ off  → MemPool is the source (confirms code claim)
#   - arena_fromblob ≈ arena_mempool > off → cost is in the cuMemMap path
#     itself, not in PyTorch's MemPool registration

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/dasync_bisect_arena_path}
GPU=${GPU:-3}
TP=${TP:-1}
PORT=${PORT:-30077}
WORKLOAD_S=${WORKLOAD_S:-180}
MEM_FRAC=${MEM_FRAC:-0.7}
N_REPS=${N_REPS:-3}
mkdir -p "$OUT_DIR"

run_phase() {
    local rep="$1"
    local label="$2"
    local env_str="$3"
    local log="$OUT_DIR/rep${rep}.${label}.server.log"
    local bench="$OUT_DIR/rep${rep}.${label}.bench.json"
    local benchlog="$OUT_DIR/rep${rep}.${label}.bench.log"
    rm -f "$log" "$bench" "$benchlog"

    echo "[rep $rep / $label] boot (env: $env_str)"
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_str \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp $TP --mem-fraction-static $MEM_FRAC \
            --max-running-requests 256 \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --log-level info > "$log" 2>&1 &
    local pid=$!

    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[rep $rep / $label] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[rep $rep / $label] server died"; tail -20 "$log"; return 1
        fi
    done
    [ $waited -ge 1200 ] && { echo "[rep $rep / $label] TIMEOUT"; kill -9 $pid; return 1; }

    echo "[rep $rep / $label] bench"
    $VENV -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$MODEL_DIR" --tokenizer "$MODEL_DIR" \
        --dataset-name random \
        --random-input-len 1024 --random-output-len 128 \
        --request-rate 4 --num-prompts $((WORKLOAD_S * 4)) \
        --output-file "$bench" \
        > "$benchlog" 2>&1 || echo "[rep $rep / $label] bench rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4
}

for r in $(seq 1 $N_REPS); do
    echo "=========================================================="
    echo "  rep $r / $N_REPS"
    echo "=========================================================="
    run_phase $r off            ""                                                          || exit 1
    run_phase $r arena_mempool  "SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=0"            || exit 1
    run_phase $r arena_fromblob "SGLANG_ARENA_SHARED=1 SGLANG_ARENA_FROM_BLOB=1"            || exit 1
done

echo
echo "=== Aggregation ==="
$VENV -c "
import glob, json, statistics, math

def L(p):
    try: return json.load(open(p))['mean_ttft_ms']
    except: return None

cfgs = ('off', 'arena_mempool', 'arena_fromblob')
data = {c: [] for c in cfgs}
for r in range(1, $N_REPS+1):
    for c in cfgs:
        v = L(f'$OUT_DIR/rep{r}.{c}.bench.json')
        if v: data[c].append(v)

for c in cfgs:
    xs = data[c]
    if xs:
        m = statistics.mean(xs); s = statistics.stdev(xs) if len(xs)>1 else 0
        print(f'  {c:16s} N={len(xs)}  mean={m:6.2f} ± {s:.2f} ms  raw={[round(x,2) for x in xs]}')

def paired_delta(a, b):
    diffs = [(y-x)/x*100 for x,y in zip(data[a], data[b])]
    if not diffs: return
    m=statistics.mean(diffs); s=statistics.stdev(diffs) if len(diffs)>1 else 0; se=s/math.sqrt(len(diffs))
    print(f'  {b:16s} vs {a:16s}: per-rep Δ={[f\"{x:+5.2f}\" for x in diffs]}  mean={m:+.2f}% SE={se:.2f}')

print()
paired_delta('off',           'arena_mempool')
paired_delta('off',           'arena_fromblob')
paired_delta('arena_mempool', 'arena_fromblob')
"
