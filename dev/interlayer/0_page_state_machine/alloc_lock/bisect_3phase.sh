#!/usr/bin/env bash
# 3-phase bisection for the +5% TTFT regression on idle_no_regression 9B activev2.
#
# Existing N=6 idle_no_regression data showed mean=+5.07% ± 3.77 pp, |Δ|/SE = 3.29 →
# the regression IS real, not noise. Micro-tests exonerated 4 candidates
# (lock, MemPool, budgeter-tick, arena-kernel-perf) but didn't identify
# the source. This bisection isolates the contributions of two binary
# components — arena tensor backing and budgeter dispatch — by running
# THREE configs back-to-back, N=3 reps each:
#
#   off          : no arena, no budgeter   (= idle_no_regression's "off" phase)
#   arena_only   : SGLANG_ARENA_SHARED=1   (arena tensor backing on; no budgeter)
#   inter        : SGLANG_HIMA=1            (both on; = idle_no_regression's "inter" phase)
#
# Decision logic:
#   - If arena_only ≈ off and inter ≈ off + 5%
#       → +5% comes from budgeter (planner.decide / fire dispatch / ...)
#   - If arena_only ≈ off + 5% and inter ≈ off + 5%
#       → +5% comes from arena tensor backing under real inference
#   - If arena_only ≈ off and inter ≈ off + 5%, with arena_only=baseline
#       → budgeter+arena interaction (e.g. CUDA graph + arena state)
#
# Each rep: 3 phases × ~3 min wall = ~9 min. N=3 reps → ~27 min total.

set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/dasync_bisect_3phase}
GPU=${GPU:-3}
TP=${TP:-1}
PORT=${PORT:-30077}
WORKLOAD_S=${WORKLOAD_S:-180}
MEM_FRAC=${MEM_FRAC:-0.7}
N_REPS=${N_REPS:-3}
mkdir -p "$OUT_DIR"

run_phase() {
    local rep="$1"
    local label="$2"        # off / arena_only / inter
    local env_str="$3"      # extra env vars to set (or empty)
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
    run_phase $r off        ""                                                   || exit 1
    run_phase $r arena_only "SGLANG_ARENA_SHARED=1"                              || exit 1
    run_phase $r inter      "SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_HIMA_LOG=$OUT_DIR/rep${r}.inter.budgeter.jsonl" || exit 1
done

echo
echo "=== Aggregation ==="
$VENV -c "
import glob, json, statistics
import math

def load_ttfts(label):
    paths = sorted(glob.glob('$OUT_DIR/rep*.%s.bench.json' % label))
    out = []
    for p in paths:
        try:
            out.append(json.load(open(p))['mean_ttft_ms'])
        except Exception as e:
            print('  skip %s: %s' % (p, e))
    return out

def stats(label, xs):
    if not xs: return None
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return (m, s, len(xs))

def pct_se(a, b):
    am, asd, an = a
    bm, bsd, bn = b
    pp = (bm - am) / am * 100
    se = math.sqrt(asd**2/an + bsd**2/bn) / am * 100
    return pp, se

off  = load_ttfts('off')
arn  = load_ttfts('arena_only')
inter= load_ttfts('inter')
print('Per-rep mean TTFT (ms):')
print(f'  off       :', [f'{x:6.2f}' for x in off])
print(f'  arena_only:', [f'{x:6.2f}' for x in arn])
print(f'  inter     :', [f'{x:6.2f}' for x in inter])
print()
S_off, S_arn, S_int = stats('off', off), stats('arena_only', arn), stats('inter', inter)
for label, S in [('off', S_off), ('arena_only', S_arn), ('inter', S_int)]:
    if S:
        m, s, n = S
        print(f'  {label:12s}: mean={m:6.2f} ± {s:.2f} ms (N={n})')
print()
if S_off and S_arn:
    pp, se = pct_se(S_off, S_arn)
    print(f'  arena_only vs off : Δ = {pp:+5.2f}% ± {se:.2f} pp  (|Δ|/SE = {abs(pp)/max(se,1e-9):.2f})')
if S_off and S_int:
    pp, se = pct_se(S_off, S_int)
    print(f'  inter      vs off : Δ = {pp:+5.2f}% ± {se:.2f} pp  (|Δ|/SE = {abs(pp)/max(se,1e-9):.2f})')
if S_arn and S_int:
    pp, se = pct_se(S_arn, S_int)
    print(f'  inter vs arena_only: Δ = {pp:+5.2f}% ± {se:.2f} pp  (|Δ|/SE = {abs(pp)/max(se,1e-9):.2f})')
"
