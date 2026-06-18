#!/usr/bin/env bash
# Static mamba-size sweep (NO mechanism — plain `off` servers). Decouples
# "does a bigger mamba pool recover cache_hit on this workload" from the
# cross-pool mechanism, to test the theory that a static mismatch leaves a
# realizable gain. If cache_hit rises monotonically with --max-mamba-cache-
# size, the gain is real + reachable (and the mechanism's failure to show
# it is an implementation issue: pool grew but cache didn't follow). If it
# stays flat ~0.2 across 64..512, mamba size is not the cache_hit lever
# here.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }
TRACES_FILE=${TRACES_FILE:-/scratch/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl}
OUT=${OUT:-/tmp/d10_mamba_sweep}
NUM_CONCURRENCY=${NUM_CONCURRENCY:-14}
MAX_TIME_MIN=${MAX_TIME_MIN:-10}

run_one() {
    local size="$1" gpu="$2" port="$3"
    local cell="$OUT/m$size"
    rm -rf "$cell"; mkdir -p "$cell/metrics"
    pkill -9 -f "launch_server.*--port $port" 2>/dev/null; sleep 2
    CUDA_VISIBLE_DEVICES=$gpu \
        env \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $port \
            --tp 1 --mem-fraction-static 0.55 --max-running-requests 256 \
            --max-mamba-cache-size $size \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --enable-metrics --enable-request-time-stats-logging \
            --export-metrics-to-file --export-metrics-to-file-dir "$cell/metrics" \
            --log-level info > "$cell/server.log" 2>&1 &
    local pid=$!
    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited+10))
        [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$port/health 2>/dev/null)" = "200" ] && break
        kill -0 $pid 2>/dev/null || { echo "[sweep m$size] server died"; return 1; }
    done
    echo "[sweep m$size] ready after ${waited}s; replay ${MAX_TIME_MIN}min"
    $VENV dev/eval/main/cc_trace_replay.py \
        --api-base "http://127.0.0.1:$port" --model "$MODEL_DIR" \
        --traces "$TRACES_FILE" --num-concurrency "$NUM_CONCURRENCY" \
        --max-time-min "$MAX_TIME_MIN" --max-tokens 1024 \
        --min-turns 15 --min-chars 30000 \
        --output-file "$cell/bench.json" > "$cell/bench.log" 2>&1 || true
    kill -9 $pid 2>/dev/null; sleep 3
    echo "[sweep m$size] done"
}

pids=()
i=0
for size in 64 128 256 512; do
    gpu=$((4+i)); port=$((30101+i)); i=$((i+1))
    run_one "$size" "$gpu" "$port" &
    pids+=($!); sleep 5
done
for p in "${pids[@]}"; do wait $p || true; done

echo
echo "=== mamba-size sweep cache_hit (Sigma cached / Sigma prompt) ==="
$VENV - "$OUT" <<'PY'
import sys,glob,os,json
out=sys.argv[1]
def ch(cell):
    logs=glob.glob(os.path.join(cell,'metrics','sglang-request-metrics-*.log'))
    if not logs: return (None,0)
    newest=max(logs,key=os.path.getmtime); c=p=0;n=0
    for ln in open(newest):
        ln=ln.strip()
        if not ln: continue
        try: d=json.loads(ln)
        except: continue
        a,b=d.get('cached_tokens'),d.get('prompt_tokens')
        if isinstance(a,(int,float)) and isinstance(b,(int,float)): c+=a;p+=b;n+=1
    return (c/p if p else None, n)
for size in (64,128,256,512):
    r,n=ch(os.path.join(out,f'm{size}'))
    print(f'  max-mamba-cache={size:4d}: cache_hit={r if r is None else round(r,4)}  (nreq={n})')
PY
echo "[sweep] done"
