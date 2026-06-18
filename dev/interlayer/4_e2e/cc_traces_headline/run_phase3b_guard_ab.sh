#!/usr/bin/env bash
# Phase 3b (#282 A1) — both-full guard A/B at the conc22 inert window.
#
# Both cells run the SAME cross-fire mechanism (Budgeter + Admitter + LPB);
# the ONLY difference is SGLANG_XPOOL_BOTH_FULL_GUARD. At conc22 with the
# default over-provisioned mamba=256 both pools read occupancy-full, so the
# guard (default on) suppresses every fire → mechanism is inert (prior data:
# fires 27→≤1, cache_hit ~0.32). A1 made KV genuinely growable, so with the
# guard OFF an m2k fire should now DURABLY grow KV (KVArenaActuator capacity
# logs, live_tokens rising above boot) and cache_hit should climb toward the
# static envelope (~0.52 @ mamba≈160-equivalent KV bytes).
#
# This isolates the toggle, not baseline-vs-mechanism. Goes/no-goes:
#   - guard_off shows ≥1 "KVArenaActuator: capacity ->" with live > boot AND
#     cache_hit_off > cache_hit_on by a clear margin → A1 captures the win
#     dynamically → proceed to 3c (N=3 paired-delta + regression cells).
#   - no KV grow / no cache_hit lift → diagnose (fire gating, NB threshold,
#     admitter cross-fire seeding) before any N=3.
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
HUB=/scratch/yuzhou/.cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_DIR=$(ls -d "$HUB/models--Qwen--${MODEL_NAME}/snapshots/"* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "$MODEL_NAME not in HUB"; exit 1; }

OUT_DIR=${OUT_DIR:-/tmp/phase3b_guard_ab}
GPU=${GPU:-0}
TP=${TP:-1}
PORT=${PORT:-30078}
TRACES_FILE=${TRACES_FILE:-/scratch/yuzhou/projects/sglang/dev/eval/datasets/cc_long_traces.jsonl}
NUM_CONCURRENCY=${NUM_CONCURRENCY:-22}
MAX_TIME_MIN=${MAX_TIME_MIN:-8}
MAX_SESSIONS=${MAX_SESSIONS:-80}   # request-bounded → comparable cache_hit/tail
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_MAMBA_CACHE=${MAX_MAMBA_CACHE:-256}   # default over-provisioned: the inert window
EVICTION_POLICY=${EVICTION_POLICY:-lpb}   # A1 grow requires LPB (#280)
BOOT_LIVE=1525471                          # boot KV live tokens (for grow check)

mkdir -p "$OUT_DIR/guard_on" "$OUT_DIR/guard_off"

run_cell() {
    local cell="$1"          # guard_on | guard_off
    local guard="$2"         # 1 | 0
    local cell_out="$OUT_DIR/$cell"

    rm -f "$cell_out"/*.log "$cell_out"/*.json "$cell_out"/*.jsonl
    pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null
    sleep 3

    local env_str="\
        SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=2.0 \
        SGLANG_HIMA_LOG=$cell_out/budgeter.jsonl \
        SGLANG_HIMA_ADMITTER_LOG=$cell_out/admitter.jsonl \
        SGLANG_XPOOL_QUEUE_WAIT_US=125000 \
        SGLANG_XPOOL_BOTH_FULL_GUARD=$guard"

    echo "[3b/$cell] boot (both_full_guard=$guard, conc=$NUM_CONCURRENCY, mamba=$MAX_MAMBA_CACHE, $EVICTION_POLICY)"
    CUDA_VISIBLE_DEVICES=$GPU \
        env $env_str \
        nohup $VENV -m sglang.launch_server \
            --model-path "$MODEL_DIR" --host 127.0.0.1 --port $PORT \
            --tp $TP --mem-fraction-static 0.55 \
            --max-running-requests 256 \
            --max-mamba-cache-size $MAX_MAMBA_CACHE \
            --radix-eviction-policy "$EVICTION_POLICY" \
            --reasoning-parser qwen3 --enforce-piecewise-cuda-graph \
            --enable-metrics --enable-request-time-stats-logging \
            --export-metrics-to-file \
            --export-metrics-to-file-dir "$cell_out/metrics" \
            --log-level info > "$cell_out/server.log" 2>&1 &
    local pid=$!

    local waited=0
    while [ $waited -lt 1200 ]; do
        sleep 10; waited=$((waited + 10))
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
                "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
            echo "[3b/$cell] ready after ${waited}s"; break
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[3b/$cell] server died"; tail -25 "$cell_out/server.log"; return 1
        fi
    done
    [ $waited -ge 1200 ] && { echo "[3b/$cell] TIMEOUT"; kill -9 $pid; return 1; }

    echo "[3b/$cell] CC replay (${MAX_TIME_MIN}min cap, ${MAX_SESSIONS} sessions)"
    $VENV dev/eval/main/cc_trace_replay.py \
        --api-base "http://127.0.0.1:$PORT" --model "$MODEL_DIR" \
        --traces "$TRACES_FILE" --num-concurrency "$NUM_CONCURRENCY" \
        --max-time-min "$MAX_TIME_MIN" --max-sessions "$MAX_SESSIONS" \
        --max-tokens "$MAX_TOKENS" --min-turns 15 --min-chars 30000 \
        --output-file "$cell_out/bench.json" \
        > "$cell_out/bench.log" 2>&1 || echo "[3b/$cell] bench rc=$?"

    kill -9 $pid 2>/dev/null
    sleep 4

    # --- per-cell signal extraction ---
    # NOTE: the cross-fire GROW path exposes granted KV pages via
    # `unmark_pages_capped`, NOT `KVArenaActuator.set_capacity_tokens`, so the
    # "KVArenaActuator: capacity ->" log does NOT fire for cross-fire grows.
    # The honest KV-growth signal is the allocator's total capacity
    # (kv_used + kv_evictable + kv_available) rising above boot — read it from
    # the budgeter snapshots. Fires are counted from the planner's FIRE log.
    local nfire maxtot
    nfire=$(grep -c "FIRE tick=" "$cell_out/server.log" 2>/dev/null || echo 0)
    maxtot=$($VENV - "$cell_out/budgeter.jsonl" <<'PY'
import sys, json
mx = 0
try:
    for line in open(sys.argv[1]):
        d = json.loads(line)
        tot = (d.get("kv_used_tokens", 0) + d.get("kv_evictable_tokens", 0)
               + d.get("kv_available_tokens", 0))
        mx = max(mx, tot)
except Exception:
    pass
print(mx)
PY
)
    echo "[3b/$cell] m2k/k2m_fires=$nfire  max_total_kv_tokens=$maxtot (boot=$BOOT_LIVE, grow=$((maxtot - BOOT_LIVE)))"
    # Plan-source breakdown: how the granted pages were sourced (the
    # load-bearing detail — free=idle pages only vs drain=cold-cache evict).
    echo "[3b/$cell] fire plan sources:"
    # `|| true`: under `set -o pipefail` a no-match grep returns non-zero
    # (e.g. guard_on legitimately fires 0×), which would otherwise fail the
    # cell and abort the whole A/B before guard_off runs.
    { grep "XPoolFirePlanner.build" "$cell_out/server.log" 2>/dev/null \
        | grep -oE "\(free=[0-9]+ drain=[0-9]+ migrate_pages=[0-9]+ migrate_moves=[0-9]+\)" \
        | sort | uniq -c | sed 's/^/    /'; } || true
    return 0
}

run_cell guard_on  1 || { echo "[3b] guard_on failed"; exit 1; }
run_cell guard_off 0 || { echo "[3b] guard_off failed"; exit 1; }

echo
echo "=== Phase 3b summary ==="
$VENV - "$OUT_DIR" <<'PY'
import json, sys, os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".",
))
# Reuse the canonical cache-hit extractor (Σcached/Σprompt from the exported
# per-request metrics log) so this matches validate_cc.py exactly.
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/4_e2e/cc_traces_headline")
from validate_cc import _cache_hit_from_metrics
out = sys.argv[1]
def bench(cell):
    p = os.path.join(out, cell, "bench.json")
    try:
        with open(p) as f: return json.load(f)
    except Exception as e:
        return {"_err": str(e)}
rows = {}
for cell in ("guard_on", "guard_off"):
    d = bench(cell)
    ch = _cache_hit_from_metrics(os.path.join(out, cell))
    rows[cell] = (ch, d)
    if "_err" in d:
        print(f"  {cell:9s}: cache_hit={ch}  (bench.json: {d['_err']})"); continue
    print(f"  {cell:9s}: cache_hit={ch}  mean_ttft={d.get('mean_ttft_ms'):.0f}  "
          f"p99_ttft={d.get('p99_ttft_ms'):.0f}  out_tps={d.get('output_tps'):.0f}  "
          f"reqs={d.get('num_requests_valid')}/{d.get('num_requests_total')}  "
          f"errors={d.get('num_errors')}")
on, off = rows["guard_on"][0], rows["guard_off"][0]
if on is not None and off is not None:
    print(f"\n  cache_hit delta (off - on) = {off - on:+.4f}")
print("\nGo/no-go: guard_off should show KV-capacity-changes>=1 with max_live>boot")
print("AND cache_hit(guard_off) clearly > cache_hit(guard_on).")
PY
