#!/bin/bash
# Run one (case, static-split) cell of the waste matrix. The static split is set
# via the design's own knob --mamba-full-memory-ratio ("default" = sglang's
# out-of-box 0.9, no override). Sweeping the ratio finds each workload's
# static-best split, and shows case3 (dynamic) has no single split that serves
# both phases. We never use --max-mamba-cache-size.
# Usage: run_split.sh <case> <ratio|default> <gpu> <port> <outdir>
set -u
CASE=$1; RATIO=$2; GPU=$3; PORT=$4; OUT=$5
HERE=/scratch/yuzhou/projects/sglang/reproduce/waste
VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python
mkdir -p "$OUT"
SERVE_LOG=$OUT/server.log
RATIO_ARG=""; [ "$RATIO" != "default" ] && RATIO_ARG=$RATIO

SERVE_LOG=$SERVE_LOG bash "$HERE/serve.sh" "$GPU" "$PORT" 262144 "$RATIO_ARG" >/dev/null
echo "[$CASE ratio=$RATIO] booting gpu=$GPU port=$PORT ..."
ready=0
for i in $(seq 1 200); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
  [ "$code" = "200" ] && { ready=1; break; }
  sleep 5
done
[ "$ready" = "1" ] || { echo "[$CASE ratio=$RATIO] BOOT TIMEOUT"; tail -20 "$SERVE_LOG"; exit 2; }

# Per-case concurrency. case2/case3-phaseB use conc 256 so the short swarm
# exceeds the default max_running (~147) and queues at the default split, with
# no need to shrink the mamba pool.
case "$CASE" in
  case1) bash "$HERE/replay.sh" "$PORT" "$HERE/case1/data/trace.jsonl"   64 0 "$OUT" case1 ;;
  case2) bash "$HERE/replay.sh" "$PORT" "$HERE/case2/data/trace.jsonl"  256 0 "$OUT" case2 ;;
  case3a)
    # temporal dynamic, duration-balanced: long phase (sessions truncated to ~50k,
    # KV-binds) then swarm phase (mamba-binds), each spanning minutes so the flip
    # is visible on one time axis.
    bash "$HERE/replay.sh" "$PORT" "$HERE/case3a/data/phase_a_long50k.jsonl" 32 0 "$OUT" case3a_long
    bash "$HERE/replay.sh" "$PORT" "$HERE/case3a/data/phase_b_swarm.jsonl"   256 0 "$OUT" case3a_swarm
    ;;
  case3b)
    # spatial dynamic: case1 long + case2 swarm interleaved, arriving concurrently.
    bash "$HERE/replay.sh" "$PORT" "$HERE/case3b/data/trace.jsonl" 128 0 "$OUT" case3b_mixed
    ;;
esac

SVPID=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1)
[ -n "$SVPID" ] && kill -9 "$SVPID" 2>/dev/null
sleep 3
$VENV "$HERE/parse_waste.py" "$SERVE_LOG" --out "$OUT" --label "$CASE ratio=$RATIO"
echo "[$CASE ratio=$RATIO] DONE -> $OUT"
