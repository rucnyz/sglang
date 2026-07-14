#!/bin/bash
# RQ1 swarm (recurrent-bound, server-bound throughput regime): N=3 base vs sys.
# Trace: cc_qwen_swarm_real (1533 real CC programs, subagents flattened to
# independent roots, shared 16K system-prompt prefix, NO duplication).
# Config: DEFAULT sglang (no MEMFRAC/RATIO override). Only workload knobs:
# --stagger 0.02 (compress the real 1277h arrival timeline to ~30s, 50 sess/s)
# and --max-concurrency 400 (> boot max_running=195 so baseline saturates).
# HiMA grows mamba from idle KV (k2m) -> admission cap rises past 195 -> more
# concurrent -> throughput-led win. Requires the #337 fix (actuator unmark ->
# mamba_allocator.live_size single source of truth).
set -eu
SCRIPT=/scratch/yuzhou/projects/sglang/reproduce/RQ1/run_arm.sh
TRACE=${TRACE:-/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_swarm_real.jsonl}
PORT=30097; GPU=7
STAG=${STAGGER:-0.02}; CONC=${CONC:-400}
OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/swarm/runs/official
mkdir -p "$OUTDIR"

while ss -ltn 2>/dev/null | grep -q ":$PORT "; do echo "Port $PORT busy, waiting..."; sleep 15; done

echo "[$(date +%H:%M:%S)] === swarm official (DEFAULT config, stagger=$STAG conc=$CONC) START ==="
for ARM in base sys; do
  for REP in 1 2 3; do
    BOOTDIR="$OUTDIR/${ARM}_boot${REP}"; mkdir -p "$BOOTDIR"
    echo "[$(date +%H:%M:%S)] $ARM rep$REP"
    PORT=$PORT GPU=$GPU bash "$SCRIPT" "$ARM" "$TRACE" "$STAG" "$CONC" - 1 "$BOOTDIR"
    cp "$BOOTDIR/${ARM}_r1.json" "$OUTDIR/${ARM}_r${REP}.json" 2>/dev/null || true
    sleep 5
  done
done

echo "[$(date +%H:%M:%S)] === swarm RESULTS ==="
for ARM in base sys; do
  python3 -c "
import json, statistics
tps,hit,ttft,ttftp99,tpot,e2e,wall,nerr=[],[],[],[],[],[],[],[]
for rep in range(1,4):
    d=json.load(open('$OUTDIR/${ARM}_r'+str(rep)+'.json'))
    tps.append(d['throughput_tok_s']); hit.append(d['cache_hit'])
    ttft.append(d.get('ttft_ms',{}).get('mean',0)); ttftp99.append(d.get('ttft_ms',{}).get('p99',0))
    tpot.append(d.get('tpot_ms',{}).get('mean',0)); e2e.append(d.get('e2e_ms',{}).get('mean',0))
    wall.append(d.get('wall_s',0)); nerr.append(d.get('n_error',0))
m=statistics.mean; s=lambda l: statistics.stdev(l) if len(l)>1 else 0
print(f'$ARM: tps={m(tps):.1f}±{s(tps):.1f} hit={m(hit):.4f} ttft={m(ttft):.0f} ttft_p99={m(ttftp99):.0f} tpot={m(tpot):.0f} e2e={m(e2e):.0f} wall={m(wall):.0f} nerr={sum(nerr)} reps={[round(t,1) for t in tps]}')
" 2>/dev/null
done
echo "=== DONE ==="
