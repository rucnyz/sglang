#!/bin/bash
# Case3 official: t6_v2 conc=128, N=3 fresh boots per arm.
set -eu
SCRIPT=/scratch/yuzhou/projects/sglang/reproduce/RQ1/run_arm.sh
T6=/scratch/yuzhou/projects/agentreplay/data/traces/cc_qwen_t6_v2.jsonl
PORT=30097; GPU=7
OUTDIR=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case3/runs/official_conc128
mkdir -p "$OUTDIR"

# Wait for GPU to be free (candidate C might still be running)
while ss -ltn 2>/dev/null | grep -q ":$PORT "; do
  echo "Port $PORT in use, waiting..."
  sleep 30
done

echo "[$(date +%H:%M:%S)] === case3 official (t6_v2, conc=128) START ==="
for ARM in base sys; do
  for REP in 1 2 3; do
    BOOTDIR="$OUTDIR/${ARM}_boot${REP}"
    mkdir -p "$BOOTDIR"
    echo "[$(date +%H:%M:%S)] $ARM rep$REP"
    PORT=$PORT GPU=$GPU bash "$SCRIPT" "$ARM" "$T6" 0.5 128 - 1 "$BOOTDIR"
    cp "$BOOTDIR/${ARM}_r1.json" "$OUTDIR/${ARM}_r${REP}.json" 2>/dev/null
    sleep 5
  done
done

echo "[$(date +%H:%M:%S)] === case3 RESULTS ==="
for ARM in base sys; do
  python3 -c "
import json, statistics
tps, hit, ttft, tpot = [], [], [], []
for rep in range(1, 4):
    f = '$OUTDIR/${ARM}_r' + str(rep) + '.json'
    d = json.load(open(f))
    tps.append(d['throughput_tok_s'])
    hit.append(d['cache_hit'])
    t = d.get('ttft_ms', {}); ttft.append(t.get('mean', 0))
    tp = d.get('tpot_ms', {}); tpot.append(tp.get('mean', 0))
    if d.get('n_error', 0) > 0:
        print(f'  WARNING: $ARM r{rep} has {d[\"n_error\"]} errors!')
m = statistics.mean; s = lambda l: statistics.stdev(l) if len(l)>1 else 0
print(f'$ARM: tps={m(tps):.1f}±{s(tps):.1f} hit={m(hit):.4f} ttft={m(ttft):.0f}ms tpot={m(tpot):.0f}ms (reps={[round(t,1) for t in tps]})')
" 2>/dev/null
done
echo "=== DONE ==="
