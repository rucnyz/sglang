#!/bin/bash
# py-spy the steady-state decode to pin the ~3.7% machinery regression. Profiles
# the GPU worker (Python stacks; a slower GPU kernel shows as more samples in the
# Python frame that launches/awaits it). Two arms on the same 6000-swarm, default
# split: base (lru, no machinery) vs budadm_lru (budgeter+arena+admitter, factor
# OFF). Compares hot-frame sample counts (mamba-layer vs attention vs alloc/free
# vs budgeter).
set -u; cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case2/data/cc_qwen_case2_swarm.jsonl; OUT=reproduce/RQ1/ablations/profile_decode; mkdir -p "$OUT"
PORT=30097; GPU=7; VENV=/scratch/yuzhou/projects/sglang/.venv/bin/python; AR=/scratch/yuzhou/projects/agentreplay
PYSPY=/scratch/yuzhou/projects/sglang/.venv/bin/py-spy
COMMON="--model-path Qwen/Qwen3.5-9B --host 127.0.0.1 --port $PORT --tp 1 --reasoning-parser qwen3 --enforce-piecewise-cuda-graph --mem-fraction-static 0.45 --enable-cache-report --context-length 262144 --log-level info"
CSIGMA="SGLANG_CSIGMA_KV_ALPHA=1.0214961938707212e-07 SGLANG_CSIGMA_KV_BETA=0.024570739655696554 SGLANG_CSIGMA_KV_GAMMA=5.97224986310455 SGLANG_CSIGMA_M_ALPHA=0.0 SGLANG_CSIGMA_M_BETA=0.0 SGLANG_CSIGMA_L_STAR=0.0"
MACH="SGLANG_HIMA=1 SGLANG_HIMA_TICK_S=1.0 SGLANG_LPB_WINDOW_S=120.0 $CSIGMA SGLANG_HIMA=1 SGLANG_XPOOL_QUEUE_WAIT_US=125000 SGLANG_XPOOL_COOLDOWN_S=1.0"

prof() {  # <name> <env...>
  local name=$1; shift; local env="$*"
  ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | grep -oE '[0-9]+' | head -1 | xargs -r kill -9 2>/dev/null
  for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done; sleep 2
  env $env CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $VENV -m sglang.launch_server $COMMON --radix-eviction-policy lru > "$OUT/server_$name.log" 2>&1 &
  local sv=$!
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break; kill -0 $sv 2>/dev/null || { echo "[$name] DIED"; return 1; }; sleep 5; done
  PYTHONPATH=$AR $VENV -m agentreplay replay --trace "$TRACE" --stagger 0.02 --gap-scale 0 --max-concurrency 256 --flush --url http://127.0.0.1:$PORT/generate --label $name --out "$OUT/$name.json" > "$OUT/$name.replay.log" 2>&1 &
  local rp=$!
  sleep 45  # warm into steady-state decode
  local wpid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU 2>/dev/null | head -1)
  echo "[$name] profiling worker pid=$wpid"
  $PYSPY record -p "$wpid" -d 45 -s -f raw -o "$OUT/$name.folded.txt" 2>"$OUT/$name.pyspy.log" || echo "[$name] pyspy err: $(tail -2 $OUT/$name.pyspy.log)"
  wait $rp 2>/dev/null
  kill -9 $sv 2>/dev/null; sleep 3
}

prof base
prof budadm "$MACH"
echo "=== top leaf frames (samples) per arm ==="
for n in base budadm; do
  echo "--- $n ---"
  awk '{c=$NF; sub(/ [0-9]+$/,"",$0); n=split($0,a,";"); leaf=a[n]; s[leaf]+=c} END{for(k in s) print s[k],k}' "$OUT/$n.folded.txt" 2>/dev/null | sort -rn | head -20
done
