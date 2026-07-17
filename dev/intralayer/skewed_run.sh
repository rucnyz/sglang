#!/bin/bash
# Launch server with tight mamba pool + run skewed_bench.py, both arms × N_TRIAL.
#
# The key knob is --max-mamba-cache-size: forcing it small relative to
# num-groups makes the mamba pool a contended resource on which the
# LRU vs LPB choice actually matters.

set -eu
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/scratch/yuzhou/projects/sglang/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
PORT=${PORT:-30099}
WARMUP_S=${WARMUP_S:-360}
N_TRIAL=${N_TRIAL:-2}

NUM_GROUPS=${NUM_GROUPS:-12}
NUM_PROMPTS=${NUM_PROMPTS:-300}
RPS=${RPS:-2.0}
ALPHA=${ALPHA:-1.5}
MAX_MAMBA=${MAX_MAMBA:-4}  # << this is the critical pressure knob

OUT_BASE=/scratch/yuzhou/projects/vllm-songyang/dev/intralayer/runs/sglang_skewed
mkdir -p "$OUT_BASE"
echo "out_base=$OUT_BASE"
echo "skewed: groups=$NUM_GROUPS num=$NUM_PROMPTS alpha=$ALPHA rps=$RPS max_mamba=$MAX_MAMBA trials=$N_TRIAL"

run_arm() {
  local arm="$1"
  local trial="$2"
  # LPB on/off is a CLI flag (--radix-eviction-policy lpb), #181.
  local extra_env=""
  local evict_policy="lru"
  if [ "$arm" = "lpb" ]; then
    extra_env="SGLANG_LPB_WINDOW_S=600.0"
    evict_policy="lpb"
  fi
  local log="$OUT_BASE/${arm}_t${trial}_server.log"
  echo "=== arm=$arm trial=$trial ==="

  pkill -f "launch_server.*--port $PORT" 2>/dev/null || true
  sleep 6

  nohup env $extra_env \
    .venv/bin/python -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
      --mem-fraction-static 0.8 --log-level info \
      --radix-eviction-policy "$evict_policy" \
      --enforce-piecewise-cuda-graph \
      --max-mamba-cache-size $MAX_MAMBA \
      --reasoning-parser qwen3 \
      >"$log" 2>&1 &
  local pid=$!
  echo "[$arm t${trial}] pid=$pid"

  local waited=0
  while [ $waited -lt $WARMUP_S ]; do
    sleep 10
    waited=$((waited + 10))
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
            "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ]; then
      echo "[$arm t${trial}] ready after ${waited}s"
      break
    fi
  done
  if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 \
          "http://127.0.0.1:$PORT/health" 2>/dev/null)" != "200" ]; then
    echo "FAIL: server did not become ready"
    tail -30 "$log"
    kill -9 $pid 2>/dev/null || true
    return 1
  fi

  echo "[$arm t${trial}] running skewed_bench..."
  .venv/bin/python dev/intralayer/skewed_bench.py \
    --arm "$arm" --trial "$trial" --port "$PORT" \
    --num-groups $NUM_GROUPS --num-prompts $NUM_PROMPTS \
    --rps $RPS --alpha $ALPHA \
    >"$OUT_BASE/${arm}_t${trial}.out" 2>&1
  echo "[$arm t${trial}] bench done"

  local hit_lines=$(grep "Prefill batch" "$log" | grep -v "cached-token: 0" | wc -l)
  local total_prefills=$(grep -c "Prefill batch" "$log" || true)
  echo "[$arm t${trial}] prefill batches: $total_prefills, with cached-token > 0: $hit_lines"

  kill -9 $pid 2>/dev/null || true
  sleep 5
}

for trial in $(seq 1 $N_TRIAL); do
  run_arm recency "$trial"
  run_arm lpb "$trial"
done

echo
echo "=== compare (all trials) ==="
.venv/bin/python <<PY
import json, statistics
OUT_BASE = "$OUT_BASE"; N_TRIAL = int("$N_TRIAL")
def load(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

print(f"\n{'arm':<8} {'trial':<6} {'mean_ms':>10} {'median_ms':>10} {'hit%':>8} {'hot_groups_mean_ms':>20}")
print('-' * 70)
for arm in ["recency", "lpb"]:
    for t in range(1, N_TRIAL + 1):
        d = load(f"{OUT_BASE}/{arm}_t{t}.json")
        if not d:
            print(f"{arm:<8} {t:<6} MISSING")
            continue
        # mean of groups 0-3 (the hot ones under alpha=1.5)
        pg = d["per_group"]
        hot = [pg[str(g)] for g in range(4) if pg[str(g)]["n"] > 0]
        hot_mean = sum(g["mean_ms"] * g["n"] for g in hot) / max(1, sum(g["n"] for g in hot))
        print(f"{arm:<8} {t:<6} {d['mean_req_ms']:>10.1f} {d['median_req_ms']:>10.1f} {d['cache_hit_pct']:>7.1f}% {hot_mean:>18.1f}")
PY
