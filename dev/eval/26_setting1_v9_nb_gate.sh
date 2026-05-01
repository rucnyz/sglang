#!/bin/bash
# Setting 1 v9-auto net-benefit gate sweep — second attempt to close
# Phase A regression after gate-retune showed MAMBA_HIGH doesn't help.
#
# The paper §design-l2 describes net-benefit gating: a fire is rejected
# unless expected benefit (avoided re-prefill from saturation pressure)
# exceeds α·C_act = 4.5s. Default SGLANG_XPOOL_NET_BENEFIT=0; here we
# enable it and sweep the cooldown ticks too.
#
# Hypothesis: Phase A's mamba saturation has no admission-pressure
# (no paused/retracted reqs since stock cache evicts aggressively),
# so net-benefit gate's B_lb computes to zero and rejects the fire.
# This eliminates the 15 actuator fires' cost on Phase A.
#
# 4 settings × ~12 min = 48 min wall.

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30300}"
RUN_NAME="${RUN_NAME:-nb-gate-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[nb-gate] root=$ROOT gpu=$GPU"

# (config_name, NET_BENEFIT, COOLDOWN)
configs=(
  "default 0 2"
  "nb_on  1 2"
  "cd_high 0 20"
  "nb+cd  1 20"
)

idx=0
for cfg in "${configs[@]}"; do
  read -r name nb cd <<< "$cfg"
  out_dir="$ROOT/$name"
  mkdir -p "$out_dir"
  port=$((PORT_BASE + idx))
  idx=$((idx + 1))
  echo
  echo "=========================================================="
  echo "[nb-gate] $name (NET_BENEFIT=$nb COOLDOWN=$cd) port=$port"
  echo "=========================================================="
  ONLY_L1=1 ONLY_L2=1 \
    CUDA_VISIBLE_DEVICES=$GPU PORT=$port OUT_DIR="$out_dir" \
    SGLANG_K_BIG_AUTO_THRESHOLD=0.5 \
    SGLANG_XPOOL_NET_BENEFIT=$nb \
    SGLANG_XPOOL_COOLDOWN=$cd \
    bash dev/eval/21_setting1_v9_pool_binding.sh \
    2>&1 | tee "$out_dir/runner.log" || echo "[nb-gate] $name FAILED"
done

echo
echo "=========================================================="
echo "[nb-gate] SUMMARY ($ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os
root = "$ROOT"
print(f"\n{'config':<14}{'phase':<6}{'mean_ttft':>11}{'p99_ttft':>11}{'tps':>10}{'fires':>7}")
print("-"*60)
for name in ["default", "nb_on", "cd_high", "nb+cd"]:
    bj = f"{root}/{name}/L11_L21_budgeter.jsonl"
    fires = 0
    if os.path.exists(bj):
        try:
            with open(bj) as f:
                for line in f:
                    if '"xpool_direction":' in line and '"none"' not in line:
                        fires += 1
        except Exception:
            pass
    for ph in ["A","B","C"]:
        fp = f"{root}/{name}/L11_L21_phase_{ph}_bench.json"
        if not os.path.exists(fp): continue
        with open(fp) as f:
            lines = [l for l in f if l.strip()]
        d = json.loads(lines[-1])
        f_disp = fires if ph == "A" else ""
        print(f"{name:<14}{ph:<6}{d['mean_ttft_ms']:>11.1f}{d['p99_ttft_ms']:>11.1f}{d['input_throughput']:>10.0f}{f_disp!s:>7}")

print("\nReference: (0,0) Phase A 1818ms, (1,1) default Phase A 3105ms")
PY
