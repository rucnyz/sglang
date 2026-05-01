#!/bin/bash
# Concrete L2-positive workload search — under-allocated mamba pool
# (mamba_full_memory_ratio=0.5) on v9-auto trace. Hypothesis: with
# mamba pool cut roughly in half, L1 alone has to evict more
# aggressively, so L2's cross-pool transfer can genuinely add capacity
# to mamba and improve recovery TTFT / hit rate.
#
# 2 cells (L10, L11) × 3 trials × 3 phases ≈ 30 min on GPU 2.
# Success: L11 statistically separable from L10 on Phase A
# (|Δμ| > combined σ).

set -euo pipefail
cd /scratch/yuzhou/projects/sglang
export PATH=/scratch/yuzhou/projects/sglang/.venv/bin:$PATH
export PYTHONPATH=/data/yuzhou/projects/sglang/python:${PYTHONPATH:-}
GPU="${GPU:-2}"
PORT_BASE="${PORT_BASE:-30500}"
N_TRIALS="${N_TRIALS:-3}"
RUN_NAME="${RUN_NAME:-l2-positive-$(date +%Y%m%d-%H%M%S)}"
ROOT="dev/eval/runs/$RUN_NAME"
mkdir -p "$ROOT"
echo "[l2-positive] root=$ROOT gpu=$GPU n_trials=$N_TRIALS"

CELLS=("1 0" "1 1")
idx=0
for trial in $(seq 1 $N_TRIALS); do
  for pair in "${CELLS[@]}"; do
    set -- $pair
    L1=$1; L2=$2
    cell="L1${L1}_L2${L2}"
    out_dir="$ROOT/trial${trial}_${cell}"
    mkdir -p "$out_dir"
    port=$((PORT_BASE + idx))
    idx=$((idx + 1))
    echo
    echo "[l2-positive] trial=$trial cell=$cell port=$port (mamba_ratio=0.5)"
    # MAMBA_FULL_MEMORY_RATIO is engine flag, not env. Pipe through
    # sglang's launch_server via EXTRA_FLAGS in _common.sh
    ONLY_L1=$L1 ONLY_L2=$L2 \
      CUDA_VISIBLE_DEVICES=$GPU PORT=$port OUT_DIR="$out_dir" \
      SGLANG_K_BIG_AUTO_THRESHOLD=0.5 \
      EXTRA_FLAGS="--mamba-full-memory-ratio 0.5" \
      bash dev/eval/21_setting1_v9_pool_binding.sh \
      2>&1 | tee "$out_dir/runner.log" || echo "[l2-positive] $cell trial$trial FAILED"
  done
done

echo
echo "=========================================================="
echo "[l2-positive] SUMMARY ($ROOT)"
echo "=========================================================="
python3 - <<PY
import json, os, statistics
root = "$ROOT"
n_trials = $N_TRIALS

cells = [(1,0), (1,1)]
phases = ["A", "B", "C"]
print(f"\n{'cell':<10}{'phase':<6}{'mean_ttft':>14}{'p99_ttft':>14}{'tps':>10}")
print("-"*54)
data = {}
for cell in cells:
    cstr = f"L1{cell[0]}_L2{cell[1]}"
    for phase in phases:
        ttfts, p99s, tpss = [], [], []
        for t in range(1, n_trials+1):
            fp = f"{root}/trial{t}_{cstr}/{cstr}_phase_{phase}_bench.json"
            if not os.path.exists(fp): continue
            with open(fp) as f:
                lines = [l for l in f if l.strip()]
            d = json.loads(lines[-1])
            ttfts.append(d['mean_ttft_ms'])
            p99s.append(d['p99_ttft_ms'])
            tpss.append(d['input_throughput'])
        if not ttfts: continue
        data[(cstr, phase)] = (ttfts, p99s, tpss)
        m_t = statistics.mean(ttfts); s_t = statistics.stdev(ttfts) if len(ttfts)>1 else 0
        m_p = statistics.mean(p99s); s_p = statistics.stdev(p99s) if len(p99s)>1 else 0
        m_tps = statistics.mean(tpss)
        print(f"{cstr:<10}{phase:<6}{m_t:>8.1f}±{s_t:<5.1f}{m_p:>8.1f}±{s_p:<5.1f}{m_tps:>10.0f}")

print()
# success check
for phase in phases:
    if ('L11_L20' if False else 'L11_L21', phase) in data and ('L11_L20', phase) in data:
        pass
    if (f"L11_L20", phase) in data and (f"L11_L21", phase) in data:
        l10_t, l10_p, _ = data[("L11_L20", phase)]
        l11_t, l11_p, _ = data[("L11_L21", phase)]
        m10 = statistics.mean(l10_t); s10 = statistics.stdev(l10_t) if len(l10_t)>1 else 0
        m11 = statistics.mean(l11_t); s11 = statistics.stdev(l11_t) if len(l11_t)>1 else 0
        delta = m11 - m10; combined_sigma = s10 + s11
        sig = "STATISTICALLY SEPARATED" if abs(delta) > combined_sigma else "within noise"
        sign = "+" if delta > 0 else ""
        print(f"Phase {phase}: L11 - L10 mean_ttft = {sign}{delta:.1f} ms (combined σ {combined_sigma:.1f}) → {sig}")
PY
