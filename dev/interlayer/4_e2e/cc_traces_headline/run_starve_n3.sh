#!/usr/bin/env bash
# N=3 median run of the mamba-starve regime (max-mamba-cache-size 64) to
# filter the single-run variance seen across the 2026-06-03 starve runs
# (off cache_hit swung 0.15 / 0.21 / 0.27). Three runs in parallel on
# separate GPUs, then median validation. Builtin cost curves (the saved
# profile has κ_M=0, #276).
set -uo pipefail
cd "$(dirname "$0")/../../../.." || exit 2

BASE=${BASE:-/tmp/d10_n3}
pids=()
for i in 1 2 3; do
    gpu=$((3 + i))          # GPU 4, 5, 6
    port=$((30090 + i))
    rm -rf "${BASE}_$i"
    MAX_MAMBA_CACHE=64 OUT_DIR="${BASE}_$i" GPU=$gpu PORT=$port \
        bash dev/interlayer/4_e2e/cc_traces_headline/run_cc.sh \
        > "${BASE}_$i.log" 2>&1 &
    pids+=($!)
    echo "[n3] launched run $i on GPU $gpu port $port (pid ${pids[-1]})"
    sleep 5   # stagger boots slightly
done

rc=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "[n3] run $((idx + 1)) exited non-zero (validator gate or boot)"
        rc=1
    fi
done

echo
echo "=== N=3 median validation ==="
/scratch/yuzhou/projects/sglang/.venv/bin/python \
    dev/interlayer/4_e2e/cc_traces_headline/validate_cc.py \
    --out-dirs "${BASE}_1" "${BASE}_2" "${BASE}_3"
echo "[n3] done (per-run rc=$rc; median verdict above)"
