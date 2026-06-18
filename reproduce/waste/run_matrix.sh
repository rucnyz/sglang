#!/bin/bash
# Static-split sweep: each case across mamba_full_memory_ratio in
# {0.05, 0.3, default(=0.9), 2.0}, spanning very-KV-heavy -> sglang-default ->
# mamba-heavy. The split is set ONLY via --mamba-full-memory-ratio (the design's
# own knob); "default" passes no flag (sglang out-of-box). static-best per case
# = the ratio minimizing queue; case3 (dynamic) should have no good single
# ratio. Three per-GPU sequences run in parallel (GPU 0/5/6).
set -u
H=/scratch/yuzhou/projects/sglang/reproduce/waste
RATIOS="0.05 0.3 default 2.0"
run_gpu() {  # <gpu> <port> <case>
  local gpu=$1 port=$2 c=$3
  for r in $RATIOS; do
    echo "=== $(date) $c ratio=$r (gpu $gpu) ==="
    bash "$H/run_split.sh" "$c" "$r" "$gpu" "$port" "$H/$c/results/ratio_${r}" 2>&1 | sed "s/^/[$c.$r] /"
  done
}
run_gpu 0 30098 case1 > "$H/case1/sweep.log" 2>&1 &
run_gpu 5 30099 case2 > "$H/case2/sweep.log" 2>&1 &
run_gpu 6 30100 case3a > "$H/case3a/sweep.log" 2>&1 &
wait
echo "=== $(date) MATRIX DONE ==="
