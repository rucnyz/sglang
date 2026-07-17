#!/bin/bash
set -u
HERE=/scratch/yuzhou/projects/sglang/reproduce/waste
GPU=0; PORT=30098
for c in case3 case1; do
  echo "=== $(date) START $c ==="
  bash "$HERE/run_case.sh" "$c" "$GPU" "$PORT" 2>&1 | sed "s/^/[$c] /"
done
echo "=== $(date) ALL DONE ==="
