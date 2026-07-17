#!/bin/bash
# Per-case waste at sglang's DEFAULT static split (no override). Thin wrapper
# over run_split.sh with ratio=default. Use run_matrix.sh to sweep the split and
# find static-best. Usage: run_case.sh <case> <gpu> <port>
set -u
CASE=$1; GPU=$2; PORT=$3
HERE=/scratch/yuzhou/projects/sglang/reproduce/waste
bash "$HERE/run_split.sh" "$CASE" default "$GPU" "$PORT" "$HERE/$CASE/results/default"
