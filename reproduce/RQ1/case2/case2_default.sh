#!/bin/bash
# CORRECTED case2 (mamba-bound, k2m) at the DEFAULT mamba/KV split.
#
# METHODOLOGY (load-bearing): this driver does NOT export RATIO, so run_arm.sh
# omits --mamba-full-memory-ratio and BOTH arms boot at sglang's DEFAULT split
# (mamba_full_memory_ratio=0.9 -> mamba ~21GB / max_running ~147, KV ~24GB /
# ~560k tokens). The prior case2_n3.sh forced the regime with RATIO=0.1; that is
# now forbidden. The mamba-bound regime here comes ENTIRELY from the workload:
# the trace cc_qwen_case2_swarm.jsonl is ~288 SHORT (~3.5k-prompt + ~192-out)
# childless roots that swarm in over [0,18s] at --max-concurrency 256, so
# max_running ~147 binds (queue builds) while KV sits ~idle -> mamba-bound. sys
# grows mamba from idle KV (k2m) so max_running rises past 147 and the swarm
# drains; base (static default) cannot.
#
# array so the dynamic cap can grow max_running past the default 147. Allowed.
# It is exported for the sys arm only (run_arm.sh sys reads SGLANG_* envs); the
# base arm ignores it.
#
# N reps on ONE server per arm (run_arm.sh's native NREPS loop, --flush between
# reps); the #327 flush-boundary crash is fixed, so per-rep reboot is gone.
# --stagger 0.02 = swarm (uniform tight stagger, NOT the trace timeline).
set -u
cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case2/data/cc_qwen_case2_swarm.jsonl
OUT=reproduce/RQ1/case2/runs
# NO RATIO export -> default split. CTXLEN bounds single-seq length (fits the
# ~560k KV pool); STAGGER 0.02 swarm; --max-concurrency 256.
NREPS="${NREPS:-1}"   # NREPS=1 smoke (default); set NREPS=3 for the structured run.

echo "[case2-default] $(date) base N=$NREPS"
bash reproduce/RQ1/run_arm.sh base "$TRACE" 0.02 256 - "$NREPS" "$OUT" 2>&1 | sed 's/^/[base] /'
echo "[case2-default] $(date) sys N=$NREPS"
bash reproduce/RQ1/run_arm.sh sys  "$TRACE" 0.02 256 - "$NREPS" "$OUT" 2>&1 | sed 's/^/[sys] /'
echo "[case2-default] $(date) DONE"
