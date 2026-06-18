#!/bin/bash
# CORRECTED case3 (dynamic, workload-driven binding FLIP) at the DEFAULT split.
#
# METHODOLOGY (load-bearing): this driver does NOT export RATIO, so run_arm.sh
# omits --mamba-full-memory-ratio and BOTH arms boot at sglang's DEFAULT split
# (mamba_full_memory_ratio=0.9 -> mamba ~21GB / max_running ~147, KV ~24GB /
# ~560k tokens). The prior case3_dyn.sh forced the regime with RATIO=0.1; now
# forbidden. The A->B binding flip comes ENTIRELY from the workload on ONE
# timeline (trace cc_qwen_case3_default.jsonl):
#   Phase A [0,150s]:   ~10 real LONG (60-200k) root-only sessions. At the
#       default split a few long contexts saturate the ~560k KV pool before
#       max_running 147 -> KV-bound, mamba idle. sys grows KV from idle mamba (m2k).
#   Phase B [~1000,1018s]: the SAME ~288 short swarm as case2_default, arriving
#       after A drains. At the default split max_running ~147 binds with KV idle
#       -> mamba-bound. sys grows mamba from idle KV (k2m).
# At ONE fixed default split the bind flips A(KV)->B(mamba) purely from load;
# base (static default) is wrong-sized for one phase. sys must reallocate BOTH ways.
#
# Arrival = absolute trace t (STAGGER="-", run_arm.sh drops --stagger) so the
# recorded A->B timeline drives the flip; a fixed uniform stagger would erase it.
# gap-scale 0 zeroes tool gaps (arrival-driven timeline).
#
# the phase-B k2m grow can lift max_running past 147, as in case2. Allowed.
# N reps on ONE server per arm (run_arm.sh's native NREPS loop, --flush between
# reps); the #327 flush-boundary crash is fixed, so per-rep reboot is gone.
set -u
cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case3/data/cc_qwen_case3_default.jsonl
# NO RATIO export -> default split.
NREPS="${NREPS:-1}"   # N=1 per arm smoke; widen to 3 once the flip is confirmed.

OUT=reproduce/RQ1/case3/runs
echo "[case3-default] $(date) base N=$NREPS"
bash reproduce/RQ1/run_arm.sh base "$TRACE" - 256 - "$NREPS" "$OUT" 2>&1 | sed 's/^/[base] /'
echo "[case3-default] $(date) sys N=$NREPS"
bash reproduce/RQ1/run_arm.sh sys  "$TRACE" - 256 - "$NREPS" "$OUT" 2>&1 | sed 's/^/[sys] /'
echo "[case3-default] $(date) DONE"
