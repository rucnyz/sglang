#!/bin/bash
# case1 (KV-bound, m2k WIN): \sys vs baseline on a BURST of the LONGEST real cc
# roots (cc_qwen_case1_longkv = the 52 longest roots from the 2.3G corpus, prompt
# p50 92k / max 196k, built by case_default_build.py case1). At the sglang DEFAULT
# split only ~6 of 52 fit the ~560k KV pool, so KV binds and the rest QUEUE; each
# session is one O(1) mamba slot (52 << 147 -> mamba idle); the running set (~6)
# is small so prefill is admission-bound with HEADROOM (not compute-bound, which is
# why the old medium-ctx conc-64 trace only won +1.7%). \sys (run_arm sys) = lpb +
# Budgeter + Admitter cross-fire + PF64 + calibrated c_M=0; it grows KV from idle
# mamba (m2k) -> admits more long contexts -> prefill throughput UP, queue drains,
# ttft DOWN. base = lru, no Budgeter. N=3 both arms, conc 48 (deep queue), GPU 7.
set -u
cd /scratch/yuzhou/projects/sglang
TRACE=/scratch/yuzhou/projects/sglang/reproduce/RQ1/case1/data/cc_qwen_case1_longkv.jsonl
OUT=reproduce/RQ1/case1/runs
mkdir -p "$OUT"

echo "[case1] $(date) START base N=3"
bash reproduce/RQ1/run_arm.sh base "$TRACE" 0.02 48 - 3 "$OUT" 2>&1 | sed 's/^/[base] /'
echo "[case1] $(date) START sys N=3"
bash reproduce/RQ1/run_arm.sh sys  "$TRACE" 0.02 48 - 3 "$OUT" 2>&1 | sed 's/^/[sys] /'
echo "[case1] $(date) DONE -> $OUT"
