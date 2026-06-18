# case1 data

`cc_qwen_case1_longkv.jsonl` (~707 MB, gitignored) — case1's trace: the 52 LONGEST
real-CC root sessions (prompt p50 92k / max 196k) selected from the 2.3 GB corpus
`cc_qwen3p5_9b.jsonl`. The KV-bound m2k burst (a handful saturate the ~560k KV
pool, the rest queue, mamba idle). Build:
`python reproduce/RQ1/case_default_build.py case1`.
