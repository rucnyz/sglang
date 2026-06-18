# case3 trace

Holds `cc_qwen_case3_default.jsonl` (~313 MB, gitignored): the dynamic A->B trace
case3 replays (long KV-bound phase, then the case2 swarm). BUILT from case1's
`cc_qwen_t6.jsonl` (swarm source) + the agentreplay corpus `cc_qwen3p5_9b.jsonl`
(long phase-A source, ~2.3 GB, stays in the corpus).

Populate: `python reproduce/RQ1/case_default_build.py` (writes here + case2/data;
needs case1's trace + the corpus long source).
