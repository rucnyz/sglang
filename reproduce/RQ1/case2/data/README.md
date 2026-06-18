# case2 trace

Holds `cc_qwen_case2_swarm.jsonl` (~169 MB, gitignored): the short-swarm trace
case2 replays (mamba-bound, k2m direction). BUILT from case1's `cc_qwen_t6.jsonl`.

Populate: `python reproduce/RQ1/case_default_build.py` (writes here + case3/data;
needs case1's trace present as the swarm source).
