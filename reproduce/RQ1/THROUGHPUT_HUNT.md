# RETRACTED: this doc's "no throughput win" conclusion was wrong

**The RQ1 throughput win is real and documented in `FINDINGS.md`**
(Qwen3.5-9B, H200, agentreplay token-exact, N=3): Case1 long-horizon KV-bound
+4.4%, Case2 swarm +5.5%, Case3 dynamic +7.0% tps, with TTFT −45% to −70%.

This doc previously argued "no throughput win, HiMA is only a TTFT/latency
optimizer." That conclusion was reached by building EXTREME synthetic
constructions and then over-generalizing from them, overriding the real
FINDINGS result. It was wrong. Do not trust the retracted conclusion.

## The one technical fact worth keeping

Two EXTREME constructions do push the GPU compute-bound and show flat/negative
throughput, but they are NOT the realistic operating point:

- **`build_longburst`** (187 independent, co-arriving 76k prompts): binds KV
  hard, but at that context length the GPU is compute-bound (util 99% at the
  base admission cap), so growing KV admits more at ~0 extra throughput → flat.
- **hard-truncated 512-tok swarm** (`cc_qwen_swarm_short/tiny`): pushes past the
  GPU-efficient concurrency → −4/−5% (see `swarm/README.md`).

These are stress probes, not the RQ1 workloads. The realistic multi-turn CC
traces (`t6_v2`, `t12`) at conc 64/128 — the actual RQ1 Cases 1/2/3 — bind their
pool BELOW the compute knee, which is exactly where HiMA's cross-pool resize
earns throughput (+4.4–7%) on top of the TTFT win. The lesson: a harder,
non-representative construction does not override an N=3 win on the real
workload.

Source of truth: `FINDINGS.md` + `run_official_case123.sh`.
