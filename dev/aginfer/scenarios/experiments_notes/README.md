# experiments_notes — cross-cutting non-experiment docs

Findings, debugging notes, simulation baselines, and gap catalogs
that don't belong to one specific scenario.

| file | what |
|---|---|
| [`GAPS.md`](GAPS.md) | catalog of promised-but-untested mechanisms (G1–G11) |
| [`instrument_chain.md`](instrument_chain.md) | observability scaffold story (structured `aginfer_metric` logging) |
| [`runaway_tail.md`](runaway_tail.md) | 1 % requests = 80 % wall finding (root cause of weak §8 N=3 signal) |
| [`ttft_analysis.md`](ttft_analysis.md) | per-request TTFT distribution + cache hit ratio ceiling |
| [`swa_assert_hypothesis.md`](swa_assert_hypothesis.md) | code-trace hypothesis for the SWA assert (now fixed) |
| [`controller_decline_root.md`](controller_decline_root.md) | (C-deeper) load_back declines → SWA sub-pool root cause |
| [`algo_baselines_sim.md`](algo_baselines_sim.md) | paper §8 algorithm simulation: LRU / TA / InferCept / Continuum / KVFlow / Ours under same trace |

Optional: `evidence/` subfolder with the cycle dirs that back
specific findings (e.g. instrument_chain cycles).
