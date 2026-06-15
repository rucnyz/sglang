# archive — legacy, kept for the record (do NOT use for new work)

## scenarios_harbor/
The old **harbor + standalone-sglang** experiment platform: per-scenario run folders
(`1_swebench_default`, `2_hbm_pressure`, `3_high_concurrency`, `4_ablation`,
`5_eviction_characterization`, `donoharm_fix228`, `replay`, `_legacy`, `_shared`,
`experiments_notes`) with raw harbor/daemon/sglang logs + analyses. Superseded by the
**Dynamo** platform (`dev/dynamo/`) + the paper reproduce packages
(`sglang/reproduce/RQ1/scenarios/`). The `#230` eviction-characterization suite
(Tier-1 CPU sim + Tier-2 e2e) lived under `scenarios_harbor/5_eviction_characterization/`
— resurrect from here if that task is revived.

## workload/
The §2.4 Agent-DAG data model (`agent_dag.py`) — online-revealed DAG + event stream for the
offline simulation harness. Its only importer was `scenarios_harbor/5_eviction_characterization/`
(also archived); no live code uses it. NOTE: `baselines/` is NOT here — despite its name it is
the **live** daemon's core policy library (types/cost/OursGreedy/knapsack/adapter), kept in place.
