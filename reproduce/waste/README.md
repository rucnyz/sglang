# reproduce/waste: pool waste under the static split

Measures the HBM that the fixed boot split (`mamba_full_memory_ratio`) wastes on
each of the three design workloads (design.md "The problem" / "Where we win").
Waste is the idle fraction of the pool that is NOT binding while the other pool
is saturated, i.e. the memory the cross-pool layer could reclaim.

All three workloads are curated from the real cc agentreplay trace
(`make_slices.py`): case1 selects the longest sessions, case2 truncates first
turns into a short-Q&A swarm, case3 concatenates a long phase then a short phase.

## The two commands (per case)

```bash
# 1. serve a BASELINE sglang (LRU, no Budgeter) on a given GPU + port
bash serve.sh <gpu> <port>            # writes the batch-usage log we parse

# 2. replay the case's curated trace through the token-exact harness
bash replay.sh <port> <case> <outdir> # case in {case1,case2,case3}
```

`run_case.sh <case> <gpu> <port>` wraps both (serve, wait-ready, replay, kill,
parse) and writes `case<N>/results/`. `run_all.sh` runs all three sequentially.

## The parse script

```bash
python parse_waste.py <server.log> --out <dir> --label <name>
```

Reads the per-batch usage sglang already logs (`full token usage`,
`mamba usage`, `#running-req`, `#queue-req`), and emits:
- `waste.csv` per-tick timeseries,
- `waste.png` KV and mamba usage over wall-clock time (the flip is visible in case3),
- `summary.json` headline waste numbers.

Headline fields: `kv_bound_frac` (ticks KV saturated), `mamba_idle_at_kv` (idle
mamba while KV is the wall, the case-1 waste), `mamba_bound_frac` /
`kv_idle_at_mamba` (the case-2 waste), and `queue_*` (the cost the waste imposes).

Note: `mamba usage` is total occupancy (live + cached); cached snapshots are
reclaimable, so `1 - usage` is a lower bound on the truly borrowable mamba (the
live-only figure in design.md is larger).

## Cases

| case | workload | binding pool | waste measured |
|---|---|---|---|
| case1 | long-horizon agents (longest cc sessions) | KV | idle mamba while KV full |
| case2 | large-concurrency short Q&A (truncated swarm) | mamba | idle KV while mamba full |
| case3 | dynamic (long phase then short phase) | flips | both, over time |

Results land in `case<N>/results/{waste.csv,waste.png,summary.json,server.log}`.
