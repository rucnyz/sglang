# bench — cost micro-benches

Standalone scripts measuring the cost constants the cost model
consumes. Each maps to a `c_*` quantity in
[`../design.md`](../design.md) §"Shared cost model".

| file | measures | design.md ref |
|---|---|---|
| `bench_cumem_costs.py` | `c^xfer`: cuMemUnmap + cuMemMap host wall per chunk, p50/p99 across n ∈ {4, 8, 16, 30} | §"Shared cost model" → `c^xfer(X)` |
| `bench_snapshot_cost.py` | `c_m`: side-stream data-copy cost for `migrate_slot` (per-slot bytes × side-stream BW) | §"Shared cost model" → `c_m(X)` |
| `bench_graph_unmap_race.py` | reproduces the `cudaErrorIllegalAddress` race between captured CUDA graph replay and concurrent cuMemUnmap (closed by A1 property; see [`../0_page_state_machine/`](../0_page_state_machine/)) | §"Threading model" property A1 |

Run:

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python dev/interlayer/bench/<file>
```
