# T10 / T11 — first working anywhere-free smoke

Single-trial 2-cell (L2 on/off) smoke at M2 swarm conc=800, D(256,256),
8 min on Qwen3.5-35B-A3B / H200. Validates the post-T8 / T9 / T10 / T11
stack actually fires cross-pool transfers under live traffic without
crashing.

## Runs

```
results/t10-smoke-20260505-012447/   # 1st smoke — exposed OwnerMap dedup bug
                                     # plans=0 (refused on assert_complete: 120 pages
                                     # double-counted as free + active because stale
                                     # req_to_token row on released req_pool_idx)

results/t10-smoke-fix1-013709/       # post-OwnerMap dedup — exposed page-vs-token-slot
                                     # confusion: planner emitted token-slot ids as if
                                     # they were chunk ids, way out of arena's chunk
                                     # range, shrink_explicit silently dropped them
                                     # (fire 1 plan, but unmapped=0).

results/t11-cleanup-021204/          # post-T11 page-only refactor — *first physical
                                     # cross-pool transfer to actually move pages*.
                                     # 1 plan, 0 abort, unmap=20, grant=20,
                                     # fire wall-time 67 ms.
```

## What t11-cleanup-021204 proves

- The page-only API (planner sees pages; token-slot translation hidden in
  ArenaActuator) prevents the chunk-vs-page confusion that broke fix1.
- cap-barrier translates 1 page → 2048 token-slots correctly (52 ms — dominated
  by `mark_pages_capped` on a 2048-element tensor).
- `cuMemUnmap` × 20 KV subpools: 20 pages physically released.
- `cuMemMap` × 20 mamba subpools: 20 handles bound, mamba pool grew.
- verify did not abort.
- Engine kept serving for the full 8 min after the fire; no illegal-access
  crash (the T7 v3 class is fixed).

## Why only 1 fire in the 8-min bench

`cross_pool_planner` triggers a fire when `nb_direction` clears the gate
(benefit > α · c_a). After the first fire moved 20 pages, mamba pressure
relieved enough that subsequent ticks couldn't clear the threshold. Plus
the cooldown is 30 ticks × 2 s = 60 s, which kept the planner muted for
the first minute post-fire.

This is a **decision-frequency knob**, separate from the fire mechanism.
For multi-fire smokes:
- shrink `SGLANG_XPOOL_COOLDOWN` (e.g. 5 ticks)
- or run longer benches (e.g. 30 min)
- or higher pressure (e.g. concurrency 1000+)

## Files per cell

```
{cell}_server.log         # SGLang server log; grep "execute[seq=" for fire events
{cell}_client.log         # genai-bench client log
{cell}_budgeter.jsonl     # per-tick budgeter snapshot (KV/mamba usage,
                          # plan_direction, plan_reason, t8 fields)
genai_results/            # per-request bench output (TTFT, TPOT, etc.)
```

## How to read budgeter.jsonl

Each line = one tick. Useful fields:

- `xpool_plan_direction`: planner's verdict (none / kv_to_mamba / mamba_to_kv)
- `xpool_plan_reason`: why fire/skip
- `xpool_plan_executed`: True if a plan was attempted this tick
- `xpool_t8_plan_seq`: plan_seq when T8 path fires
- `xpool_unmapped_total` / `xpool_granted_total`: physical move counts
- `xpool_fire_total_us`: wall time for the fire
