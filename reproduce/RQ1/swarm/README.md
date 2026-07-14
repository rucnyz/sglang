# RQ1 Swarm (recurrent-bound, k2m) — crash-free + oscillation-free; throughput win NOT available on 9B/H200

The k2m swarm was pursued as a THROUGHPUT headline (grow the mamba pool for
higher admission concurrency).
Outcome after four clean root-cause fixes: the swarm is now fully crash-free and
the planner behaves correctly (no k2m/m2k oscillation), but there is **no
throughput win** on Qwen3.5-9B / H200 with the default static split.
The mechanism is correct; the regime does not reward it.

## Four root-cause fixes (each: reproducing test FIRST, then fix)

| # | fix | root cause | test |
|---|-----|-----------|------|
| #339 | `PrefillAdder.rem_mamba_slots` per-req mamba budget | mamba admission had no per-req budget; a COW-hit req counts its own shared-prefix snapshot as reclaimable, but it is locked as the COW source and evaporates → over-admit → `alloc_req_slots` RAISE | `1_dyn_admission_cap/test_mamba_allocator.py` |
| #340 | actuator `grow_headroom_pages()` clamp | arena chunk-id space (80 GiB headroom) >> allocator `CappedFreeList` ceiling, so a k2m grow near the ceiling hands out chunk ids past `max_size` → `unmark: id N exceeds ceiling` | same file |
| #341 | k2m KV working-set floor `kv_used + chunk` | k2m drained KV below one prefill chunk → `alloc_token_slots` OOM; must reserve genuinely-free (not evaporate-able evictable) capacity | `3_budgeter/mamba_drain_floor/test_kv_working_set_floor.py` |
| #342 | LPB-loss eviction cost (planner) | planner priced eviction by RAW token/slot COUNT, so the swarm's low-reuse KV churn (hit 0.15) read `R_kv=50157` vs `R_m=0` → drained mamba every tick, fighting the Admitter's grows → oscillation | `3_budgeter/payback/test_lpb_loss_eviction_cost.py` |

#342 is the substantive one: the LPB loss of never-reused cache is ~0, so the
planner stops firing spurious m2k.
Oscillation went from 278 k2m / 389 m2k (fighting) to 574 k2m / 60 m2k
(coherent grow), `R_m >> R_kv`, mamba grows to its max.

## E2E A/B (short-prompt swarm, default config, conc 400, N=1, crash=0 both arms)

| trace | arm | tps | peak / avg conc | wall |
|-------|-----|-----|-----------------|------|
| p50=1587 | base | 1344 | 195 | 472 |
| p50=1587 | sys  | 1271 (−5%) | 246 | 500 |
| 512-tok (hit 0.87) | base | 2570 | 195 / 127 | 247 |
| 512-tok (hit 0.87) | sys  | 2466 (−4%) | 246 / 147 | 258 |

## Why no throughput win (the decisive finding)

`sys` sustains higher concurrency (avg 147 vs 127) and fewer decode steps, but
each step is slower: a larger decode batch loads more full-attention KV, so the
per-step cost rises faster than the batch.
At batch ~195 a 9B hybrid on H200 is already compute/attention-bound, so growing
mamba past the base cap adds no throughput, only HiMA overhead (fires +
Budgeter/Admitter/LPB machinery) → −4/−5%.
The default static mamba pool (ratio 0.9 → cap 195) already matches the
GPU-efficient concurrency; the swarm often does not even saturate it (avg 127 <
195).
A k2m throughput win would need an under-provisioned static mamba split (a low
`--mamba-full-memory-ratio`, a server setting we do not change), or a model whose
default split is bad for the workload.

The RQ1 throughput win is the **m2k / KV-bound (long-horizon)** direction, where
the default over-provisions mamba and HiMA drains it to grow KV (documented
+10.8%, N=3).
The swarm's contribution is crash-free + correct-planner robustness and a TTFT
lever, not throughput.

## Reproduce

```
bash reproduce/RQ1/run_arm.sh base <trace> 0.02 400 - 1 <out>   # then sys
# traces: agentreplay/data/traces/cc_qwen_swarm_short.jsonl (p50=1587),
#         cc_qwen_swarm_tiny.jsonl (512-tok truncation)
```
