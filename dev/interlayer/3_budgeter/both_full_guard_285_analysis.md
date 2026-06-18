# #285 / #297 — both-full guard & working-set floor: analysis + planned A/B

Study of whether to relax the cross-fire **both-full guard** (#285) and/or
rebuild the **allocator floor** (#297) as the design's working-set ideal.
No code changed yet — this is the measurement-gated prep.

## What the guards actually are (read of `xpool_planner.py`)

The NB direction planner has TWO guards before it commits a fire:

1. **Per-direction saturation guard** — already ACTIVE-aware:
   - `kv_active_for_guard = usage_kv`; `nb_k2m = -inf` if it ≥ `kv_high_water`.
   - `m_active_for_guard = snapshot["usage_mamba_active"]` (active slots only,
     NOT cache); `nb_m2k = -inf` if it ≥ `mamba_high_water`.
   - So a pool is blocked from *shrinking* only when its **active/live** usage
     is saturated — cold cached snapshots do NOT count. This is the
     working-set-correct behavior, and it is already shipped.

2. **Both-full guard** (the #285 target) — CACHE-INCLUSIVE:
   - `snap_occ_kv = snapshot["pool_occupancy_kv"]`,
     `snap_occ_m = snapshot["pool_occupancy_mamba"]` (these INCLUDE cached
     bytes), suppresses BOTH directions when both ≥ high-water.
   - Added (per its comment) to prevent a real harm: when both caches are
     full, draining a "cold" mamba snapshot orphans a still-hot PAIRED KV
     prefix (`evict_full` drops both) → cache_hit crater (observed −40pp,
     m2k×27 on cc/conc22). The NB could not see this because it priced the
     drain on the victim's OWN reuse (mamba, tiny), not the paired KV.

## The inconsistency, and why #285 is now safe

The two guards disagree on what "full" means: the per-direction guard uses
**active**, the both-full guard uses **cache-inclusive**. So when BOTH pools
are full of COLD cache (active LOW, cache-inclusive HIGH), the per-direction
guards would allow a fire (real reclaimable slack), but the both-full guard
**overrides and suppresses it** — exactly the regime where reclaiming cold
cache is beneficial. That over-suppression is #285.

The both-full guard was a blunt bolt-on for the NB's pricing blind spot. **#268
(landed) fixes that blind spot**: `predict_evict_cost_us` now prices an
internal/KV-stays mamba drain by the whole-prefix TOTAL (`c_kv + c_m`), so the
NB itself sees the paired-KV value and rejects a fire that would orphan hot
cache. With the harm now priced in the NB, the blunt cache-inclusive guard can
be relaxed.

**#285 = key the both-full guard on `usage_*_active` instead of
`pool_occupancy_*`.** Note this makes it *redundant* with the per-direction
active guards (if both actives ≥ high-water, both directions are already
`-inf`), so the practical effect is to **remove the cold-cache
over-suppression**. There is already a flag: `SGLANG_XPOOL_BOTH_FULL_GUARD`
(default 1) — so the A/B is a one-env-var change, no code edit needed to test.

## #297 is likely SUBSUMED by #285

The static floor, when `shared_arena` (WIDE on), is
`mamba_static_min_chunks = 1` (`memory_pool.py`) — i.e. the pool may shrink to
1 chunk. Combined with the per-direction ACTIVE guard (won't shrink below the
active working set), the shipped behavior is already "shrink toward the active
working set" — which IS the working-set floor the design (#297) wants. The
only thing forcing it to be more conservative is the cache-inclusive both-full
guard. So **relaxing the guard (#285) delivers the #297 ideal**; re-architecting
the floor (`kv_min_now`/`mamba_min_now`/`safety_margin`) is probably
unnecessary. Defer #297 unless, after #285, a measurement still shows the
1-chunk static floor blocking a beneficial fire.

## The measurement that decides #285 (BLOCKED on a free GPU)

A/B under MAMBA PRESSURE (`MAX_MAMBA_CACHE=64` → mamba binds, both pools go
cache-full), request-bounded, with #268 in place:

| cell | `SGLANG_XPOOL_BOTH_FULL_GUARD` | expectation |
|---|---|---|
| guard-on (shipped) | 1 | both-full guard suppresses fires → few/0 fires, no cold-cache reclaim |
| guard-off | 0 | fires allowed; #268 prices the coupling harm → **cache_hit must NOT crater** (no −40pp), and tps/TTFT should improve from reclaiming cold cache |

**Decision rule:** if guard-off (with #268) fires more, improves tps/TTFT, AND
cache_hit holds (the −40pp does not return) → #285 is validated: relax the
both-full guard to active-based (or default the flag off). If guard-off
re-craters cache_hit → #268 alone is insufficient and the guard stays.

Node is currently fully contended (all 8 GPUs busy); this A/B is queued behind
a free GPU (see the auto-waiter). The first mamba-pressure run was killed when
a neighbor grabbed the GPU mid-load (OOM) — a measurement-hygiene hazard, not a
mechanism result.
