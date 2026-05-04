# T3: Smart over-cap selection via free-page mask

Task #75. Paper §3.2.2 second half: "When the inter-pool decision rule
fires a transfer of $n$ pages from $\sigma_{\text{src}}$ to
$\sigma_{\text{dst}}$, the actuator queries the allocator's per-page
free-mask ... and picks $n$ pages that are currently logically free ---
hold no live block and no cached block needing eviction."

## What changes

Today's `ChunkArena.shrink(pool, n, evict_policy="tail")` always picks
the **highest-indexed mapped slots** for unmap. This works only if the
tail is empty, which depends on natural request completion + placement
bias (T2). Under high churn or skewed allocation, the tail can have
live blocks; today the actuator's drain-ready check fails and the fire
aborts (~88% abort rate at chunk-grain).

T3 inverts the dependency: instead of "pick the tail and hope it's
empty," **the actuator asks the allocator which slots are currently
free and picks N of those, regardless of position**. This decouples
drain success from where free indices happen to be in the pool.

## Three-piece change

1. **Allocator API** (`BaseTokenToKVPoolAllocator`, `MambaPool`):
   - `free_page_mask() -> torch.Tensor[bool]` of size `pool.size`,
     `True` at indices currently in `free_pages` (no in-flight req
     holds them, no cached block needs eviction first).
   - `select_drain_pages(n) -> torch.Tensor[int]` returns up to n
     free-page indices, preferring contiguous high-index runs (so
     the unmap can batch into a single VA range when possible).

2. **Chunk-arena explicit-target shrink** (`ChunkArena`):
   - New `shrink_explicit(pool_name, slot_indices) -> int`: unmap the
     given slot indices, return how many actually unmapped. Fails
     gracefully on slots that aren't currently mapped.
   - Old `shrink(..., evict_policy="tail")` retained for back-compat.

3. **Actuator wiring** (`KVArenaActuator.set_capacity_pages`,
   `MambaArenaActuator.set_capacity_slots`):
   - When `SGLANG_SMART_OVERCAP=1`, query `allocator.select_drain_pages`
     to choose drain targets, then call `chunk_arena.shrink_explicit`.
   - Default path unchanged (tail-evict via existing `shrink`).

## Why this matters

Without T3, drain success at chunk-grain is ~12% (paper §3.2.1 / T1
`notes.md`). T1+T2 together push it higher by making tail naturally
free, but tail-evict still fails when the allocator's high indices
happen to hold any live block (random under bursty traffic, even with
placement bias). T3 picks any free slot, not just tail, so success rate
becomes pure availability: "$|free\_pages| \geq n$" — under T2 plus a
moderately under-pressured pool, this is virtually 100%. The remaining
abort cases (every free slot held by a live req) are what T4 (atomic
migration) covers as a final fallback.

## Flag

`SGLANG_SMART_OVERCAP=1` enables the new path. Default behavior
unchanged. Layered with T1 (`SGLANG_ARENA_CHUNK_BYTES=2 MiB`) and T2
(`SGLANG_ALLOCATOR_PLACEMENT_BIAS=1`), the three flags compose to give
the §3.2 ideal mode.

## How to verify

1. **Smoke** (`test/test_smoke.sh`): boot with all three flags on,
   serve a few prompts.
2. **Drain success rate** (`reproduce.sh`): run a benchmark that
   triggers cross-pool transfers (multi-turn swarm or M3 phase-shift
   trace) with all three flags on. Capture the budgeter log's
   `xpool_unmapped_total` per fire — fires-with-movement / total fires
   gives the commit-success rate. T1+T2 alone should give ~50–80%; T3
   should push it to ~95%.

## Status

- [x] Design note
- [x] `free_page_mask()` and `select_drain_pages(n)` on allocator
  (`python/sglang/srt/mem_cache/allocator.py:76-130`); unit test PASS
  in `results/unit_test_allocator_api.txt`
- [ ] `shrink_explicit(pool, slots)` on `ChunkArena`
- [ ] Actuator wiring under `SGLANG_SMART_OVERCAP=1`
- [ ] Smoke serving works under combined flags
- [ ] Drain success rate captured (log analysis)

## Followups

T4 (atomic migration) handles the residual case where every free slot
in the over-cap range is `< n`: actively migrate live blocks out so the
target range can drain.
