# T3 notes

## Three-part change recap

| Part | File | Change |
|---|---|---|
| 1 | `python/sglang/srt/mem_cache/allocator.py` | Added `free_page_mask()` and `select_drain_pages(n, prefer)` to `BaseTokenToKVPoolAllocator` |
| 2 | `python/sglang/srt/arena/chunk_arena.py` | Added `ChunkArena.shrink_explicit(pool, slot_indices)` |
| 3 | `python/sglang/srt/arena/cross_pool_actuator.py` | Added `_select_drainable_chunks()` helper + env-gated branch in shrink loop |

## Verification

| Test | Verifies | Result |
|---|---|---|
| `test/test_allocator_api.py` | Part 1 API correctness in isolation: mask shape, sentinel slot 0, alloc/free updates, prefer="high"/"low" semantics, capped n | PASS |
| `test/test_shrink_explicit.py` | Part 2 unmap mechanics: unmaps specified slots, skips already-unmapped / OOB silently, accepts list or tensor | PASS |
| `test/test_smoke.sh` | T1+T2+T3 flags compose: boot succeeds, T2 prerequisite log appears, 5 generates return | PASS (boot 110 s) |

## What this verifies

1. The allocator can answer "which pages are free" without going through HTTP / Python introspection.
2. The chunk arena can unmap an arbitrary explicit slot list (not just tail).
3. The actuator's env-gated branch picks chunks via the new helper without crashing the engine on boot or simple serving.

## What this does NOT verify

The smoke does not exercise the **fire path** — i.e., the cross-pool actuator deciding to shrink one pool and grow the other. That requires a workload that pushes a pool toward the admission ceiling (M2 swarm conc=800, or M3 phase-shift). Until T7 runs that workload with `SGLANG_SMART_OVERCAP=1`, the commit-success-rate uplift over T1+T2 alone is unobserved.

Specifically untested:
- Whether `_select_drainable_chunks` returns the **right** chunks under
  contention (tested only that the shape / count is sensible).
- Whether `shrink_explicit` plus the env-gated branch **commits** more
  fires than the `shrink("tail")` path on a real workload.
- Whether the page→chunk grouping (a `mask.view(n_chunks, tokens_per_chunk).all(dim=1)`) handles the case where `size` isn't a multiple of `tokens_per_chunk` (currently rounded down via `n_pages // tokens_per_chunk`; a few trailing pages may be hidden from the smart-selection scan but they're equally hidden from the tail-scan, so no asymmetric regression).

## Followup items

T4 (atomic page migration) will use the same `free_page_mask` API
plus a new "migrate live block out of these pages" primitive.

T7 (M2 swarm conc=800 validation) is where commit-success-rate is
actually measured — at that point the budgeter log's
`xpool_unmapped_total` per fire under all three flags vs T1+T2 only
gives the on/off delta this milestone delivers.

## Status

T3 done. Code change spans 3 files (~110 lines net). Two unit tests +
one smoke test all pass. Real-workload uplift deferred to T7.
