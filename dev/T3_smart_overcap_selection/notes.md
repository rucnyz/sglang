# T3 notes

## Three-part change recap

| Part | File | Change |
|---|---|---|
| 1 | `python/sglang/srt/mem_cache/allocator.py` | `free_page_mask()`, `select_drain_pages(n, prefer)`, `mark_pages_capped(ids)`, `unmark_pages_capped(ids)` on `BaseTokenToKVPoolAllocator` |
| 2 | `python/sglang/srt/arena/chunk_arena.py` | `ChunkArena.shrink_explicit(pool, slot_indices)` |
| 3 | `python/sglang/srt/arena/cross_pool_actuator.py` | `_select_drainable_chunks()` helper + env-gated `SGLANG_SMART_OVERCAP=1` branch in `shrink_then_grow` that picks chunks via the helper, calls `shrink_explicit`, then `mark_pages_capped` to prevent the allocator from handing out pages whose VA was just unmapped |

## Verification

| Test | Path | Verifies | Result |
|---|---|---|---|
| `test_allocator_api.py` | CPU/CUDA | Part 1 free_page_mask + select_drain_pages: shape, sentinel slot 0, alloc-then-prefer="high"/"low" semantics, capped-on-overrequest | PASS |
| `test_shrink_explicit.py` | small CUDA arena | Part 2: unmap given slots, return count, skip OOB / already-unmapped, accept list and tensor | PASS |
| `test_select_drainable_chunks.py` | pure Python with fake allocator | Helper picks all-free chunks, prefers high index, handles edge cases (nothing free / sparse free / no allocator) | PASS |
| `test_mark_pages_capped.py` | small CUDA | **Correctness invariant**: alloc never returns a page that was passed to mark_pages_capped, even when mask shows "free". Round-trip via unmark restores original state. | PASS |
| `test_smoke.sh` | full engine boot | T1+T2+T3 flags compose, server boots, 5 generates return, T2 prerequisite log line appears | PASS (boot 110 s) |

## Hidden bug found and fixed

The first version of T3 part 3 wired `shrink_explicit` directly without
also telling the allocator that the unmapped pages were no longer
mappable. This would have allowed the next `alloc()` to hand out a page
id whose underlying VA had just been `cuMemUnmap`'d → next kernel touch
would `cudaErrorIllegalAddress`. Caught by reasoning through the
allocator-state path, not by the smoke (smoke doesn't exercise fire +
re-alloc against the same pool).

Fix: added `mark_pages_capped(page_indices)` + `unmark_pages_capped()`
on `BaseTokenToKVPoolAllocator` and wired the cap-page call right after
`shrink_explicit` in cross_pool_actuator.py. The new
`test_mark_pages_capped.py` unit test asserts the invariant explicitly
(alloc(60) after marking 91..100 capped never returns 91..100).

## What remains unverified

The smoke does not exercise the **fire path** — i.e., budgeter actually
deciding to call `kv_to_mamba_chunks` or `mamba_to_kv_chunks`. The
five-generate workload doesn't approach admission ceiling, so the
budgeter never fires. Verifying that:

1. The `T3 smart over-cap selection` log line actually appears under load
2. The `T3 mark_pages_capped` log line appears with non-zero `moved`
3. Subsequent allocations after fire don't crash on unmapped pages

requires a workload that triggers cross-pool transfer. T7 (M2 swarm
conc=800) is that workload; it will run with `SGLANG_SMART_OVERCAP=1`
and can grep server log for both lines + count successful fires.

The conservative path (smart_overcap=0, env unset) is unchanged from
T2, so any regression on the fire path can be A/B'd by toggling the
flag.

## Status

T3 done.

Code changes (4 files):
- python/sglang/srt/mem_cache/allocator.py        (~110 lines added)
- python/sglang/srt/arena/chunk_arena.py          (~30 lines added)
- python/sglang/srt/arena/cross_pool_actuator.py  (~50 lines added)

Tests (4 unit + 1 smoke), all PASS:
- dev/T3_smart_overcap_selection/test/test_allocator_api.py
- dev/T3_smart_overcap_selection/test/test_shrink_explicit.py
- dev/T3_smart_overcap_selection/test/test_select_drainable_chunks.py
- dev/T3_smart_overcap_selection/test/test_mark_pages_capped.py
- dev/T3_smart_overcap_selection/test/test_smoke.sh

Real-workload verification: T7.
