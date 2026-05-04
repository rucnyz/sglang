# T2 notes

## A/B run on Qwen3.5-35B-A3B / H200 GPU 2

```
[bias_on]  wall=20.70s successes=32/32 placement_log=1
[bias_off] wall=20.73s successes=32/32 placement_log=0
```

32 sequential prompts × 128-token output. Δ wall = 0.03 s = 0.15% (noise).
All 32 generates returned successfully on both arms. Server log shows the
"T2 placement bias active" line on the bias-on arm only — confirming the
env override path is wired correctly.

## What this verifies

1. **env override flips `need_sort=True` correctly**: log-line presence on
   bias-on, absence on bias-off, no other config differences.
2. **No serving regression**: throughput / latency / success rate identical
   within run-to-run noise. The arena-on path (T1's contribution) plus
   placement bias adds zero observable cost on the smoke workload.
3. **Smoke compatibility**: 32 successive admit→serve→complete cycles
   pass under bias-on. The kept-sorted free-list invariant survives
   alloc + free + merge_and_sort_free interactions over a sustained run.

## What this does NOT verify

The reproduce can only show "T2 doesn't break serving"; it cannot directly
observe live-block clustering at pool head. Two reasons:

1. **No `/dump_state` endpoint** in SGLang HTTP API. Allocator state is
   only inspectable via internal Python attributes, not HTTP.
2. **Per-page free mask** isn't exposed yet — that's T3's job. Once T3
   implements `KVAllocator.free_page_mask()`, we can re-test T2 by
   querying the mask under sustained load with bias on/off and showing
   the bit-distribution histograms differ (bias-on: free bits cluster at
   high indices; bias-off: scattered).

So the **direct observation of placement bias** lands as a follow-up
experiment alongside T3's free-mask exposure. T2 milestone scope is just
"flag works, doesn't degrade" — both met.

3. **Sub-pool-pressure regime not exercised**: the smoke workload keeps
   pool fill < 5% (kv_usage 0.001, mamba_usage 0.039 from a prior /metrics
   run, deleted in cleanup). Placement bias only matters when allocator
   has to make eviction-vs-replacement choices, i.e. pool > 50% fill. A
   workload pushing pool fill into that regime would expose any
   ordering-dependent behavior. T7 (M2 swarm at conc=800) naturally
   exercises that regime end-to-end.

## Behavior under hood (no code change beyond the env flag)

`need_sort=True` was already in the codebase, gated on
`disaggregation_mode in ("decode", "prefill")`. Two paths it touches:

- `BaseTokenToKVPoolAllocator.merge_and_sort_free()`: adds a `torch.sort`
  after merging `release_pages` into `free_pages`. Cost: O(N log N) on
  the merged free list, called only when `alloc()` finds insufficient
  contiguous head pages.
- `TokenToKVPoolAllocator.free()`: appends to `release_pages` (deferred)
  rather than directly to `free_pages` (immediate). Avoids per-free
  re-sort.

Net runtime cost: amortized O(log N) per-alloc, near-zero per-free.
Correctness invariant: at any point an allocator hands out indices, the
first `n` are the smallest `n` currently free. With `need_sort=False`
this invariant is broken (FIFO over the free list), but no caller
depends on it being broken.

## Status

T2 done. Code change: 7 lines in `model_runner_kv_cache_mixin.py:574-585`
(env override + log line). A/B captures throughput parity. Observed
log line. `dev/CLAUDE.md` workflow followed.
