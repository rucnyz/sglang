# T2: Allocator placement bias (live → head, free → tail)

Task #74. Paper §3.2.2 first half: "Each pool's block allocator runs first-fit
lowest-address: new live allocations bias toward the head of the pool's
mapped region (near $V_i$), so the tail naturally accumulates free or
evictable cached space."

## What changes

The `BaseTokenToKVPoolAllocator` already has a `need_sort` parameter that
maintains a sorted-ascending `free_pages` list:
- `free()` defers freed indices to `release_pages` rather than appending to
  `free_pages` directly.
- `alloc()` lazy-merges `release_pages` into `free_pages` (with
  `torch.sort`) when `free_pages` runs short.
- `alloc()` then returns `free_pages[:n]` — the smallest n free indices.

Net effect: live allocations cluster at low indices (pool head), free
indices accumulate at high indices (pool tail). Exactly what T3 (smart
over-cap selection) needs to find contiguous unmapped pages near the tail.

The catch: `need_sort` is enabled only in disaggregation mode (line 574 of
`model_runner_kv_cache_mixin.py`):

```python
need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
```

Single-engine deployments (our paper setup) leave `need_sort=False`, so
the allocator's free list is FIFO — freed pages go to the tail of
`free_pages` and only get reused after a wrap, breaking the
"high indices = always free" invariant.

T2 adds an env override that forces `need_sort=True` whenever the arena
substrate is on:

```python
need_sort = (
    self.server_args.disaggregation_mode in ("decode", "prefill")
    or os.environ.get("SGLANG_ALLOCATOR_PLACEMENT_BIAS", "0") == "1"
)
```

Default behavior unchanged (no env var → no change). Setting
`SGLANG_ALLOCATOR_PLACEMENT_BIAS=1` enables low-address placement.

## Why bother if `need_sort=True` already exists

The mode is reachable today via disaggregation, but the paper claim is
that single-engine inter-pool reallocation works — which requires the
placement bias to be on in single-engine mode too. Without this env
override, T3's smart over-cap selection on a single-engine deployment
would be searching the tail for free pages, but live blocks would be
randomly scattered (FIFO-recycled) — making the search succeed only by
luck.

## Flag

`SGLANG_ALLOCATOR_PLACEMENT_BIAS=1` enables low-address placement. The
arena-on path already needs `SGLANG_ARENA_SHARED=1`; it's safe to also
set placement bias in that case (no semantic conflict, same correctness
invariants).

## How to verify

1. **Smoke**: `bash dev/T2_allocator_placement_bias/test/test_smoke.sh`
   boots Qwen3.5-35B-A3B with placement-bias on, runs 5 prompts, exits
   non-zero on failure.
2. **Allocation distribution**: `bash dev/T2_allocator_placement_bias/reproduce.sh`
   runs the smoke and dumps the in-flight `req_to_token` row distribution
   from `/dump_state` — placement-bias-on should show live indices
   clustered near 0; placement-bias-off should show them scattered.

## Status

- [x] Design note
- [x] Env override added in `model_runner_kv_cache_mixin.py:574-585` (+ log line)
- [x] Smoke serving works under `SGLANG_ALLOCATOR_PLACEMENT_BIAS=1`
- [x] A/B vs placement-bias-off: 20.70 s vs 20.73 s (Δ = 0.15%, noise);
  32/32 success either arm; log line confirms env override path.
- [x] **Direct placement-bias mechanism**: `test_placement_bias_direct.py`
  proves bias-on alloc returns indices in sorted ascending order
  (lowest free first); bias-off returns FIFO order (freed-back indices
  go to tail and are handed out 50× later). Mechanism verified at
  invariant level, not statistical workload.

## Followups

T3 (smart over-cap selection) consumes T2's placement bias — without
T2, T3's free-page query would too often find no contiguous free pages
near the tail.
