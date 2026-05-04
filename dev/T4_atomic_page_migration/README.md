# T4: Atomic page migration for residual live pages

Task #76. Paper §3.2.3: "If smart selection cannot find $n$ contiguous-or-
scattered free pages because some target pages hold live blocks, the
actuator migrates those blocks out of the to-be-unmapped pages before
issuing `cuMemUnmap`."

## Why this exists

T3's smart over-cap selection (`_select_drainable_chunks`) returns chunks
where **every page is currently free**. Under T2's placement bias plus
moderate workload churn, this is enough — the highest chunks naturally
empty. But under aggressive churn (bursty traffic, or any single-layer
allocator quirk that leaves a live page in an otherwise-free chunk), the
helper returns fewer than `n` chunks. T3 then falls back to the legacy
tail-shrink path, which fails with `drain_pending` if the tail isn't
empty.

T4 closes this last gap: when smart selection returns `< n` chunks, we
**actively migrate** the live blocks out of the chosen drain range
before unmapping. After migration the target chunks become drainable.

## Scope (mamba-first)

Mamba pool is the easier case and we'll implement T4 for it first:

- Each mamba slot is its own page (T1 gift: `tokens_per_chunk = 1` at
  2 MiB grain).
- A live mamba slot holds a single recurrent-state vector (~1 MB).
- Migration = D2D copy of that vector + update `mamba_pool_idx` in
  the owning request.
- Cost: ~1 ms per migrated slot.

Paged-KV migration is a future extension — KV blocks live in
`tokens_per_chunk` (=2048 at T1) bundles per chunk, and migration would
need to copy the entire block plus update `req_to_token` for every
in-flight request that referenced that block. ~3×–10× more work.
**Out of scope for T4 minimum**; KV path stays on the T3 fallback
("tail-shrink, abort if not free").

## API

```python
# On MambaPool (and analogous on KV later):
def migrate_slot(self, src_idx: int, dst_idx: int) -> bool:
    """Copy mamba slot src → dst, update owning request's
    mamba_pool_idx, mark src as free. Returns True if migrated;
    False if src wasn't held by any in-flight request (so no-op).

    Caller's responsibility: ensure no kernel is mid-flight on the
    pool. Wrap with cuStreamSynchronize before + after.
    """
```

The cross-pool actuator picks src indices (live pages in the over-cap
range) and dst indices (free pages elsewhere in the pool, taken from
`select_drain_pages` with `prefer="low"` to keep the head dense).

## Race-free protocol

1. Scheduler is between steps (cross_pool actuator already runs there).
2. `cuStreamSynchronize` — wait for last decode kernel to finish.
3. For each (src, dst) pair: D2D copy slot bytes; update
   `req.mamba_pool_idx`; mark allocator: `src` becomes free, `dst`
   becomes capped.
4. `cuMemUnmap` the now-empty src chunks.
5. `cuMemMap` to dst pool.
6. Scheduler resumes.

The whole thing is single-thread inside the actuator; "atomic" refers
to "kernel-observable atomicity": between two kernel launches, all
mappings and req_to_token updates are coherent.

## Flag

`SGLANG_ATOMIC_MIGRATION=1` enables T4. Layered with T1+T2+T3:

```
SGLANG_ARENA_SHARED=1
SGLANG_ALLOCATOR_PLACEMENT_BIAS=1
SGLANG_SMART_OVERCAP=1
SGLANG_ATOMIC_MIGRATION=1
```

When `SMART_OVERCAP=0`, `ATOMIC_MIGRATION` is ignored (the legacy path
doesn't have a migration hook). When `SMART_OVERCAP=1` and migration is
on, the actuator goes:

```
chunks = _select_drainable_chunks(src, n, tpc)
if len(chunks) < n:
    if SGLANG_ATOMIC_MIGRATION:
        # Pick (n - len(chunks)) more chunks by allowing partial-free.
        # For each live page in those chunks, migrate it elsewhere.
        chunks = _expand_via_migration(src, chunks, n)
    else:
        # T3-only fallback to tail-shrink (may abort).
```

## How to verify

1. **Unit test** (`test/test_migrate_slot.py`): construct a small
   mamba-like pool, mark some slots live, run `migrate_slot`, verify
   the bytes copied + the owning-req-pointer is updated.
2. **Helper unit test** (`test/test_expand_via_migration.py`):
   given an over-cap range with N requested chunks but only M
   drainable, verify `_expand_via_migration` returns exactly N chunks,
   migrating the right number of slots in.
3. **Smoke** (`test/test_smoke.sh`): boot with all 4 flags on, serve a
   few prompts, confirm log shows "T4 atomic migration active" line.

End-to-end fire-path verification with **counted migrations under
load** is T7's job.

## Status

- [x] Design note
- [x] `MambaPool.migrate_slot(src, dst)` API
  (`python/sglang/srt/mem_cache/memory_pool.py:537-585`); unit test PASS
- [x] `_expand_via_migration` helper in cross_pool_actuator
  (`python/sglang/srt/arena/cross_pool_actuator.py:48-115`); unit test PASS
- [x] Env-gated wiring in cross_pool_actuator under
  `SGLANG_ATOMIC_MIGRATION=1`; CrossPoolTransferActuator accepts a
  `mamba_slot_migrator` callback at construction
- [x] Unit tests:
  - `test_migrate_slot.py` (PASS) — byte-copy + alloc state + edge cases
  - `test_expand_via_migration.py` (PASS) — 5 edge cases
- [x] Smoke under T1+T2+T3+T4 flags (PASS, boot 110 s)
- [-] Budgeter-level migrator-callback plumbing: deferred. T7's M2
  swarm validation will plumb the callback to walk
  `scheduler.running_batch` for the owning request and update
  `req.mamba_pool_idx`.

## Followups

- KV migration (multi-token-per-page) deferred.
- Lock-free RCU migration (vs cuStreamSynchronize fence) is a future
  optimization in `dev/future/01_lockfree_rcu_migration.md`.
