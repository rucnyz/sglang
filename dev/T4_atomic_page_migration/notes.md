# T4 notes

## Three-piece delivery

| Part | File | Change |
|---|---|---|
| 1 | `python/sglang/srt/mem_cache/memory_pool.py` | `MambaPool.migrate_slot(src, dst)`: copies state across all `mamba_cache.conv` and `mamba_cache.temporal` tensors, marks src into `_capped_slots`, removes dst from `free_slots`. Returns False on src==dst or non-free dst. |
| 2 | `python/sglang/srt/arena/cross_pool_actuator.py` | `_expand_via_migration(src_act, drainable_chunks, n_target, tpc, migrator)` helper that, for `tpc==1` (mamba page-grain), walks the live-but-not-yet-drainable tail and calls migrator on each, expanding the drainable chunk list. tpc>1 (KV) is no-op (out of T4 scope). |
| 3 | `python/sglang/srt/arena/cross_pool_actuator.py` | `CrossPoolTransferActuator.__init__` accepts `mamba_slot_migrator` callback. The shrink-then-grow path calls `_expand_via_migration` when `SGLANG_ATOMIC_MIGRATION=1` and smart-overcap returned fewer chunks than requested. |

## Verification

| Test | Path | Verifies | Result |
|---|---|---|---|
| `test_migrate_slot.py` | small CUDA | byte-copy from src to dst, src in `_capped_slots`, dst removed from `free_slots`, edge cases (dst not free, src==dst) | PASS |
| `test_expand_via_migration.py` | pure-Python with fake pool / counting migrator | extends a 2→4 chunk list by calling migrator with correct (src,dst) pairs in the right order; tpc>1, no-migrator, already-at-target, migrator-fails edge cases | PASS |
| `test_smoke.sh` | full engine boot | T1+T2+T3+T4 four flags compose: boot succeeds (110 s), T2 prerequisite log appears, 5 generates return cleanly | PASS |

## Honest scope (deliberately narrow)

T4 only covers **mamba-pool migration with `tokens_per_chunk == 1`** (the
gift of T1's 2 MiB page granularity: every mamba slot is its own
chunk). KV pool migration would need:

- per-token block copy across all (layer × kv_head × head_dim) bytes
- `req_to_token` table update for every in-flight request that
  references the moved block
- 2-3× the work of mamba migration

KV migration is left to a follow-up. Under T1+T2+T3+T4 the actuator
falls back to legacy tail-shrink for KV transfers (paper §3.2.3 says
this is an acceptable degradation since KV blocks under T2 placement
bias keep the tail mostly free anyway).

## What this does NOT verify

The cross_pool_actuator's wiring **uses** `migrate_slot` only when:
1. `SGLANG_ATOMIC_MIGRATION=1`, AND
2. `SGLANG_SMART_OVERCAP=1` already chose < n drainable chunks, AND
3. The fire is on the mamba src side

The smoke does not satisfy condition (2)+(3) — smoke serving never
fires the actuator at all. Migration is exercised via the unit tests.
End-to-end fire-path migration with real workload + req-pointer
update is T7's job.

The `mamba_slot_migrator` callback wiring at the budgeter level (i.e.,
how the agent constructs a callback that finds the owning request in
`scheduler.running_batch` and updates `req.mamba_pool_idx`) is **not
yet plumbed** — T4 reserves the constructor slot but the budgeter
agent does not currently pass a non-None migrator. T7 wiring at the
budgeter level closes this.

## Status

T4 implementation done in milestone-scope sense (API + helper + env
gate + smoke). Real fire-path migration deferred to T7.
