# Dynamic admission cap — tests + sglang-specific impl notes

Test suite + profiling artifacts for the dynamic admission cap
subsystem. Spec is in [`../design.md`](../design.md) §"Dynamic
admission cap (coupling with pool growth)" (within §Mechanism):
when the cross-pool actuator grows `mamba_pool.size` from N → M,
scheduler per-request bookkeeping arrays grow correspondingly so
admission can accept up to `floor(M / ratio)` running reqs at
runtime.

## sglang components

All four components named in design.md and their production
locations. `design.md` §"Page ownership state" + §"Transfer protocol"
enumerate the cap-barrier / mark / unmark / `_capped_*` API surface
explicitly — names below match that section.

| design.md component | production location | role |
|---|---|---|
| `ReqTokenVAArena` | `python/sglang/srt/arena/req_token_arena.py` | VA-stable wrapper over `chunk_arena.SharedHandlePool` + `from_blob_ext.tensor_from_va`; backs `ReqToTokenPool.req_to_token` |
| `ReqToTokenPool` | `python/sglang/srt/mem_cache/memory_pool.py:137` | per-req token-id table; has `max_size`, `grow(new_size)`, `shrink(new_size)` |
| `HybridReqToTokenPool` | `python/sglang/srt/mem_cache/memory_pool.py:1134` | extends `ReqToTokenPool`; pre-allocates small mapping tensors (`req_index_to_mamba_index_mapping`, ping_pong) at `max_size` directly |
| `FutureMap` | `python/sglang/srt/managers/overlap_utils.py:45` | overlap-mode req-state; pre-allocates `token_ids_buf` for `max_running_requests_max`; updates live `future_limit` / `max_running_requests` on grow |
| `MambaPool` (dynamic resize) | `python/sglang/srt/mem_cache/memory_pool.py:317` | has `live_size`, `set_capacity_slots(n_slots)`; Budgeter polls `mamba_pool.live_size` per iteration and re-issues `set_capacity_slots` after a fire |

VA-stable wrapping via `chunk_arena.SharedHandlePool` +
`from_blob_ext.tensor_from_va` means physical pages are mapped
only for live rows but captured Triton kernels keep their
`data_ptr` valid post-grow — so dynamic grow / shrink does not
require CUDA-graph re-capture.

## Disaggregated serving

`ReqToMetadataIdxAllocator` and `MetadataBuffers` (in
`python/sglang/srt/disaggregation/utils.py`) are now boot-allocated
for the live cap's UPPER BOUND, not its init value. The size is
computed by `disagg_metadata_buffer_size(...)`:

| mode | upper bound |
|---|---|
| `DECODE` | `req_to_token_pool.size + pre_alloc_size` — `DecodeReqToTokenPool` does not inherit Phase 7's dynamic-cap mechanism; it has its own `pre_alloc_size` headroom (`scheduler.py:1184-1190`) |
| `PREFILL` | `max(max_running_requests, req_to_token_pool.max_size)` |

Multiplied by ×2 for the original headroom convention. Pre-allocating
to the upper bound (instead of resizing on grow) is required because
the RDMA transport registers metadata tensors' `data_ptr` once at
boot; a runtime resize would invalidate the registration. Mirrors
Phase 7's approach for `FutureMap.max_running_requests_max`.

Unit-tested at
[`test_disagg_buffer_sizing.py`](test_disagg_buffer_sizing.py)
(`#208`).

## Files

### Tests

`test_phaseN.py` numbering is historical — it refers to the dev
phases laid out in [`../archive/dyn_admission_cap_history/plan.md`](../archive/dyn_admission_cap_history/plan.md),
not the project-level Phase 0-6 roadmap in `../PLAN.md`. Each file
maps to one production surface in design.md §"Dynamic admission cap".

| file | what it pins |
|---|---|
| `test_phase1.py` | `ReqTokenVAArena` unit invariants |
| `test_phase2.py` | `ReqToTokenPool` grow/shrink against `max_size` |
| `test_phase3.py` | `HybridReqToTokenPool` lockstep grow with mapping tensors |
| `test_phase4.py` | `FutureMap` dynamic grow up to `max_running_requests_max` |
| `test_phase5.py` | Budgeter post-fire admission cap update — `_maybe_update_admission_cap` triggers on `mamba_pool.size` change |
| `test_phase5b_cuda_graph.py` | CUDA-graph capture survives `grow` (P0 invariant: `data_ptr` stable post-grow) |
| `test_phase7.py` | `MambaPool` dynamic resize via `set_capacity_slots` |
| `test_xpool_balanced_atomic.py` | LCM-balanced cross-pool transfer math (atomic across sub-pools) |
| `test_owner_map_vectorized.py` | vectorized `_compute_fully_free_pages` matches the per-page Python loop reference; **test_7** pins #226 padded-slot-0 safety (chunk 0 unconditionally excluded) |
| `test_mark_no_realloc.py` | `mark_pages_capped` does not reallocate the large `free_pages` tensor (fast path) plus a growing suite of cap-barrier / cross-pool invariants. See "Notable fixes" below — each row tells you which tests in this file pin it. |
| `test_disagg_buffer_sizing.py` | #208 — RDMA-registered `MetadataBuffers` pre-allocated for live-cap upper bound |

### Profiling

| file | what it measures |
|---|---|
| `profile_fire_impact.py` | CUDA-event GPU-side fire impact measurement |
| `profile_cumem_decode_impact.py` | direct measurement that the cuMem ops themselves do not slow decode kernels (decoupled from the `torch.cuda.synchronize()` issue surfaced by #205) |
| `profile_arena_decode.py` | arena tensor backing decode-stream impact |
| `D8_bisect_arena_only.sh` | D8 regression bisect driver: arena cost vs Budgeter tick vs cap-grow |
| `d8_inter_blocking.sh` | D8 inter-fire decode blocking measurement driver |

## #213 — chunk_arena.grow returns mapped slot IDs (architectural cleanup)

Audit-driven refactor (PLAN.md #213): replaced 4 parallel dst cap-bump
paths with a single ID-flow. `ChunkArena.grow(name, n)` now returns
`list[int]` of mapped slot IDs (was `int` count); `xpool_actuator`
pipes them straight to `dst_act.unmark_token_slots(ids)` which dispatches
to `allocator.unmark_pages_capped(tensor)` (KV) or `MambaPool.unmark_slots(tensor)`
(mamba). Removed in Phase E:

- `unmark_lowest_capped_after_grow` helper (xpool_actuator.py) — replaced by
  using the IDs arena.grow actually returned, no sort-and-take-lowest derivation.
- `KVArenaActuator.cap_allocator_only` + `MambaArenaActuator.cap_allocator_only`
  — actuators only ever need the ID-restore surface now.
- `MambaPool._migrated_capped_slots` state (populated by `migrate_slot`) +
  the blacklist filter in `set_capacity_slots` GROW — cross-pool path no
  longer routes through `set_capacity_slots` GROW; migrated slots can't be
  re-exposed because the actuator restores ONLY the IDs arena.grow returns.

Fail-fast assertion (#205 spirit) added at the new dispatch site: if the
lockstep invariant ever breaks (sub-pools return mismatched ID prefixes),
`xpool_actuator._execute_async_locked` raises `RuntimeError` immediately
instead of exposing a slot whose chunk isn't physically backed.

Regression: tests 19/20/21/22/23 in this folder's `test_mark_no_realloc.py`
pin the new contracts (Phase B grow contract, Phase C `unmark_slots`,
Phase D mamba & KV actuator wrappers, Phase D KV wrapper E2E, Phase D
lockstep fail-fast). Tests 22 + 23 were added in a post-#213 audit
pass that caught a real bug: `KVArenaActuator.unmark_token_slots`
called `torch.tensor(...)` without `import torch` in `kv_actuator.py`,
which would `NameError` on the first k2m fire in production. Test_21
bypassed the wrapper and missed it; test_22 drives the wrapper
end-to-end and would have caught it pre-ship. Test 16 rewritten to
verify EXACT-ID semantics (not sort-based). Test 13 narrative
updated to flag it as historical regression for the deleted blacklist
flow. Test 14 unchanged (atomicity invariant `min(granted_per_subpool)`
still load-bearing).

## Notable fixes

### KV cap state: the `CappedFreeList` model (the file's namesake)

The arena KV allocator (`TokenToKVPoolAllocator`) holds the whole
FREE/CAPPED split in one `CappedFreeList` (`capped_free_list.py`). A
capped page-id is NEVER in the free list: the capped set is an implicit
integer tail `tail_lo` (the contiguous unbacked range `[tail_lo, size]`;
`_NO_TAIL` = nothing capped) plus a small `marks` set for the mid-range
ids a Drain unmapped. `live() = size − n_capped`, where `n_capped` =
tail length plus the marks count, is pure integer arithmetic (the tail is
never materialized).

This is why mark/unmark are cheap and the decode path pays nothing:

- **No per-fire realloc.** `mark`/`unmark` are O(K) in the marked-id
  count: they edit `marks` (or move `tail_lo`), never reallocate the
  free-list tensor. Only the gentle `set_cap`/boot-cap path rebuilds the
  free list, not a per-fire drain.
- **No per-token filter.** `free()` is append-only (a capped id is never
  LIVE, so it can never be freed back, hence no filter is needed; an
  over-add dedup is gated behind an O(1) `available() > live()` check).
  `alloc()` pops the lowest free ids, checking nothing when no drain is
  in flight (else one small `isin` over the tiny `marks`).

The earlier design materialized the full `[tail_lo, size]` tail as a
`_capped_pages` tensor and ran a per-`alloc()`/`free()` `isin` filter
against it. At saturation that ~1.5 M-id tensor and its per-token filter
were the decode tax; `CappedFreeList` removes both. The compatibility
accessors (`free_pages`, `_capped_pages`, `_cap`, `_capped_lo`,
`live_size`) still exist as read-only projections of the `CappedFreeList`
state for external readers (pool-leak checker, OwnerProvider, telemetry,
tests).

Regression: this folder's entire `test_mark_no_realloc.py` is the
pin — test_1 (alloc skips capped), test_2 (no realloc per mark),
test_3 (mark/unmark idempotent), test_4 (KV-scale 1669×
benchmark), test_8 (free of capped doesn't increase `available`),
etc. The KV-side filter contract derived here is what #174 / #221
mirror onto Mamba (the mamba side still uses the `_capped_slots`
tensor + `free()` filter; see #174 below).

### #228 — Mamba `live_size` mirrors KV (semantic rewrite)

Pre-#228, `MambaPool` carried its own `_cap_slots` integer +
maintained `self.size` as the LIVE cap (mutated by
`set_capacity_slots`). The KV side has `self.size = max`
(immutable) and `_cap` as the live cap separately, with
`live_size = size − _capped_pages.numel()`. The asymmetry meant
mirror invariants (#162, #174) couldn't apply verbatim to Mamba.

Fix: rewrite `MambaPool.live_size` as
`size − (capped ≤ size).count()`, drop `_cap_slots` state. The
masked subset (capped ≤ size) is needed on the Mamba side
because `_capped_slots` also carries boot-deferred IDs in
`(size, max_size]` whose chunks are unmapped and don't subtract
from the live count.

#228 is the foundational rewrite that makes #221 (mamba invariant
mirror), #222 (mamba lock), #224 (mamba clear mirror) all
expressible as KV-side parallels. test_29 + test_30 pin the new
contract.

### #174 — MambaPool.free leaks capped slots (D10 N=3 v3 root cause)

Symptom (2026-05-30): N=3 v3 D10@C=56 — all 3 inter cells crashed
~2 min after m2k fires with `Pointer argument (at 7) cannot be
accessed from Triton (cpu tensor?)` in
`unified_linear_attention_with_output`.

Root cause: `MambaPool.free()` filtered freed slots only via
`_cap_slots` (the integer live cap), but the cross-pool actuator's
`_MambaCapAllocator.mark_pages_capped` records individual slot
IDs in `_capped_slots` **without** lowering `_cap_slots`. So when
an in-flight req or radix-cache eviction subsequently freed a slot
that was in `_capped_slots` but still `<= _cap_slots`, the slot
went back into `free_slots`. The next admission picked it up — but
the underlying chunk had been `cuMemUnmap`'d by the m2k worker, so
the next forward pass touched unmapped VA → Triton kernel crash.

Fix: at `MambaPool.free` (memory_pool.py), filter `free_index`
against `_capped_slots` before any other accounting. Symmetric to
the KV side's filter in `TokenToKVPoolAllocator.free`
(allocator.py, D11 #154 fix).

Regression: `test_mark_no_realloc.py` tests 17 / 17b / 17c / 17d
together pin the contract from four angles (subagent depth audit
caught the original single-slot test as SHALLOW given the live
bug's blast radius vs KV-side 5-test coverage of the symmetric
filter):

- **17** (single-slot): `mark([50])` + `free([50])` → 50 ∉ free_slots.
  The minimal live-crash reproduction.
- **17b** (mixed batch): `mark([50, 51])` + `free([50, 51, 52, 53])`
  → {50, 51} dropped, {52, 53} preserved. Pins against an over-eager
  filter that drops the entire batch on any overlap (real radix-cache
  eviction frees N slots at once).
- **17c** (ordering): `_cap_slots=49` + `mark([50])` + `free([50])` —
  slot 50 is both in `_capped_slots` AND above `_cap_slots`. The
  filter must run first; `_capped_slots` stays exactly `[50]` (no
  double-add via the above-cap branch).
- **17d** (non-capped negative control): `mark([50])` + `free([7])`
  → 7 reaches `free_slots` normally; 50 stays out. Pins the filter's
  specificity (no cross-contamination, no wrong-tensor regression).

### #223 — uneven `granted_per_subpool` leaks SharedHandlePool handles

Symptom (latent, observable as slow SharedHandlePool depletion):
`dst._arena.grow(name, per_dst)` may return fewer chunk IDs than
requested if `SharedHandlePool._free_handles` partially exhausts
between sub-pool grow calls. Pre-#223 the cap was correctly bumped by
`min(granted_per_subpool)` (test_14), but the over-granted chunks in
the leading sub-pools stayed `cuMemMap`'d and unowned — their handles
never returned to the shared free-list. Repeated fires under memory
pressure compound the leak until SharedHandlePool runs dry.

Fix: in `XPoolActuator._execute_async_locked` (`xpool_actuator.py`),
after computing `actual_per_dst = min(granted_per_subpool)`, walk
each sub-pool's granted ID list and call
`dst._arena.shrink_explicit(name, extra_ids)` for any sub-pool with
`len(sub_ids) > actual_per_dst`. The unmap returns the handles to
`SharedHandlePool._free_handles`. Cleanup chosen over a hard assert
because partial exhaustion is a legitimate runtime condition under
memory pressure, not a contract violation — assert would kill an
otherwise-recoverable fire.

Regression: `test_mark_no_realloc.py::test_33` drives
`_execute_async_locked` with stub arenas where `p0=[5,6,7]` and
`p1=[5,6]`; verifies `dst_arena.shrink_calls` contains exactly
`("p0", [7])` after the fire completes.

### #162 KV capped invariant assertion (Audit Phase 9 D4, HIGH)

Symptom (latent, not yet observed in production): a capped-id count that
exceeds `size` would drive `live() = size − n_capped` negative, the
pool-leak detector trips with the wrong root cause, and `available()`
returns garbage, all far from the actual mutation site.

Fix (current, post-`CappedFreeList`): `CappedFreeList.mark` fail-fasts on
any marked id `> size`, so the capped set can never push `n_capped` past
`size`; the arena allocator's `_assert_capped_invariant` then verifies the
free list has no duplicate ids (so `available()` can't outrun `live()`).
Loud-crash, not a silent fix, aligning with the #205 fail-fast principle:
it surfaces the bug at the moment of cause.

Regression: `test_mark_no_realloc.py::test_18` drives an over-cap; the
assertion must fire. Falsification: a `D8 long-run trace (≥ 30 min)`
should never trip it under normal workload.

### #221 — `_capped_slots` invariant assertion (mamba mirror of #162)

Mamba-side parallel of #162. `MambaPool._assert_capped_slots_invariant()`
called at every `_capped_slots` mutation (`migrate_slot`,
`set_capacity_slots` SHRINK, `MambaPool.free` above-cap branch,
`_MambaCapAllocator.mark/unmark_pages_capped`). Loud crash if
`_capped_slots.numel() > max_size`.

Regression: `test_mark_no_realloc.py::test_24` (real-pool artificial
violation → assertion fires), test_26 (mark path symmetry pin),
test_27 (no-defensive-hasattr regression guard).

### #222 — `MambaPool._alloc_lock` for scheduler/worker race

Symptom (latent): the cross-pool actuator's worker thread mutates
`free_slots` / `_capped_slots` / `self.size` via `set_capacity_slots`
+ `unmark_slots` while the scheduler thread is concurrently in
`alloc` / `free` / `migrate_slot`. Read-then-rebind in either side
can race the other → mask shape mismatch (IndexError) OR stale cat
(same slot handed out twice).

Fix: `MambaPool._alloc_lock = threading.Lock()` (mirrors KV-side
`BaseTokenToKVPoolAllocator._alloc_lock`). Every mutator wraps its
body in `with self._alloc_lock:` (six methods). `_MambaCapAllocator
.mark/unmark_pages_capped` (lives in `mamba_actuator.py`) also
acquires the same lock.

Regression: `dev/interlayer/0_page_state_machine/alloc_lock/test_mamba_alloc_lock.py` —
5 sub-tests including a 2b that drives `_MambaCapAllocator` under
contention.

### #224 — `MambaPool.clear()` orphans slots above the live cap

Symptom: after a cross-pool actuator shrink (`set_capacity_slots(N)`
with N < max_size), the slots in `(N, max_size]` have unmapped
chunks. Pre-fix `clear()` reset `_capped_slots = empty` and
`free_slots = [1..N]`, leaving those slots in neither tensor.
Trigger: `/flush_cache` → `HybridReqToTokenPool.clear` →
`mamba_pool.clear()` — next GROW couldn't restore the unmapped
chunks; `live_size` math went stale.

Fix mirrors `BaseTokenToKVPoolAllocator.clear` (KV side): when
`self.size < self.max_size`, stage `_capped_slots = arange(size+1,
max_size+1)`; else empty.

Regression: `test_mark_no_realloc.py::test_34` (production
`MambaPool.clear` rebound to a `_FakePool`; both branches —
shrunk + full-cap — covered).

### #225 — m2k `_capped_slots` accumulation + drain pin

Coverage gap closer (no production bug, no fix). Pre-existing
`test_25` drove only the k2m grow direction (single batch);
`test_28` drove only a single-step m2k→k2m round-trip. `test_35`
adds 4 scenarios on a real `MambaPool`:

- **A**: 3-cycle `_MambaCapAllocator.mark_pages_capped`
  accumulation, numel 12→21 monotone with #221 invariant after
  every op.
- **B**: rollback drain via `_MambaCapAllocator.unmark_pages_capped`
  (4 production sites: verify-failure, queue-full, worker-noop,
  worker-exception; all size-invariant).
- **C**: mixed 5-mark / 2-drain → net {4,5,7,8,9,10}.
- **D**: production k2m drain via `MambaPool.unmark_slots` (the
  inner path of `xpool_actuator._execute_async_locked` →
  `dst_act.unmark_token_slots`). Pins cap-as-max: below-cap drain
  keeps size=20, above-cap drain bumps size to max restored ID (23).

### #226 — padded-slot-0 safety (chunk 0 must stay mapped)

Latent bug: chunk 0 carries padded slot 0 — design.md §"Per-unit
sizes" says slot 0 is the padded-output target; the kernel writes
to slot 0 every forward pass. Pre-fix
`_compute_fully_free_pages` marked slot 0 as "free unconditionally"
so page 0 could enter the fully-free candidate set;
`expand_pages_to_token_slots` used `start = max(1, p*tps)` — for
KV (tps>1) it returned `[1..tps)` for page 0, for mamba (tps=1)
it returned `[]`. Either way the upstream actuator would unmap
chunk 0 → next padded write → `cudaErrorIllegalAddress`.

Selectability: `fire_planner` picks largest IDs first so page 0
is selected last; healthy workloads rarely hit it, but under
sustained shrink it becomes selectable.

Fix (two layers, defense-in-depth):
- `_compute_fully_free_pages` drops `fully_free_mask[0]` post-
  reshape (single source of truth — chunk 0 never enters the
  candidate set).
- `expand_pages_to_token_slots` + `page_is_fully_free` on both
  KV/Mamba actuators raise `ValueError` / return `False` on
  page 0 (loud fail-fast if upstream filter is ever broken).

Regression: `test_owner_map_vectorized.py::test_7` (3 shapes:
KV tps=4, mamba tps=1, n_pages=1 edge); `test_mark_no_realloc.py
::test_36` (KV+Mamba raise contract);
`dev/interlayer/4_e2e/byte_transfer/test_chunk_slot_unit.py`
tests 1/4 rewritten to assert the new raise.

### #227 — Delete `KVArenaActuator.shrink_fraction`

Dead-code cleanup. The actuator's real shrink API is
`set_capacity_tokens(n)`; `shrink_fraction(frac)` was a 2-line
convenience wrapper with zero callers anywhere in `*.py` / `*.md`.

### #282 — KV pool growable (m2k grow-KV), the dynamic-cap port

Symmetric counterpart to `MambaPool`'s #279/#228 dynamic cap.
Before this, an m2k fire grew the KV *arena* (physical handles
mapped) but the `TokenToKVPoolAllocator`'s free-page accounting
was pinned at the boot ceiling — granted chunks never became
allocatable, so KV could not durably grow and m2k was inert
(the root cause behind the cc-traces m2k regression, not pool
coupling). The fix mirrors Mamba: `BaseTokenToKVPoolAllocator`
takes an optional `max_size` (the page-id ceiling, default =
`size` → boot-only back-compat); `size` becomes the live cap;
`set_capacity_pages(n)` shifts ids between the live free set and
the deferred `(cap, max_size]` headroom (`_capped_pages`) without
ever mutating the realized free tensor. `available_size` and the
scheduler's `full_token_usage` read `live_size = size -
len(_capped_pages)`, so occupancy is honest at every cap.
Covered by `test_kv_growable.py` (back-compat, grow-past-boot,
alloc-never-returns-headroom, #283 double-count repro, R1 idle +
under-load usage, out-of-range clamp, clear preserves cap,
actuator round-trip, many-cycle no-leak, concurrency, leak
diagnostic excludes headroom).

**Boot wiring + the hybrid `_kv_arena` fix.** The allocator's
`max_size` is wired only on the `page_size == 1` path: when the
KV cache is arena-backed it is set to `boot ×
SGLANG_XPOOL_KV_MAX_FACTOR` (default 2.0), clamped to the arena's
VA `max_tokens` — NOT the raw VA ceiling (tens of millions of
tokens), which would size the `free_pages` tensor / alloc masks
to hundreds of MB. On a hybrid (Mamba) model the KV cache is a
`HybridLinearKVPool` whose arena lives on its inner
`full_kv_pool`, so the original direct `token_to_kv_pool._kv_arena`
read raised `AttributeError` at boot. Fix follows the codebase's
"declare unconditionally, read directly (no `getattr` default)"
convention: `_kv_arena` is now a class attribute on the `KVCache`
base (default `None`), and `HybridLinearKVPool` forwards it to
`full_kv_pool` via a property (consistent with its ~20 other
forwards). The wiring site reads `token_to_kv_pool._kv_arena`
directly and works for both plain and hybrid pools. Pinned by
`test_kv_growable.py::test_23`. Boot sanity (Qwen3.5-9B hybrid,
cross-fire on, mamba 256, LPB, GPU): `boot ok=1`, `KVArenaActuator
max_tokens=43468800 (arena ceiling) live=1525471`, smoke generate
correct — confirms A1 takes effect end-to-end with no boot crash.

### #282 audit (2026-06-07) — two reproduced bugs in the grow path

An adversarial re-audit of A1 found two real bugs, both now fixed
test-first:

- **CRITICAL: `clear()` reverted cross-fire grows.** The production grow
  path is `unmark_pages_capped` (the actuator's `unmark_token_slots`), NOT
  `set_capacity_pages`; it never advanced the boot `_cap`. `clear()`
  (flush_cache) rebuilt `_capped_pages` from `_cap`, so a flush silently
  reverted grown KV to boot AND orphaned the arena handles mamba donated —
  the KV twin of the #224 MambaPool bug. Fix: `clear()` now PRESERVES
  `_capped_pages` (the single source of truth for un-backed ids); the boot
  headroom is staged once in `__init__`. The symmetric cross-fire SHRINK
  (`mark_pages_capped`) is covered by the same change (a flush would
  otherwise reinstate the unmapped tail → alloc hands out an unbacked
  slot). `test_kv_growable.py::test_24` (grow survives flush) +
  `test_25` (shrink survives flush).
- **HIGH: out-of-ceiling grant silently dropped.** The allocator page-id
  ceiling is `max_size` (boot × `SGLANG_XPOOL_KV_MAX_FACTOR`) but the arena
  VA `max_tokens` is far larger; a granted slot id `> max_size` was silently
  dropped by `unmark_pages_capped`'s `torch.isin` while the arena had
  cuMemMap'd the chunk → donated-handle + HBM leak. Fix follows #162/#205:
  `unmark_pages_capped` fail-fasts on ids `> size` (`test_26`), and
  `BudgetAgent._maybe_fire` refuses a grow fire when the destination pool is
  already at its ceiling (`test_budgeter_drain_fire.py::test_D`) so
  production degrades gracefully instead of crashing.

Both bug classes here are now structurally precluded by the
`CappedFreeList` model (see "KV cap state" above): `tail_lo` IS the live
cap, so `clear()`/`reset` can never reconstruct a stale cap and revert a
grow, and `mark`/`unmark` fail-fast on out-of-ceiling ids. The
`set_capacity_pages` / `_capped_pages` / `_cap` symbols described above are
the superseded KV mechanism, retained here as the historical WHY; the live
KV state lives in `CappedFreeList`.

## Residual measurement (post-fix)

The most recent measured state of the system **with dynamic
admission cap fully enabled**: D8 N=5 saturated single-pool
shows `+0.61 % ± 0.05` TPOT regression vs `off` baseline
(1.4 σ — borderline; within day-to-day system noise).
design.md §saturated_bubble specifies the headline target as `+10 %
throughput`, with the regression budget for the resize
machinery being roughly zero. The 0.61 % residual is documented
explicitly in design.md as a known caveat.

## Cross-references

- Spec: [`../design.md`](../design.md) §"Dynamic admission cap
  (coupling with pool growth)".
- Implementation plan: [`../PLAN.md`](../PLAN.md) Phase 0 (this
  folder's alignment audit) + Phase 5 (#100 D8d uses dynamic
  admission cap as one isolation cell).
- Pre-consolidation sub-design / audit / progress files (the
  implementation history this folder grew out of) live in
  [`../archive/dyn_admission_cap_history/`](../archive/dyn_admission_cap_history/).
