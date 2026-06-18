# kv_migrate_slot — #271 KV-slot migration feasibility spike

**Verdict (2026-06-07): FEASIBLE — captured-graph replay PROVEN (#291) for
flashinfer decode.** Live-KV-slot migration (Stage-3, `cross_migrate(src=kv)`)
is empirically safe. `test_kv_migrate_replay.py` (real CUDA, 2/2) proves the
data primitive (A) and that the index-fill is data-driven (B); the owed
load-bearing proof — that a CAPTURED decode graph re-derives the indices on
replay, so a between-replay migration is picked up — is now closed by
`test_kv_captured_replay.py` (#291, real CUDA capture+replay, exit 0):

- **A — data primitive.** `MHATokenToKVPool.move_kv_cache([d],[s])`
  relocates a single KV token-slot `s→d` **byte-exactly across every
  layer's k AND v buffer** (and leaves `s` intact for the actuator to
  cap/unmap afterward). A KV `migrate_slot(s,d)` is a thin wrapper over it.
- **B — index fill is data-driven.** Re-running the real production
  `create_flashinfer_kv_indices_triton` after a `req_to_token[req,pos]=d`
  edit reads the new slot. (`test_B` runs the fill directly, outside any
  graph — necessary but not sufficient; the captured-graph half is C.)
- **C — captured-graph replay equivalence (#291, PROVEN).**
  `test_kv_captured_replay.py` captures a real `torch.cuda.CUDAGraph` over
  the production fill (`create_flashinfer_kv_indices_triton`) PLUS a KV
  gather (attention-read stand-in), with `req_to_token` as the captured-input
  tensor. baseline (req→s) and a neg-control (req→d pre-move, |Δ|=512, so the
  graph IS sensitive) bracket the result: after `move_kv_cache(d←s)` +
  `req_to_token[pos]=d`, the replay output equals baseline **byte-exactly
  (|Δ|=0.0)** — the captured graph re-derives `kv_indices` from `req_to_token`
  on every replay, indices are NOT baked-stale at capture. Capturing the fill
  INSIDE the graph proves the strongest form; production runs the fill just
  outside the graph into the same persistent buffer, so it is a fortiori safe.

This is the KV analog of the mamba `step2_migrate_slot_replay_invariant`
spike, and confirms the same two-phase contract works for KV: move bytes
(`move_kv_cache`), then rewrite the owning req's pointer (`req_to_token`).

## Caveats / what the spike does NOT yet cover

- **Backend coverage — now COMPLETE for all CUDA backends (#294b).** The
  captured-graph replay safety property is proven per index-fill KERNEL, and
  every CUDA decode backend uses one of two kernels:
  - `create_flashinfer_kv_indices_triton` — **flashinfer** (empirical: #291),
    and **triton** + **aiter** reuse the identical kernel (kernel-identity
    asserted by `test_kv_backend_index_coverage.py`, so #291's proof
    transfers; if any swaps kernels the guard fails loudly).
  - `normal_decode_set_metadata` (the FA3 fused `_fused_metadata_kernel_
    ps1_no_swa`) — **FA3** (empirical: #294b, `test_kv_captured_replay_fa3.py`).
  The uniform CUDA-graph machinery (`cuda_graph_runner.replay` →
  `init_forward_metadata_replay_cuda_graph` runs the fill OUTSIDE
  `graph.replay()` into persistent buffers the captured graph reads) means a
  between-replay req_to_token rewrite always propagates. aiter is AMD/ROCm
  (kernel-identity checked; runtime untested on NVIDIA). MLA pools are
  excluded by `can_move_kv_cache()` (no migration), so their backend path is
  moot.
- **Sync discipline.** A live slot's bytes race the in-flight decode that
  reads/writes them. Migration must run with the same synchronization mamba
  uses in Stage-0 (sync before the byte-copy so in-flight kernels on `s`
  finish; sync after so the copy + the `req_to_token` rewrite are visible
  before the next replay). The cross-fire actuator's "Layer-0 / no defensive
  sync" model (#205) covers FREE pages only — migration of LIVE slots is the
  one path that genuinely needs the explicit sync.

## #271 build (steps 1–4 SHIPPED, step 5 GATED on #291)

Steps 1–4 are implemented and unit-covered (CPU, real allocator + fake
actuator/cache); step 5 is deliberately held behind a fail-closed env gate
until the captured-graph proof (#291) lands.

1. **DONE — KV `migrate_slot(s, d)`.** `TokenToKVPoolAllocator.migrate_slot`
   (a thin wrapper over `kvcache.move_kv_cache([d],[s])`) relocates the slot
   under `_alloc_lock`, then swaps free (`dst` out, `src` in). Per Convention
   A it does NOT cap `src` — the freed page rejoins `free_pages`. Fail-fast
   on `page_size != 1`, slot-0, dst-not-free/capped, or a kvcache lacking
   `move_kv_cache`.
2. **DONE — pool-agnostic Stage-0.** `xpool_actuator._run_stage0` calls
   `src_act.migrate_slot` (KV→allocator, mamba→pool) and dispatches the
   pointer rewrite per direction; the migration loop is bracketed by
   `torch.cuda.synchronize()` (guarded by `is_available`). Tests
   `test_kv_pointer_rewrite.py::test_D/F`.
3. **DONE — KV pointer rewrite.** `SchedulerStage0Handler.
   rewrite_kv_token_indices(s, d)` finds the running req holding `s` in
   `req_to_token[req_pool_idx, :seqlen]` and rewrites it; fails loud if no
   live owner. Bounded to `[0, seqlen)` so a stale tail entry is not a false
   match (`test_kv_pointer_rewrite.py::test_A/B/C/E`, 6/6 total).
4. **DONE — KV cost-order walk.** `SchedulerOwnerProvider.
   _kv_live_pages_in_cost_order` (vectorized): SOURCE = fully-live-uncached
   page, DONORS = free slots on partial kept pages; whole-free / capped /
   page-0 / cached slots excluded → `[(freed_page, ((src,dst),...))]`
   (`test_kv_live_pages.py`, 4/4).
5. **GATED — enable `allow_migrate` for k2m.** The walk is wrapped in a
   fail-closed env gate `SGLANG_XPOOL_KV_MIGRATE` (default `0` → returns
   `[]`, so a `cross_migrate(kv_to_mamba)` candidate degrades to
   free-only/drain even when the planner passes `allow_migrate=True`).
   Remaining prerequisites before flipping to `1`: (a) #291's captured-graph
   replay proof — **DONE** (`test_kv_captured_replay.py`, flashinfer); (b)
   `enable_kv_cache_copy` wiring on the pool — **DONE** (#294a): a single
   source of truth `memory_pool.kv_live_migration_enabled()` (read by BOTH
   this walk gate AND boot-time pool construction, so they can't disagree)
   drives `enable_kv_cache_copy`, and `HybridLinearKVPool` forwards it to its
   inner full KV pool; (c) per-backend captured-graph coverage — **DONE**
   (#294b: flashinfer #291, FA3 empirical, triton/aiter kernel-identical).
   `test_kv_live_pages.py::test_4` pins the env gate, `test_kv_copy_wiring.py`
   pins the enable_kv_cache_copy wiring + the capability gate.

   **Step 5 is now fully wired.** `BudgetAgent._maybe_fire` always passes
   `allow_migrate=True` to the planner (the OwnerProvider walk self-gates: KV
   migrates only under `SGLANG_XPOOL_KV_MIGRATE` + `can_migrate_slot`, mamba is
   atomic-inert). So the ONLY remaining step to turn live KV migration ON in
   production is operational: set `SGLANG_XPOOL_KV_MIGRATE=1` at launch.
   `test_budgeter_drain_fire.py` test_C/test_H pin that the agent forwards
   `allow_migrate=True` for both directions.

The cold-cache DRAIN side of KV-source cross-fire is already shipped
(#271a / #289); this folder is purely the LIVE-slot MIGRATION half.

## Run

```bash
D=dev/interlayer/0_page_state_machine/kv_migrate_slot
# Feasibility spike + captured-graph proofs (real CUDA):
CUDA_VISIBLE_DEVICES=0 .venv/bin/python $D/test_kv_migrate_replay.py        # A+B: 2/2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python $D/test_kv_captured_replay.py       # C/#291 flashinfer: exit 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python $D/test_kv_captured_replay_fa3.py   # #294b FA3: exit 0
.venv/bin/python $D/test_kv_backend_index_coverage.py                       # #294b coverage guard: 2/2

# Build steps 1–4 + audit fixes (CPU unit tests):
.venv/bin/python $D/test_kv_pointer_rewrite.py     # steps 1–3: 6/6
.venv/bin/python $D/test_kv_live_pages.py           # step 4 + gate: 4/4
.venv/bin/python $D/test_kv_cached_locked.py        # HIGH-1: 2/2
.venv/bin/python $D/test_kv_migration_validate.py   # HIGH-2: 1/1
.venv/bin/python $D/test_kv_copy_wiring.py          # #294 enable_kv_cache_copy + capability gate: 3/3
.venv/bin/python $D/test_kv_backend_index_coverage.py  # #294b per-backend guard: 2/2
# (mamba fail-closed gate lives with the budgeter tests:)
.venv/bin/python dev/interlayer/3_budgeter/no_spike/test_mamba_migrate_failclosed.py  # 2/2
.venv/bin/python $D/test_migration_engages_planner.py  # #295 engagement: 1/1
```

## Does it earn its keep? (#295 — e2e win, GPU-pending)

Correctness/safety is done; the *value* is a separate, owed measurement
(tracked as task #295). Split into a no-GPU half (done) and a GPU half (owed):

- **DONE (no-GPU): engages + is load-bearing.** `test_migration_engages_
  planner.py` drives the real `XPoolFirePlanner.build` → `SchedulerOwnerProvider`
  → `TokenToKVPoolAllocator` in a forced fragmentation layout (KV near-full,
  scattered live slots, drain exhausted). Migration fires (`plan.migrations`
  non-empty) AND is load-bearing: with `SGLANG_XPOOL_KV_MIGRATE` OFF the
  identical layout **refuses** the k2m fire (`free+drain < n`, `refuse_count++`).
  So a page is freed *only because* migration consolidated it. This pins the
  exact regime migration needs: **`free + drain < n` for a k2m fire**.
  Attribution is instrumented: the budgeter jsonl fire record logs
  `fire_migrate_moves` / `fire_drain_pages` (+ `fire_refuse_count`).
- **OWED (idle GPU): the win number.** A/B `SGLANG_XPOOL_KV_MIGRATE` 1 vs 0
  on a workload that reaches the regime above. First check whether natural cc
  traces ever do (`fire_migrate_moves>0`); if drain always suffices there,
  that is the finding → demonstrate the win on a synthetic fragmentation
  workload. Target-pinned (a k2m fire the OFF baseline refuses now succeeds;
  cache_hit +X pp predicted from pages consolidated; out_tps within tol;
  output equivalence per #291). Bench discipline: idle GPU only.

## Post-build audit findings (all fixed, test-first)

Three subagent audits hardened this path; all findings lived in code reachable
only with the step-5 gate ON (so no live hazard today) but would corrupt state
the moment migration is enabled — each fixed test-first.

### Round 3 (allow_migrate=True wiring) — mamba migration fail-closed

Flipping `BudgetAgent` to always request `allow_migrate=True` made the mamba
source (mamba_to_kv) walk reachable. Mamba live-migration was inert only
*incidentally* (mamba runs `tps==1` → no partial pages → no donors, #269) —
NOT gated. Unlike the KV side it has no opt-in flag or captured-graph replay
proof, so a fragmentable layout (`tps>=2`, anticipated for TP/bf16 ssm) would
silently relocate LIVE recurrent state. **Fix:** `_live_pages_in_cost_order(
"mamba")` refuses (`[]` + one-time warning) for `tps != 1`, making inertness a
fail-closed GATE; `tps==1` keeps the normal walk. `test_mamba_migrate_
failclosed.py` (2/2). (Audit also flagged a LOW: env-off mamba_to_kv now runs
the cheap mamba live-set walk before returning `[]` — bounded, pure-read,
accepted.)

### Round 2 (steps 1–5) — capability gate

The env gate alone is insufficient: `kv_live_migration_enabled()` is read
per-walk but `enable_kv_cache_copy` is read ONCE at boot, so flipping
`SGLANG_XPOOL_KV_MIGRATE` on AFTER boot — or an MLA/NPU hybrid whose inner pool
has no usable `move_kv_cache` — would let the walk emit moves that then assert
(or `AttributeError`) inside `migrate_slot` mid-fire, swallowed by `agent.tick`
= the same half-applied class as HIGH-2. (The `move_kv_cache` `_kv_copy_config`
assert is NOT a reliable guard: the `SGLANG_NATIVE_MOVE_KV_CACHE` path returns
before it, and a `HybridLinearKVPool` wrapping MLA passes a bare `hasattr`.)
**Fix:** a real capability predicate `KVCache.can_move_kv_cache()` (base False;
MHA True once warmed or under the native path; Hybrid forwards to its inner
pool → MLA inherits False) + `TokenToKVPoolAllocator.can_migrate_slot()`
(page_size==1 ∧ kvcache can move). The Stage-3 walk consults `can_migrate_slot`
and refuses with ZERO migration (one-time warning) if the live pool can't
migrate; `migrate_slot`'s own pre-mutation guard now uses `can_move_kv_cache`
(authoritative, replacing the bare `hasattr`). `test_kv_copy_wiring.py::test_3`
pins it (env ON + incapable pool → `[]`).

### Round 1 (steps 1–4)

- **HIGH-1 — locked shared-prefix slots were migratable.** The Migration
  source set is LIVE-UNCACHED = allocated − free − capped − CACHED, and
  `SchedulerOwnerProvider._cached_kv_slots` supplied the CACHED set. It was
  built from `_iter_drain_victims` (the cost-order EVICTION-victim walk),
  which seeds only from `evictable_leaves` and skips LOCKED parents
  (`RadixCache._iter_evict_victims`: `if parent.lock_ref != 0: continue`). So
  KV slots backing a locked shared-prefix node — held by MULTIPLE running
  reqs — were absent from the cached set, classified LIVE-uncached, and could
  be selected as a migration source. Migrating one rewrites only the FIRST
  owner's `req_to_token` (`rewrite_kv_token_indices` returns on first hit), so
  every co-sharer would read the freed `src` = silent KV corruption + an
  orphaned radix node. **Fix:** `_cached_kv_slots` now walks the WHOLE tree
  from `root_node` (locked AND evictable nodes), unioning every node's KV
  `value`. `test_kv_cached_locked.py` (2/2): the cached set includes locked
  slots, and a fully-locked-prefix page is never a migration source.

- **HIGH-2 / M2 — `_run_stage0` had no validate-then-apply.** The migration
  loop mutated as it iterated (`migrate_slot` → `rewrite_*` per move). If a
  later move's rewrite raised (its src has no live owner) AFTER `migrate_slot`
  already freed that src and moved its bytes, the owning req would read a
  freed slot and `dst` would be live-but-orphaned — and the raise is swallowed
  by `agent.tick`'s `except Exception`, so the corruption survives silently;
  earlier moves are left applied with no rollback. **Fix:** `_run_stage0` now
  validates that EVERY source has a live owner (`stage0_handler.has_live_owner`,
  a pure read) BEFORE relocating any bytes — an invalid plan aborts with zero
  mutation. `test_kv_migration_validate.py` (1/1): a 2-move plan whose 2nd src
  has no owner aborts before any `move_kv_cache`, leaving the free set and the
  good req's pointer untouched.
