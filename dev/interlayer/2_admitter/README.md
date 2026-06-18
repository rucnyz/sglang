# Admitter — implementation tests

Test suite for the per-arrival Admitter cost decision. Spec is in
[`../design.md`](../design.md) §"Admitter — per-arrival cost decision".

## Status

The Admitter ships the **full 7-action** cost program, and (P4.1) is
**symmetric** — a cross-fire's destination is whichever pool blocks the
arrival (`dst_pool` ∈ {kv, mamba}), not a fixed `dst=kv`:

| Action | Status | Production location |
|---|---|---|
| `own_free` | shipped | `python/sglang/srt/budgeter/admitter.py` (`_TIE_BREAK_ORDER`, all 7) |
| `own_evict` | shipped | same |
| `cross_free` | shipped | same |
| `cross_evict` | shipped | same |
| `own_migrate` | shipped (#183) | same |
| `cross_migrate` | shipped (#183; #271 KV live-migration) | same |
| `defer` | shipped | same |

**`own_evict` is LIVE (#180/#259).** The scheduler wires the radix
tree into the cost model via `CostModel.set_evict_cache("kv",
tree_cache)` right after constructing the Admitter (in
`Scheduler.init_running_status`, gated on `SGLANG_HIMA`). The
cache exposes `predict_evict_cost_us(num_tokens, pool="kv")`, which
sums `Σ n_b · c_i(s_b)` over the exact set the cache's own eviction
would pick — obtained from the SAME pure-read victim-selection
generator the real eviction consumes (`RadixCache._iter_evict_victims`
/ `MambaRadixCache._iter_evict_full_victims`), so the priced set is
byte-identical to the evicted set by construction (no drift).

Wired (own_evict, dst=kv):
- KV-only models: `RadixCache` (`_iter_evict_victims`).
- Hybrid models: `MambaRadixCache` (`_iter_evict_full_victims`,
  cost = `n_b·(c_kv+c_m)` per leaf — both buffers lost on evict).
- NOT wired: `HiRadixCache` / `HiMambaRadixCache` (override `evict()`
  with host-tiering the predictor doesn't model yet) and SWA — they
  stay at `c_evict_us("kv")` = +inf → own_evict infeasible
  (fail-closed). `SWARadixCache` LPB is #261.

**`#216` — byte-exact under lock**: `Admitter.decide_for_req` holds
the dst allocator's `_alloc_lock` across the capacity snapshot +
c^evict prediction + decision, so a concurrent worker-thread
`set_capacity` / alloc / free can't perturb the priced numbers
(`test_scheduler_hook` test 21).

**`MambaRadixCache.predict_evict_cost_us(pool="mamba")` is live**
(#275/#270). The mamba side now has the SAME single-source shape as
the KV side: one pure-read victim-ordering generator
`_iter_mamba_victims` (LPB Phase-1 contiguous cold-tail run, walked
lazily, then a lazily-built Phase-2 heap over hit-bearing nodes; or a
plain LRU prev-chain walk) is consumed by BOTH `evict_mamba` (which
frees each victim) and `_plan_mamba_eviction` (which adds only the
pure-read internal-tombstone vs leaf vs swept-cascade classification
the predictor needs). The earlier hand-synced double walk (the order
logic duplicated across `evict_mamba` and `_plan_mamba_eviction`) is
gone, so the priced set cannot drift from the evicted set. The
predictor prices internal victim `c_m`, leaf victim `c_kv + c_m`,
swept tombstone `c_kv`, each `n_b`-weighted under LPB. Its first
consumer is the **Budgeter's reuse-aware m2k drain cost**
(`snapshot["mamba_drain_cost_us"]`, wired in `agent._maybe_fire`),
which closes the cross-fire regression where draining a hot mamba
cache read as active-slack (#275). The Admitter's `cross_evict`
(src=mamba) decide → `set_evict_cache("mamba", ...)` → Stage-0 evict
path is the remaining end-to-end wiring tracked in #270.

**`HiMambaRadixCache` scope**: the single-source generator covers the
BASE `MambaRadixCache` only. `HiMambaRadixCache.evict_mamba` overrides
with a fundamentally different policy (pure LRU, no LPB two-phase) and
a host-offload lifecycle (`_evict_device_leaf` →
`_evict_to_host`/`_evict_regular`/`write_backup`, evicted/backuped node
states), so it keeps its own freeing path rather than consuming the
base generator. It still inherits the base `_plan_mamba_eviction` /
`predict_evict_cost_us` for cost prediction, but Hi own_evict is
intentionally fail-closed (see "NOT wired" above), so no cross-class
unification is needed. Pinned by
`test_mamba_evict_single_source.py` (LRU + LPB order/set/cost, base).

### evict ↔ predictor equivalence (why byte-exact holds)

**Note on env vars**: the Admitter and Budgeter are gated by `SGLANG_HIMA=1`.

`predict_evict_cost_us` and the real eviction share ONE pure-read
plan, so the priced set can't drift from the evicted set:

- **Plain `RadixCache`** — `evict()` and the predictor both consume
  `_iter_evict_victims(n)` (a generator). `evict()` materialises it
  fully before mutating (the generator seeds parent-promotion from
  live `len(parent.children)`).
- **Hybrid `MambaRadixCache`** — `evict_full()` and the predictor
  both consume `_plan_full_eviction(n)`, which returns
  `(victims, swept_tombstones)`. `victims` are the leaves/promoted-
  parents passed to `_evict_leaf_node`; `swept_tombstones` are the
  tombstone internal nodes (`mamba_value is None`, left by a prior
  `evict_mamba`) that `_evict_leaf_node` sweeps internally via
  `_iteratively_delete_tombstone_leaf`. The predictor prices victims
  at `n_b·(c_kv+c_m)` and swept tombstones at `n_b·c_kv` (mamba
  already gone). **Stop-point equivalence**: the planner counts
  swept-tombstone KV tokens toward the demand stop — matching
  `evict_full`'s `_evict_leaf_node` delta — and a tombstone is only
  counted in the same step its LAST child becomes a victim, so a
  "counted-but-not-freed" mismatch is structurally impossible.
- **Hybrid `MambaRadixCache` (mamba pool)** — `evict_mamba()` and
  `_plan_mamba_eviction(n)` (the predictor's source for `pool="mamba"`)
  both consume the same `_iter_mamba_victims()` generator, so the
  mamba side is single-source just like the KV side. The generator
  yields victim ORDER; the freeing (`evict_mamba`) and the
  classification+pricing (`_plan_mamba_eviction`) live in the two
  consumers because they genuinely differ (free vs price).
- **LRU semantics**: the KV `_iter_evict_victims` /
  `_plan_full_eviction` generators are **global-LRU** (heapify by
  `last_access_time`), matching the LPB path's global heap. The
  pre-refactor `evict_full` leaf-chain walk was an approximation that
  could evict a more-recent leaf before a just-promoted older parent;
  the refactor moves to consistent global-LRU (pinned by
  `test_mamba_evict_predictor` test_10). The mamba `_iter_mamba_victims`
  LRU branch instead walks the `mamba_lru_list` prev chain
  (insertion-order, tail = oldest); `last_access_time` only orders the
  LPB Phase-2 heap, not the LRU walk.

Pinned by `test_evict_predictor.py` (16, plain) +
`test_mamba_evict_predictor.py` (15, hybrid) +
`test_mamba_evict_single_source.py` (6, mamba single-source): static
oracles, predict==real-evict set, parent-promotion + multi-tombstone
cascade, demand-boundary, partial-tombstone-not-swept, LPB
no-perturbation, and the LPB-Phase-1-cold-tail / Phase-2-heap mamba
order under both `evict_mamba` and the predictor.

Tie-break ordering today: **`own_free > cross_free > own_evict >
cross_evict > defer`** (`admitter.py:_TIE_BREAK_ORDER`, strict —
the module docstring used to say `own_evict = cross_evict`, fixed
to match the constant). Design's full 7-action ordering extends
this with `> own_migrate > cross_migrate > defer` (the migrate
suffix slots in below the free / evict candidates so a free-or-
evict candidate that ties with a migrate one wins).

## Cold-start protocol (production hardening)

The Admitter applies an additional cross-* gate beyond what
design.md §"Shared cost model" specifies: until the `c^xfer` EWMA
has ≥3 observations (`cost_model.is_warmed_up()`), if any own-*
action is feasible (own_free or own_evict has finite cost), cross-*
is suppressed. This avoids polluting future arrivals' decisions
with cross-* fires priced against the conservative initial value
(3000 µs/page). Once no own-* is feasible, cross_free is used as
a probe to populate the EWMA. See the cold-start gate in
`Admitter.decide` (the `cross_gated` / `is_warmed_up()` block) for the
implementation. (Builds on top of design.md's drift gate; not a
spec-level requirement.)

Cost-model facade (`c^xfer`, `c^evict_i`, `c_i(s)`, `w_q`) is
shipped in `python/sglang/srt/budgeter/cost_model.py`. The
credibility-gate concept that earlier iterations exposed as a
separate component is now internalized into the pressure adapter
per design.md §"Empirical pressure signal" (paper appendix
`eq:nb-lb`).

## Files

### Unit tests (this folder)

| file | what it covers | design.md ref |
|---|---|---|
| `test_admitter_no_cross.py` | own-free / own-evict / defer paths (skeleton Admitter; cross-* gated off) | §"Admitter — per-arrival cost decision" |
| `test_cost_model_facade.py` | shared cost-model facade + c^xfer producer wiring | §"Shared cost model" |
| `test_evict_predictor.py` | #180 policy-aware `c^evict_i(X)` predictor for plain `RadixCache` (16 tests) — `predict_evict_cost_us` + `evict()` share `_iter_evict_victims`; falsification tests cross-check predicted set vs REAL `evict()` + static oracle + cascade/locked/degenerate coverage; `CostModel.c_evict_us(pool,...)` routes per-pool | §"Why exact c^evict" + §"LPB and the Admitter" |
| `test_mamba_evict_predictor.py` | #259/#270 `MambaRadixCache.predict_evict_cost_us` — `evict_full`/`evict_mamba` + predictor share `_plan_full_eviction`/`_plan_mamba_eviction` (15 tests: static LRU oracle, predict==real-`evict_full` set cost, LPB order, parent promotion, no-perturbation, tombstone cascades, and `pool="mamba"` predict==real-`evict_mamba` set cost incl. internal/leaf/swept cascade for the #275/#270 drain cost) | §"Why exact c^evict" |
| `test_sync_fire.py` | sync fire path (cross-free / cross-evict) + `_fire_inflight` mutex | §"Transfer protocol" |
| `test_scheduler_hook.py` | scheduler hook + per-arrival JSONL log (`SGLANG_HIMA`, `SGLANG_HIMA_ADMITTER_LOG` gates) | §"Admitter" (implementation-level) |
| `test_migrate_probe.py` | #182 boot-time `c_m` probe — `migrate_probe.measure_mamba_migrate` returns per-slot mamba migration wall; `CostModel.c_migrate_us(pool, n_slots)` scales linearly; cold-start + KV-side both fail-closed to +inf | §"Shared cost model" `c_m(X)` row |

### Verification tests (sub-folders)

| folder | what it covers | design.md ref |
|---|---|---|
| [`cost_picks_xfree/`](cost_picks_xfree/) | live workload: Admitter picks cross_free when cheap; cross_free dominates cross-pool decisions (≥95%) | §cost_picks_xfree + sweep arm |
| [`calibration_sanity/`](calibration_sanity/) | cost-model meaningfulness — calibration ratios + action coverage + top-2 discrimination (3 sub-tests for §calibration_sanity's three sub-conjectures) | §calibration_sanity-cal / §calibration_sanity-cov / §calibration_sanity-disc |
| [`own_evict_when_hot/`](own_evict_when_hot/) | negative: Admitter prefers own-evict when src cache is hot (cost model discriminates against blind transfers) | §own_evict_when_hot |
| [`cxfer_ewma_self_suppress/`](cxfer_ewma_self_suppress/) | negative: c^xfer EWMA spike self-suppresses over-fire via the fire-gate threshold | §cxfer_ewma_self_suppress |

## c_xfer page-count rounding (LCM wire-in)

`Admitter.decide()` prices `c_xfer_total = n_pages_rounded × c_xfer_per_page`
where `n_pages_rounded = ceil(n_pages / lcm_pages) × lcm_pages` (#219).
The LCM equals `lcm(actuator.n_kv_subpools, actuator.n_mamba_subpools)`
— the minimum page count the actuator atomically transfers (smaller
requests round to zero pages and abort, see `XPoolActuator.execute`).

The Admitter does not learn `lcm_pages` at construction; the actuator
isn't built until `BudgetAgent` lazy-constructs the chain on its first
tick. `BudgetAgent._ensure_actuator_chain()` (#229) builds the chain
EAGERLY at the top of `_maybe_fire` — before the planner decides
whether to fire — so even on low-load workloads where the Budgeter
chooses `direction=None`, the Admitter still receives `lcm_pages` on
the first tick. The chain-build calls `_wire_admitter()` which pushes
`(actuator, planner, lcm_pages)` into `scheduler.admitter` and the
field-default `1` is overwritten with `actuator.lcm_pages`. Until
that push, `decide()` under-prices cross-* by an LCM factor (e.g. 12×
on a `(4, 6)` subpool layout, larger on bigger models) and biases
admission toward cross-* over defer. The push is idempotent so
subsequent ticks don't re-overwrite. `XPoolActuator.lcm_pages` is a
`@property` so it remains a single source of truth — the actuator's
own `execute()` also reads it (no inline `math.lcm(...)`).

Pinned by `test_scheduler_hook.py` tests 18-20a:
- test 18 — `lcm_pages` property semantics across three geometries
- test 19 — pre/post-wire cross_free price scales by exactly the LCM
- test 20 — `_wire_admitter` push + idempotency in isolation
- test 20a — `_ensure_actuator_chain` actually calls `_wire_admitter`
  during first chain-build (integration guard for the load-bearing
  call site)

## Historical artifacts

Pre-consolidation sub-design / plan / progress / audit files (the
implementation work the above tests sprang from) live in
[`../archive/admitter_history/`](../archive/admitter_history/).
Their content is now subsumed by the top-level [`design.md`](../design.md)
and [`PLAN.md`](../PLAN.md); they remain for traceability only.

## Phase 3 Steps 2–5 — migrate inputs, three-stage knapsack, Stage-0, refuse counter (#183)

Step 1 (committed) extended `Admitter.decide()` to seven candidates with
default-inert migrate params. Steps 2–5 wire the migrate path end-to-end:

**Step 2 — `decide_for_req` feeds migrate inputs** (`admitter.py`).
Inside the `kv_alloc._alloc_lock` block, computes `n_migrate_slots =
ceil(x_tokens / tokens_per_page)`, `src_migratable` (mamba LIVE state
defraggable to free slots, in KV-token-equiv), `dst_migratable=0` (KV has
no migrate primitive), `c_migrate_src_us = c_migrate_us("mamba",
n_migrate_slots)`, `c_migrate_dst_us = c_migrate_us("kv", …) (== +inf)`,
and passes all into `decide()`. `src_migratable` comes from
`_get_mamba_migratable_kv_equiv` — a CONSERVATIVE lower bound
`min(live_slots, free_slots) × tokens_per_page` where `free_slots =
MambaPool.available_size()` and `live_slots = MambaPool.live_size −
free_slots` (a migrate needs both a LIVE slot to move and a FREE dst slot).
Stub seam: `scheduler.get_mamba_migratable_kv_equiv()` (mirrors
`get_mamba_free_kv_equiv`). Pinned by `test_scheduler_hook.py` tests 23–25
(finite cross_migrate under a stub; +inf under c_m cold-start;
real-path conservative-min formula). `StubMambaPool` gained a `live_size`
attr to match the real pool (no production getattr fallback).

**Step 3 — FirePlan fields + three-stage knapsack** (`fire_plan.py`,
`owner_provider.py`, `scheduler_owner_provider.py`, `fire_planner.py`).
`FirePlan` gained `drains: tuple = ()` and `migrations: tuple = ()`
(empty ⇒ free-only plan byte-unchanged). `OwnerMap` gained
`cached_pages_in_cost_order` / `live_pages_in_cost_order`
(`None` unless expansion is requested — Stage-1-only callers stay
zero-cost). `XPoolFirePlanner.build()` gained `allow_drain` /
`allow_migrate` flags and accumulates free → (drain CACHED, cheapest-
first) → (migrate LIVE, ascending c_m) until `n` pages; sets
`plan.drains` / `plan.migrations`; on exhaustion increments
`self.refuse_count` (init in `__init__`) and returns None. The real
`SchedulerOwnerProvider._expansion_lists` is now implemented (#267, see
below); the free-only path (`allow_*=False`) still returns
`(None, None)` zero-cost. Pinned by `test_fire_planner_stages.py` tests
1–5 (Stage-1 wins on free≥n; Stage-2 fills from cached; Stage-3 fills
from live; refuse_count monotonic; empty-expansion plan byte-identical
to today).

**Step 4 — Actuator Stage-0 execution** (`xpool_actuator.py`).
`cap_barrier` runs Stage-0 BEFORE the free-page cap-barrier, guarded on
`plan.drains or plan.migrations` (empty ⇒ byte-identical to the pre-#183
free-only path). `_run_stage0` drains cached pages (CACHED→FREE via the
scheduler-coupled `stage0_handler.evict_pages`) and migrates each LIVE
slot (`MambaPool.migrate_slot(src, dst)` — byte-exact, property A2 —
plus `stage0_handler.rewrite_ssm_state_indices(src, dst)`, mandated by
the migrate_slot docstring). Missing handler on a Drain/Migration plan
fails loud. GPU byte-exact verification (`test_stage0_transfer.py`, real
`MambaPool` on cuda:0, 3/3): post-migrate the dst slot is byte-exact to
the src's pre-migration conv+temporal contents; the ssm rewrite fires
with (src,dst); the src slot lands in `_capped_slots` (FREE-but-held,
the exact state cap_barrier unmaps) and leaves `free_slots`. Sample
output: `slot 1->2 state relocated, ssm rewritten, src capped/unmappable`.

**Step 5 — refuse counter surfacing** (`agent.py`, `admitter.py`).
`BudgetAgent._snapshot` reads `self._fire_planner.refuse_count` into the
per-tick JSONL as `fire_refuse_count` (absent until the chain builds).
`Admitter.execute_decision`'s `plan is None` branch surfaces
`refuse_count` in the defer reason. Pinned by
`test_fire_planner_stages.py` test 6 (snapshot surfacing) and
`test_sync_fire.py` test 10 (build→None defer reason carries
`refuse_count`, monotonic).

Regression: full 2_admitter suite green (test_scheduler_hook 26/26,
test_sync_fire 13/13, test_admitter_seven 6/6, + all others);
test_lpb_policy 16/16; test_owner_map_vectorized 7/8 (the 1 failure is a
pre-existing GPU-contention timing flake, identical on the clean
baseline).

## #267 — `_expansion_lists` + Stage-0 handler + cross-fire wiring (FINAL ring)

The migration MECHANISM (Steps 2–5) was verified with FAKE
providers/handlers. #267 fills the production gap so the three-stage fire
planner runs end-to-end with cross-fire ON:

**Real `SchedulerOwnerProvider._expansion_lists`**
(`scheduler_owner_provider.py`). No longer raises NotImplementedError.
- `_cached_pages_in_cost_order` (Drain): reuses the SAME pure-read victim
  selector `c^evict` and the real eviction consume —
  `MambaRadixCache._plan_full_eviction` on a hybrid cache,
  `RadixCache._iter_evict_victims` on a KV-only cache. Maps each victim's
  freed slots (`node.value` for `pool='kv'`, `node.mamba_value` for
  `pool='mamba'` — full eviction frees both buffers) → page-ids, emitting
  a page the first tick it becomes FULLY covered (every slot free), in
  eviction cost order. Byte-identical to a real evict by construction.
  The `_plan_full_eviction` budget is always `full_evictable_size()+1`
  (it counts FULL/KV tokens regardless of which buffer is harvested —
  feeding the mamba count truncates the walk early).
- `_live_pages_in_cost_order` (Migration): mamba LIVE slots NOT held by a
  cached `mamba_value` snapshot (live = allocated − free − capped −
  cached, excl. padded slot 0), mapped to pages, ordered ascending slot
  id. Tie-break note: `c_m` is a per-slot CONSTANT, so ascending-slot-id
  IS the ascending-`c_m` cost order — an LRU-ish FIFO. Bounded by the
  free-dst budget (one free dst per moved slot). KV pool returns `[]`
  (no migrate primitive).

**Real `SchedulerStage0Handler`** (`scheduler_stage0_handler.py`,
scheduler-coupled). Implements the actuator's Stage-0 collaborator:
- `evict_pages(direction, drains)` → CACHED→FREE via `tree_cache.evict`
  (`num_tokens` for KV-source, `mamba_num` for mamba-source).
- `alloc_free_dst_slot(pool)` → PEEK a free mamba slot (the following
  `migrate_slot` consumes it); excludes padded slot 0; fails loud on
  exhaustion.
- `rewrite_ssm_state_indices(src, dst)` → repoints the running req that
  owns `src` at `dst`, moving BOTH `req.mamba_pool_idx` AND the mirrored
  `HybridReqToTokenPool.req_index_to_mamba_index_mapping` (what the
  attention backend reads via `get_mamba_indices`). Fails loud if no
  running req owns `src`.
Wired in `BudgetAgent._ensure_actuator_chain` as
`XPoolActuator(stage0_handler=SchedulerStage0Handler(...))`.

**Cross-fire path** (`admitter.py`, `scheduler.py`).
`Admitter.execute_decision` now handles `cross_migrate` (in addition to
`cross_free`/`cross_evict`) and maps the action to the planner's
expansion gates: `cross_free` → Stage 1 only; `cross_evict` →
`allow_drain=True`; `cross_migrate` → `allow_drain=allow_migrate=True`.
`Scheduler._add_request_to_queue` fires for `cross_migrate` too.

GPU validation (`test_expansion_lists.py`, real `MambaRadixCache` +
`MambaPool` on cuda:0, 5/5): Migration list = live-uncached pages in
ascending order (cached excluded); Drain list == `_plan_full_eviction`
victim order; handler moves BOTH req pointers + `evict_pages` frees the
cached slot; **end-to-end `XPoolActuator._run_stage0` with the
PRODUCTION handler is byte-exact** (slot relocated, both pointers
rewritten, src capped/unmappable); free-only path returns `(None,None)`.

**Live engine validation** (Qwen3.5-9B hybrid, GPU 0,
`SGLANG_HIMA=1`, mem-fraction 0.42, mamba=400, lpb):
6 real Admitter-driven `cross_free` fires executed on GPU —
`execute_async DONE dir=mamba_to_kv unmapped=768 granted=768` (byte-
conserving), 0 aborts, 0 verify failures, 240/240 requests returned 200,
no crash/NaN.

**Design ambiguity resolved — why no NATURAL cross_migrate/cross_evict
fire on this HW.** The Admitter's arrival path is fixed
`src=mamba, dst=kv` (KV-token demand, mamba is the only other pool).
For that direction:
- `cross_evict` (src=mamba) stays infeasible on the ADMITTER arrival
  path: `c_evict_us("mamba", …)` is +inf because `set_evict_cache(
  "mamba", …)` is never wired into the Admitter. (The underlying
  `MambaRadixCache.predict_evict_cost_us(pool='mamba')` predictor now
  EXISTS — its `evict_mamba` pure-read mirror landed for the Budgeter's
  reuse-aware m2k drain cost, #275/#270 — but routing it into the
  Admitter's `cross_evict` decide + Stage-0 evict is the remaining
  end-to-end wiring in #270.)
- `cross_migrate` (src=mamba) is structurally dominated by `cross_free`:
  `src_migratable = min(live, free) ≤ free = src_free` (the conservative
  bound — a migrate needs a free dst), so whenever cross_migrate is
  feasible (`src_migratable ≥ x`) so is the cheaper `cross_free`
  (`src_free ≥ x`), and `cost(cross_free)=c_xfer < c_xfer+c_m=
  cost(cross_migrate)`. cross_migrate appears as a finite CANDIDATE in
  the cost vector (confirmed live) but can never WIN.
So a natural cross_migrate/cross_evict fire cannot be triggered for the
arrival direction the Admitter uses on this hardware. The Stage-0 /
expansion machinery is therefore validated by the deterministic GPU
tests above (real pools, real handler, byte-exact). A separate finding:
under heavy KV pressure the Admitter's `cross_free` over-harvests the
mamba pool (the bottleneck), which can starve the running batch
(`cache_unfinished_req`: "Can not alloc mamba cache") — a `cross_free`
property predating #267, orthogonal to the Stage-0 work.

## Migration feasibility — verified model (#269)

Full verification of what Migration (`cross_migrate` / `own_migrate`)
can actually do, done before touching code (the earlier `min(live,free)`
framing was wrong twice — recorded here so the lineage is honest).

**The transfer unit is the VMM chunk, not sglang's attention `page_size`.**
`tokens_per_chunk = chunk_bytes // per_token_bytes` (default `chunk_bytes`
= 2 MiB, the H200 VMM granularity). Page selection has **no contiguity
requirement** (`fire_planner`: "any K free pages work equally well").

**Migration frees a whole chunk only by relocating its occupied slots
into SCATTERED free slots** — free slots sitting on *partially-live*
chunks. Whole-free chunks are Stage-1's transfer payload (the planner
unmaps them on fire), so they are NOT migration destinations. Hence the
only capacity Migration adds is consolidation of fragmentation, and that
exists only when a chunk can be partially free, i.e. `tokens_per_chunk
≥ 2`.

**Atomic vs fragmentable is a per-pool, config-dependent property:**

| cfg (Qwen3-Next, heads=32 hd=128 state=128) | mamba per_token | tps | |
|---|---|---|---|
| tp=1, fp32 ssm (default) | 2 MiB = 1 chunk | **1** | **atomic** |
| tp=1, bf16 ssm | 1 MiB | 2 | fragmentable |
| tp=2, fp32 | 1 MiB | 2 | fragmentable |
| tp=4/8, fp32 | 512/256 KB | 4/8 | fragmentable |

KV is **always** fragmentable (`per_token = kv_heads·head_dim·dtype` ≈
1–4 KB → `tps` ≈ 512–2048). The single-GPU headline bench
(`4_e2e/cc_traces_headline/run_cc.sh`: `TP=1`, no `SGLANG_MAMBA_SSM_DTYPE`
→ fp32) sits exactly on the mamba-atomic corner, so `cross_migrate(src=
mamba)` is genuinely inert there — not a bug, the geometry.

**Consequence for the seven candidates (current src=mamba/dst=kv):**
- `cross_migrate(src=mamba)` — inert at tp=1/fp32 (atomic); active under
  TP/bf16. mamba already has `migrate_slot`.
- `own_migrate` — no admission mechanism in any direction: admission
  allocates scattered free slots, so intra-pool defrag never adds an
  allocatable slot. (Candidate kept for design-structure parity; stays
  `+inf` via `c_migrate_dst_us`.)
- The genuinely-valuable migration is `cross_migrate(src=kv)` in the
  `dst=mamba` direction (KV is fragmentable) — needs a KV `migrate_slot`
  primitive (#271) + the dst=mamba direction (#159). This is where #100
  ("Migration earns its keep") lives.

**Fix landed (#269, then hardened under audit — feasibility +
execution).** `SchedulerOwnerProvider._live_pages_in_cost_order` returns
concrete migration MOVES — `[(freed_page_id, ((src,dst),...))]` — not
page-ids:
- **Sources are FULLY-LIVE chunks** (all `tps` slots live-uncached);
  **destinations are SCATTERED free slots on KEPT partial chunks**. The
  sets are disjoint, so a migration never uses the freed chunk's own slot
  as its destination (no self-destination) and never lands on a whole-free
  chunk (Stage-1's transfer payload); each donor is assigned once (no
  double-spend). The classify loop is bounded to `[1, n_pages)`, so the
  padded tail slot can't leak into the donor pool.
- **Atomic pools (`tps==1`) yield `[]`** — no partial chunks ⇒ no donors.
- `FirePlan.migrations` carries the `(src,dst)` pairs; `XPoolActuator.
  _run_stage0` runs `migrate_slot(src, dst)` directly (no page-id used as
  a slot-id, no dst guessing — the obsolete handler `alloc_free_dst_slot`
  was removed). This fixes the latent tps≥2 execution bugs (page-id↔
  slot-id, self-dst, arbitrary dst) that were dormant because
  cross_migrate never fired at the tps=1 bench.
- `Admitter._mamba_feasibility` reads `mamba_free` (fully-free chunks) and
  `src_migratable` (consolidatable chunk count) from the SAME
  owner-provider computation, so decide and planner agree by construction.
  It is GATED: returns early when `cross_fire` is off, and on an ATOMIC
  layout takes a CHEAP path (`available_size()·tps`, 0) WITHOUT building
  the owner map — no per-arrival GPU sync under the KV `_alloc_lock`. Only
  a fragmentable layout pays for the walk, which snapshots the mamba pool
  under its own `_alloc_lock` (#222) for consistency.
- `src_migratable` (and `src_evictable`, `mamba_free`) are the per-
  mechanism QUANTITIES; the cost program then combines them CUMULATIVELY
  (#273): cross_evict feasible iff `free + evictable ≥ X`, cross_migrate
  iff `free + evictable + migratable ≥ X`, mirroring the planner's
  free→drain→migrate fill. Each cross cost charges its mechanism only for
  the shortfall it covers (cross_migrate = c^xfer(X) + c^evict(drain part)
  + c_m(migrate part)), so the Admitter's predicted cost equals the fire's
  actual byte cost; decide_for_req prices c^evict once at the shortfall
  target `min(X−free, evictable)` under the #216 lock. Zero-downside holds
  (when migration isn't needed the migrate part is 0 → cross_migrate ties
  and loses the tie-break). The planner's fixed free→drain→migrate order
  is not always cost-optimal (long cached prefixes can make drain pricier
  than migrate) — a cost-merge optimization is tracked as Phase-6 #274.

Audit hardening (post-#273):
- **LCM-rounding consistency** — cross feasibility AND the drain/migrate
  shortfall use `x_eff = n_pages_rounded·tps` (the rounded harvest the
  planner actually frees, via `_round_up_pages`), not raw `x_tokens`.
  Otherwise the Admitter could pass feasibility against X while the
  planner refused for want of the rounded count, or under-price the
  rounded-up remainder (same spirit as #219).
- **`c_evict_us(pool, 0) == 0`** — the "0 tokens → 0 cost" short-circuit
  precedes the cache-None check, so a zero-drain cumulative candidate is
  never poisoned to +inf when the evict cache is unwired.
- **mamba `_live_pages_in_cost_order` tail** — when `size % tps != 0` the
  trailing partial page is intentionally excluded (conservative
  under-count, never over-selects); documented rather than asserted, since
  the actuator marks the real slots from `expand_pages_to_token_slots`,
  so a non-divisible size is valid and a hard assert would crash
  legitimate fragmentable configs.

Tests: `test_expansion_lists.py` test_1 (atomic → `[]`), test_1b
(fragmentable tps=2: fully-live source, donors on other pages, no
self-dst, budget cap, move shape), test_1c (capped chunk excluded from
donors), test_1d (mixed `[free,cached]` chunk: cached excluded from
source, free sibling still a donor), test_3/3b (Stage-0 runs the
planner-assigned `(src,dst)`); `test_stage0_transfer.py` test_1b
(multi-move `_run_stage0` byte-exact); `test_fire_planner_stages.py`
test_3 (planner flattens page moves); `test_admitter_seven.py` test_2b
(cumulative free+evict feasibility), test_2c (free+evict==X boundary
tie), test_2d (free≥X three-way tie → cross_free), test_3 (3-part cost
composition); `test_scheduler_hook.py` test_25 (no provider →
`src_migratable=0`), test_26 (atomic cheap-path asserts NO owner-map
build; fragmentable feasibility boundary), test_27 (shortfall>evict
split: drain capped at evictable + migrate remainder).

**#272 — FIXED.** `MambaPool.__init__` used to clamp `tokens_per_chunk =
max(1, chunk_bytes // per_token_bytes)`, while `MultiTensorArena` strictly
requires `chunk_bytes % per_token == 0`. So a model whose per-layer SSM state
exceeded one chunk (or did not divide it) got a silently-wrong
`tokens_per_chunk` (the clamp returned 1, or the `//` floored), was sized with
that wrong value, and only crashed later as a `MultiTensorArena` RuntimeError
mid-boot. The clamp is now `_arena_tokens_per_chunk(chunk_bytes,
per_token_bytes)` (module-level in `memory_pool.py`), which enforces the same
`% == 0` invariant at config time and raises an actionable `ValueError` (which
multiple to set `SGLANG_ARENA_CHUNK_BYTES` to) instead of clamping. Pinned by
`dev/interlayer/0_page_state_machine/test_chunk_alignment_272.py` (5 tests:
exact divisor, non-dividing → raise, per-token-state-larger-than-chunk → raise
not clamp, non-positive, arena-constraint sweep).

**The genuinely-valuable migration (`cross_migrate(src=kv)`, #271)** — KV
is always fragmentable, so consolidating KV chunks to donate to mamba is
where Migration earns its keep; it needs a KV `migrate_slot` primitive +
the dst=mamba direction (#159). The execution machinery above already
generalizes to it (same `(src,dst)` move plan via `build_kv_owner_map`).

## #268 — internal mamba victim under-priced in `predict_evict_cost_us`

(Distinct from `#298`, the parallel degenerate-curve gate on the *migrate*
actuator cost `c_migrate` — still open, not touched here.)


`MambaRadixCache.predict_evict_cost_us(pool="mamba")` priced an INTERNAL
mamba victim (snapshot dropped, KV value kept for descendants) by `c_m`
alone. Under the real calibration `κ_M = 0` (#276) that is ~0, so a HOT
internal snapshot's drain cost collapsed to 0 and the Budgeter's m2k NB
gate could read draining it as free → over-harvest (#268).

Subtlety that makes it narrow: when the internal victim's KV is ALSO freed
this pass (its leaf cascade tombstones it into `swept`), the `c_kv` is
counted in `swept`, so `c_m` (internal) + `c_kv` (swept) already sums to the
whole-prefix total — correct, and naively adding `c_kv` to every internal
victim double-counts (broke `test_7`). The gap is ONLY the internal victim
whose KV STAYS (live descendants, never swept): it gets `c_m` alone.

Fix (swept-aware, `predict_evict_cost_us`): an internal victim NOT in
`swept` is priced by the whole-prefix total `c_kv + c_m`, because recovering
its snapshot still needs a full prefix re-prefill (attention and recurrent
layers interleave; the kept KV cannot stand in for the rest of the forward —
the single-curve recovery model of paper §3). Matches a leaf victim and
`eviction_priority()` (which already prices a KV-present internal node by
`c_kv + c_m`, so it needed no change). Reproducing test:
`test_mamba_drain_overharvest_298.py`; `test_mamba_evict_predictor.py` (15)
stays green (the KV-swept `test_7` case is unchanged).

Live exposure is small: under LPB, `eviction_priority` already protects hot
internal nodes (high ℓ), so a drain only reaches a warm KV-stays internal
node when its volume exceeds all cold cache (bounded by fire magnitude).
This is a correctness/consistency fix, not a large e2e win.

## #298 — migrate has no degenerate-curve gate, and that is correct

The cross-fire DRAIN has a degenerate-curve fail-closed gate
(`BudgetAgent._cross_drain_allowed`): the drain evicts cached entries, so its
reuse-aware cost is a recovery *curve* that can collapse to ~0 (e.g. κ_KV all
zero) and then cannot price hot cache, so the gate falls closed to free-only.
`#298` asked whether the migrate side needs the same gate. It does not, and the
asymmetry is intentional:

- The migrate cost `c_migrate` is a boot-probed **scalar** (`BootProbedMigrateCost`,
  per-slot µs), not a fitted curve, so there is nothing to "go degenerate."
- Migration relocates **LIVE** state byte-exact (`MambaPool.migrate_slot`); it
  evicts no cache and triggers no recompute, so there is no hot-cache
  mispricing for a curve gate to guard.
- It already fail-closes a different (correct) way: cold-start `c_migrate = +inf`
  (infeasible until the boot probe runs), `SGLANG_XPOOL_KV_MIGRATE` default-off,
  and the `can_migrate_slot()` self-gate; KV migration is priced `+inf` (no
  primitive), and mamba migration is atomic-inert at tp=1/fp32 (#269).

The cold-start `+inf` fail-closed contract is pinned by
`test_migrate_probe.py` (test_3 cold-start + KV → +inf; test_2 post-probe
linear). So `#298` is resolved as a justified asymmetry: no gate added (adding
one would guard a risk that does not exist); the rationale is documented in
`_cross_drain_allowed`'s docstring. Distinct from `#268` (the drain-side
over-harvest fixed above).

## Phase 4 — symmetric (bidirectional) Admitter (P4.1)

`decide_for_req` was KV-dst only: it always priced "grow KV by draining
mamba" (m2k) and never checked whether mamba could give the arriving req its
state slot. A hybrid arrival needs room in BOTH pools, so a mamba-pressure
burst (mamba full of live state, KV slack) was reported `own_free` off the KV
slack alone — admitting the req into a pool that then crashed at
`cache_unfinished_req` ("Can not alloc mamba cache", #312). That gap is *why*
the Budgeter needed a static working-set floor at all.

P4.1 makes the Admitter symmetric:
- mamba scarce (`free+evictable < _mamba_arrival_need_slots`), KV can donate
  → **grow mamba from KV** (`dst=mamba`, `src=kv`; `_decide_grow_mamba`).
- KV scarce (or neither) → the existing **grow KV from mamba** path.
- both scarce → **defer** (opposite cross directions; neither serves — fall
  back to sglang back-pressure, never a doomed fire).
- the chosen action vs defer is cost-driven: an empty queue prefers the free
  defer; a backlog (a burst) makes deferring expensive so the grow fires.

`AdmitterDecision` now carries `dst_pool`/`src_pool`; the scheduler hook
(`_maybe_admitter_fire`) fires in that direction. Pinned by
`test_symmetric_admit_p4.py` (mamba-scarce never owns; under queue pressure
grows mamba from KV; both-scarce defers; KV-scarce regression; neither owns).
This is the burst-safety mechanism that will let the Budgeter floor drop to a
grow-latency buffer (P4.5, after the e2e validates the grow lands in time).

Note: `test_stage0_transfer.py` has a pre-existing failure (`_FakeStage0Handler`
lacks `has_live_owner`, a stub drift from the prod Stage0 handler) — unrelated
to P4.1; confirmed failing identically with P4.1 changes stashed.

### P4.2 — fork-lifecycle sizing

P4.1's scarcity test used `_mamba_arrival_need_slots = 1` (the active SSM slot
only). But a hybrid req also forks ONE mamba slot at `cache_unfinished_req`
(copies its prefix state into a new locked node, keeping its active slot —
net +1). So a req admitted when mamba has exactly 1 free slot gets its active
slot, then its mid-prefill fork finds the pool full → #312. P4.2 sizes the
arrival need at `active(1) + fork(1) = 2` slots (sglang physical
constraint from `MambaRadixCache.cache_unfinished_req`): a single free slot is still
scarce. Pinned by `test_fork_sizing_p4.py` (one free slot → not own_*; under
pressure → grow mamba; two free slots → own_free).

This also fixed a unit bug in `_decide_grow_mamba`: the grow demand `x` must
be the FULL lifecycle need (`need · tokens_per_page`), not the shortfall —
otherwise `own_free` (dst_free ≥ x) is spuriously feasible against the
shortfall and the req is admitted short of its fork slot. `decide` then judges
own-vs-grow exactly as the dst=kv side does with `x_tokens`.

P4.3 (both pools scarce → defer) is already covered: P4.1's both-scarce branch
returns `defer` with a complete candidate vector, and the scheduler hook
no-ops on own/defer (only cross_* fires), so the req simply stays queued —
sglang's normal back-pressure, no doomed fire, no crash.

### P4.1-B1 — the k2m grow fire must be sized in mamba chunks (audit fix)

Audit of P4.1/P4.2 found the mamba-grow firing path mis-sized the transfer:
`_maybe_admitter_fire` passed the req's KV input length (`x_tokens`) to
`execute_decision`, which sized `n_pages` from it — so a mamba grow transferred
the req's KV-input worth of pages (e.g. 4096 tokens → ~4 chunks, rounded to the
actuator LCM), unrelated to the mamba slot need. And even sized by the need, it
converted slots→pages with the KV `tokens_per_page` (1024), over-growing mamba
by `tokens_per_chunk×` on a fragmentable layout (tps>1). Inert on the atomic
cc config (tps=1) but wrong elsewhere.

Fix: `AdmitterDecision` carries `fire_x_tokens` (the priced dst demand);
`execute_decision` sizes the fire from THAT, not the req's KV input.
`_decide_grow_mamba` expresses the demand in mamba CHUNKS —
`ceil(need / mamba_tokens_per_chunk)` — read via `_mamba_tokens_per_chunk`
(owner provider → scheduler seam → arena → 1). Pinned by
`test_grow_fire_sizing_p4.py`: the grow fires `kv_to_mamba` with
`n_pages = ceil(need/tps)` (atomic tps=1 → 2 chunks; fragmentable tps=2 →
1 chunk), not the KV input. This test also closes the audit's E1 gap (no test
previously drove the actual `execute_decision`/FirePlan firing path).
