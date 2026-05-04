# T8 — cross_pool layering refactor

**Status:** design (pre-code).
**Triggered by:** T7 swarm conc=800 smoke v3 crashed with `CUDA illegal memory access`
in `process_batch_result_prefill` immediately after a fire commit, despite drain
having evicted 59231 cached tokens. Root cause: actuator unmapped chunks containing
pages held by active reqs.

## TL;DR

Move the **owner-aware decision** of *which* chunks to unmap, *which* tree refs
to drain, and *which* active-req pages to migrate **out of the actuator** and
**into a scheduler-side planner**. Actuator becomes a pure mechanism that
executes a fully-specified `FirePlan`. Eliminates the entire class of
"actuator unmapped a chunk that an active req owned" bugs by construction —
the planner owns ground truth (running batch, tree, req_to_token_pool), so
plans cannot reference active-req pages without an accompanying migrate step.

## Why the current layering is wrong

`cross_pool_actuator.py` (795 lines) currently does both:

1. **Policy / inference** — `_select_drainable_chunks`, smart-overcap, tail-shrink
   fallback, `mark_pages_capped` post-fact wiring, tpc resolution defaults.
2. **Mechanism** — `arena.shrink_explicit`, `arena.grow`, allocator capacity calls,
   tensor-handle bookkeeping.

The policy half is making decisions with **insufficient information**: it only
sees `allocator.free_page_mask`. It cannot distinguish a page held by a tree
node (drainable) from a page held by an active req (must migrate). Lacking
that, it falls back to heuristics (smart-overcap → tail-shrink), each layer
silently degrading to a less-correct fallback. T7 hit the bottom of that chain
and unmapped a tail chunk containing an active req's `out_cache_loc`.

The "decoupled" framing — actuator only depends on pool/allocator — sounds
clean but actively hides the inputs the operation requires. It is missing
abstraction, not extra abstraction.

## The new layering

```
┌─────────────────────────────────────────────────────────────┐
│  Scheduler (event_loop_normal, between two forward steps)   │
│  ─────────────────────────────────────────────────────────  │
│  budgeter.agent.tick(state)                                 │
│      → CrossPoolPlanner.decide() → PlanDecision             │
│      → XPoolFirePlanner.build(decision) → FirePlan          │
│      → CrossPoolActuator.execute(plan)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ pure mechanism
┌─────────────────────────────────────────────────────────────┐
│  CrossPoolActuator.execute(plan)                            │
│  ─────────────────────────────────────────────────────────  │
│  for chunk in plan.chunks_to_unmap:                         │
│      arena.shrink_explicit(chunk)                           │
│  for src,dst,req,slot in plan.pages_to_migrate:             │
│      D2D copy, atomic kv_indices update, free old           │
│  arena.grow(plan.chunks_to_map_on_dst)                      │
│  no decisions, no fallbacks, no heuristics                  │
└─────────────────────────────────────────────────────────────┘
```

`CrossPoolPlanner` (existing) keeps its job: deciding *when* + *direction* +
*magnitude*. We add `XPoolFirePlanner` (new, scheduler-side) that turns that
high-level intent into a concrete chunk-level plan, with full ownership info.

## Data structures

```python
# python/sglang/srt/arena/fire_plan.py
@dataclass(frozen=True)
class MigrateOp:
    src_page: int             # capped page-id
    dst_page: int             # head-region page-id reserved for the migrate
    req_pool_idx: int         # which req owns this page
    slot_in_req: int          # offset into req.kv_indices

@dataclass(frozen=True)
class FirePlan:
    direction: str                       # "kv_to_mamba" | "mamba_to_kv"
    capped_page_range: tuple[int, int]   # [low, high) — to mark before migrate
    chunks_to_unmap_src: list[int]       # chunk ids on src arena
    pages_to_drain: list[int]            # tree refs to evict before unmap
    pages_to_migrate: list[MigrateOp]    # active-req refs to copy before unmap
    chunks_to_map_dst: int               # n chunks to grow on dst
    expected_unmap_pages: int            # for verify step
    plan_seq: int                        # monotonic, for log correlation
```

```python
# python/sglang/srt/arena/owner_provider.py
class OwnerProvider(Protocol):
    """Scheduler-implemented; passed to planner at construction time.
    Built once per fire (NOT cached): scheduler holds the lock
    between forward steps and walks running_batch + tree_cache fresh.
    """
    def build_owner_map(self) -> "OwnerMap": ...

class OwnerMap:
    free_pages: set[int]
    tree_pages: dict[int, "TreeNodeRef"]
    active_pages: dict[int, tuple[int, int]]  # page → (req_pool_idx, slot)
    @property
    def total(self) -> int: return len(free) + len(tree) + len(active)
```

## What dies in the refactor

From `cross_pool_actuator.py`:
- `_expand_via_migration` (lines 41-113) — silent fallback when `target_pool=None`.
- `_select_drainable_chunks` (lines 115-173) — heuristic; planner replaces.
- `_drain_complete` (line 277) — replaced by explicit `pages_to_drain` list.
- The `else: arena.shrink(name, n_per_src_subpool)` tail-shrink fallback in
  `_do_transfer` — gone. There is only one path: execute the plan.
- All tpc-resolution fallback chains (already partly cleaned in pre-T8 sweep).

Final actuator size: target ≤ 250 lines (from 795).

## Hazard table — before vs. after

| Hazard | Pre-T8 | Post-T8 |
|--------|--------|---------|
| Active req page in unmapped chunk | possible (T7 v3 hit it) | impossible: every active page either pinned (planner skips chunk) or migrated (D2D copy + kv_indices swap) |
| New alloc lands on capped page | racy: `mark_pages_capped` is post-fact | impossible: cap-barrier is step 1 of execute |
| `shrink_explicit` silently no-ops on out-of-range chunk | yes (T7 v1 root cause) | impossible: planner emits chunk ids it has just verified |
| Tpc-default of 1 corrupting indices | latent (defended by raise) | irrelevant: planner has already converted to page-id space |
| Smart-overcap insufficient → tail-shrink degrades | yes | impossible: no fallback path exists |
| CUDA graph captured kv_indices invalidated | handled by T6 | unchanged (T6 stays) |
| Spec-decoding state lost on migrate | handled by T4 | unchanged (T4 stays, called from migrate op) |

## Concurrency story

Whole `execute(plan)` runs in scheduler thread, between two forward steps,
with `scheduler.lock` held. No async, no background thread. This is the
same window where `set_capacity_pages` already runs today — we just
guarantee the plan was built in the same window.

## Per-fire cost (typical)

- Plan build: O(running_batch_size + tree_node_count). Numpy-vectorized;
  expect <1 ms on swarm conc=800.
- Cap-barrier: µs (allocator bitmap op).
- Drain (tree evicts): existing radix-tree evict path; µs–tens of µs.
- Migrate: 0 in the common case (smart selection picks all-free chunks);
  worst case ≤ ~50 D2D copies × 1.3 µs = ~70 µs.
- Unmap + map: ~5.7 ms (unchanged from current).
- Verify: assert over capped-page set ⊆ free-set; µs.

Net: indistinguishable from current fire when smart selection wins; ≤ 5 ms
overhead in worst case. **Fire frequency is seconds-apart**, not in the
forward critical path.

## Migration plan (engineering steps)

Each step lands as one commit. Default behavior unchanged at every step
until the final flag flip.

1. **Define `FirePlan` and `OwnerProvider`** in `python/sglang/srt/arena/`.
   No call sites yet. Pure types.
2. **Implement `XPoolFirePlanner`** in `python/sglang/srt/budgeter/fire_planner.py`.
   Inputs: `OwnerProvider`, `KVArenaActuator`, `MambaArenaActuator`, direction,
   target delta. Output: `FirePlan`. Flag-gated: `SGLANG_T8_PLANNER=1`.
3. **Implement `CrossPoolActuator.execute(plan)`** as a new method, parallel to
   the existing `_do_transfer`. Behavior identical to `_do_transfer` when
   `pages_to_migrate=[]` and `pages_to_drain=[]` (smart-overcap-only plan).
   Flag: `SGLANG_T8_EXECUTE=1` (must be on for `_do_transfer` to dispatch
   into the new path).
4. **Wire scheduler-side `OwnerProvider`** that walks `running_batch`,
   `tree_cache`, `req_to_token_pool`. New file
   `python/sglang/srt/budgeter/scheduler_owner_provider.py`. Constructed by
   `BudgetAgent` when scheduler is available.
5. **Implement migrate path** inside `execute(plan)` — D2D copy of KV slices
   + atomic `req.kv_indices[slot] = new_page` + free old. Reuses
   `MambaPool.migrate_slot` for spec-decoding case (already exists, T4).
6. **Flip default**: `SGLANG_T8_PLANNER=1` and `SGLANG_T8_EXECUTE=1` become
   default-on; legacy `_do_transfer` body becomes a deprecation stub.
7. **Delete legacy code**: remove `_select_drainable_chunks`,
   `_expand_via_migration`, tail-shrink else branch, fallback chains.
   File shrinks from 795 → ≤250 lines.
8. **Reproduce M2 swarm conc=800** with all flags on. Expect: zero crashes,
   `xpool_fire_total_us` similar to current, `xpool_active_pages_migrated`
   logged per fire (typically 0).

## Reproduce / validate

`reproduce.sh` runs M2 swarm conc=800 with `SGLANG_T8_PLANNER=1
SGLANG_T8_EXECUTE=1`, captures budgeter.jsonl + server.log, asserts:
- no `CUDA error` in server.log
- ≥ 1 fire with `xpool_unmapped_total > 0` AND `xpool_granted_total > 0`
- `xpool_active_pages_migrated` field present in every fire row
- final throughput ≥ T7-baseline-without-fire (i.e. fire didn't hurt)

`test/` contains:
- `test_owner_map_correctness.py` — synthetic running batch + tree, verify
  `OwnerMap.total == allocator.size`.
- `test_plan_no_active_in_capped.py` — random scenarios, planner output
  must satisfy `set(plan.capped_pages) ∩ active_pages == {p for p in
  plan.pages_to_migrate}` (i.e. only migrated pages may be in the capped
  range with active owners).
- `test_execute_atomicity.py` — inject D2D copy failure, verify abort path
  leaves allocator + arena in consistent state (no half-unmapped chunks).

## Out of scope

- Cross-rank / TP coordination (single-rank only; existing T1–T7 are too).
- Mamba ↔ KV symmetry: same protocol both directions (just swap roles);
  no extra design work.
- Eviction policy changes inside the radix tree — drain step calls existing
  `tree_cache.evict` API unchanged.
- Pause/retract decisions — planner does NOT retract reqs; if migrate budget
  is insufficient it shrinks the plan (smaller delta) rather than retracting.
  Retract remains the engine's existing alloc-fail fallback.
