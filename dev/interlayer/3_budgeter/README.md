# 3_budgeter — Budgeter validation tests

Verification tests for the Budgeter (steady-state pressure rebalance)
described in [`../design.md`](../design.md) §"Budgeter — steady-state
pressure rebalance".

The Budgeter rides on top of the [Admitter](../2_admitter/) (per-arrival
cost decisions) and drives asynchronous cross-pool transfers via the
[xpool_actuator](../0_page_state_machine/) when EWMA-smoothed pressure
crosses configured thresholds. Its correctness has two dimensions: it
must NOT over-react to single-tick noise (this folder), AND it must
converge to the right direction under steady-state drift (covered by
the e2e workload runs under `../4_e2e/`).

## Sub-folders

| folder | what it pins | design.md ref |
|---|---|---|
| [`no_spike/`](no_spike/) | (a) EWMA on `_mamba_above_high_consec` smooths a single-tick pressure spike (`test_budgeter_no_spike.py`, 6 sub-tests); (b) the multi-source `nb_direction_aware` counter genuinely distinguishes radix-cached "phantom" saturation from live admission ceiling (`test_nb_multisource_unit.py`, 25 sub-tests, including the active-vs-total falsy-zero bug and the both-full guard toggle `test_V/V2/W`); (c) the steady-state fire drains cold cache, not just free pages, and now the m2k call site applies the #312 working-set floor (`test_budgeter_drain_fire.py`, 13 sub-tests A–M, #270) | §no_spike + §"Budgeter — steady-state pressure rebalance" |
| [`tau_invariance/`](tau_invariance/) | the control loop is τ-invariant (#302): FLOW signals priced as per-second rates (÷dt), cooldown/amortize horizons in WALL SECONDS not tick counts, and the reuse-aware grow signal's fractional cap (`test_tau_invariance_302.py`, 3 sub-tests) | §"Empirical pressure signal" + §"Three separated concerns" |
| [`mamba_drain_floor/`](mamba_drain_floor/) | the m2k cross-fire's working-set floor (#312/#297): cap the drain so `live_size ≥ max_running + mamba_protected_size() + fork_headroom`, computed in slots and converted to pages via the arena `tokens_per_chunk` (the F2 overshoot), else a later `cache_unfinished_req` fork crashes "Can not alloc mamba cache" (`test_mamba_drain_floor_312.py`, 7 sub-tests) | §"Allocator floor: working set only" |
| [`mamba_pool_perf/`](mamba_pool_perf/) | the arena per-layer-list `MambaPool` adds no per-step overhead over the stacked baseline where no cross-fire is active: `free` skips the isin + `>size` mask + device sync (the `_no_cross_fire` fast path), `alloc` hoists the scalar zero out of the per-layer loop, and the whole cap machinery conserves slots. Microbench + N=3 perf-target asserts (`bench_mamba_pool_ops.py`, `test_mamba_pool_perf_targets.py`) + correctness/conservation (`test_mamba_free_fastpath.py` 4, `test_mamba_pool_invariants.py` 8) | §"Cap-barrier internals" + §"FREE/CAPPED invariants (production)" (mamba analog of #322) |

## Notable fixes

### #312 — m2k drain needs a working-set floor (else `cache_unfinished_req` crashes)

An m2k fire on a near-full mamba pool drained it below the burst working
set; a later request burst re-prefilled and could not fork a fresh active
SSM slot → `AssertionError: "Can not alloc mamba cache"`. Root cause (not a
workaround): no floor on `mamba_pool.live_size`. `_maybe_fire` now caps the
drain so `live_size ≥ max_running + mamba_protected_size() + fork_headroom`
(all read LIVE each fire), computed in slots and converted to pages via the
arena `tokens_per_chunk`. Full writeup, dead ends, and the F2 unit subtlety:
[`mamba_drain_floor/`](mamba_drain_floor/). This realizes the TARGET
allocator floor (design.md §"Allocator floor: working set only") for the m2k
donor direction.

### #270 — steady-state fire must DRAIN cold cache (`allow_drain=True`)

`BudgetAgent._maybe_fire` built the FirePlan with `allow_drain=False`
(the default), so the steady-state m2k/k2m fire harvested only
genuinely-FREE source pages and skipped Stage-2 (Drain-expansion). At
steady saturation the source pool's reclaimable slack is its cold
cached snapshots (full-but-quiescent), not free slots, so the fire was
inert: `cc_traces_headline` 3b measured 12 m2k fires at `occ_m=0.945` /
`usage_mamba_active=0.175` (~77% cold cache) all planning
`free=4 drain=0` — KV grew ~+9K tokens and cache_hit didn't move.

This contradicted design.md on two counts: §"Budgeter — steady-state
pressure rebalance" defines this exact scenario ("mamba pool sits
half-empty holding cold cache ... cache hit rate slowly bleeds") as the
Budgeter's job; and §"Grow benefit and drain cost are both reuse-aware"
has `nb_m2k` SUBTRACT the reuse-aware `mamba_drain_cost_us` once per
fire — the decision priced a drain the execution then refused.

Fix: pass `allow_drain=True` (and `allow_migrate=False` — the NB prices
no migration term; KV slot migration is #271). The reuse-aware drain
cost is the gate (a hot cache resists the drain; only cold pages are
harvested), so `allow_drain` is the execution side of a cost the
decision already paid for. Pinned by
`no_spike/test_budgeter_drain_fire.py`. GPU end-to-end confirmation
(`cc_traces_headline` 3c) is pending a free GPU.

**Audit (2026-06-07) follow-up — drain made fail-closed.** A re-audit of
the unconditional `allow_drain=True` found it could re-open the #275
mamba-starve regression, so the drain is now gated by
`BudgetAgent._cross_drain_allowed`: it returns True only for **m2k** AND
under **LPB** AND with a **non-degenerate** mamba cost curve. Each failure
mode falls back to free-only (with a one-time operator warning on a
degenerate curve):

- **k2m** has no reuse-aware KV drain cost yet (design defers it to #271),
  so draining KV would be priced only by the legacy active estimate —
  gated off until #271. (`test_H`)
- **LRU** gives `n_b ≡ 1`, so the drain cost can't tell hot from cold —
  same gate the grow benefit uses (#280). (`test_G`)
- **Degenerate κ_M** (`m_alpha = m_beta = 0`, #276) collapses the drain
  cost to ~0 regardless of policy, defeating the gate. (`test_F`)

Also fixed the drain **volume** mismatch: the reuse-aware
`mamba_drain_cost_us` was priced for `dst_chunks_per_action` (the planner
threshold unit, default 1) but the fire drains up to `_n_pages_per_fire`
(default 4) — a ~4× under-count. Now priced for the fire magnitude.
(`test_E`). Suite: `test_budgeter_drain_fire.py` 8/8.

### MambaPool hot-path — no per-step overhead where no cross-fire is active

The arena's per-layer-list temporal layout (needed so cross-pool transfer can
map physical bytes per sub-pool) made every per-request `MambaPool` op loop
over `num_layers` tensors instead of one stacked tensor, and `free` paid a
`torch.isin` + `>size` mask `.any().item()` device sync on EVERY call even with
no cross-fire ever active. On a high-concurrency swarm those launches dominated
and showed up as the ~2-3.7% default-split decode regression in the agentreplay
A/B. Fix (the mamba analog of #322's KV-allocator isin tax): `free` takes a
`_no_cross_fire` fast path (capped empty AND `size == max_size`) that returns
ids straight to `free_slots`, no isin / mask / sync; `alloc` hoists the scalar
zero out of the per-layer loop. Strictly free (no time/space cost), and the
fast path flips off the instant any path caps a slot, so the #312/#329
unmapped-VA guard is preserved (modulo the pre-existing `clear()`/#327
flush-boundary case, which drops below-cap actuator marks and crashes
identically on the slow path, orthogonal to this fast path). The residual `copy_from` / `alloc` per-layer
launches are inherent to the separate-VA-per-layer arena and flagged for an
arena-layout decision, not fixed. Full writeup, microbench, and the
correctness/conservation + N=3 perf-target suites:
[`mamba_pool_perf/`](mamba_pool_perf/).

## Cross-references

- [`../2_admitter/cxfer_ewma_self_suppress/`](../2_admitter/cxfer_ewma_self_suppress/) —
  the Admitter-side EWMA self-suppression test (§cxfer_ewma_self_suppress). The Budgeter's
  EWMA shares the same shape but operates on a different signal (pool
  pressure, not c^xfer).
- [`../4_e2e/idle_no_regression/`](../4_e2e/idle_no_regression/) — the
  e2e companion: under idle traffic the Budgeter must not fire spuriously.
- Production code: `python/sglang/srt/budgeter/xpool_planner.py`
  (`_decide_inner`, `_pick_direction_by_nb`); `MambaPool` in
  `python/sglang/srt/mem_cache/memory_pool.py` (`free`, `_no_cross_fire`,
  `alloc`) for the hot-path fix.
