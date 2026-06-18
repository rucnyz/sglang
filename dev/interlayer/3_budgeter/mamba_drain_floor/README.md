# mamba_drain_floor — working-set floor for the m2k cross-fire (#312 / #297)

## Symptom

e2e crash on `cc_zero_downside` with the `inter_admitter` arm, surfaced by
the τ-invariance refactor (#302):

```
AssertionError: "Can not alloc mamba cache"
  at mamba_radix_cache.cache_unfinished_req  (assert mamba_value_forked is not None)
```

The Budgeter fired `mamba_to_kv` on a near-full mamba pool; a later request
burst spiked active demand on the SHRUNKEN pool and a `cache_unfinished_req`
fork could find no slot → SIGQUIT.

## Mechanism (derived from sglang code, not e2e tuning)

A mamba fork (`MambaRadixCache.cache_unfinished_req` → `fork_from`) fails iff
`mamba free == 0` **and** there is no UNLOCKED evictable cache (`evict_mamba`
skips any node with `mamba_lock_ref > 0`). The m2k cross-fire reduces
`mamba_pool.live_size` (the allocatable capacity; `size` stays constant —
the cap-barrier marks slots off via `_capped_slots`). If a fire drains below
what running requests hold, a later active-slot alloc / `cache_unfinished_req`
fork finds no slot → assert. The floor must keep the **live working set**:

```
floor = (m_used − mamba_evictable_size()) + safety_margin
      = active_SSM_states + protected_locked_cache + safety_margin
```

where `m_used = live_size − available_size`, so `m_used − evictable` is the
active+locked quantity `_get_mamba_token_info` reports as `mamba_num_used`.
Donatable = `available + evictable − safety_margin`: free slack PLUS unlocked
evictable cached snapshots (the plan's Drain stage frees them CACHED→FREE
before unmap). This is the TARGET working-set floor (design.md §"Allocator
floor: working set only") realized for the m2k donor.

### #297: the floor must NOT be the nominal `max_running` cap

The original #312 floor used `max_running_requests + protected` — the nominal
concurrency cap, not the live active set. That re-prefill reasoning ("a burst
of up to `max_running` reqs each needs a fresh active slot") over-reserves:
`max_running ≈ pool/3` slots (sglang caps it at `max_mamba_cache_size / 3`),
so the floor withholds ≈ two-thirds of the pool **regardless of live load**.
In a KV-bound long-context regime KV binds concurrency at a few requests, so
the active set is tiny and the cap-sized floor refuses the m2k donate that is
the whole inter-layer win. Observed: **59/72 m2k fires aborted** on the
nominal-cap floor in the 262k agentreplay run (`live_size=194 < max_running=147
+ protected=53 = 200`), even though most of the pool was idle evictable cache.
The fix reserves the live working set and delegates bursts to the active-slot
grow hook (`_mamba_active_grow_hook`, synchronous k2m from idle KV), so the
static cap reserve is unnecessary.

## Fix

A single `BudgetAgent._mamba_working_set_floor_slots(m_used, evictable)` returns
`max(0, m_used − evictable) + safety_margin`, wired into BOTH m2k floor sites:
the periodic tick (`_maybe_fire`) and the on-demand KV-grow hook
(`_grow_kv_from_mamba`, previously floored at `m_used` alone → donated only
free slack, never evictable cache). It is applied through
`_mamba_drain_floor(live_size, floor_slots, slots_per_page, requested_pages)`:

- The cap is in SLOTS, converted to PAGES via `slots_per_page` (the arena
  `tokens_per_chunk`). **F2 (load-bearing unit fix):** a page-unit cap drains
  `pages × tokens_per_chunk` slots, overshooting the slot floor by
  `tokens_per_chunk×` when it is >1. The cc config has `tokens_per_chunk == 1`,
  so this bug is invisible there; the unit test pins it at `tps ∈ {4, 12}`.
- Fail-closed: `slots_per_page <= 0` (non-arena pool) → drain 0 → refused.
  `live_size <= floor` → drain 0 → refused.

The scheduler / tree-cache / arena state is read DIRECTLY (no
`getattr(..., default)`): a missing `mamba_evictable_size`, `available_size`,
or arena attribute is a real integration bug and should fail loud, not silently
fail-open to an under-floor (the health check `_do_health_check` requires this
API up front when a mamba pool is present).

`safety_margin` is `SGLANG_XPOOL_MAMBA_FLOOR_SLOTS` (default 32) — the buffer
for single-tick allocation noise, NOT a burst reserve (the grow hooks self-heal
bursts).

## Tests

Test-first (bug-workflow): each test bakes the RED condition next to the GREEN
assertion, so it demonstrably catches the bug it guards.

`test_mamba_drain_floor_312.py` — the pure `_mamba_drain_floor` page-cap (floor
passed in, floor-agnostic):

| test | guards |
|---|---|
| `test_floor_caps_drain_to_keep_live_above_floor` | a no-floor drain leaves `live < floor` (RED); the cap holds `live >= floor` (GREEN) |
| `test_tps_gt1_no_overshoot` | **F2**: page-unit cap overshoots the slot floor at `tps=12` (RED); tps-aware cap does not (GREEN) |
| `test_unknown_granularity_fails_closed` | `slots_per_page <= 0` → refuse (return 0) |
| `test_floor_reached_returns_zero` | `live <= floor` → refuse |
| `test_invariant_over_grid` | post-drain `live >= floor` over a `(live, floor, tps, req)` grid |

`test_mamba_working_set_floor_297.py` — the floor *computation* (drives the real
`_maybe_fire`, reproducing the 262k regime):

| test | guards |
|---|---|
| `test_tick_m2k_fires_when_pool_holds_idle_evictable_cache` | the headline #297 case: `live < max_running + protected` but the active set is tiny + the pool is idle evictable cache → the working-set floor fires where the nominal-cap floor refused (RED→GREEN) |
| `test_drain_never_breaches_active_plus_protected` | #312 safety: the fired drain never shrinks `live` below `active + protected`, over a realistic grid |
| `test_genuinely_full_pool_refuses` | no free + no evictable slack → refuse |
| `test_working_set_floor_independent_of_nominal_cap` | donatable volume is invariant to `max_running` (design §501) |

`test_mamba_drain_floor_integration_312.py` — the `_maybe_fire` wiring: F1 (the
health check requires `mamba_evictable_size` + `available_size`) and F2 (the
page→slot conversion at `tps>1`, end-to-end).

Run:

```bash
.venv/bin/python dev/interlayer/3_budgeter/mamba_drain_floor/test_mamba_working_set_floor_297.py
```

Integration coverage of the steady-state `_maybe_fire` flags also lives in
`../no_spike/test_budgeter_drain_fire.py`.

## Dead ends (so the lineage doesn't repeat)

- **fix-1 (reclaimable):** capped against free + evictable cache. Insufficient
  — the burst re-prefills, so cold cache doesn't count.
- **fix-2 (`.size`):** `.size` is constant under the cap-barrier; capping
  against it never refused → still crashed.
- **fix-3 (live_size + fixed 128 margin):** no longer crashed on the cc config
  but GUESSES `protected`; under-reserves whenever `protected > margin`.
- **fix-4 (live_size + actual protected + headroom):** the exact form — but
  initially capped in PAGES against a SLOT floor (the F2 overshoot), corrected
  to the slot-converted form here.
- **fix-5 (`max_running + protected`, the #312 floor):** crash-safe but
  reserved the nominal concurrency cap, which is ≈ `pool/3` slots regardless of
  live load. In a KV-bound regime that withheld ≈ two-thirds of the pool and
  refused 59/72 m2k fires (#297) — safe but it neutered the win. Superseded by
  the live working set `(m_used − evictable) + margin`, with the active-slot
  grow hook (not the static cap) providing burst concurrency safety.

## #329: the floor alone cannot prevent the crash in the m2k regime

The #297 working-set floor reserves the LIVE active set, not the nominal cap.
That is the correct floor, but it is a genuine LOWER bound only when capacity
can be recovered on demand. In the m2k regime (mamba borrowed FOR KV, so KV is
FULL by design) the active-slot / fork grow hook fires a k2m grow, but there is
no idle KV to lend back, so the grow returns 0. The floor can then sit exactly
at the working set with no slack, and a COW copy / caching fork / active-slot
alloc that needs ONE more slot finds nothing evictable (the rest is locked /
active). On the 262k agentreplay run this reintroduced the #312 crash via the
COW path (`_match_post_processor`, `assert dst_index is not None`, n_err 3848,
cache_hit collapsed to 0.31, SIGQUIT).

The floor stays the PRIMARY mechanism (keep mamba above its working set so the
fast path never needs to evict-or-grow). But when the m2k regime genuinely
removes all slack, the only crash-proof answer is **graceful degrade**, not
assert. This is the defensive backstop the rejected-alternative section below
foreshadowed: it does not replace the floor, it catches the residual case the
floor cannot (no idle KV ⇒ no recovery).

### Three sibling alloc sites, one disease

An over-drained mamba pool must back-pressure, not crash. Each mamba-alloc site
degrades on final failure (all in `mamba_radix_cache.py` / `memory_pool.py`):

| site | symbol | degrade |
|---|---|---|
| COW copy | `MambaRadixCache._match_post_processor` via `_cow_mamba_slot_or_none` | alloc → evict+alloc → grow-hook+alloc → None ⇒ `_no_mamba_match_result()`: a mamba cache MISS (empty `device_indices` + root `last_node`). The request re-prefills from scratch. The matched KV prefix is coupled to the cached mamba state (both end at `best_value_len`) so it is dropped with it; `req.mamba_pool_idx` stays None and a fresh active slot is taken at alloc time. |
| caching fork | `MambaRadixCache._fork_mamba_with_recovery` → `cache_unfinished_req` | returns None ⇒ caller falls back to `_skip_cache_unfinished_req` (don't deposit the snapshot; the request keeps its live state and retries next round). |
| active slot | `HybridReqToTokenPool.alloc` via `_alloc_active_mamba_slot` + `_rollback_active_alloc` | rolls back this batch's fresh req_pool slots + fresh mamba slots/buffers and returns None, the same None the req-slot-exhausted branch already returns, so the scheduler back-pressures. |

`test_mamba_alloc_degrade_329.py` pins all three (RED: the pre-fix asserts
"Can not alloc mamba cache" / "Not enough space for mamba cache"; GREEN: each
degrades). It also pins the success path is unchanged (COW copies, fork
succeeds, active alloc returns slots) so the degrade is reachable only on the
genuine-scarcity branch.

**Baseline safety:** with the Budgeter OFF (no grow hooks wired, default
split), mamba is never drained below boot, so the first `mamba_pool.alloc`
always succeeds and no degrade branch is entered — byte-identical to stock
sglang. Only the terminal failure changed (assert → degrade).

## Alternative considered and partially superseded: best-effort fork

The crash sites are `assert ... is not None` ("Can not alloc mamba cache"). A
tempting fix is **best-effort** — skip / degrade instead of asserting. We
originally rejected this as the PRIMARY fix:

- Caching forks never fail in **stock** sglang, because sglang sizes the mamba
  pool to `ratio·max_running` (`ratio = MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
  = 3`, up to 5 with the extra buffer) and caps `max_running = pool // ratio`
  (`model_runner_kv_cache_mixin`). So the assert is a legitimate "can't happen"
  invariant for them. **Our** cross-pool draining is what removes that slack and
  makes the fork fail — the invariant violation is ours.
- Patching sglang's assert to tolerate a state **we** create treats the symptom
  while the root cause (the Budgeter over-drained) stays.

That reasoning still holds for why the FLOOR is the primary fix. What #329
established is that the floor cannot be a HARD guarantee in the m2k regime (no
idle KV ⇒ no recovery), so the degrade is the necessary backstop for the
residual case, not a substitute for the floor. The degrade is designed to be
correct (a real cache miss / clean back-pressure), not a silent corruption: the
request always proceeds from a consistent state.

## Known latent issue

`protected` growth BETWEEN fires (more cache gets locked after the fire is
sized) is not re-checked mid-fire (MED, latent). The fire is a single
quantum and the cooldown bounds how often it repeats, so a single fire can't
chase a moving `protected`; worth a guard if a future workload locks cache at
fire cadence.
