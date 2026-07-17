# no_spike — Budgeter ignores single-tick pressure spike (EWMA smooth)

What it tests: the `slow_recovery_len_*` EWMA in
`mem_cache/common.py:record_recovery_len_{kv,rec}` (α=0.05, ~14-event
half-life) absorbs a single 10× spike so the planner's NB gate doesn't
fire on noise. Recovery length L is the dominant driver of `c_KV(L)`
(quadratic) and `c_M(L)` (linear); a 10× spike without smoothing would
scale `c_KV` by up to 100× and trip the threshold immediately.

6 sub-tests:
- test_0 NEGATIVE CONTROL — raw 10×L spike (bypass EWMA) DOES fire
  (NB=90493us > 60000us threshold) — proves threshold reachable
- test_1: settled EWMA absorbs 1×(10×L_base) event → L:1000→1450
  (matches α=0.05·10000+0.95·1000) → no fire
- test_2: sustained 10× load fires at tick 29 (closed-form predicts
  ~30, bound 45) — EWMA not over-suppressing real load
- test_3: half-life ratio 0.488 after 14 events (predicted 0.5),
  150-event drift 0.02%
- test_4: mamba-side rec EWMA smooths independently of kv-side
- test_5: cold-start branch — first event seeds L_EWMA = L_first
  directly (no α dilution against initial=0)

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
.venv/bin/python dev/interlayer/3_budgeter/no_spike/test_budgeter_no_spike.py
```

Pure-Python; takes ~1s; no GPU.

## Result

6/6 PASS. The sustained-load fire at tick 29 matches the closed-form
prediction (~30) — strong evidence the test is exercising the actual
planner code path, not a vacuous check.

Commit: `28d9480ef9`

## Sibling: `test_nb_multisource_unit.py`

Same folder, different planner concern. Locks in the multi-source NB
extension from commit `0e4051b988` (xpool_planner.py
`_pick_direction_by_nb` now sums queue_us / paused_us / retract_us /
persist_us alongside the eviction-cost term — per paper
§appendix-trigger:564-566 + design.md §"Budgeter — steady-state pressure rebalance"). 5 sub-tests verify
the new signal aggregation:

- A: L=0 + mamba-saturated + queue → fire kv_to_mamba (pre-fix
  refuses with "no recovery_len observed")
- B: mirror, kv-saturated → fire mamba_to_kv
- C: L=0 + zero signals → no fire (no over-eager regression)
- D: L observed + no queue → c(L)·P_save still works (back-compat)
- E: L + queue compose ADD (verified by NB delta)

Test-first protocol: pre-fix → 2/5 PASS (only C + D, which test the
non-multi-source paths); post-fix → 5/5 PASS.

Reproduce:
```bash
.venv/bin/python dev/interlayer/3_budgeter/no_spike/test_nb_multisource_unit.py
```

**Sub-test F added 2026-05-26**: regression guard for a Python
falsy-zero bug in the `usage_*_active` fallback path. Pre-fix used
`snap.get(k, fallback) or fallback` which treats the legit value
0.0 (idle workload with all mamba slots cached) as falsy and swaps
in the total usage — defeating the active-fix entirely. Test F sets
`usage_mamba_active=0.0` while `usage_mamba=0.99` and asserts that
20 ticks at high TOTAL don't increment `_mamba_above_high_consec`.
Caught by reading the JSONL of an active-fix-v1 run that still showed
fires despite logging `usage_m_active=0.000`. The bug + fix loop
(falsy-zero → explicit `in` check → re-bench) is documented in
`../../4_e2e/idle_no_regression/README.md` and `../../0_page_state_machine/alloc_lock/TODO.md`.

**Sub-tests G/H/I added 2026-05-30** (task #165, the "direction
half-fix"): the active-vs-cache distinction (test F, task #113) was
plumbed into `_classify`, persist-consec, and the high-water guard
— but the NB calculation itself in `_pick_direction_by_nb` was still
called with *raw* `usage_kv` / `usage_mamba`. Lines 281-284 of
`xpool_planner.py` computed `P_save_m` from total mamba occupancy,
so on CC traces with hot mamba cache (raw 0.95 / active 0.20) the
planner read the cache fill as pressure and chose `kv_to_mamba` —
shrinking the *real* bottleneck (KV) to expand a pool with 75%+
admission slack. cc_traces_headline (2026-05-30) hard-failed: mean_ttft +27%,
p99 +66%, output_tps -62%, cache hits 8.5M vs 10.5M baseline.

- G: hot mamba CACHE (raw 0.95, active 0.20) + queue → must NOT
  trigger k2m (cache is not pressure).
- H: kv_active 0.92 saturating + mamba 95% cache → must fire m2k
  (give mamba's nominal capacity to the real bottleneck).
- I: positive control — mamba_active genuinely 0.92 → still fire
  k2m (guards against over-correction).

Pre-fix: H FAILS with `P_save: kv=0.73 m=0.83 → NB[k2m]=106383 >
NB[m2k]=93617` (debug reason captured 2026-05-30). Fix swaps the
call site at line 670 of `xpool_planner.py` to pass
`usage_*_active` (already classified above) instead of raw. Why old
tests A/B/D/E missed it: they leave `usage_*_active` out of the
snapshot, so the fallback makes raw == active and the bug is
invisible. The CC case is the *only* one where raw and active
genuinely diverge.

**Audit follow-up 2026-05-30 — sub-tests J/K/K2/L added (tasks
\#166-168)**: a subagent audit after \#165 surfaced three more
"fallback-laundering" / "coincidental-equality" gaps with the same
shape as the bug we just fixed. Each gap got a dedicated reproducer.

- **J (gap 1.1 regression guard)** — `xpool_planner.py:274` silently
  fallbacks `L_m = L_kv` when `slow_recovery_len_rec` is missing or
  zero. All prior tests either set L_rec=0 (→ fallback) or set
  L_rec = L_kv lockstep. Test J drives EWMAs to distinct targets
  (regime 1: L_kv=2k, L_rec=16k; regime 2: swap) and asserts c_kv vs
  c_m scale with the correct L. **No current bug**, but locks the
  invariant so any future `c_m_us(L_kv)`-typo class fix is caught.
- **K + K2 (gap 1.2 fix)** — `edge_trigger=True` branch at lines
  704-705 was still calling `_classify(usage_kv, ...)` on raw. Test K
  configures `edge_trigger=True` + `nb_direction_aware=False`
  (critical: nb_direction_aware preempts edge_trigger via return at
  line 684/692) with mamba raw=0.99 / active=0.20 and asserts
  `_mamba_state != ABOVE_HIGH`. Pre-fix FAILS with
  `_mamba_state='above_high'`. K2 mirrors with active=0.99 (genuine
  pressure) to guard against over-correction. Fix: classify on
  `usage_*_active` at lines 704-705 with the same `in`/`is None`
  fallback guard used in the nb_direction_aware branch. Production
  dormant today (nb_direction_aware=True default), but the toggle
  is a deployment trap.
- **L (gap 1.3)** — A/B/D/E/G/H/I all use `_seed_consec` to write
  `_kv_above_high_consec` directly, never exercising the
  `_classify → consec++` natural path with `_active` distinct from
  raw. Test L runs 5 ticks with raw=0.99 / active=0.50 and asserts
  consec stays 0; then 5 ticks with active=0.99 and asserts the
  classifier propagates. Currently passing — confirms \#113 wired
  this correctly; locks in for future half-fixes.

Sibling test 17 in `dev/interlayer/2_admitter/test_scheduler_hook.py`
(gap 1.4): pins per-pool `c_evict_us("kv"/"mamba", ...)` routing in
`admitter.py:529-530` against a label swap.

**Sub-tests M/M2 added 2026-05-30 (task #170, live-cc_traces_headline-surfaced
bug)**: re-running cc_traces_headline after the #165 P_save fix dropped the
regression catastrophe from -62% throughput to ~neutral, but the
budgeter JSONL still showed 3 wrong k2m fires at
kv_active=0.26..0.42, mamba_active=0.09..0.10 — both pools far
below low_water=0.85 with no memory pressure at all. Root cause at
`xpool_planner.py:304-311`: when `total_excess==0`, queue/pause/
retract pressure was split 50/50 → `NB[k2m]==NB[m2k]` exact tie →
`>=` tie-break picks k2m → wastes a ~300ms unmap/map cycle to fix
a non-memory stall.

- **M**: kv_active=0.30, mamba_active=0.10, queue=200 → must return
  `direction=None` (queue stall is compute/batch-bound, not memory).
  Pre-fix FAILS with `NB[k2m]=NB[m2k]=100000us` tie → k2m.
- **M2**: kv_active=0.92 (above low_water), mamba_active=0.10,
  queue=200 → m2k SHOULD still fire (over-fix guard — one-side-
  pressed must still attribute pressure).

Fix: when `total_excess == 0`, set `pressure_to_kv = pressure_to_m =
0.0` (skip the 50/50 split). Tie-break is no longer reached because
both NB terms collapse to zero from queue contribution.

**Sub-tests N/O added 2026-05-30 (task #171, architectural fix)**:
re-running D10 @ C=56 (the C bump intended to engage the mechanism
after C=14 showed no pressure) revealed 2 more wrong k2m fires
within the post-#170 code. Live: `kv_active=0.18..0.42,
mamba_active=0.41`. mamba was just 0.01 above the production
default `mamba_low_water=0.40`, but the post-#170 binary excess-
share split still gave m_share=1.0 → full queue pressure went to
mamba → NB[k2m]=6M → fires k2m even with KV at 82% headroom and
mamba at 59% headroom.

The root cause is more fundamental than #165, #170: the binary
excess-share attribution treats "excess > 0" as full responsibility,
breaking the saturation-weighted semantics the paper's Eq p-loss-save
encodes. Two prior fixes were bandages on top of the same broken
attribution framework. Architectural fix replaces the entire share-
split with `pressure_to_σ = admit_aggregate × P_save_σ` — the same
ramp `c_σ(L) × P_save_σ` uses for the eviction-cost term. Below
low_water (P_save_σ=0) the signal does not attribute as memory-bound
at all; ramps linearly from 0 at low_water to 1 at full saturation.

- **N (architectural fail-fast)**: kv=0.30 (deeply slack),
  mamba=0.71 (0.01 above low_water=0.70). Asserts no-fire. Pre-fix
  FAILS (fires k2m by binary share=1.0); post-fix PASSES (P_save_m
  ≈ 0.033 → marginal NB).
- **O (architectural invariant)**: NB[m2k] at kv=0.85 vs kv=0.949
  must scale with P_save_kv (0.50 → 0.83, ratio 1.66). Pre-fix
  ratio = 1.0 (binary share gives equal pressure regardless);
  post-fix ratio = 1.66 (linear in P_save_kv as expected).

Both sub-tests doubled as the TDD red for the architectural refactor.

design.md attribution section (§"Per-pool attribution") updated
to describe the saturation-weighted attribution explicitly.

## R — reuse-aware drain cost (task #275, cross-fire regression)

The `cc_traces_headline` mamba-starve run exposed a NET-NEGATIVE
cross-fire on a high-reuse (89% cache hit) workload: TTFT +23%,
cache_hit −5.8pp. Root cause — the `nb_m2k` drain penalty
`c_m × p_loss_m` prices draining mamba off its **active**
utilization, which is blind to cache reuse. Mamba sat at 95%
occupancy / 40% active (`P_save_m ≈ 0.01`, read as slack) with
`c_m = 0` at `L=0`, so the drain penalty collapsed to ~0 and the
Budgeter repeatedly fired m2k — draining a HOT cache to grow an
only-mildly-pressured KV pool.

The asymmetry the fix encodes: the **grow** benefit stays
active-based (don't grow a pool whose active load is slack even
when its cache is full — `test_G`), but the **drain** cost must be
the reuse-aware (hit-weighted) eviction cost of the snapshots the
drain forces out — NOT the active `P_loss`. The agent supplies it
via `snapshot["mamba_drain_cost_us"]`; `nb_m2k` subtracts it once
per fire (not scaled by `P_loss_m`). Absent the field, NB falls
back to the legacy active estimate byte-for-byte.

- **R (reproducing test, TDD red→green)**: same setup as test_H
  (KV saturating, mamba active-slack, cache full) but with
  `mamba_drain_cost_us = 10M` (HOT cache). Asserts
  `direction != "mamba_to_kv"`. Pre-fix FAILS (m2k fires, blind to
  the cost); post-fix PASSES (reuse-aware cost drives `nb_m2k`
  negative). test_H (cold cache → m2k fires) stays green —
  cold-cache drains are still correct.

Scope split: the NB-consumption half (planner reads the field) +
the design.md §"The grow benefit is active-based; the drain cost
is reuse-aware" landed here. The cache-side half — populating
`mamba_drain_cost_us` from `MambaRadixCache.predict_evict_cost_us`
over the hit-weighted LRU/LPB eviction victims — is tracked in
**#270** (the symmetric KV-source drain cost rides #271).

## Recovery-length plumbing + the grow-side eviction gap (#277, 2026-06-03)

The cc_traces_headline mamba-starve re-run exposed why cross-fire never
GROWS the starved mamba pool. Two gaps:

**Gap 1 — eviction-cost term was dead in production (FIXED).** The
slow-recovery-length EWMAs (`_slow_recovery_len_kv_ewma` / `_rec_ewma`,
written by `record_recovery_len_kv` / `_rec` on every eviction, read by
`_pick_direction_by_nb` as `c_σ(L)`) were NEVER plumbed into
`BudgetAgent._snapshot`. So `L=0` every tick → `c_kv=c_m=0` → the entire
eviction-cost half of the NB never contributed (confirmed: `L=0` across
all 297 starve ticks despite heavy thrashing). The planner had only ever
fired on queue/persist. Fix: `agent.py` pre-inits the EWMAs in
`_do_health_check` and reads them into the snapshot.
`test_snapshot_recovery_len.py` pins it.

**Gap 2 — grow benefit is active-gated (the mirror of #275, REMAINING).**
Even with `L` plumbed, the grow term `c_σ(L) × P_save_σ` is gated by
ACTIVE utilization. Probe (prod defaults, starve snapshot: mamba occ
0.969, active 0.406 = low_water, kv active 0.42, `L_rec=2000`):

    c_m = 11330us @ L=2000   (alive now)
    P_save_m = 0.01          (active-based → ~0)
    NB[k2m] = 16 × (11330 × 0.01) = 1813us  <  threshold 7500us  → NO fire

So a pool that is occ-full and actively evicting hot snapshots
(cache_hit bleeding) still reads as "not pressured" because its
active-slot utilization is moderate. This is the exact symmetric image of
#275: #275 found the DRAIN cost blind to hot-cache REUSE; this is the
GROW benefit blind to hot-cache EVICTION. The fix (a reuse-aware eviction
RATE term, not `c × p_save_active`) is tracked in #277 and needs the same
design alignment #275 had.
