# PLAN — Budgeter τ-invariance (#302, merges #304)

## Problem (proven, RED)

`test_tau_invariance_302.py` fails on current code:

- `NB[k2m]` is **15× different** at τ=30s vs τ=2s for the *same* real
  eviction rate (1000 tok/s) — the flow signal is a raw per-tick count
  with no `/dt`, and `lifetime = cooldown_ticks` is a τ-blind tick count.
- The inter-fire lockout is **32s vs 480s** at τ=2 vs τ=30 — the cooldown
  decrements once per `decide()` call, so its wall-clock length is
  `cooldown_ticks × τ`.

Root cause: the control loop is parameterized in **tick-counts + per-tick
constants**, so its real-time behaviour silently scales with
`SGLANG_HIMA_TICK_S` (τ). All harnesses run τ∈{1,2}s; the 30s default
is exercised by no harness, so this is a real but untested latent defect.

## Target — already the design's shipped-variant SPEC

`design.md` §"Empirical pressure signal" + §"Trigger rule" already specify
the τ-invariant form: signals as **per-second rates** (`raw / dt`),
**EWMA-smoothed** (η, default 0.1), `cooldown_min` / `amortize_horizon` in
**seconds** with `cooldown_min ≥ amortize_horizon`, and τ as
"per scheduler iteration (no knob)". The code never implemented the `/dt` +
EWMA and fused both horizons into one `cooldown_ticks`. So this is **make
code match design**, not a design change.

`τ` becomes a pure **sampling rate** (default 1s, behaviour-invariant). The
"30s problem" is a *cadence* problem — fixed by running frequently + `/dt` +
seconds-horizons — and is **estimator-agnostic**: it is NOT a reason to
adopt BOCPD. BOCPD stays the documented north-star for *phase-transition
dynamics* (measurement-gated, separate; #201/#202).

## Why `cooldown_min` and `amortize_horizon` stay SEPARATE (not fused)

Without change-point detection (the shipped EWMA variant), `design.md`
requires `cooldown_min > amortize_horizon`: the gap past payback is an
**oscillation buffer** — re-firing the instant payback completes risks
reversing on noise. Equality is safe only WITH change-point detection
(BOCPD regime). The code's fusion forces `cooldown_min == amortize_horizon`
(the unsafe equality) AND ties both to τ. The fix restores the two seconds
knobs.

## Unit model (the core of the fix)

| signal | kind | conversion |
|---|---|---|
| `num_evicted_tokens_recent`, `{kv,mamba}_evicted_*_recent`, retracted delta | FLOW | rate = count / dt |
| `num_queue_reqs`, `num_paused_reqs` | STOCK (instantaneous depth) | standing rate = depth × penalty_per_item_per_s (NO /dt) |
| persist (`{kv,mamba}_above_high_consec`) | DWELL | seconds = consec × dt (track dwell-seconds) |
| `slow_recovery_len_{kv,rec}` | LEVEL (tokens) | unchanged |

Rates are EWMA-smoothed with a **time-constant in seconds** (η adapted to
dt) so smoothing is τ-invariant. NB benefit = `rate × amortize_horizon_s`;
one-time costs (reuse-aware drain, `c^xfer`) are unchanged.

## Anchoring (protect measured conclusions)

Pick seconds defaults + per-signal coefficients so the **real-time
behaviour at the cc/headline operating point** (τ=2, cooldown=6 → 12s
lockout, 12s amortize) is preserved. The τ=2 cell (headline cc win + the
#285 A/B) stays ~equivalent; the τ=1 cells shift to the unified semantics
and must be re-benched to confirm their *conclusions* survive (they assert
direction / no-regression / soft bounds, not exact magnitudes).

## Steps (ordered, gated)

1. **[DONE]** Reproducing test `test_tau_invariance_302.py` (RED).
2. **Core code** (dt REQUIRED, no fallback — per no-fallbacks principle):
   - `pressure_adapter.py`: emit per-second rates (flows `/dt`, stocks as
     standing rates, persist as dwell-seconds); EWMA the rate stream with a
     seconds time-constant. Retire the dead `_ema_*` fields or move EWMA
     here.
   - `xpool_planner.py`: `cooldown_ticks` → `cooldown_min_s` +
     `amortize_horizon_s` (seconds); `lifetime` → `amortize_horizon_s`;
     cooldown gate → wall-clock (`now − last_fire_time ≥ cooldown_min_s`);
     fail-loud config assert `cooldown_min_s ≥ amortize_horizon_s` (raise on
     `<`). Consume `snapshot["dt"]`.
   - `agent.py`: compute `dt = now − prev_last_tick`, put on snapshot;
     default `SGLANG_HIMA_TICK_S` 30 → 1s; remove dead `_ema_kv/_ema_mamba`.
   - Gate: reproducing test GREEN; τ-invariance asserted.
3. **Migrate `no_spike/` suite** (30+ tests) to seconds-config + supply
   `dt`; keep GREEN (assert direction-preserved). Same for any other test
   constructing `XPoolPlanner` / setting `cooldown_ticks`.
4. **Translate harness envs**: `SGLANG_XPOOL_COOLDOWN` (ticks) →
   `SGLANG_XPOOL_COOLDOWN_S` / `SGLANG_XPOOL_AMORTIZE_S`; default τ→1s,
   across the ~10 `dev/interlayer/**` + `dev/eval/**` scripts (translate
   each script's `ticks×τ` into the equivalent seconds).
5. **design.md sync** (small): shipped-variant DESCRIPTION
   (`cooldown_ticks` → seconds knobs; "30s tick" → τ-invariant sampling) +
   2 clarifying notes (BOCPD assumes *signal shape*, not arrival
   distribution; 30s is a *cadence* issue, estimator-agnostic). SPEC parts
   (rate / EWMA / seconds) already correct — leave them.
6. **Re-bench** (measurement-gated): τ=1 harness suite + cc headline +
   #285 under new code → confirm conclusions survive. Check GPU free first.
7. **Subagent audit** of the whole change (impl ↔ design ↔ tests).

## Out of scope (separate, documented)

- **BOCPD adoption** (#201/#202) — measurement-gated upgrade, evaluated
  against THIS corrected baseline; its edge is at transitions, design says
  stable workloads are equivalent.
- **Learned horizon** (EWMA-of-dwell / inter-crossing interval feeding
  `amortize_horizon`, or BOCPD `E[run-length]`) — lightweight bridge; only
  if a hand-set `amortize_horizon_s` proves insufficient.
- **#285 both-full guard** — orthogonal to τ; let the running A/B finish
  (its answer holds under current semantics), re-confirm post-fix.
