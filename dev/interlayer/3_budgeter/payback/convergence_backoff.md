# Convergence backoff — the PaybackPlanner must verify its own self-convergence

## Problem (observed)

35B swarm (t12 @ conc64, current build) regresses under full HiMA:
`full=766.4 < base=777.9` (−1.5%, both n=3, n_err=0, identical out-tokens). The
Admitter no-backlog gate is already in; this is a **Budgeter** (tick-path
`PaybackPlanner`) overhead, not an Admitter cost.

Budgeter log: **3549 fires (2618 k2m + 931 m2k)** over the run, ~2.3/s. Every
fire is `r_evict`-driven (`admit=0` on all), i.e. mamba/KV radix-cache eviction
loss, with the queue ≈ empty (mean 0.13) and running (max 66) never at the mamba
cap (min 67). So the workload is not admission-bound; the fires only shuffle
memory and pay cuMem cost.

## Root cause

`PaybackPlanner`'s docstring asserts self-convergence: *"pool grows → harm rate
drops → fires stop."* The code **never verified it** — it fired whenever the
instantaneous harm difference cleared the payback threshold.

Time-series (`convergence.py`) shows the assumption is false on 35B swarm: the
mamba eviction loss stays pinned at ~1,000,000 µs/s across the whole run and
**does not drop when mamba_cap grows** (cap oscillates 78↔104). The working set
exceeds total memory, so the harm is **not capacity-elastic** — growing the pool
cannot reduce it. The planner chases an unachievable target forever.

Contrast 9B swarm (helps +8.7%): mamba eviction loss is small (~10–45k) and
**drops to 0 when mamba grows** (cap → 229–246) → self-convergence holds → fires
stop → win. So the firing mechanism is sound; only the unverified-assumption
failure mode is the bug.

## Fix

Close the loop the design already assumed. Per direction, remember the target
pool's harm `r_dst` at the last committed fire. If a subsequent fire finds
`r_dst` has **not dropped by ≥ `converge_eps` (5%)**, that fire was ineffective
(harm inelastic) → widen this direction's cooldown **2× per consecutive miss**
(`converge_backoff_cap=4` → ≤16×). A single effective fire (harm dropped) resets
the streak. AIMD control — standard for an unresponsive actuator.

- 9B: fires stay effective → no backoff → converge → keep the win.
- 35B: fires ineffective → cooldown widens to 16 s → fire rate drops ~16× →
  overhead → ~0 → recover parity.
- Genuine shift (each direction effective in its phase) → no backoff → responsive.

The widened cooldown still lets a periodic probe fire re-measure elasticity, so
the planner resumes immediately if the workload becomes elastic again — nothing
is permanently disabled. No fallback branch, no dead code; the existing
fire-commit block is modified in place. `xpool_planner.py:PaybackConfig` +
`decide()`.

## Test

`test_payback_planner.py::TestConvergenceBackoff` (4 new; 14/14 total pass):
- `test_unresponsive_harm_backs_off` — constant harm → <40 fires / 200 ticks,
  gaps widen.
- `test_responsive_harm_keeps_firing` — harm drops per fire → keeps firing at
  base cooldown until converged.
- `test_backoff_beats_no_backoff_on_inelastic` — cap=4 fires strictly fewer than
  cap=0 on the same inelastic feed.
- `test_effective_fire_resets_backoff` — streak resets when harm becomes elastic.
Existing idle/steady/shift/cooldown/convergence/margin invariants unchanged.

## A/B (perf gate) — RESULT

`backoff_validate.sh` (swarm t12 @ conc64, current build, n≥2):
- **9B full (backoff): 799.4 (+11.6% vs base 717)** — the win is kept and in fact
  grew (was +8.7%); fires 1155 → ~190. Clean net-positive.
- **35B full (backoff): 764.1 (−1.8% vs base 778)** — fires 3549 → 779, but
  throughput did NOT recover.

The backoff throttled the thrash but 35B stayed regressed, so the fire *overhead*
was not the whole story: the residual fires still drifted the split mamba-ward.
The backoff is necessary (it removes the inelastic-harm oscillation and grows the
9B win) but not sufficient. The completing fix is the **active-usage guard** — the
planner was growing the pool with LESS live demand. See
[active_usage_guard.md](active_usage_guard.md). With both, the budgeter is
cost-neutral on 35B (`wo_admt` 779.8 ≈ base 778) and the 9B win holds (+13.3%).
