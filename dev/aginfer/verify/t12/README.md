# T12 — h_τ(occ) shape calibration (PLAN §1)

Falsify the spec's linear holding-cost placeholder `h_(τ, sp)(occ) =
α × occ` against two convex alternatives (power-law, hyperbolic).

## STATE OF THE WORLD (2026-06-01)

T12 has two halves; the second is gated on T11.

| half | needs | status |
|---|---|---|
| (1) Fitter + picker + log parser | nothing | **DONE — this verify** |
| (2) Run the fitter on real `(occ, marginal_V_u)` quadruples from a scenario cycle | T11 owns the scenario data path | **DEFERRED** — PLAN §1 T11/T12 + task #173 |

The deferred half is mechanical once T11 lands: enable the daemon's
calibration log line, run a cycle, feed the lines through
`parse_t12_log_lines` → `fit_all` → `best_by_aic`.  All three
shapes are wired and tested against synthetic ground truth.

## SCOPE

### Candidate shapes

| name | functional form | params | rationale |
|---|---|---|---|
| `linear` | `α · occ` | 1 (α) | DESIGN §7 placeholder |
| `power` | `α · occ^γ` | 2 (α, γ ∈ [0.5, 10]) | right-tail-heavy regime — small cost until near full, then steep |
| `hyperbolic` | `α / (1 − occ)` | 1 (α) | diverges as occ → 1, matches §9 admission cap |

### Picker

Akaike Information Criterion (Gaussian residuals):
`AIC = n · log(RSS / n) + 2k`.  Lower is better; the `+2k`
penalty makes the picker non-trivially prefer the 1-param models
over the 2-param `power` unless `power`'s residual is materially
smaller.  Ties (numerical) break in favour of the simpler model.

Edge case: clean ground-truth data → RSS ≈ 0 → `log(0)` would
raise.  Floored at `1e-300` so a perfect fit gets a very-large-
negative AIC (still the best by comparison).

### Log format

The daemon emits one calibration sample per event when the
calibration-log mode is enabled (wire-up lands with T11):

```
aginfer_metric event=t12_calibration tier=HBM subpool=kv \
    occ=0.62 marginal_v_u=-1.234 n_units=42
```

`marginal_V_u` is the V_u of the lowest-scoring resident in the
(tier, subpool) bucket at that snapshot's occupancy — the unit
that's first-to-evict at this load.

`parse_t12_log_lines` extracts the four required fields, groups
by `(tier, subpool)`, silently drops malformed lines.

## STAGES (10)

```
A. Clean-data recovery (picker returns the true shape)
  A0 linear → linear            (α=2.0 recovered within 1%)
  A1 power(α=1.5, γ=2.5) → power (both params recovered within ±5%)
  A2 hyperbolic → hyperbolic     (α=0.1 recovered within 5%)
B. Robustness
  B0 5% Gaussian noise: ≥ 4/5 seeds recover the true shape
  B1 fit_one with n < 2 raises ValueError
  B2 fit_all omits non-converging shapes (NaN-in-y forces fit
     failure for every shape; ≥1 must be dropped)
  B3 best_by_aic ties: 1-param wins over 2-param (Occam); same-k
     ties resolve to first-inserted ("linear"); fit_all iteration
     order is locked at linear→power→hyperbolic so production
     same-k ties match the in-isolation test
  B4 power γ saturation: γ_true ∈ {15, 0.1} → saturated=True (both
     bounds); γ_true ∈ {0.55, 9.95} (near-bound but feasible)
     → saturated=False (false-positive guards on both bounds)
C. Log parser
  C0 groups by (tier, subpool); ignores non-t12_calibration lines
  C1 malformed lines (missing field / unparseable float) → silently dropped
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t12/verify.py
```

Runs in ~1 s (pure numpy + scipy.optimize.curve_fit; no I/O,
no subprocesses).

## RESULTS

**PASSED** — all 10 stages.

* date: 2026-06-01
* raw logs:
  * `results/20260601_t12_initial_pass.log` — initial 9-stage pass
  * `results/20260601_t12_post_175_pass.log` — post-#175 (10 stages,
    saturation diag + tightened B2/B3)
  * `results/20260601_t12_post_175r2_pass.log` — post-#175-round-2
    (saturation false-positive fix; γ=0.55 / γ=9.95 guarded)

| Stage | Result |
|---|---|
| A0 linear recovery | PASS — α = 2.0000 |
| A1 power(γ=2.5) recovery | PASS — α≈1.50, γ≈2.50 |
| A2 hyperbolic recovery | PASS — α = 0.1000 |
| B0 noisy recovery (4/5 seeds) | PASS for linear / power / hyperbolic |
| B1 < 2 samples raises | PASS — ValueError |
| B2 non-converge omitted | PASS — NaN-in-y forces ≥1 shape drop |
| B3 AIC tie → simpler + same-k → first-inserted + fit_all iter order locked | PASS |
| B4 saturation: both bounds caught + both feasible-near false-positive guards | PASS |
| C0 parser groups | PASS — 2 (tier, subpool) buckets; foreign lines ignored |
| C1 parser drops malformed | PASS — only valid line survives |

## WHEN T11 LANDS

1. Add an env-gated emit in `daemon/kv_scheduler.py` (in the
   `state_fetched` instrumentation block — same hot path as the
   `_m("state_fetched", ...)` call), one line per `(tier, subpool)`
   with the current `occ` and the policy's computed `marginal_V_u`
   for that bucket.
2. Run a scenario cycle with `AGINFER_T12_CALIBRATION_LOG=1`.
3. Feed the daemon log through `parse_t12_log_lines`, then call
   `fit_all` per `(tier, subpool)` bucket, then `best_by_aic`.
4. If the picked shape is NOT `linear` AND the residual gap is
   material (say AIC improvement > 10), swap the placeholder in
   `baselines/costs.py:holding_unit_cost` or thread the picked
   shape into the policy.

The fitter is stable across noise levels and degenerate inputs
(per B0–B2), so the calibration step is a single short Python
script — no algorithmic work left in T12 itself, only data
collection.
