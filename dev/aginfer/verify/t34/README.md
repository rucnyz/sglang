# T34 — multi-axis sparse 0/1 knapsack DP (#156, DESIGN §9)

The two `joint_decide` primitives — the exact-DP core of the policy:

* **`knapsack_min_cost_multi`** (pressure phase): min-total-V_u-cost
  subset of {Migrate-HBM-out, Pause} that frees ≥ `bytes_needed` from
  every pressured HBM subpool (≥ axes) without overflowing any
  destination `cap_left` (≤ axes).
* **`knapsack_max_value_multi`** (headroom phase): max-total-V_u-gain
  subset of {Resume} within each HBM subpool's `budget` (≤ axes).

Multi-axis because items consume bytes from *different* (tier, subpool)
budgets; each axis quantises at its own `page_bytes` (per-axis
`bucket_size`, no global collapse).  Sparse: only reachable `dp` cells
materialise (a dense table is exponential in axis count).

Lives in `baselines/knapsack.py` with the candidate contract
(`Migrate(cost, relief, acquired)`, `Pause(cost, relief)`,
`Resume(gain, re_use)`; sparse `{tier: {subpool: bytes}}`, string tier
keys matching the §5/§6 wire).

## TWO DESIGN-vs-CODE CORRECTIONS

1. **Parent-pointer reconstruction** (not the DESIGN's subtract-the-
   delta traceback).  The relief axis is CAPPED via `min(W, …)` on the
   forward step, so the state is NON-invertible — subtracting an item's
   relief delta cannot recover the predecessor once a transition
   saturated the cap, and the reconstructed subset is wrong (the DP
   *cost* `dp[s_pick]` stays correct; only the chosen-subset readout
   breaks).  Recording the predecessor state per improving transition
   makes the readout exact.  **Stage A2 caught the subtract version
   returning a too-cheap (cost 1.83 vs true 17.09), infeasible subset.**
2. **Infeasibility as an exception** (`KnapsackInfeasibleError` with a
   forensic context dict), not a direct `fatal()` call — keeps the
   primitive pure + the infeasible path unit-testable.  The daemon's
   `joint_decide` (deferred — see SCOPE) maps it to
   `fatal("joint_decide_infeasible", **ctx)` per DESIGN §9/§10.

## STAGES (12)

```
A. knapsack_min_cost_multi (pressure)
  A0 single relief axis: cheapest subset hitting bytes_needed
  A1 multi-axis: 2 HBM relief axes + 1 DRAM cap axis
  A2 EXACT vs brute force — 60 random fixtures (Migrate+Pause mix,
     bucket_size=1): DP chosen-subset cost == brute-force min feasible
     cost AND the subset is feasible
  A3 destination cap is a HARD constraint: a cheap Migrate that
     overflows DRAM is rejected for a (pricier) Pause
  A4 returned subset is itself feasible + optimal
B. quantisation
  B0 relief rounds DOWN (per item — 63B → 0 buckets, so 63+63 still
     0); destination acquire rounds UP (1B → 1 bucket)
C. infeasibility
  C0 total relief < need → KnapsackInfeasibleError w/ forensic ctx
     (bytes_needed / cap_left / n_items / dp_size + caller context)
  C1 DROP/Pause always feasible: destinations FULL (0 cap) → the Pause
     (no acquired) still satisfies the target
D. knapsack_max_value_multi (headroom)
  D0 single budget axis: max-gain subset within budget
  D1 EXACT vs brute force — 60 random fixtures (bucket_size=1)
  D2 budget HARD; re_use rounds UP (1B → 1 bucket → only one fits a
     1-bucket budget)
  D3 returned subset optimal
```

The brute-force oracle (exhaustive 2^K subset enumeration) is the gold
standard for an exact DP; at `bucket_size=1` the DP works in raw bytes
so DP-optimum == brute-optimum exactly (120 random fixtures total).

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t34/verify.py
```

Pure-Python; ~0.3 s.  No GPU, no sglang.

## RESULTS

**PASSED** — all 15 stages (12 + E0/E1/E2 from the audit).

* date: 2026-06-03
* raw logs: `results/20260603_t34_initial_pass.log` (12),
  `results/20260603_t34_post_audit_pass.log` (15)

## AUDIT CLOSURE (2026-06-03)

Adversarial audit verified the DP CORE correct (~270k randomized
trials incl. the parent-pointer reconstruction, bs>1 multi-axis, ties,
over-relief clamping) — no correctness bug.  Closed the findings:

* **#9 (the real one) — performance/DoS**: the auditor broke the
  "microseconds" claim (|dp| → ~1.8M cells / 34 s with large distinct
  deltas; no guard).  Added a ``max_dp_cells`` ceiling (default
  10×10⁵): past it the DP raises ``KnapsackBudgetExceededError`` (→
  ``fatal`` in ``joint_decide``, crash-only) instead of stalling the
  event loop.  Stage **E1** trips it; DESIGN §9 "microseconds" claim
  corrected.
* **#8** — ``KnapsackInfeasibleError.context`` now carries the
  candidate ``items`` (per DESIGN's ``fatal(candidates=…)``), so ops
  can see WHICH candidates were available (top-k undersizing vs a
  filter drop).  C0 asserts it.
* **#10 / #11 / #12** — added **E0** (exact-vs-brute at bucket>1 with
  MULTIPLE relief AND cap axes — the original A2 was 1+1 at bucket=1)
  and **E2** (empty items / zero need / zero budget).
* **contract / NIT** — docstrings now state the phase precondition
  (min-cost takes Migrate/Pause, max-value takes Resume),
  ``bucket_size > 0``, and the safe-direction quantisation-halt note.

## SCOPE BOUNDARY (deferred)

This task is the DP PRIMITIVES + candidate contract + infeasibility,
per the PLAN §4 T34 entry.  The full **`joint_decide` integration** —
replacing the current greedy `OursGreedyPolicy.decide` (single-axis
`cap_total − used`) + the sequential admission `_on_pressure` /
`_on_resolved` (the "Gauss-Seidel decompose" DESIGN §9 supersedes)
with one `joint_decide` that builds Migrate/Pause/Resume candidates
(`migrate_candidates` relief/acquired/cost; `pause_candidates`;
`resume_candidates`), computes `forecast`/`bytes_needed`, normalises
the flat→nested candidate shapes, and calls these primitives — is a
large rewire of the live decision path, tracked as a follow-on.
