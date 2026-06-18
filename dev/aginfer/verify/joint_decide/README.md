# joint_decide — DESIGN §9 union-action decision (#194)

Wires the T34 multi-axis DP (#156) into the **live decision path**,
replacing the two separate, sequential decision modules:

* `OursGreedyPolicy.decide` — per-unit greedy migrate, single-axis
  `cap_total − used` capacity check;
* the admission `_on_pressure` / `_on_resolved` loops — a separate
  handler composed on top of kv_scheduler (the "Gauss-Seidel
  decompose" DESIGN §9 supersedes).

with ONE `joint_decide(state, event)` over the union action space
`A = {unit migrate} ∪ {program pause/resume}`.

**Value-gated rewrite (2026-06-07).**  §9 is value-gated, not cover: every
phase takes ONLY net-positive actions and may no-op; relief (migrate) and
resume COEXIST (not mutually exclusive); there is no forced relief and no
infeasibility.  The **Pause lever is DORMANT** — not generated (its cost
misses the paused agent's forgone progress and its OOM-benefit is
unmodelled, §8).  The old min-cost-cover pressure phase (which forced
relief and caused the A3 agent-stall regression) and its `best_effort`
fallback were removed; the live joint is over `{migrate-relief} ∪
{resume}`.

## Pieces

| piece | where | role |
|-------|-------|------|
| `migrate_candidates(state, D_t)` | `baselines/ours_greedy.py` | §7 unit-level `Migrate(cost, relief, acquired)` generator |
| `forecast` / `pause_candidates` / `resume_candidates` | `daemon/admission_controller.py` | §8 program-level generator + per-HBM-subpool forecast |
| `joint_decide` | `daemon/joint_decide.py` | §9 phase selector + DP call + mixed-plan return |
| `_dispatch_plan` | `daemon/kv_scheduler.py` | splits the plan: Migrate→POST, Resume→tracker.resume+PUT (the Pause branch exists for when the lever is enabled, but `joint_decide` emits none) |

The kv_scheduler `handle` now runs `joint_decide` (thresholds from the
router, costs from the shared policy) and dispatches the mixed plan;
admission is the §8 candidate generator, not a separate handler.
`KvScheduler.admission_enabled` gates the Resume lever (off = the kv-only
migrate-only ablation arm, Run K).

## DESIGN-vs-CODE CORRECTIONS

1. **Multiple-choice, not plain 0/1, over a unit's transitions.**
   `migrate_candidates` emits several transitions per unit (evict /
   spill / DROP).  A plain 0/1 knapsack (DESIGN §9 "exact 0/1
   knapsack") can pick TWO transitions of the same unit — physically
   incoherent: their relief double-counts the unit's bytes and the
   costs aren't additive (each is scored as a marginal change from the
   ORIGINAL residence).  The DP treats items sharing a non-None `group`
   (= the unit hash) as **at-most-one** (multiple-choice knapsack).
   t34 stage G0 pins it against a grouped brute oracle.

2. **Destination budget clamps to ≥ 0.**  `cap − used` can be negative
   when a destination tier (DRAM/DISK) is over-subscribed.  A negative
   budget makes the DP reject even zero-consume candidates (the
   accumulator starts at 0, already `>` a negative bound).  `joint_decide`
   clamps to `max(0, cap − used)` (no room is 0 room).  Stage G pins it.

3. **Value-gated → no infeasibility, no `best_effort`.**  An earlier
   revision used a min-cost-cover pressure phase that *forced* relief and,
   when it couldn't reach the cover target (in-flight-dominated pressure
   migration can't touch — the A3 swa regime), fell back to a
   `best_effort` max-relief subset.  Forcing relief against an
   unrelievable subpool paused active agents that could never be usefully
   resumed → agent-stall timeouts worse than baseline.  The value-gated
   rewrite removes the forcing: the same value-max DP that picks
   beneficial actions also picks the empty set when none is beneficial, so
   "nothing relieves the pressure" is simply the no-op — no infeasibility,
   no best-effort.  `fatal` is reserved for the `max_dp_cells` DP blow-up.

## Stages

```
A migrate_candidates  — cost = V(cur)−V(next)+M_eff; relief/acquired
                        per (tier,sp); relief>0 filter; (uid,add,remove) id
A-leaf                — mirrors sglang's 3 apply-site leaf guards (#210)
B forecast            — per-HBM-subpool used_bytes (+ inflight term, 0
                        under the T26/T11 placeholders); horizon=heartbeat_s
C pause/resume_cands  — pause_candidates (generator, dormant in §9) +
                        Resume(gain, re_use) for PAUSED passing capacity_fits
D joint_decide select — VALUE-GATED: net-positive relief acts; dead-zone +
                        unrelievable-swa no-op (do-no-harm); resume; pauses
                        never appear
D-starve              — #211 dropped-unit PAUSED programs still resume
D-press-resume        — #213 relief migrate + un-starve resume coexist in
                        one plan under pressure; no pause
E DP correctness      — knapsack_max_value_multi: no same-group double-pick,
                        empty-on-no-pay, exact vs brute (40, multi-axis),
                        budget held, blow-up → raise
F live dispatch       — KvScheduler.handle → _dispatch_plan: pressure →
                        relief Migrate (POST); headroom → Resume (tracker.
                        resume + PUT{pre_pause_state}); no pause ever;
                        admission OFF → relief-only (kv-only arm)
G robustness          — destination-budget ≥0 clamp; value-gate excludes a
                        HOT (cost≥0) relief candidate end-to-end
H targeting (SF-3)    — relief targets the PRESSURED subpool only; never
                        churns a healthy subpool nor grows the pegged one
```

## AUDIT CLOSURE (2026-06-04)

Adversarial audit confirmed the DP core correct (MCKP at-most-one,
parent-pointer reconstruction, negative costs, pressure-suppresses-
headroom). It surfaced one **real bug** + test/doc gaps:

* **BUG — `joint_decide` was never called when `D_t` was empty.**
  `KvScheduler.handle` had a greedy-era `if not decision_set: return`
  that short-circuited before the joint decision, so the admission
  (pause/resume) half **never ran on LLM_PREFILL** (D_t always ∅) or any
  empty-top-k event — directly violating DESIGN §9. Caught by the new
  **stage F** (resume on an empty-D_t PRESSURE_RESOLVED returned
  nothing). Fixed: the handler now always runs `joint_decide`; hints are
  a no-op on empty D_t.
* **test-gaps closed**: stage F (the mixed-plan dispatch — brand-new
  code that had zero positive-path coverage); stage G (the destination-
  budget clamp); t187 C2 now injects a REAL migrate-dispatch error.
* **doc**: the SESSION_END pressure-gating (DESIGN §9) and the §8
  holder-divided V_u interim (DESIGN §8) are now documented in DESIGN,
  not just docstrings.

## VALUE-GATED REWRITE CLOSURE (2026-06-07)

The §9 decision was changed from min-cost-COVER to value-gated MAX-VALUE
(see the top-of-file note + the CORRECTIONS section).  The suite was
rewritten accordingly and re-audited:

* stage **D** now pins value-gated selection: net-positive relief acts +
  collapses a unit's transitions to ONE (group exclusion wiring); the
  hysteresis dead-zone AND an unrelievable pegged `swa` subpool both
  NO-OP (do-no-harm); pauses never appear.
* stages **D-press-resume** (#213) and **H** (SF-3) pin relief+resume
  coexistence and pressured-subpool targeting.
* stage **E** swapped the deleted min-cost oracle for the
  `knapsack_max_value_multi` brute oracle (grouping, multi-axis, negative
  gains, empty-on-no-pay).
* stage **G** swapped the `best_effort` assertion for an end-to-end
  value-gate exclusion of a HOT (cost≥0) candidate.
The "pressure-suppresses-headroom" property and the `best_effort`
fallback are GONE — the audit confirmed neither survives in code or
tests.

## Dependencies / honest degradation

`forecast_inflight_demand` needs `decode_throughput` (T26 —
`decode_per_program` ships empty), `E[remaining_tokens]` (T11 / #126),
and `bytes_per_token_in_subpool` (a model-architecture constant not in
`/aginfer/state`).  Until those land, `forecast[sp] == used_bytes` — so
§9's relief/resume triggers reduce **exactly** to the allocator-truth
HBM occupancy, and become trajectory-aware once the inputs are measured.
The **Pause lever is dormant** independently of these (it is not
generated by `joint_decide` at all until both its progress-cost and its
OOM-benefit are modelled, §8) — so `pause_candidates` / `pause_relief`
are exercised only as generators in stage C, never on the live decision
path.  Tracked as #194 follow-ons.

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/joint_decide/verify.py
```

Pure-Python; no GPU, no sglang.  The live wiring is exercised end-to-end
by `verify/integration_stress` (flavors A/D/G) on the real stack.

## RESULTS

**PASSED** — all 11 stages (A, A-leaf, B, C, D, D-starve, D-press-resume,
E, F, G, H) under the value-gated rewrite.  The live A3 daemon dump
(`/tmp/live_state.json`, pegged `swa` ≈ 0.85, all-in-flight) feeds
`joint_decide` to a **0-action plan** offline — the do-no-harm no-op the
value-gate is designed for, the opposite of the old cover-forced pauses.

* date: 2026-06-07 (value-gated rewrite)
