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

## Pieces

| piece | where | role |
|-------|-------|------|
| `migrate_candidates(state, D_t)` | `baselines/ours_greedy.py` | §7 unit-level `Migrate(cost, relief, acquired)` generator |
| `forecast` / `pause_candidates` / `resume_candidates` | `daemon/admission_controller.py` | §8 program-level generator + per-HBM-subpool forecast |
| `joint_decide` | `daemon/joint_decide.py` | §9 phase selector + DP call + mixed-plan return |
| `_dispatch_plan` | `daemon/kv_scheduler.py` | splits the plan: Migrate→POST, Pause→tracker.pause+PUT, Resume→tracker.resume+PUT |

The kv_scheduler `handle` now runs `joint_decide` (thresholds from the
router, costs from the shared policy) and dispatches the mixed plan;
admission is the §8 candidate generator, not a separate handler.
`KvScheduler.admission_enabled` gates the Pause/Resume levers (off =
the kv-only migrate-only ablation arm, Run K).

## THREE DESIGN-vs-CODE CORRECTIONS (this task)

1. **Multiple-choice, not plain 0/1, over a unit's transitions.**
   `migrate_candidates` emits several transitions per unit (evict /
   spill / DROP).  A plain 0/1 knapsack (DESIGN §9 "exact 0/1
   knapsack") can pick TWO transitions of the same unit — physically
   incoherent: their relief double-counts the unit's bytes and the
   costs aren't additive (each is scored as a marginal change from the
   ORIGINAL residence).  The DP now treats items sharing a non-None
   `group` (= the unit hash) as **at-most-one** (multiple-choice
   knapsack).  Singleton groups (`group=None`) reproduce the original
   0/1 behaviour exactly, so t34 is unchanged.  Stage E pins "never two
   transitions of one unit" against a brute-force oracle.

2. **`cap_left` clamps to ≥ 0.**  `capacity_left_bytes = cap − used`
   can be negative when a destination tier (DRAM/DISK) is over-
   subscribed.  A negative budget makes the DP reject even zero-acquire
   DROP candidates (the cap accumulator starts at 0, already `>` a
   negative `Wcap`) → spurious infeasibility.  `joint_decide` clamps to
   `max(0, cap − used)` (no room is 0 room, never "less than zero").
   **Caught by integration_stress** (DRAM at −32 GB `cap_left` fatal'd
   the daemon mid-load).

3. **Pressure infeasibility → best-effort, NOT fatal.**  DESIGN §9/§10
   claim "infeasible = top-k undersized = an algorithm bug → `fatal`".
   That is false in the common case integration_stress exercises:
   **in-flight-dominated pressure** (most of HBM is decode bytes, tiny
   radix footprint) that migration cannot touch, with **no Pause
   candidate available** (sglang's `per_program_usage` ships empty pre-
   measurement, like `prefill_bps=0.0`).  Then no subset reaches
   `bytes_needed` — a *workload reality*, not a bug.  Crashing the
   daemon on transient over-pressure is wrong; the §6 fire-and-forget
   contract already says "re-evaluate on the next event", and sglang's
   own eviction is the backstop.  The pressure phase now runs the DP in
   `best_effort` mode (free the max-relief subset, log the shortfall)
   and reserves `fatal` for the genuine misconfiguration: the
   `max_dp_cells` DP blow-up.

## Stages

```
A migrate_candidates  — cost = V(cur)−V(next)+M_eff; relief/acquired
                        per (tier,sp); relief>0 filter; (uid,add,remove) id
B forecast            — per-HBM-subpool used_bytes (+ inflight term, 0
                        under the T26/T11 placeholders); horizon=heartbeat_s
C pause/resume_cands  — Pause(cost=V_u_program+marginal, relief=inflight+
                        committed) for REASONING/ACTING; Resume(gain, re_use)
                        for PAUSED passing capacity_fits
D joint_decide select — pressure / headroom / dead-zone; pressure
                        suppresses headroom; LLM_PREFILL (empty D_t) still
                        runs the admission generators
E DP correctness      — no same-unit double-pick (MCKP) vs brute-force
                        oracle (40 fixtures); under-relievable pressure →
                        best-effort (no fatal); DP blow-up → still raises
F live dispatch       — KvScheduler.handle → _dispatch_plan: pressure →
                        Pause (tracker.pause + PUT{PAUSED,pre}); headroom
                        → Resume (tracker.resume + PUT{pre_pause_state});
                        admission OFF → no Pause (kv-only arm)
G robustness          — cap_left≥0 clamp (over-subscribed destination
                        keeps zero-acquire DROP feasible); best_effort
                        relieves ALL pressured axes when caps allow
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
  code that had zero positive-path coverage); stage G (the cap_left
  clamp + multi-axis best_effort); t187 C2 now injects a REAL migrate-
  dispatch error (the old `_RaisingPolicy.decide` was never reached
  post-joint) and the dead `_DemoteAllPolicy`/`_RaisingPolicy` stubs +
  `--max-pauses-per-event` arg were removed.
* **doc**: the SESSION_END pressure-gating (DESIGN §9) and the §8
  holder-divided V_u interim (DESIGN §8) are now documented in DESIGN,
  not just docstrings.

## Dependencies / honest degradation

`forecast_inflight_demand` and `pause_relief`'s `future_inflight_savings`
term need three inputs that are **not yet wired**: `decode_throughput`
(T26 — `decode_per_program` ships empty), `E[remaining_tokens]` (T11 /
#126), and `bytes_per_token_in_subpool` (a model-architecture constant
not in `/aginfer/state`).  Until those land, `forecast[sp] == used_bytes`
and `pause_relief = inflight + committed` (snapshot only) — so §9's
pressure/headroom triggers reduce **exactly** to the allocator-truth HBM
occupancy the admission loop used pre-rewrite (behaviour-preserving),
and become trajectory-aware once the inputs are measured.  `marginal_
pause_cost` is 0 while `prefill_bps == 0.0`.  Tracked as #194 follow-ons.

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/joint_decide/verify.py
```

Pure-Python; no GPU, no sglang.  The live wiring is exercised end-to-end
by `verify/integration_stress` (flavors A/D/G) on the real stack.

## RESULTS

**PASSED** — all 5 stages (A–E), plus the live path end-to-end via
`verify/integration_stress` (7 flavors green on the real B300 stack):
stage D migrate-under-traffic (200 batches), stage G SESSION_END
demoted 6→5 HBM units, stage F no spurious `joint_decide` fatal under
sustained load (forensic: 0) — confirming the best-effort + cap-clamp
fixes hold against real pressure.

* date: 2026-06-04
* raw logs: `results/20260604_joint_decide_pass.log`,
  `results/20260604_integration_stress_pass.log`
