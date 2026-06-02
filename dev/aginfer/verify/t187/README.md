# T187 — SESSION_END migrate D_t (#187, DESIGN §4 / §7)

Sibling of F5 (#185).  F5 (#185) owns the SESSION_END state-
transition + gate-release + PUT.  This task wires the **migrate**
half: on `SESSION_END` for program `p`, the daemon re-scores
`D_t = session_scoped_units(p)` (units held ONLY by `p`) and
demotes/drops them — once `p` ends those units have no other holder
and `p` contributes **0** to their future `p_hat`
(DESIGN §4 event table, §7 `decision_set`, "SESSION_END normal
path").  Units `p` shared with a still-live program are untouched
(they survive `p`).

## WHAT THIS ADDS (3 coupled changes)

1. **`build_paper_state` p_hat rule** (`kv_scheduler.py`): an ENDED
   holder no longer counts as "alive".  Before #187, `State.ENDED`
   (introduced in #185) is `!= None`, so `any_alive` was True and a
   unit held only by an ended program kept `p_hat = 1.0` **forever**
   — it would never be demoted.  Now ENDED is excluded from the
   alive set, so such a unit falls back to the workload-prior
   `min(1, hits/age)`.  A still-live co-holder keeps `p_hat = 1.0`
   (the unit survives the ended program).
2. **`_build_decision_set` SESSION_END branch** (`kv_scheduler.py`):
   returns `session_scoped_units(p)` = units whose holders are
   exactly `{p}` (reuses `_units_for_session`, which already
   implements the exclusive form).  Shared units (≥2 holders) are
   excluded.
3. **Composed SESSION_END handler** (`event_router.py`): the handler
   now runs `tracker.end(p)` FIRST (so the scorer sees ENDED), THEN
   `kv_scheduler.handle(event, router)` (the migrate D_t), THEN the
   PUT `{ENDED}`.  Ordering is load-bearing — end() before handle()
   or the scorer would see `p` alive (`p_hat = 1.0`) and keep the
   units.  `make_session_end_handler` / `attach_session_end_handler`
   take an optional `kv_scheduler`; `main.py` passes `sched`
   (None when `--kv-scheduler=disabled` → pure F5).

## STAGES (10)

```
A. decision_set
  A0 SESSION_END(p) → session_scoped (exclusive) units only; a unit
     shared by p+q is EXCLUDED
  A1 SESSION_END(p) with only-shared units → empty D_t
B. p_hat ENDED-exclusion (the scoring anchor — RED before #187)
  B0 unit held ONLY by an explicitly end()-ed program → workload-
     prior p_hat (< 0.1), NOT 1.0
  B1 unit held by an ENDED + a LIVE program → p_hat == 1.0 (the live
     co-holder dominates; the unit survives p)
  B2 never-seen holder → workload-prior (regression — the carve-out
     doesn't disturb the unknown-holder path)
  B3 the carve-out is EVENT-AGNOSTIC: on a MEMORY_PRESSURE event a
     leftover ENDED-held unit also scores the workload-prior p_hat
     and becomes a top-k demote candidate (audit G2 — the change
     fires on every event, the intended latent-bug fix, not just
     SESSION_END)
C. composed handler (stub demote-all / raising policy)
  C0 end → migrate(session_scoped) → PUT, migrate ENQUEUED BEFORE
     the PUT, migrate targets p's exclusive units only
  C1 kv_scheduler=None → pure F5 (ENDED + PUT, no migrate)
  C2 if the migrate step (kv_scheduler.handle) RAISES, the F5 PUT
     {ENDED} is STILL enqueued + the program STILL ENDED (audit B1 —
     handle() only guards its own fetch/build, so the handler wraps
     the migrate step; F5 state-transition + PUT is the contract,
     migrate is best-effort on top)
D. real composed router (main.py attach order)
  D0 attach_kv_scheduler THEN attach_session_end_handler(…, sched):
     a real SESSION_END ends the program AND runs the migrate
     (migrate_calls advanced) AND enqueues the PUT
E. real policy (scoring drives the decision)
  E0 with two otherwise-identical cold HBM units, the real
     OursGreedyPolicy assigns a strictly LOWER keep-in-HBM value to
     the ENDED-holder unit than the live one (V_u = p_hat·[R(DROP) −
     R(HBM)] − holding, R(DROP) > R(HBM) ⇒ lower p_hat ⇒ lower keep-
     value ⇒ demoted first), and puts the ENDED unit in its demote
     plan.  (The absolute keep-vs-demote threshold is a value-rule
     property tested in kv_scheduler_value_rule; here we pin the
     COMPARATIVE effect, which is what SESSION_END relies on.)
F. shared survives via D_t exclusion
  F0 p+q hold a unit; SESSION_END(p) via the real policy → that unit
     is NOT in the migrate plan because holders ⊋ {p} excludes it
     from D_t (the survival mechanism is exclusion, not a scorer
     keep-decision — so #187 cannot over-evict a surviving program's
     KV: every unit in p's D_t belongs exclusively to the ending p)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t187/verify.py
```

Pure-Python (asyncio); ~0.5 s.  No GPU, no sglang launch.

## RESULTS

**PASSED** — all 12 stages (10 initial + 2 from the audit: B3, C2).

* date: 2026-06-02
* raw logs: `results/20260602_t187_initial_pass.log` (10),
  `results/20260602_t187_post_audit_pass.log` (12)

## REGRESSION SANITY

* kv_scheduler_value_rule: PASS (19) — incl. the existing
  `stage_c2_p_hat_alive_vs_ended`, which stays green because it uses
  NEVER-SEEN holders (tracker.state → None), a path the ENDED carve-
  out does not touch.  NB that test is misnamed ("…_vs_ended") — it
  never calls `tracker.end()`, so real `State.ENDED` p_hat is
  covered ONLY by t187 B0/B1 (audit N4).
* T41 SESSION_END (F5): PASS (17) — the handler composition is
  additive; pure-F5 path (kv_scheduler=None) preserved
* T40 hint emitter: PASS — SESSION_END now also pushes hints for the
  (low-p_hat) session_scoped units via the same handle() pipeline
* T6 program_tracker: PASS
* integration_stress: full-stack sglang + daemon (6 flavors), run as
  a broad sanity.  **It does NOT exercise SESSION_END** (grep:
  0 hits) — so it is NOT a regression guard for this path.

## COVERAGE HONESTY (audit G3)

There is **no end-to-end (real sglang) test of the SESSION_END
migrate** landing a demote/drop.  What IS covered:

* the migrate **wire** (`POST /aginfer/migrate` → sglang applies the
  residence-set transition) is e2e-proven by T20 / T33 / the
  e2e_smoke migrate loop — #187 does not change that path;
* the SESSION_END **decision** (p_hat carve-out + D_t selection +
  handler composition) is daemon-pure logic, fully covered here
  incl. the real composed router (D0).

The untested seam is the live webhook → daemon handler → migrate
POST → sglang-demote chain end-to-end.  A SESSION_END flavor for
integration_stress is the natural home; tracked as a follow-on.

## RELATIONSHIP TO #185

#185 (F5) and #187 are the two halves of SESSION_END.  #185 = the
control-plane response (ENDED state, gate-release-with-499 for a
parked PAUSED request, the PUT).  #187 = the data-plane response
(demote/drop the ending program's exclusive KV).  They share one
handler: `end()` (F5) runs first precisely so the migrate (T187)
scores against the post-END `p_hat`.
