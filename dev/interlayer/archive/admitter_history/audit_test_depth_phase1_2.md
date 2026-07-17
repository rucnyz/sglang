# Test depth audit — Phase 1 + Phase 2 (2026-05-29)

Subagent audit of `test_cost_model_facade.py` (Phase 1, 6 tests) and
`test_admitter_no_cross.py` (Phase 2, 7 tests). Identified 3 critical
gaps + several smaller ones.

## Critical gaps identified

### Gap 1: warm-up gate not wired into `Admitter.decide()`

`admitter.py:decide()` never called `self.cost_model.is_warmed_up()`.
Design.md cold-start protocol (§354-356; admitter/design.md:152-167)
requires cross-* to be suppressed when EWMA reflects only the
conservative initial value, unless no own-* alternative is feasible.

**Why it matters:** the Admitter is the source of c^xfer observations.
If it fires unwarmed, the 3000 µs/page initial dominates decisions
before any real measurement lands. D6/D6n/D10 would measure against
a permanently-cold curve.

**Fix landed:** added `cross_gated` second-tier check in
`admitter.py:127-141`. When `cross_fire_enabled=True` AND
`not is_warmed_up()` AND any own-* is feasible → cross-* are forced
to +inf. Cold-start probe (cross-* when own-* infeasible) still
fires to warm up the EWMA.

**Test landed:** `test_8_warmup_gate_wired_into_decide` verifies the
cold→warm transition: same inputs produce own_evict cold, cross_free
warm. Also asserts `candidate_costs_us["cross_free"] == +inf` cold.

### Gap 2: tie-break order under-tested (1/4 design pairs)

Only `test_2_own_evict_wins_over_defer` exercised a tie. Three other
ties from design.md §372 were untested:
- own_free vs cross_free (both at cost 0)
- cross_free vs own_evict (same finite cost)
- own_evict vs cross_evict (same finite cost)

**Why it matters:** `_TIE_BREAK_ORDER` (admitter.py:60-66) is the
only static encoding of design.md §372. A reviewer swapping any pair
would silently change admission policy with no test failure.

**Test landed:** `test_7_tiebreak_all_four_design_pairs` constructs
identical-cost inputs for each of the 4 pairs and asserts the
design-prescribed winner. The 4 winners are observed by construction;
none coincidental.

### Gap 3: producer guard at `agent.py:835` untested

The `if not result.aborted and result.granted_pages > 0:` check
guarding `cost_model.update_xfer` is the production invariant audit
flagged. Subagent says lower urgency — Phase 4 will add a second
caller path so a regression test makes more sense there.

**Status:** deferred to Phase 4.

## Smaller gaps addressed

- `tokens_per_page` was never varied (all tests used default 1024
  with x_tokens ≤ 500 → n_pages=1) → `test_9_tokens_per_page_rounding`
  added, sweeps 6 cases (1, 1024, 1025, 2048, 3000 tokens; tps=64
  and 1024).
- `x_tokens=0` was untested → Phase 4 will revisit when the
  scheduler-hook wrapper is added (Phase 5 hook may filter 0-token reqs
  before they hit decide).

## Smaller gaps deferred (intentional)

- Negative inputs (`queue_len<0`, `x_tokens<0`, negative `c_evict_*`):
  the audit notes production never produces these, but the lack of
  asserts means a bug elsewhere causes silent wrong decisions. Defer
  to Phase 5 when the scheduler-hook wrapper validates inputs.
- `cross_fire_enabled` mutation mid-run: Phase 4 wires this through
  config; if it stays construction-only, no test needed.
- `c_xfer_per_page_us` passed as arg vs read from `cost_model`: the
  pure-function interface is intentional; the wrapper in Phase 5 is
  what would carry a mis-unit bug. Defer to Phase 5 integration test.

## Subagent recommendation

> **Soft no-go** for Phase 3. Land tie-break sweep + warm-up wiring
> first; Phase 3 will rely on `decide()` returning correct cross-*
> decisions when c^evict is finite.

## Action taken

- Tests 7, 8, 9 added → 10/10 PASS in `test_admitter_no_cross.py`
- `admitter.py` cold-start gate landed (lines 127-141)
- All existing Phase 1 (6/6), dyn_admission_cap Phase 2/7 (7+6), 
  owner_map (6), balanced_atomic (7), mark_no_realloc (4) tests still PASS

→ Phase 3 unblocked.

## Genuinely deep tests (per subagent verdict)

These remain the strongest assurances after the round of additions:

1. `test_2_warmup_gate_boundary` (cost_model_facade) — N=2→N=3 transition
2. `test_6_singleton_shared_across_facade_instances` — defends facade design
3. `test_5 + 5b` paired positive/negative gate check
4. `test_4_Q_times_w_q` — multi-Q sweep, catches sign/off-by errors
5. `test_1_xfer_ewma_convergence` — two-axis check (fixpoint + linear scaling)
6. **NEW:** `test_7_tiebreak_all_four_design_pairs` — locks down §372
7. **NEW:** `test_8_warmup_gate_wired_into_decide` — locks down §354-356
8. **NEW:** `test_9_tokens_per_page_rounding` — locks down page-rounding
