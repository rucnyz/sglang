"""#183 — Admitter seven-candidate cost program (adds own/cross_migrate).

design.md §"Page selection" extends the decision from five candidates
to seven: own/cross × {free, evict, migrate} + defer. The cross
candidates are CUMULATIVE free → drain → migrate (#273), mirroring the
planner's Stage 1→2→3 fill: each layers its mechanism on top of free, and
the cost charges each mechanism only for the shortfall it covers (so
`c_evict_src_us` / `c_migrate_src_us` passed to `decide()` are already
the drain-part / migrate-part costs, 0 when that part is empty).

Pins:
  1. own_migrate is inert for dst='kv' (c_migrate_dst_us=+inf, no KV
     migrate primitive) — self-gating, ready for dst=mamba (#159).
  2. cross_migrate wins when free + evict can't cover X but free + evict +
     migrate can, and c^xfer + drain + migrate < defer.
  3. cross_migrate cost = c_xfer_total + c_evict_src_us(drain part) +
     c_migrate_src_us(migrate part) — cumulative composition.
  4. Migration loses a tie to free / evict (tie-break suffix order).
  5. cross_migrate obeys the cross-* cold-start gate (suppressed while an
     own-* alternative is feasible and the c^xfer EWMA is unwarmed).
  6. The candidate set is exactly the seven design.md actions.
"""
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

_SEVEN = {"own_free", "own_evict", "cross_free", "cross_evict",
          "own_migrate", "cross_migrate", "defer"}


def _admitter(warm=True):
    from sglang.srt.budgeter.cost_model import (
        CostModel, get_runtime_actuator_cost, reset_cost_model,
    )
    reset_cost_model()
    if warm:
        rac = get_runtime_actuator_cost()
        for _ in range(3):  # is_calibrated after 3 live observations
            rac.update(1000.0, 1)
    from sglang.srt.budgeter.admitter import Admitter
    return Admitter(cost_model=CostModel())


def _base(**over):
    """Default decide() kwargs: nothing feasible but defer. Override per test."""
    # `c_evict_src_us` is the cross DRAIN-part cost (cumulative model):
    # 0 means "no drain needed/available" (the default here, src_evictable=0).
    # `c_migrate_src_us` is the MIGRATE-part cost; +inf default = no migrate.
    # Demand = exactly one page (1024 tokens, tps=1024, lcm=1) so the
    # LCM-rounded effective demand `x_eff` equals x_tokens — feasibility
    # values below are then plain token counts against a 1-page target.
    kw = dict(
        x_tokens=1024, dst_pool="kv", dst_free=0, dst_evictable=0,
        src_pool="mamba", src_free=0, src_evictable=0, queue_len=10,
        c_evict_dst_us=float("inf"), c_evict_src_us=0.0,
        c_xfer_per_page_us=10.0,
        dst_migratable=0, src_migratable=0,
        c_migrate_dst_us=float("inf"), c_migrate_src_us=float("inf"),
        tokens_per_page=1024, lcm_pages=1,
    )
    kw.update(over)
    return kw


def test_1_own_migrate_inert_for_kv_dst():
    a = _admitter()
    # dst_migratable >= X but c_migrate_dst_us=+inf (no KV migrate primitive)
    dec = a.decide(**_base(dst_migratable=10_000, c_migrate_dst_us=float("inf"),
                           queue_len=0))
    assert dec.candidate_costs_us["own_migrate"] == float("inf"), dec.candidate_costs_us
    assert dec.action == "defer", dec.action
    print("  PASS  1  own_migrate inert for dst=kv (c_migrate_dst_us=+inf)")


def test_2_cross_migrate_wins_when_free_evict_infeasible():
    a = _admitter()
    # No free / evict anywhere; src has migratable LIVE state; migrate cheap.
    # cumulative: free(0)+evict(0)+migrate ≥ X ⇒ feasible; drain part 0.
    dec = a.decide(**_base(
        src_migratable=10_000, c_migrate_src_us=50.0,  # c_xfer(1)*10 + 0 + 50 = 60
        queue_len=10_000,  # defer = 10000 * w_q >> 60
    ))
    assert dec.action == "cross_migrate", f"{dec.action}: {dec.candidate_costs_us}"
    print("  PASS  2  cross_migrate wins when free+evict infeasible, migrate < defer")


def test_2b_cumulative_free_plus_mechanism_feasibility():
    """#273: cross candidates are FREE + their mechanism. Neither free (60)
    nor evict (60) covers X=100 ALONE, but free+evict=120 ≥ 100 ⇒
    cross_evict feasible (the old migrate-alone/evict-alone predicate would
    have called it infeasible). cross_free still infeasible (60<100).
    cross_migrate also feasible (free+evict+migrate) but ties cross_evict
    when no migration is needed, losing the tie-break."""
    a = _admitter()
    # X=1024 (1 page). free=600, evict=600: neither ≥ 1024 alone, sum 1200 ≥ 1024.
    dec = a.decide(**_base(
        src_free=600, src_evictable=600, c_evict_src_us=30.0,  # drain 424 tok
        src_migratable=10_000, c_migrate_src_us=50.0,
        c_xfer_per_page_us=10.0, queue_len=10_000,
    ))
    assert dec.candidate_costs_us["cross_free"] == float("inf"), (
        f"free(600) < X(1024) → cross_free infeasible: {dec.candidate_costs_us}"
    )
    assert dec.candidate_costs_us["cross_evict"] == 10.0 + 30.0, (
        f"free+evict(1200) ≥ 1024 → cross_evict feasible at c_xfer+drain: "
        f"{dec.candidate_costs_us}"
    )
    # No migration needed (free+evict already ≥ X) → migrate part 0 →
    # cross_migrate ties cross_evict; tie-break picks cross_evict.
    assert dec.candidate_costs_us["cross_migrate"] == 10.0 + 30.0 + 50.0, (
        dec.candidate_costs_us["cross_migrate"]
    )
    assert dec.action == "cross_evict", (
        f"free+evict covers X → cross_evict (tier 3) beats cross_migrate "
        f"(tier 5): {dec.action}"
    )
    print("  PASS  2b cumulative free+evict feasibility (neither alone) → "
          "cross_evict selected")


def test_2c_cross_evict_migrate_tie_at_boundary():
    """#273: at the EXACT boundary free+evict == X, no migration is needed,
    so decide_for_req sets the migrate part to 0 (c_migrate_src_us=0).
    cross_evict and cross_migrate then tie at c_xfer+drain; the tie-break
    (tier 3 < tier 5) picks cross_evict."""
    a = _admitter()
    # free 512 + evict 512 = 1024 == X. migrate part 0 → c_migrate_src_us 0.
    dec = a.decide(**_base(
        src_free=512, src_evictable=512, c_evict_src_us=30.0,
        src_migratable=10_000, c_migrate_src_us=0.0,
        c_xfer_per_page_us=10.0, queue_len=10_000,
    ))
    assert dec.candidate_costs_us["cross_evict"] == 40.0, dec.candidate_costs_us
    assert dec.candidate_costs_us["cross_migrate"] == 40.0, (
        f"migrate part 0 at the boundary → cross_migrate ties cross_evict: "
        f"{dec.candidate_costs_us}"
    )
    assert dec.action == "cross_evict", dec.action
    print("  PASS  2c free+evict==X boundary → cross_evict/migrate tie → "
          "cross_evict wins")


def test_2d_cross_free_wins_three_way_tie():
    """#273 zero-downside: when free ≥ X, the drain and migrate parts are
    both 0, so all THREE cross candidates cost exactly c_xfer_total; the
    tie-break picks cross_free (tier 1) over cross_evict (3) / cross_migrate
    (5). Pins that a 'finite' cross_evict/cross_migrate never displaces the
    cheaper cross_free."""
    a = _admitter()
    dec = a.decide(**_base(
        src_free=10_000, src_evictable=10_000, c_evict_src_us=0.0,
        src_migratable=10_000, c_migrate_src_us=0.0,
        c_xfer_per_page_us=10.0, queue_len=10_000,
    ))
    c = dec.candidate_costs_us
    assert c["cross_free"] == 10.0 and c["cross_evict"] == 10.0 and \
        c["cross_migrate"] == 10.0, c
    assert dec.action == "cross_free", (
        f"three-way tie at c_xfer_total must go to cross_free (tier 1): "
        f"{dec.action}"
    )
    print("  PASS  2d free≥X three-way tie at c_xfer_total → cross_free wins")


def test_3_cross_migrate_cost_composition():
    a = _admitter()
    # Three-part cumulative fill: free(0) + drain(40 tok @ c_evict=20) +
    # migrate(rest @ c_migrate=50). cross_migrate feasible iff
    # free+evict+migrate ≥ X; cost = c_xfer_total + drain + migrate.
    dec = a.decide(**_base(
        src_free=0, src_evictable=40, c_evict_src_us=20.0,
        src_migratable=10_000, c_migrate_src_us=50.0,
        c_xfer_per_page_us=10.0, queue_len=10_000,
    ))
    # X=1024 → n_pages=1 → c_xfer = 1*10. + drain 20 + migrate 50 = 80.
    assert dec.candidate_costs_us["cross_migrate"] == 10.0 + 20.0 + 50.0, (
        dec.candidate_costs_us["cross_migrate"]
    )
    # cross_evict infeasible here (free+evict = 40 < 1024), so cross_migrate
    # is the harvest that wins.
    assert dec.candidate_costs_us["cross_evict"] == float("inf"), (
        dec.candidate_costs_us["cross_evict"]
    )
    assert dec.action == "cross_migrate", dec.action
    print("  PASS  3  cross_migrate cost = c_xfer + drain + migrate (80.0); "
          "cross_evict infeasible (free+evict<X)")


def test_4_migrate_loses_tie_to_free_and_evict():
    a = _admitter()
    # own_evict and cross_migrate both cost 0-ish; evict must win the tie.
    dec = a.decide(**_base(
        dst_evictable=10_000, c_evict_dst_us=60.0,           # own_evict = 60
        src_migratable=10_000, c_migrate_src_us=50.0,         # cross_migrate = 60
        queue_len=10_000,
    ))
    assert dec.candidate_costs_us["own_evict"] == 60.0
    assert dec.candidate_costs_us["cross_migrate"] == 60.0
    assert dec.action == "own_evict", (
        f"tie must go to own_evict (tier 2) over cross_migrate (tier 5): "
        f"{dec.action}"
    )
    print("  PASS  4  migration loses a tie to evict (tie-break: evict<migrate)")


def test_5_cross_migrate_cold_start_loses():
    # Unwarmed c^xfer EWMA uses the sentinel high cost (3000 us/page),
    # so cross_migrate is priced far above own_free (cost 0) and loses.
    a = _admitter(warm=False)
    dec = a.decide(**_base(
        dst_free=10_000,                       # own_free feasible (cost 0)
        src_migratable=10_000, c_migrate_src_us=1.0,
        queue_len=10_000,
    ))
    assert dec.candidate_costs_us["cross_migrate"] > dec.candidate_costs_us["own_free"], (
        "cold-start cross_migrate must cost more than own_free"
    )
    assert dec.action == "own_free"
    print("  PASS  5  cross_migrate loses to own_free under cold-start sentinel cost")


def test_6_candidate_set_is_seven():
    a = _admitter()
    dec = a.decide(**_base())
    assert set(dec.candidate_costs_us) == _SEVEN, set(dec.candidate_costs_us)
    print("  PASS  6  candidate set is exactly the seven design.md actions")


def main():
    tests = [test_1_own_migrate_inert_for_kv_dst,
             test_2_cross_migrate_wins_when_free_evict_infeasible,
             test_2b_cumulative_free_plus_mechanism_feasibility,
             test_2c_cross_evict_migrate_tie_at_boundary,
             test_2d_cross_free_wins_three_way_tie,
             test_3_cross_migrate_cost_composition,
             test_4_migrate_loses_tie_to_free_and_evict,
             test_5_cross_migrate_cold_start_loses,
             test_6_candidate_set_is_seven]
    print(f"\n#183 seven-candidate Admitter tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}"); traceback.print_exc()
    print(f"#183 decide: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
