"""#271 step-5 audit (MEDIUM) — mamba live-migration is FAIL-CLOSED, not
incidentally inert.

`BudgetAgent._maybe_fire` now passes `allow_migrate=True` for BOTH directions
(the KV walk self-gates on SGLANG_XPOOL_KV_MIGRATE + can_migrate_slot). The
mamba source (mamba_to_kv) has NO such env/capability gate — it was inert only
because mamba runs `tps == 1` (atomic: no partial pages → no donors, #269).
That makes inertness an unasserted runtime accident: a fragmentable mamba
layout (`tps >= 2`, anticipated for TP / bf16 ssm — see admitter
`_mamba_free_and_migratable`) would otherwise silently migrate LIVE recurrent
state with no opt-in and no captured-graph replay proof (unlike the KV side,
#291/#294b).

Fix: `_live_pages_in_cost_order("mamba")` refuses (returns []) for `tps != 1`,
so mamba migration is GATED off until a mamba-side gate + replay proof land.
`tps == 1` keeps the normal walk (which correctly yields [] via donor
emptiness). CPU-only — stub the live/free helpers to inject a fragmentable
layout that WOULD migrate pre-fix.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _provider(tps):
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider
    mamba_act = types.SimpleNamespace(
        _tokens_per_page=lambda: tps, n_pages=4,
        pool=types.SimpleNamespace(_capped_slots=None),
    )
    prov = SchedulerOwnerProvider(
        scheduler=types.SimpleNamespace(),
        kv_actuator=None, mamba_actuator=mamba_act,
    )
    # Fragmentable layout (tps=2, 4 pages, slots 0..7):
    #   p1 [2,3] fully-live -> SOURCE; p2 [4 free|5 live], p3 [6 free|7 live]
    #   -> 2 donors. Pre-fix this yields [(1, ((2,4),(3,6)))].
    prov._mamba_live_uncached_slots = lambda pool: [2, 3, 5, 7]
    prov._free_slot_set = lambda name: {4, 6}
    return prov


def test_tps2_mamba_migration_fails_closed():
    prov = _provider(tps=2)
    out = prov._live_pages_in_cost_order("mamba")
    assert out == [], (
        f"mamba live-migration must be FAIL-CLOSED for tps!=1 (ungated, "
        f"unproven) — got {out}"
    )
    print("  PASS  tps=2 fragmentable mamba -> [] (fail-closed, not incidental)")


def test_tps1_mamba_still_inert_no_crash():
    """tps==1 (the real mamba config) keeps the normal walk and yields []
    (atomic-inert, #269) — the gate must not change this path."""
    prov = _provider(tps=1)
    # tps=1: stub live/free so no page is both source and has a donor.
    prov._mamba_live_uncached_slots = lambda pool: [1, 2, 3]
    prov._free_slot_set = lambda name: set()
    out = prov._live_pages_in_cost_order("mamba")
    assert out == [], f"tps=1 mamba is atomic-inert; expected []; got {out}"
    print("  PASS  tps=1 mamba -> [] (atomic-inert path unchanged)")


def main() -> int:
    tests = [
        test_tps2_mamba_migration_fails_closed,
        test_tps1_mamba_still_inert_no_crash,
    ]
    print(f"\n#271 mamba migration fail-closed tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\nmamba fail-closed: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
