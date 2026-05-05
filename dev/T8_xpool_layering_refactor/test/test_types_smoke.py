"""FirePlan + OwnerMap type smoke."""

import sys
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def test_fire_plan_construction():
    from sglang.srt.arena.fire_plan import FirePlan, FirePlanResult

    plan = FirePlan(
        direction="kv_to_mamba",
        pages_to_unmap=[5, 7, 11],
        pages_to_map_dst=3,
        plan_seq=1,
    )
    assert plan.direction == "kv_to_mamba"
    assert plan.pages_to_unmap == [5, 7, 11]

    raised = False
    try:
        plan.plan_seq = 2  # type: ignore[misc]
    except Exception:
        raised = True
    assert raised, "FirePlan must be frozen"

    result = FirePlanResult(
        plan_seq=1, direction="kv_to_mamba",
        unmapped_pages=3, granted_pages=3,
        cap_barrier_us=12, unmap_us=2173, map_us=3509, total_us=5694,
    )
    assert result.aborted is False
    print("[step1] FirePlan + FirePlanResult OK")


def test_owner_map_basic():
    from sglang.srt.arena.owner_provider import OwnerMap, OwnerProvider

    om = OwnerMap(pool_name="kv", n_pages=100, free_pages={3, 5, 7})
    assert om.n_pages == 100
    assert 3 in om.free_pages

    class FakeProvider:
        def build_kv_owner_map(self): return om
        def build_mamba_owner_map(self): return None
    assert isinstance(FakeProvider(), OwnerProvider)
    print("[step1] OwnerMap + Protocol OK")


def main():
    test_fire_plan_construction()
    test_owner_map_basic()
    print("\nT8 step1 type smoke PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
