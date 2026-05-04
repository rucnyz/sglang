"""T8 step 1 smoke: FirePlan and OwnerProvider types import cleanly,
construct, and enforce their invariants. No engine deps required.
"""

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def test_fire_plan_construction():
    from sglang.srt.arena.fire_plan import FirePlan, FirePlanResult, MigrateOp

    op = MigrateOp(src_page=12345, dst_page=42, req_pool_idx=7, slot_in_req=3)
    assert op.src_page == 12345
    assert op.dst_page == 42

    plan = FirePlan(
        direction="kv_to_mamba",
        capped_page_range=(354839, 879127),
        chunks_to_unmap_src=[2046, 2047],
        pages_to_drain=[354900, 354901, 354902],
        pages_to_migrate=[op],
        chunks_to_map_dst=30,
        expected_unmap_pages=4096,
        plan_seq=1,
    )
    assert plan.direction == "kv_to_mamba"
    assert len(plan.pages_to_migrate) == 1
    assert plan.expected_unmap_pages == 4096

    # Frozen dataclass — must reject mutation.
    raised = False
    try:
        plan.plan_seq = 2  # type: ignore[misc]
    except Exception:
        raised = True
    assert raised, "FirePlan must be frozen — caught a silent mutation"

    result = FirePlanResult(
        plan_seq=1,
        direction="kv_to_mamba",
        unmapped_pages=4096,
        granted_chunks=30,
        drained_pages=3,
        migrated_pages=1,
        cap_barrier_us=12,
        drain_us=45,
        migrate_us=130,
        unmap_us=2173,
        map_us=3509,
        total_us=5869,
    )
    assert result.aborted is False
    print(f"[step1] FirePlan + FirePlanResult construct OK (plan_seq={result.plan_seq})")


def test_owner_map_invariants():
    from sglang.srt.arena.owner_provider import OwnerMap, OwnerProvider, TreeNodeRef

    node_ref = TreeNodeRef(node=object(), page_offset=0)

    om = OwnerMap(
        pool_name="kv",
        n_pages=10,
        free_pages={0, 1, 2, 3},
        tree_pages={4: node_ref, 5: node_ref, 6: node_ref},
        active_pages={7: (0, 0), 8: (0, 1), 9: (1, 0)},
    )
    assert om.coverage() == 10
    om.assert_complete()  # must not raise

    # Break the invariant — page 9 owned by both tree and active.
    om_bad = OwnerMap(
        pool_name="kv",
        n_pages=10,
        free_pages={0, 1, 2, 3},
        tree_pages={4: node_ref, 5: node_ref, 6: node_ref, 9: node_ref},
        active_pages={7: (0, 0), 8: (0, 1), 9: (1, 0)},
    )
    raised = False
    try:
        om_bad.assert_complete()
    except RuntimeError as e:
        raised = True
        assert "coverage broken" in str(e)
    assert raised, "double-owned page must trip assert_complete"

    # OwnerProvider is a runtime_checkable Protocol — a class with the
    # right methods passes isinstance even without subclassing.
    class FakeProvider:
        def build_kv_owner_map(self) -> OwnerMap:
            return om

        def build_mamba_owner_map(self):
            return None

    fp = FakeProvider()
    assert isinstance(fp, OwnerProvider), "Protocol structural check failed"
    assert fp.build_kv_owner_map() is om
    assert fp.build_mamba_owner_map() is None

    print("[step1] OwnerMap + OwnerProvider invariants OK")


def main():
    test_fire_plan_construction()
    test_owner_map_invariants()
    print("\nT8 step1 type smoke PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
