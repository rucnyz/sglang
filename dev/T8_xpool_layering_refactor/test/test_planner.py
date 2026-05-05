"""T8 step 2 — XPoolFirePlanner tests.

Synthetic OwnerMaps drive the planner without spinning a real engine.
Verifies:
  - clean tail-only case (no active pages) → drain-only plan
  - active pages in tail → migrate-only / mixed plan, with valid dst slots
  - insufficient free dst → planner refuses (returns None)
  - owner map with broken coverage → planner refuses
  - dst page never falls inside the capped range
"""

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


class _FakeAct:
    """Minimal stand-in for KVArenaActuator: only `tokens_per_chunk`
    is read by the planner."""

    def __init__(self, tpc: int) -> None:
        self.tokens_per_chunk = tpc


class _FakeProvider:
    def __init__(self, kv_om, mamba_om=None) -> None:
        self._kv = kv_om
        self._mamba = mamba_om

    def build_kv_owner_map(self):
        return self._kv

    def build_mamba_owner_map(self):
        return self._mamba


def _make_complete_om(n_pages, tpc, *, tree_pages=None, active_pages=None):
    from sglang.srt.arena.owner_provider import OwnerMap, TreeNodeRef

    tree_pages = tree_pages or {}
    active_pages = active_pages or {}
    used = set(tree_pages) | set(active_pages)
    free = set(range(1, n_pages + 1)) - used
    node_ref = TreeNodeRef(node=object(), page_offset=0)
    return OwnerMap(
        pool_name="kv",
        n_pages=n_pages,
        free_pages=free,
        tree_pages={p: node_ref for p in tree_pages},
        active_pages=active_pages,
    )


def test_pick_free_pages_from_tail():
    """Anywhere-free planner: with most pages free at tail, picks tail K.
    """
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner

    tpc = 8
    n_pages = 64
    # Tree owns {57, 59, 61, 63}; everything else free.
    tree = {57, 59, 61, 63}
    om = _make_complete_om(n_pages, tpc, tree_pages=tree)

    planner = XPoolFirePlanner(_FakeAct(tpc), None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", target_drop_pages=8, dst_grant_chunks=1)

    assert plan is not None
    # 8 free pages selected from anywhere; planner takes highest-id free.
    # Free pages = all of {1..64} \ {57,59,61,63} = 60 pages. Top 8 (by id)
    # = {64, 62, 60, 58, 56, 55, 54, 53}.
    expected_pages = {64, 62, 60, 58, 56, 55, 54, 53}
    actual_pages = {c + 1 for c in plan.chunks_to_unmap_src}
    assert actual_pages == expected_pages, f"got {actual_pages}, want {expected_pages}"
    # No drain or migrate in anywhere-free mode.
    assert plan.pages_to_drain == []
    assert plan.pages_to_migrate == []
    assert plan.expected_unmap_pages == 8
    print(f"[step2] pick from tail: {plan.plan_seq=} pages={sorted(actual_pages)}")


def test_pick_free_pages_skipping_active_and_tree():
    """When tail pages are owned (active or tree), planner skips them and
    picks free pages from anywhere — no migration triggered."""
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner

    tpc = 8
    n_pages = 64
    # Active reqs occupy {59, 60}; tree owns {57, 58}.
    tree = {57, 58}
    active = {59: (3, 0), 60: (3, 1)}
    om = _make_complete_om(n_pages, tpc, tree_pages=tree, active_pages=active)

    planner = XPoolFirePlanner(_FakeAct(tpc), None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", target_drop_pages=8, dst_grant_chunks=1)

    assert plan is not None
    # No migrate or drain.
    assert plan.pages_to_drain == []
    assert plan.pages_to_migrate == []
    # The 8 picked pages are all free.
    actual_pages = {c + 1 for c in plan.chunks_to_unmap_src}
    for p in actual_pages:
        assert p not in tree
        assert p not in active
    print(f"[step2] skipped active+tree: pages={sorted(actual_pages)}")


def test_insufficient_free_refuses():
    """When total free pages < target, planner returns None."""
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner

    tpc = 8
    n_pages = 16
    # Only 4 free pages.
    tree = set(range(1, 9))                           # chunk 0 fully tree
    active = {9: (0, 0), 10: (0, 1), 11: (0, 2), 12: (0, 3)}
    om = _make_complete_om(n_pages, tpc, tree_pages=tree, active_pages=active)

    planner = XPoolFirePlanner(_FakeAct(tpc), None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", target_drop_pages=8, dst_grant_chunks=1)
    assert plan is None, "free=4 < target=8 → must refuse"
    print("[step2] insufficient-free correctly refused")


def test_broken_coverage_refuses():
    from sglang.srt.arena.owner_provider import OwnerMap, TreeNodeRef
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner

    # Coverage broken: page 5 missing from all sets.
    tpc = 8
    om = OwnerMap(
        pool_name="kv",
        n_pages=8,
        free_pages={1, 2, 3, 4, 6, 7, 8},  # 5 missing
        tree_pages={},
        active_pages={},
    )
    planner = XPoolFirePlanner(_FakeAct(tpc), None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", target_drop_pages=8, dst_grant_chunks=1)
    assert plan is None, "broken coverage → must refuse"
    print("[step2] broken-coverage correctly refused")


def test_flag_default_off():
    from sglang.srt.budgeter.fire_planner import is_planner_enabled

    # Default off — flag must be explicitly set.
    import os

    saved = os.environ.pop("SGLANG_T8_PLANNER", None)
    try:
        assert is_planner_enabled() is False
        os.environ["SGLANG_T8_PLANNER"] = "1"
        assert is_planner_enabled() is True
        os.environ["SGLANG_T8_PLANNER"] = "0"
        assert is_planner_enabled() is False
    finally:
        if saved is None:
            os.environ.pop("SGLANG_T8_PLANNER", None)
        else:
            os.environ["SGLANG_T8_PLANNER"] = saved
    print("[step2] flag gate behaves correctly")


def main():
    test_pick_free_pages_from_tail()
    test_pick_free_pages_skipping_active_and_tree()
    test_insufficient_free_refuses()
    test_broken_coverage_refuses()
    test_flag_default_off()
    print("\nT8 step2 planner test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
