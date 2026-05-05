"""XPoolFirePlanner tests under the chunk-keyed OwnerMap (post-cleanup)."""

import os
import sys
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


class _FakeProvider:
    def __init__(self, kv_om=None, mamba_om=None):
        self._kv = kv_om
        self._mamba = mamba_om
    def build_kv_owner_map(self):
        return self._kv
    def build_mamba_owner_map(self):
        return self._mamba


def _make_om(n_pages, free):
    from sglang.srt.arena.owner_provider import OwnerMap
    return OwnerMap(pool_name="kv", n_pages=n_pages, free_pages=set(free))


def test_pick_tail_when_all_free():
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
    om = _make_om(n_pages=10, free=set(range(10)))
    planner = XPoolFirePlanner(None, None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", n_pages_target=3)
    assert plan is not None
    # Highest-id free pages first → 9, 8, 7. Sorted ascending in plan.
    assert plan.pages_to_unmap == [7, 8, 9]
    assert plan.pages_to_map_dst == 3
    print(f"[step2] tail pick: {plan.pages_to_unmap}")


def test_pick_skips_non_free():
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
    # Pages 7, 9 are not free (e.g., one token-slot in those pages is live);
    # planner skips them.
    om = _make_om(n_pages=10, free={0, 1, 2, 3, 4, 5, 6, 8})
    planner = XPoolFirePlanner(None, None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", n_pages_target=3)
    assert plan is not None
    # Free sorted desc: 8, 6, 5, ... → top 3 = {8, 6, 5}.
    assert plan.pages_to_unmap == [5, 6, 8]
    print(f"[step2] skip non-free: {plan.pages_to_unmap}")


def test_insufficient_free_refuses():
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
    om = _make_om(n_pages=10, free={0, 1})  # only 2 free
    planner = XPoolFirePlanner(None, None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", n_pages_target=3)
    assert plan is None
    print("[step2] insufficient-free correctly refused")


def test_target_exhausts_pool_refuses():
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
    om = _make_om(n_pages=10, free=set(range(10)))
    planner = XPoolFirePlanner(None, None, _FakeProvider(om))
    plan = planner.build("kv_to_mamba", n_pages_target=10)
    assert plan is None
    print("[step2] target=n_pages correctly refused")


def test_flag_default_off():
    from sglang.srt.budgeter.fire_planner import is_planner_enabled, is_executor_enabled
    saved_p = os.environ.pop("SGLANG_T8_PLANNER", None)
    saved_e = os.environ.pop("SGLANG_T8_EXECUTE", None)
    try:
        assert is_planner_enabled() is False
        assert is_executor_enabled() is False
    finally:
        if saved_p is not None:
            os.environ["SGLANG_T8_PLANNER"] = saved_p
        if saved_e is not None:
            os.environ["SGLANG_T8_EXECUTE"] = saved_e
    print("[step2] flag gate OK")


def main():
    test_pick_tail_when_all_free()
    test_pick_skips_non_free()
    test_insufficient_free_refuses()
    test_target_exhausts_pool_refuses()
    test_flag_default_off()
    print("\nT8 step2 planner test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
