"""CrossPoolTransferActuator.execute(plan) tests under the page-only API."""

import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _make_actuator():
    from sglang.srt.arena.cross_pool_actuator import CrossPoolTransferActuator

    a = CrossPoolTransferActuator.__new__(CrossPoolTransferActuator)

    kv_arena_inner = MagicMock()
    kv_arena_inner.shrink_explicit = MagicMock(return_value=2)
    mamba_arena_inner = MagicMock()
    mamba_arena_inner.grow = MagicMock(return_value=1)

    kv = MagicMock()
    kv._arena = kv_arena_inner
    kv.n_layers = 1
    kv.n_kinds = 2
    kv._pool_name = lambda i: f"kv_{i}"

    mamba = MagicMock()
    mamba._arena = mamba_arena_inner
    mamba.n_layers = 1
    mamba.n_kinds = 1
    mamba._pool_name = lambda i: f"m_{i}"

    a.kv = kv
    a.mamba = mamba

    alloc = MagicMock()
    alloc.device = "cpu"
    alloc.free_pages = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    alloc.mark_pages_capped = MagicMock(return_value=8)
    alloc.unmark_pages_capped = MagicMock(return_value=0)

    kv_act = MagicMock()
    kv_act.allocator = alloc
    kv_act.expand_pages_to_token_slots = lambda pages: [
        s for p in pages for s in range(p * 8 + 1, (p + 1) * 8 + 1)
    ]
    kv_act.live_capacity_tokens = MagicMock(return_value=64)
    kv_act.cap_allocator_only = MagicMock()

    mamba_act = MagicMock()
    mamba_act.expand_pages_to_token_slots = lambda pages: list(pages)
    mamba_act.live_capacity_tokens = MagicMock(return_value=8)
    mamba_act.cap_allocator_only = MagicMock()

    a.kv_actuator = kv_act
    a.mamba_actuator = mamba_act
    a.shared = MagicMock()
    return a, alloc, kv_act, mamba_act, kv_arena_inner, mamba_arena_inner


def _make_plan(pages=(7,)):
    from sglang.srt.arena.fire_plan import FirePlan
    return FirePlan(
        direction="kv_to_mamba",
        pages_to_unmap=list(pages),
        pages_to_map_dst=len(pages),
        plan_seq=42,
    )


def test_happy_path():
    a, alloc, kv_act, mamba_act, kv_inner, mamba_inner = _make_actuator()
    plan = _make_plan()

    res = a.execute(plan)
    assert res.aborted is False
    # Cap-barrier called with all token-slots in page 7: [57..65).
    args, _ = alloc.mark_pages_capped.call_args
    assert args[0].tolist() == list(range(57, 65))
    # shrink_explicit called per src subpool (k+v = 2) with [7].
    assert kv_inner.shrink_explicit.call_count == 2
    for call in kv_inner.shrink_explicit.call_args_list:
        assert call.args[1] == [7]
    assert res.unmapped_pages == 4  # 2 subpools × 2 each (return_value=2)
    # grow on dst: 1 subpool, 1 page.
    assert mamba_inner.grow.call_count == 1
    print(f"[step3] happy: unmapped={res.unmapped_pages} granted={res.granted_pages}")


def test_verify_aborts_when_target_slot_in_free():
    a, alloc, _, _, kv_inner, _ = _make_actuator()
    # Page 7 → slots 57..64. Stage one of those still in free_pages.
    alloc.free_pages = torch.tensor([1, 2, 60], dtype=torch.int64)
    plan = _make_plan()
    res = a.execute(plan)
    assert res.aborted is True
    assert "verify failed" in res.abort_reason
    assert kv_inner.shrink_explicit.call_count == 0
    alloc.unmark_pages_capped.assert_called_once()
    print(f"[step3] verify abort: {res.abort_reason}")


def main():
    test_happy_path()
    test_verify_aborts_when_target_slot_in_free()
    print("\nT8 step3 executor test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
