"""T8 step 3 — CrossPoolTransferActuator.execute(plan) unit tests.

Mocks the arena + allocator + actuator interfaces just enough to drive
execute() through its 6 steps. Verifies:
  - happy path with empty drain/migrate runs cap → unmap → map → uncap
  - non-empty drain without callback raises (no silent no-op)
  - non-empty migrate without callback raises (no silent no-op)
  - verify-step abort path: if free_pages still has capped-range pages
    after drain/migrate, executor unmark_pages_capped + returns aborted
  - drain/migrate callbacks ARE called with the correct lists
"""

import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _make_actuator():
    """Build a CrossPoolTransferActuator with everything mocked.

    Bypasses __init__ since the real constructor needs MultiTensorArena
    and SharedHandlePool instances. We set the attributes execute()
    actually reads.
    """
    from sglang.srt.arena.cross_pool_actuator import CrossPoolTransferActuator

    a = CrossPoolTransferActuator.__new__(CrossPoolTransferActuator)

    # Build a fake KV / mamba arena pair. Only methods execute() touches.
    kv_arena_inner = MagicMock()
    kv_arena_inner.shrink_explicit = MagicMock(return_value=8)  # per-subpool unmapped pages
    mamba_arena_inner = MagicMock()
    mamba_arena_inner.grow = MagicMock(return_value=1)  # per-subpool granted chunks

    kv = MagicMock()
    kv._arena = kv_arena_inner
    kv.tokens_per_chunk = 8
    kv.n_layers = 1
    kv.n_kinds = 2  # k + v subpools
    kv._pool_name = lambda i: f"kv_{i}"

    mamba = MagicMock()
    mamba._arena = mamba_arena_inner
    mamba.tokens_per_chunk = 1
    mamba.n_layers = 1
    mamba.n_kinds = 1
    mamba._pool_name = lambda i: f"m_{i}"

    a.kv = kv
    a.mamba = mamba

    # Allocator stub with the methods execute() exercises.
    alloc = MagicMock()
    alloc.device = "cpu"
    alloc.free_pages = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    alloc.mark_pages_capped = MagicMock(return_value=2)
    alloc.unmark_pages_capped = MagicMock(return_value=0)

    kv_act = MagicMock()
    kv_act.allocator = alloc
    kv_act.live_capacity_tokens = MagicMock(return_value=64)
    kv_act.cap_allocator_only = MagicMock()

    mamba_act = MagicMock()
    mamba_act.allocator = MagicMock()
    mamba_act.allocator.device = "cpu"
    mamba_act.allocator.free_pages = torch.tensor([], dtype=torch.int64)
    mamba_act.allocator.mark_pages_capped = MagicMock(return_value=0)
    mamba_act.allocator.unmark_pages_capped = MagicMock(return_value=0)
    mamba_act.live_capacity_tokens = MagicMock(return_value=8)
    mamba_act.cap_allocator_only = MagicMock()

    a.kv_actuator = kv_act
    a.mamba_actuator = mamba_act
    a.shared = MagicMock()

    return a, alloc, kv_act, mamba_act, kv_arena_inner, mamba_arena_inner


def _make_plan(*, drain=(), migrate=()):
    from sglang.srt.arena.fire_plan import FirePlan, MigrateOp

    return FirePlan(
        direction="kv_to_mamba",
        capped_page_range=(57, 65),
        chunks_to_unmap_src=[7],
        pages_to_drain=list(drain),
        pages_to_migrate=list(migrate),
        chunks_to_map_dst=1,
        expected_unmap_pages=8,
        plan_seq=42,
    )


def test_happy_path_no_drain_no_migrate():
    a, alloc, kv_act, mamba_act, kv_inner, mamba_inner = _make_actuator()
    plan = _make_plan()

    res = a.execute(plan)

    assert res.aborted is False
    assert res.plan_seq == 42
    assert res.direction == "kv_to_mamba"

    # cap-barrier was called with the planner's specific page list.
    # plan.chunks_to_unmap_src=[7] → page-id 7+1=8 (with 1-indexed offset).
    args, _ = alloc.mark_pages_capped.call_args
    capped_arg = args[0]
    assert capped_arg.tolist() == [8]

    # shrink_explicit invoked once per src subpool (k + v = 2).
    assert kv_inner.shrink_explicit.call_count == 2
    for call in kv_inner.shrink_explicit.call_args_list:
        assert call.args[1] == [7]
    # 2 subpools * 8 pages each = 16 unmapped.
    assert res.unmapped_pages == 16

    # grow invoked once per dst subpool (mamba: 1 subpool).
    assert mamba_inner.grow.call_count == 1
    assert res.granted_chunks == 1

    # uncap dst.
    mamba_act.cap_allocator_only.assert_called_once()

    # No drain/migrate calls.
    assert res.drained_pages == 0
    assert res.migrated_pages == 0
    print(f"[step3] happy path: unmapped={res.unmapped_pages} granted={res.granted_chunks}")


def test_drain_without_callback_raises():
    a, _, _, _, _, _ = _make_actuator()
    plan = _make_plan(drain=[57, 58])

    raised = False
    try:
        a.execute(plan)
    except RuntimeError as e:
        raised = True
        assert "pages_to_drain" in str(e)
        assert "no drain_callback" in str(e)
    assert raised, "non-empty drain without callback must raise"
    print("[step3] drain-without-callback correctly raised")


def test_migrate_without_callback_raises():
    from sglang.srt.arena.fire_plan import MigrateOp

    a, _, _, _, _, _ = _make_actuator()
    op = MigrateOp(src_page=58, dst_page=1, req_pool_idx=0, slot_in_req=0)
    plan = _make_plan(migrate=[op])

    raised = False
    try:
        a.execute(plan)
    except RuntimeError as e:
        raised = True
        assert "pages_to_migrate" in str(e)
        assert "no migrate_callback" in str(e)
    assert raised, "non-empty migrate without callback must raise"
    print("[step3] migrate-without-callback correctly raised")


def test_callbacks_invoked_with_correct_lists():
    from sglang.srt.arena.fire_plan import MigrateOp

    a, alloc, _, _, _, _ = _make_actuator()
    # Free_pages is a torch tensor without any capped-range pages so
    # verify passes after callbacks return.
    alloc.free_pages = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    op1 = MigrateOp(src_page=58, dst_page=1, req_pool_idx=0, slot_in_req=0)
    op2 = MigrateOp(src_page=60, dst_page=2, req_pool_idx=1, slot_in_req=3)
    plan = _make_plan(drain=[57, 59], migrate=[op1, op2])

    drain_called_with = []
    migrate_called_with = []

    def drain_cb(pages):
        drain_called_with.extend(pages)
        return len(pages)

    def migrate_cb(ops):
        migrate_called_with.extend(ops)
        return len(ops)

    res = a.execute(plan, drain_callback=drain_cb, migrate_callback=migrate_cb)

    assert drain_called_with == [57, 59]
    assert migrate_called_with == [op1, op2]
    assert res.drained_pages == 2
    assert res.migrated_pages == 2
    assert res.aborted is False
    print(
        f"[step3] callbacks invoked: drain={res.drained_pages} "
        f"migrate={res.migrated_pages}"
    )


def test_verify_aborts_when_target_page_still_in_free():
    a, alloc, _, _, kv_inner, _ = _make_actuator()
    # Stage: even after cap-barrier (mocked to no-op on free_pages here),
    # target page 8 (= chunk 7 + 1) is still in free_pages. The verifier
    # must detect this and abort.
    alloc.free_pages = torch.tensor([1, 2, 8], dtype=torch.int64)

    plan = _make_plan()  # chunks_to_unmap_src=[7] → target page 8

    res = a.execute(plan)
    assert res.aborted is True
    assert "verify failed" in res.abort_reason
    assert kv_inner.shrink_explicit.call_count == 0
    alloc.unmark_pages_capped.assert_called_once()
    print(f"[step3] verify abort: {res.abort_reason}")


def main():
    test_happy_path_no_drain_no_migrate()
    test_drain_without_callback_raises()
    test_migrate_without_callback_raises()
    test_callbacks_invoked_with_correct_lists()
    test_verify_aborts_when_target_page_still_in_free()
    print("\nT8 step3 executor test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
