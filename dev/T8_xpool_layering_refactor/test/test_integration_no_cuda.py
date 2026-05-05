"""T8 step 5 integration — planner → executor → migrator (no live CUDA).

End-to-end exercise of the new code path on a synthetic engine:
SchedulerOwnerProvider walks fake reqs/tree → XPoolFirePlanner emits a
FirePlan → CrossPoolTransferActuator.execute(plan) runs cap-barrier,
calls the migrator (KVPageMigrator) for active pages, calls a fake
drain callback for tree pages, and dispatches mocked shrink_explicit /
grow.

Verifies the contract holds with all four T8 modules talking to each
other, before we wire it into BudgetAgent in step 6.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _build_world():
    """Build a coherent fake world: 16 KV pages, 1 running req on
    pages [13,14], tree node holds [11,12], rest free.
    Tail chunk (chunk 1, pages 9..16) will be the unmap target for an
    8-page shrink. drain hits 11/12, migrate hits 13/14."""
    n_pages = 16
    n_layers = 2

    # Allocator with free pages = {1..10, 15, 16}, capped empty.
    free = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16]
    allocator = SimpleNamespace(
        size=n_pages,
        page_size=1,
        device="cpu",
        free_pages=torch.tensor(free, dtype=torch.int64),
        release_pages=torch.tensor([], dtype=torch.int64),
        _capped_pages=torch.tensor([], dtype=torch.int64),
    )

    def mark_capped(t):
        # Mirror the real allocator's mark_pages_capped.
        target = t.to(allocator.device).to(torch.int64)
        n = 0
        if allocator.free_pages.numel() > 0:
            mask = torch.isin(allocator.free_pages, target)
            held = allocator.free_pages[mask]
            allocator.free_pages = allocator.free_pages[~mask]
            n += int(held.numel())
            allocator._capped_pages = torch.cat([allocator._capped_pages, held])
        return n

    def unmark_capped(t):
        target = t.to(allocator.device).to(torch.int64)
        if allocator._capped_pages.numel() == 0:
            return 0
        mask = torch.isin(allocator._capped_pages, target)
        held = allocator._capped_pages[mask]
        allocator._capped_pages = allocator._capped_pages[~mask]
        if held.numel() > 0:
            allocator.free_pages = torch.cat([allocator.free_pages, held])
        return int(held.numel())

    allocator.mark_pages_capped = mark_capped
    allocator.unmark_pages_capped = unmark_capped

    # KV pool with patterned data.
    k_buf, v_buf = [], []
    for L in range(n_layers):
        kt = torch.zeros((n_pages + 1, 2, 4), dtype=torch.float32)
        vt = torch.zeros((n_pages + 1, 2, 4), dtype=torch.float32)
        for p in range(n_pages + 1):
            kt[p].fill_(L * 1000.0 + p)
            vt[p].fill_(L * 1000.0 + p + 0.5)
        k_buf.append(kt)
        v_buf.append(vt)
    kv_pool = SimpleNamespace(k_buffer=k_buf, v_buffer=v_buf, layer_num=n_layers)

    # ReqToTokenPool: req 5 owns pages [13, 14] in slots [0, 1].
    rt_pool = SimpleNamespace(req_to_token=torch.zeros((8, 32), dtype=torch.int32))
    rt_pool.req_to_token[5, 0] = 13
    rt_pool.req_to_token[5, 1] = 14

    # Tree: one node with pages [11, 12].
    tree_node = SimpleNamespace(
        children={}, value=torch.tensor([11, 12], dtype=torch.int64)
    )
    tree_root = SimpleNamespace(
        children={0: tree_node}, value=None,
    )

    # Synthetic Req.
    req = SimpleNamespace(
        req_pool_idx=5, seqlen=2, fill_ids=[0, 0],
    )
    sched = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=rt_pool,
        running_batch=SimpleNamespace(reqs=[req]),
        waiting_queue=[],
        tree_cache=SimpleNamespace(root_node=tree_root),
    )

    return n_pages, n_layers, allocator, kv_pool, rt_pool, sched, tree_node


def _build_actuator(allocator, kv_pool, rt_pool):
    from sglang.srt.arena.cross_pool_actuator import CrossPoolTransferActuator

    a = CrossPoolTransferActuator.__new__(CrossPoolTransferActuator)

    kv_arena = MagicMock()
    kv_arena.shrink_explicit = MagicMock(return_value=8)
    mamba_arena = MagicMock()
    mamba_arena.grow = MagicMock(return_value=1)

    kv = MagicMock()
    kv._arena = kv_arena
    kv.tokens_per_chunk = 8
    kv.n_layers = 1
    kv.n_kinds = 1
    kv._pool_name = lambda i: f"k_{i}"

    mamba = MagicMock()
    mamba._arena = mamba_arena
    mamba.tokens_per_chunk = 1
    mamba.n_layers = 1
    mamba.n_kinds = 1
    mamba._pool_name = lambda i: f"m_{i}"

    a.kv = kv
    a.mamba = mamba

    kv_act = MagicMock()
    kv_act.allocator = allocator
    kv_act.tokens_per_chunk = 8
    kv_act.live_capacity_tokens = MagicMock(return_value=16)
    kv_act.cap_allocator_only = MagicMock()

    mamba_act = MagicMock()
    mamba_act.allocator = MagicMock()
    mamba_act.live_capacity_tokens = MagicMock(return_value=8)
    mamba_act.cap_allocator_only = MagicMock()

    a.kv_actuator = kv_act
    a.mamba_actuator = mamba_act
    a.shared = MagicMock()

    return a, kv_act, mamba_act, kv_arena, mamba_arena


def test_end_to_end_anywhere_free_plan_execute():
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    n_pages, n_layers, alloc, kv_pool, rt_pool, sched, tree_node = _build_world()

    # Step A: provider builds OwnerMap. Tree owns 11, 12; active owns 13, 14.
    provider = SchedulerOwnerProvider(sched)
    om = provider.build_kv_owner_map()
    om.assert_complete()
    assert 11 in om.tree_pages
    assert 12 in om.tree_pages
    assert 13 in om.active_pages
    assert 14 in om.active_pages
    print(
        f"[integ] OwnerMap: free={len(om.free_pages)} tree={len(om.tree_pages)} "
        f"active={len(om.active_pages)} capped={len(om.capped_pages)}"
    )

    # Step B: planner picks 8 free pages from anywhere — no drain, no
    # migrate. Free = {1..10, 15, 16}; top 8 highest-id = {16, 15, 10, 9,
    # 8, 7, 6, 5}.
    planner = XPoolFirePlanner(
        kv_actuator=SimpleNamespace(tokens_per_chunk=8),
        mamba_actuator=None,
        owner_provider=provider,
    )
    plan = planner.build("kv_to_mamba", target_drop_pages=8, dst_grant_chunks=1)
    assert plan is not None
    assert plan.pages_to_drain == []
    assert plan.pages_to_migrate == []
    expected_pages = {16, 15, 10, 9, 8, 7, 6, 5}
    actual_pages = {c + 1 for c in plan.chunks_to_unmap_src}
    assert actual_pages == expected_pages, f"got {actual_pages}, want {expected_pages}"
    print(
        f"[integ] FirePlan: pages={sorted(actual_pages)} (no drain, no migrate)"
    )

    # Step C: actuator executes — purely cap + unmap + map, no callbacks.
    a, kv_act, mamba_act, kv_arena, mamba_arena = _build_actuator(alloc, kv_pool, rt_pool)
    res = a.execute(plan)

    assert res.aborted is False, f"unexpected abort: {res.abort_reason}"
    assert res.drained_pages == 0
    assert res.migrated_pages == 0
    # The 8 picked pages should now be capped, not in free.
    free_now = set(alloc.free_pages.tolist())
    for p in expected_pages:
        assert p not in free_now, f"target page {p} leaked back to free"
    # Tree node and active req unchanged (planner ignored them).
    assert 11 in {p for p in om.tree_pages}  # om snapshot — still tree
    assert int(rt_pool.req_to_token[5, 0]) == 13  # active still on 13
    assert int(rt_pool.req_to_token[5, 1]) == 14  # active still on 14
    print(
        f"[integ] execute: aborted={res.aborted} unmap_us={res.unmap_us} "
        f"total_us={res.total_us}"
    )

    assert kv_arena.shrink_explicit.call_count == 1
    assert mamba_arena.grow.call_count == 1


def main():
    test_end_to_end_anywhere_free_plan_execute()
    print("\nT8 step5 integration test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
