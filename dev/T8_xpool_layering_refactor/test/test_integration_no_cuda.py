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


def test_end_to_end_plan_execute_migrate():
    from sglang.srt.arena.kv_migrator import KVPageMigrator
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    n_pages, n_layers, alloc, kv_pool, rt_pool, sched, tree_node = _build_world()

    # Step A: provider builds OwnerMap.
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

    # Step B: planner builds a FirePlan that targets the tail chunk.
    planner = XPoolFirePlanner(
        kv_actuator=SimpleNamespace(tokens_per_chunk=8),
        mamba_actuator=None,
        owner_provider=provider,
    )
    plan = planner.build("kv_to_mamba", target_drop_pages=8, dst_grant_chunks=1)
    assert plan is not None
    assert plan.chunks_to_unmap_src == [1]
    assert plan.capped_page_range == (9, 17)
    assert sorted(plan.pages_to_drain) == [11, 12]
    assert len(plan.pages_to_migrate) == 2
    migrated_srcs = sorted(op.src_page for op in plan.pages_to_migrate)
    assert migrated_srcs == [13, 14]
    # dst pages must come from below cap_low=9 → free pages {1..10}
    # (sorted asc, planner takes from head).
    for op in plan.pages_to_migrate:
        assert op.dst_page < 9
        assert op.dst_page in {1, 2, 3, 4, 5, 6, 7, 8}
    print(
        f"[integ] FirePlan: cap=[{plan.capped_page_range[0]},{plan.capped_page_range[1]}) "
        f"drain={plan.pages_to_drain} migrate=[{','.join(f'{op.src_page}->{op.dst_page}' for op in plan.pages_to_migrate)}]"
    )

    # Step C: actuator executes plan with real migrator + fake drain cb.
    a, kv_act, mamba_act, kv_arena, mamba_arena = _build_actuator(alloc, kv_pool, rt_pool)
    migrator = KVPageMigrator(kv_pool, rt_pool, alloc)

    # Drain callback: simulates tree_cache.evict_node by moving the tree
    # pages out of the tree. After eviction, tree pages return to allocator
    # — they were already capped by cap-barrier, so they should land
    # back in capped_pages (or at least not in free_pages). For this
    # integration test we just confirm the callback was invoked with
    # the right list and we manually mark them capped to reflect
    # reality (the real path goes through allocator.free which routes
    # to capped if cap is set).
    drain_seen = []

    def drain_cb(pages):
        drain_seen.extend(pages)
        # Mark them capped to simulate eviction-into-capped state.
        t = torch.tensor(pages, dtype=torch.int64)
        alloc.mark_pages_capped(t)
        return len(pages)

    res = a.execute(plan, drain_callback=drain_cb, migrate_callback=migrator.migrate)

    assert res.aborted is False, f"unexpected abort: {res.abort_reason}"
    assert res.drained_pages == 2
    assert res.migrated_pages == 2
    assert sorted(drain_seen) == [11, 12]
    # KV data verify: dst page now contains src's pattern.
    op13 = next(o for o in plan.pages_to_migrate if o.src_page == 13)
    op14 = next(o for o in plan.pages_to_migrate if o.src_page == 14)
    for L in range(n_layers):
        assert torch.allclose(
            kv_pool.k_buffer[L][op13.dst_page],
            torch.full_like(kv_pool.k_buffer[L][op13.dst_page], L * 1000.0 + 13),
        )
        assert torch.allclose(
            kv_pool.v_buffer[L][op14.dst_page],
            torch.full_like(kv_pool.v_buffer[L][op14.dst_page], L * 1000.0 + 14 + 0.5),
        )
    # req_to_token rewritten: req 5 slots 0,1 now point to dst pages.
    assert int(rt_pool.req_to_token[5, 0]) == op13.dst_page
    assert int(rt_pool.req_to_token[5, 1]) == op14.dst_page
    # dst pages claimed (not in free_pages anymore).
    free_now = set(alloc.free_pages.tolist())
    assert op13.dst_page not in free_now
    assert op14.dst_page not in free_now
    # capped contains the originally capped range pages MINUS what
    # the drain put back PLUS what unmark would have done. Easier:
    # check that the capped range pages are not in free_pages.
    for p in range(9, 17):
        assert p not in free_now, f"capped page {p} leaked back to free"
    print(
        f"[integ] execute: aborted={res.aborted} drained={res.drained_pages} "
        f"migrated={res.migrated_pages} unmap_us={res.unmap_us} total_us={res.total_us}"
    )

    # Mock arena calls happened.
    assert kv_arena.shrink_explicit.call_count == 1
    assert mamba_arena.grow.call_count == 1


def main():
    test_end_to_end_plan_execute_migrate()
    print("\nT8 step5 integration test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
