"""T8 step 4 — SchedulerOwnerProvider tests.

Builds fake scheduler-shaped object graphs (ducktyped) and verifies
the provider walks them into a coverage-complete OwnerMap.

Cases:
  - Empty engine (no reqs, empty tree)         → all pages free
  - Single req in running batch                → its pages → active
  - Tree node with cached prefix               → pages → tree
  - Active req's prefix shared with tree node  → pages → active (not tree)
  - Pre-existing capped pages mid-fire         → pages → capped
  - Coverage assert: synthetic gap (page 5 unowned) trips
"""

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


# ---- Fake scheduler shapes ----------------------------------------------


def _make_allocator(n_pages, free, *, release=(), capped=()):
    return SimpleNamespace(
        size=n_pages,
        free_pages=torch.tensor(list(free), dtype=torch.int64) if free else torch.tensor([], dtype=torch.int64),
        release_pages=torch.tensor(list(release), dtype=torch.int64) if release else torch.tensor([], dtype=torch.int64),
        _capped_pages=torch.tensor(list(capped), dtype=torch.int64) if capped else torch.tensor([], dtype=torch.int64),
    )


def _make_req(req_pool_idx, seqlen, *, fill_ids=None):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        seqlen=seqlen,
        fill_ids=fill_ids if fill_ids is not None else list(range(seqlen)),
    )


def _make_tree_root(*node_pages_lists):
    """Build a degenerate radix tree: root with N children, each child
    has a `value` tensor of page-ids. No deeper nesting needed for unit tests."""
    children = {}
    for i, pages in enumerate(node_pages_lists):
        children[i] = SimpleNamespace(
            children={},
            value=torch.tensor(pages, dtype=torch.int64),
        )
    return SimpleNamespace(children=children, value=None)


def _make_scheduler(*, allocator, running_reqs=(), waiting_queue=(), tree_root=None,
                    max_ctx=1024):
    rt_pool = SimpleNamespace(
        req_to_token=torch.zeros((16, max_ctx), dtype=torch.int32),
    )
    # Write each running req's seqlen pages into req_to_token.
    for r in running_reqs:
        idx = r.req_pool_idx
        ids = list(range(1, r.seqlen + 1))  # arbitrary placeholder; tests override
        rt_pool.req_to_token[idx, : r.seqlen] = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=rt_pool,
        running_batch=SimpleNamespace(reqs=list(running_reqs)),
        waiting_queue=list(waiting_queue),
        tree_cache=SimpleNamespace(root_node=tree_root) if tree_root else None,
    )


# ---- Tests --------------------------------------------------------------


def test_empty_engine_all_free():
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    n = 16
    alloc = _make_allocator(n, free=range(1, n + 1))
    sched = _make_scheduler(allocator=alloc)
    om = SchedulerOwnerProvider(sched).build_kv_owner_map()

    assert om.n_pages == n
    assert len(om.free_pages) == n
    assert om.tree_pages == {}
    assert om.active_pages == {}
    assert om.capped_pages == set()
    om.assert_complete()
    print(f"[step4] empty engine: free={len(om.free_pages)}")


def test_single_running_req():
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    n = 16
    # Pages 1..10 belong to a running req (req_pool_idx=3, seqlen=10).
    # Pages 11..16 free.
    alloc = _make_allocator(n, free=range(11, n + 1))
    req = _make_req(3, 10)
    sched = _make_scheduler(allocator=alloc, running_reqs=[req])
    # Override req_to_token row 3 with pages 1..10.
    sched.req_to_token_pool.req_to_token[3, :10] = torch.arange(1, 11, dtype=torch.int32)

    om = SchedulerOwnerProvider(sched).build_kv_owner_map()
    assert len(om.active_pages) == 10
    for p in range(1, 11):
        assert om.active_pages[p][0] == 3, f"page {p} should belong to req 3"
    om.assert_complete()
    print(f"[step4] single running req: active={len(om.active_pages)} free={len(om.free_pages)}")


def test_tree_node_cached_prefix():
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    n = 16
    # Pages 1..6 owned by tree node, 7..16 free.
    alloc = _make_allocator(n, free=range(7, n + 1))
    tree = _make_tree_root([1, 2, 3, 4, 5, 6])
    sched = _make_scheduler(allocator=alloc, tree_root=tree)

    om = SchedulerOwnerProvider(sched).build_kv_owner_map()
    assert len(om.tree_pages) == 6
    for p in range(1, 7):
        assert p in om.tree_pages
    om.assert_complete()
    print(f"[step4] tree-only prefix: tree={len(om.tree_pages)} free={len(om.free_pages)}")


def test_dual_owned_resolves_to_active():
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    n = 16
    # Pages 1..10 are simultaneously held by an active req AND by a
    # tree node (sglang's prefix-lock case). Provider must classify
    # them as ACTIVE — migrating is the only safe op.
    alloc = _make_allocator(n, free=range(11, n + 1))
    req = _make_req(3, 10)
    tree = _make_tree_root([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sched = _make_scheduler(allocator=alloc, running_reqs=[req], tree_root=tree)
    sched.req_to_token_pool.req_to_token[3, :10] = torch.arange(1, 11, dtype=torch.int32)

    om = SchedulerOwnerProvider(sched).build_kv_owner_map()
    for p in range(1, 11):
        assert p in om.active_pages, f"page {p} should be active (not tree)"
        assert p not in om.tree_pages, f"page {p} must not be in tree (active wins)"
    assert len(om.tree_pages) == 0
    om.assert_complete()
    print(
        f"[step4] dual-owned → active: active={len(om.active_pages)} tree={len(om.tree_pages)}"
    )


def test_capped_pages_picked_up():
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    n = 16
    # 4 capped (mid-fire), 12 free.
    alloc = _make_allocator(n, free=range(1, 13), capped=[13, 14, 15, 16])
    sched = _make_scheduler(allocator=alloc)
    om = SchedulerOwnerProvider(sched).build_kv_owner_map()
    assert om.capped_pages == {13, 14, 15, 16}
    om.assert_complete()
    print(f"[step4] capped picked up: capped={len(om.capped_pages)}")


def test_coverage_break_trips_assert():
    from sglang.srt.arena.owner_provider import OwnerMap

    # Page 5 missing from every set.
    om = OwnerMap(
        pool_name="kv",
        n_pages=8,
        free_pages={1, 2, 3, 4, 6, 7, 8},
        tree_pages={},
        active_pages={},
        capped_pages=set(),
    )
    raised = False
    try:
        om.assert_complete()
    except RuntimeError as e:
        raised = True
        assert "coverage broken" in str(e)
    assert raised
    print("[step4] coverage assert correctly trips on gap")


def test_implements_protocol():
    from sglang.srt.arena.owner_provider import OwnerProvider
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    p = SchedulerOwnerProvider.__new__(SchedulerOwnerProvider)
    assert isinstance(p, OwnerProvider), "must structurally satisfy OwnerProvider"
    print("[step4] structural Protocol check passes")


def main():
    test_empty_engine_all_free()
    test_single_running_req()
    test_tree_node_cached_prefix()
    test_dual_owned_resolves_to_active()
    test_capped_pages_picked_up()
    test_coverage_break_trips_assert()
    test_implements_protocol()
    print("\nT8 step4 owner-provider test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
