"""T8 step 5 — KVPageMigrator tests.

CPU-tensor stand-ins for k_buffer/v_buffer + req_to_token + allocator.
Verifies:
  - K/V data correctly copied src→dst across all layers
  - req_to_token entries updated to dst pages
  - dst pages removed from allocator.free_pages
  - single fire with mixed src/dst pages preserves data integrity
  - planner-bug case (dst not in free) raises loudly
"""

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _make_kv_pool(n_layers=3, n_pages=16, head_dim=4, head_num=2):
    """K/V buffers shape [n_pages+1, head_num, head_dim] per layer.
    Page 0 is the null sentinel. Fill each page with an identifiable
    pattern: page p in layer L has values (L * 1000 + p)."""
    k_buf, v_buf = [], []
    for L in range(n_layers):
        kt = torch.zeros((n_pages + 1, head_num, head_dim), dtype=torch.float32)
        vt = torch.zeros((n_pages + 1, head_num, head_dim), dtype=torch.float32)
        for p in range(n_pages + 1):
            kt[p].fill_(L * 1000.0 + p)
            vt[p].fill_(L * 1000.0 + p + 0.5)
        k_buf.append(kt)
        v_buf.append(vt)
    return SimpleNamespace(k_buffer=k_buf, v_buffer=v_buf, layer_num=n_layers)


def _make_rt_pool(n_reqs=8, max_ctx=32):
    return SimpleNamespace(
        req_to_token=torch.zeros((n_reqs, max_ctx), dtype=torch.int32),
    )


def _make_alloc(n_pages=16, free=None):
    if free is None:
        free = list(range(1, n_pages + 1))
    return SimpleNamespace(
        size=n_pages,
        free_pages=torch.tensor(free, dtype=torch.int64),
        release_pages=torch.tensor([], dtype=torch.int64),
    )


def test_migrate_copies_kv_and_updates_req_to_token():
    from sglang.srt.arena.fire_plan import MigrateOp
    from sglang.srt.arena.kv_migrator import KVPageMigrator

    n_pages = 16
    pool = _make_kv_pool(n_layers=3, n_pages=n_pages)
    rt = _make_rt_pool(n_reqs=8, max_ctx=32)
    # Req 5 holds pages [10, 13, 14] in slots [0, 1, 2].
    rt.req_to_token[5, 0] = 10
    rt.req_to_token[5, 1] = 13
    rt.req_to_token[5, 2] = 14

    # alloc: pages 1..3 are free dst candidates, 10/13/14 not free (active).
    free = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 16]
    alloc = _make_alloc(n_pages=n_pages, free=free)

    mig = KVPageMigrator(pool, rt, alloc)

    ops = [
        MigrateOp(src_page=10, dst_page=1, req_pool_idx=5, slot_in_req=0),
        MigrateOp(src_page=13, dst_page=2, req_pool_idx=5, slot_in_req=1),
        MigrateOp(src_page=14, dst_page=3, req_pool_idx=5, slot_in_req=2),
    ]
    n = mig.migrate(ops)
    assert n == 3

    # Check K/V data: dst page should now contain src's pattern.
    for L in range(3):
        assert torch.allclose(
            pool.k_buffer[L][1], torch.full_like(pool.k_buffer[L][1], L * 1000.0 + 10)
        ), f"layer {L} K dst=1 wrong"
        assert torch.allclose(
            pool.v_buffer[L][2], torch.full_like(pool.v_buffer[L][2], L * 1000.0 + 13 + 0.5)
        ), f"layer {L} V dst=2 wrong"
        assert torch.allclose(
            pool.k_buffer[L][3], torch.full_like(pool.k_buffer[L][3], L * 1000.0 + 14)
        ), f"layer {L} K dst=3 wrong"

    # Check req_to_token rewritten.
    assert int(rt.req_to_token[5, 0]) == 1
    assert int(rt.req_to_token[5, 1]) == 2
    assert int(rt.req_to_token[5, 2]) == 3

    # Check dst pages removed from allocator.free_pages.
    free_after = set(alloc.free_pages.tolist())
    assert 1 not in free_after
    assert 2 not in free_after
    assert 3 not in free_after
    # Other free pages still there.
    assert 4 in free_after
    assert 16 in free_after
    print(
        f"[step5] migrate: 3 ops, {len(ops)*pool.layer_num*2} D2D copies, "
        f"req_to_token + free_pages updated"
    )


def test_dst_not_in_free_raises():
    from sglang.srt.arena.fire_plan import MigrateOp
    from sglang.srt.arena.kv_migrator import KVPageMigrator

    n_pages = 16
    pool = _make_kv_pool(n_layers=2, n_pages=n_pages)
    rt = _make_rt_pool()
    # alloc has dst=1 NOT free (planner bug — should never happen if
    # planner was honest, but we want to catch the bug if it does).
    free = [4, 5, 6, 7, 8]
    alloc = _make_alloc(n_pages=n_pages, free=free)

    ops = [MigrateOp(src_page=10, dst_page=1, req_pool_idx=0, slot_in_req=0)]

    mig = KVPageMigrator(pool, rt, alloc)
    raised = False
    try:
        mig.migrate(ops)
    except RuntimeError as e:
        raised = True
        assert "planner claimed" in str(e), f"unexpected msg: {e}"
        assert "wasn't free" in str(e)
    assert raised, "dst-not-in-free must raise"
    print("[step5] dst-not-in-free correctly raised")


def test_empty_ops_is_noop():
    from sglang.srt.arena.kv_migrator import KVPageMigrator

    pool = _make_kv_pool(n_layers=2, n_pages=8)
    rt = _make_rt_pool()
    alloc = _make_alloc(n_pages=8)

    mig = KVPageMigrator(pool, rt, alloc)
    n = mig.migrate([])
    assert n == 0
    # alloc.free_pages unchanged.
    assert alloc.free_pages.tolist() == list(range(1, 9))
    print("[step5] empty ops no-op OK")


def test_release_pages_also_searched_for_dst():
    """If a dst page sits in release_pages (released-but-not-yet-merged),
    the migrator must claim it from there too — release_pages count as
    free for our purposes. Otherwise we'd double-hand-out."""
    from sglang.srt.arena.fire_plan import MigrateOp
    from sglang.srt.arena.kv_migrator import KVPageMigrator

    n_pages = 16
    pool = _make_kv_pool(n_layers=2, n_pages=n_pages)
    rt = _make_rt_pool()
    alloc = SimpleNamespace(
        size=n_pages,
        free_pages=torch.tensor([5, 6, 7, 8], dtype=torch.int64),
        release_pages=torch.tensor([1, 2], dtype=torch.int64),
    )

    ops = [
        MigrateOp(src_page=10, dst_page=1, req_pool_idx=0, slot_in_req=0),  # from release
        MigrateOp(src_page=11, dst_page=5, req_pool_idx=0, slot_in_req=1),  # from free
    ]

    mig = KVPageMigrator(pool, rt, alloc)
    n = mig.migrate(ops)
    assert n == 2
    assert 1 not in alloc.release_pages.tolist()
    assert 2 in alloc.release_pages.tolist()  # untouched
    assert 5 not in alloc.free_pages.tolist()
    print("[step5] release_pages claimed correctly")


def main():
    test_migrate_copies_kv_and_updates_req_to_token()
    test_dst_not_in_free_raises()
    test_empty_ops_is_noop()
    test_release_pages_also_searched_for_dst()
    print("\nT8 step5 migrator test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
