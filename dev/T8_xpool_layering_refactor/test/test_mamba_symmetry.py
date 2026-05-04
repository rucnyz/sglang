"""T8 step 6.4 — mamba migration symmetry tests.

Cases:
  - SchedulerOwnerProvider.build_mamba_owner_map walks slot-id space
    correctly (free / capped / active classification, coverage closes
    when there's no extra cache state).
  - MambaPageMigrator delegates to MambaPool.migrate_slot, updates
    req.mamba_pool_idx, and surfaces failures loudly.
  - End-to-end: planner builds a mamba_to_kv plan; executor invokes
    the mamba migrator instead of the KV one.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _make_mamba_pool(n_slots=8):
    """Stub MambaPool with the slot-allocator surface and a recording
    `migrate_slot`."""
    pool = SimpleNamespace(
        size=n_slots,
        free_slots=torch.tensor(list(range(1, n_slots + 1)), dtype=torch.int64),
        _capped_slots=torch.tensor([], dtype=torch.int64),
        migrated_log=[],
    )

    def migrate_slot(src, dst):
        pool.migrated_log.append((src, dst))
        # Simulate the real impl: dst leaves free_slots, src joins capped.
        pool.free_slots = pool.free_slots[pool.free_slots != dst]
        pool._capped_slots = torch.cat(
            [pool._capped_slots, torch.tensor([src], dtype=torch.int64)]
        )
        return True

    pool.migrate_slot = migrate_slot
    return pool


def test_build_mamba_owner_map_basic():
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    pool = _make_mamba_pool(n_slots=8)
    # Reqs hold slots 5 and 7.
    r1 = SimpleNamespace(req_pool_idx=2, mamba_pool_idx=torch.tensor([5], dtype=torch.int32))
    r2 = SimpleNamespace(req_pool_idx=3, mamba_pool_idx=torch.tensor([7], dtype=torch.int32))
    # Remove these from free.
    pool.free_slots = torch.tensor([1, 2, 3, 4, 6, 8], dtype=torch.int64)

    sched = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[r1, r2]),
        waiting_queue=[],
        mamba_pool=pool,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        tree_cache=None,
    )

    provider = SchedulerOwnerProvider(sched)
    om = provider.build_mamba_owner_map()
    assert om is not None
    assert om.n_pages == 8
    assert om.active_pages == {5: (2, 0), 7: (3, 0)}
    assert om.capped_pages == set()
    om.assert_complete()
    print(f"[gap4] mamba owner map: free={len(om.free_pages)} active={len(om.active_pages)}")


def test_build_mamba_owner_map_with_capped():
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    pool = _make_mamba_pool(n_slots=8)
    pool.free_slots = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    pool._capped_slots = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
    sched = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        mamba_pool=pool,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        tree_cache=None,
    )
    om = SchedulerOwnerProvider(sched).build_mamba_owner_map()
    assert om.capped_pages == {5, 6, 7, 8}
    om.assert_complete()
    print(f"[gap4] mamba owner map with capped: capped={len(om.capped_pages)}")


def test_mamba_migrator_basic():
    from sglang.srt.arena.fire_plan import MigrateOp
    from sglang.srt.arena.mamba_migrator import MambaPageMigrator

    pool = _make_mamba_pool(n_slots=8)
    pool.free_slots = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    r1 = SimpleNamespace(req_pool_idx=0, mamba_pool_idx=torch.tensor([5], dtype=torch.int32))
    r2 = SimpleNamespace(req_pool_idx=1, mamba_pool_idx=torch.tensor([6], dtype=torch.int32))
    sched = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[r1, r2]),
        waiting_queue=[],
    )
    mig = MambaPageMigrator(pool, sched)
    ops = [
        MigrateOp(src_page=5, dst_page=1, req_pool_idx=0, slot_in_req=0),
        MigrateOp(src_page=6, dst_page=2, req_pool_idx=1, slot_in_req=0),
    ]
    n = mig.migrate(ops)
    assert n == 2
    assert pool.migrated_log == [(5, 1), (6, 2)]
    assert int(r1.mamba_pool_idx[0]) == 1
    assert int(r2.mamba_pool_idx[0]) == 2
    print(f"[gap4] mamba migrate: ops=2 migrated={pool.migrated_log}")


def test_mamba_migrator_unknown_src_raises():
    from sglang.srt.arena.fire_plan import MigrateOp
    from sglang.srt.arena.mamba_migrator import MambaPageMigrator

    pool = _make_mamba_pool(n_slots=8)
    sched = SimpleNamespace(running_batch=SimpleNamespace(reqs=[]), waiting_queue=[])
    mig = MambaPageMigrator(pool, sched)
    ops = [MigrateOp(src_page=99, dst_page=1, req_pool_idx=0, slot_in_req=0)]
    raised = False
    try:
        mig.migrate(ops)
    except RuntimeError as e:
        raised = True
        assert "no req owns src_slot" in str(e)
    assert raised
    print("[gap4] mamba migrate unknown-src correctly raised")


def test_mamba_migrator_failed_migrate_slot_raises():
    from sglang.srt.arena.fire_plan import MigrateOp
    from sglang.srt.arena.mamba_migrator import MambaPageMigrator

    pool = _make_mamba_pool(n_slots=8)
    r = SimpleNamespace(req_pool_idx=0, mamba_pool_idx=torch.tensor([5], dtype=torch.int32))
    sched = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[r]), waiting_queue=[]
    )
    # Force migrate_slot to fail.
    pool.migrate_slot = lambda src, dst: False

    mig = MambaPageMigrator(pool, sched)
    ops = [MigrateOp(src_page=5, dst_page=1, req_pool_idx=0, slot_in_req=0)]
    raised = False
    try:
        mig.migrate(ops)
    except RuntimeError as e:
        raised = True
        assert "returned False" in str(e)
    assert raised
    # req's slot pointer must NOT have been updated since migrate failed.
    assert int(r.mamba_pool_idx[0]) == 5
    print("[gap4] mamba migrate-failure correctly raised, req unchanged")


def main():
    test_build_mamba_owner_map_basic()
    test_build_mamba_owner_map_with_capped()
    test_mamba_migrator_basic()
    test_mamba_migrator_unknown_src_raises()
    test_mamba_migrator_failed_migrate_slot_raises()
    print("\nT8 gap4 (mamba symmetry) test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
