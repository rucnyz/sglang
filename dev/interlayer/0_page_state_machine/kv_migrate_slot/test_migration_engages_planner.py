"""#295 keystone (CPU) — does KV live-migration ENGAGE end-to-end?

The e2e perf A/B (#295) only matters if migration actually FIRES. Migration
fires only when a k2m fire's free + cold-cache-drain candidates cannot
assemble `n` whole pages, but scattered live-uncached KV slots can be
consolidated into one. This test settles, WITHOUT a GPU, whether the real
`XPoolFirePlanner.build` path engages migration in that regime — driving the
production planner + `SchedulerOwnerProvider` + a real `TokenToKVPoolAllocator`
(the only GPU-bound piece, `move_kv_cache`'s bytes, is not exercised here;
plan *selection* is pure CPU).

It also pins that migration is LOAD-BEARING: with the gate OFF the exact same
fragmented layout REFUSES the fire (free+drain < n), and only the migration
stage lets it succeed. That is the precise condition the e2e workload must
reproduce — if natural cc traces never reach "free+drain < n for a k2m fire",
migration won't fire there and #295's win must be shown on a synthetic
fragmentation workload instead.

Layout (tps=4, 6 pages, size=24, fire target n=2):
  p1 [4-7]   fully-live-uncached      -> 1 migration SOURCE
  p2 [8,9|10,11], p3 [12,13|14,15]    -> 4 scattered donor slots
  p5 [20-23] whole-free               -> 1 anywhere-free page
  drain: tree_cache=None              -> 0 cold-cache pages
  => free(1) + drain(0) + migrate(1) == n(2): migration is the marginal page.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch


def _alloc(size, free_ids):
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    class _StubKV:
        page_size = 1
        def get_kv_size_bytes(self):
            return 0
        def move_kv_cache(self, tgt, src):
            pass
        def can_move_kv_cache(self):
            return True

    a = TokenToKVPoolAllocator(
        size=size, dtype=torch.float16, device="cpu",
        kvcache=_StubKV(), need_sort=False,
    )
    a.free_pages = torch.tensor(free_ids, dtype=torch.int64)
    return a


def _planner(allocator, tps=4, n_pages=6):
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider
    from sglang.srt.budgeter.fire_planner import XPoolFirePlanner
    kv_act = types.SimpleNamespace(_tokens_per_page=lambda: tps, n_pages=n_pages)
    sched = types.SimpleNamespace(
        token_to_kv_pool_allocator=allocator, tree_cache=None,
    )
    prov = SchedulerOwnerProvider(
        scheduler=sched, kv_actuator=kv_act, mamba_actuator=None,
    )
    prov._cached_kv_slots = lambda: set()  # no cached slots
    pl = XPoolFirePlanner(kv_actuator=kv_act, mamba_actuator=None,
                          owner_provider=prov)
    return pl


# Fragmented KV near-full: 1 whole-free page (p5) + scattered donors (p2,p3)
# + a fully-live page (p1). free+drain alone = 1 page < n=2.
_FREE = [8, 9, 12, 13, 20, 21, 22, 23]


def test_migration_engages_and_is_load_bearing():
    prev = os.environ.get("SGLANG_XPOOL_KV_MIGRATE")
    try:
        # --- gate ON: migration supplies the marginal page ---
        os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "1"
        pl = _planner(_alloc(24, _FREE))
        plan = pl.build("kv_to_mamba", 2, allow_drain=True, allow_migrate=True)
        assert plan is not None, "with migration ON the fragmented fire must build"
        assert plan.migrations == ((4, 8), (5, 9), (6, 12), (7, 13)), (
            f"Stage-3 must consolidate live page 1 into the scattered donors; "
            f"got {plan.migrations}"
        )
        assert sorted(plan.pages_to_unmap) == [1, 5], (
            f"freed pages = the whole-free p5 + the migrated-empty p1; got "
            f"{plan.pages_to_unmap}"
        )
        assert pl.refuse_count == 0, "migration-ON fire must not refuse"
        # The fire's completion record logs len(plan.migrations) as
        # `fire_migrate_moves` and len(plan.drains) as `fire_drain_pages`
        # (agent.py) — the per-fire signal the #295 e2e A/B counts to ATTRIBUTE
        # a win to Migration vs Drain. Pin that mapping so the instrumentation
        # can't drift from the plan it reports: here a pure-migration fire.
        assert len(plan.migrations) == 4 and len(plan.drains) == 0, (
            "this fire is logged as fire_migrate_moves=4, fire_drain_pages=0"
        )

        # --- gate OFF: same layout, migration walk returns [] -> REFUSE ---
        os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "0"
        pl_off = _planner(_alloc(24, _FREE))
        plan_off = pl_off.build("kv_to_mamba", 2, allow_drain=True,
                                allow_migrate=True)
        assert plan_off is None, (
            "with migration OFF, free(1)+drain(0) < n(2) — the fire MUST be "
            "refused; migration is the load-bearing marginal page"
        )
        assert pl_off.refuse_count == 1, "the refused fire must bump refuse_count"
    finally:
        if prev is None:
            os.environ.pop("SGLANG_XPOOL_KV_MIGRATE", None)
        else:
            os.environ["SGLANG_XPOOL_KV_MIGRATE"] = prev
    print("  PASS  migration engages through the real planner AND is "
          "load-bearing (gate OFF -> fire refused)")


def main() -> int:
    tests = [test_migration_engages_and_is_load_bearing]
    print(f"\n#295 planner-engagement test (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
    print(f"\n#295 engagement: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
