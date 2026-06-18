"""#294 — enable_kv_cache_copy is wired for live KV migration.

`TokenToKVPoolAllocator.migrate_slot` (#271) calls the pool's
`move_kv_cache`, which ASSERTS unless the pool was built with
`enable_kv_cache_copy=True` (it initializes `_kv_copy_config`). For the HiMA
cross-pool path the KV bytes live on `HybridLinearKVPool.full_kv_pool`, which
was constructed WITHOUT that flag — so a live KV migration would assert at
runtime. #294 wires it: a single source of truth `kv_live_migration_enabled()`
(read by both boot-time pool construction and the Stage-3 walk gate) drives
`enable_kv_cache_copy`, and `HybridLinearKVPool` forwards the flag to its inner
full KV pool.

These are CPU tests:
  1. `kv_live_migration_enabled()` reads SGLANG_XPOOL_KV_MIGRATE (fail-closed).
  2. `HybridLinearKVPool` forwards `enable_kv_cache_copy` to its non-MLA inner
     pool (spy the inner pool class — no CUDA warmup needed).
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch


def test_1_flag_reads_env_fail_closed():
    from sglang.srt.mem_cache.memory_pool import kv_live_migration_enabled
    prev = os.environ.get("SGLANG_XPOOL_KV_MIGRATE")
    try:
        os.environ.pop("SGLANG_XPOOL_KV_MIGRATE", None)
        assert kv_live_migration_enabled() is False, "default must be fail-closed OFF"
        os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "0"
        assert kv_live_migration_enabled() is False
        os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "1"
        assert kv_live_migration_enabled() is True, "==1 enables migration"
    finally:
        if prev is None:
            os.environ.pop("SGLANG_XPOOL_KV_MIGRATE", None)
        else:
            os.environ["SGLANG_XPOOL_KV_MIGRATE"] = prev
    print("  PASS  1  kv_live_migration_enabled() reads SGLANG_XPOOL_KV_MIGRATE "
          "(fail-closed default OFF)")


def test_2_hybrid_forwards_enable_kv_cache_copy_to_inner_pool():
    """HybridLinearKVPool(enable_kv_cache_copy=...) must reach the inner
    full_kv_pool. Spy the inner MHA class so no CUDA/triton warmup runs."""
    import sglang.srt.mem_cache.memory_pool as mp

    recorded = {}

    class _SpyMHA:
        def __init__(self, **kw):
            recorded["enable_kv_cache_copy"] = kw.get("enable_kv_cache_copy")
        def get_kv_size_bytes(self):
            return (0, 0)

    orig = mp.MHATokenToKVPool
    mp.MHATokenToKVPool = _SpyMHA
    try:
        for flag in (True, False):
            recorded.clear()
            mp.HybridLinearKVPool(
                size=64, dtype=torch.float16, page_size=1,
                head_num=4, head_dim=64, full_attention_layer_ids=[0, 1],
                enable_kvcache_transpose=False, device="cpu",
                mamba_pool=types.SimpleNamespace(),
                use_mla=False, enable_kv_cache_copy=flag,
            )
            assert recorded["enable_kv_cache_copy"] is flag, (
                f"HybridLinearKVPool must forward enable_kv_cache_copy={flag} "
                f"to full_kv_pool; got {recorded['enable_kv_cache_copy']}"
            )
    finally:
        mp.MHATokenToKVPool = orig
    print("  PASS  2  HybridLinearKVPool forwards enable_kv_cache_copy to "
          "full_kv_pool (both True and False)")


def _walk_provider(can_move):
    """A SchedulerOwnerProvider over a real allocator whose kvcache reports
    `can_move_kv_cache() == can_move`, with a migratable layout (one fully-live
    page + scattered donors) and SGLANG_XPOOL_KV_MIGRATE=1."""
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider

    class _StubKV:
        page_size = 1
        def __init__(self, can):
            self._can = can
        def get_kv_size_bytes(self):
            return 0
        def move_kv_cache(self, tgt, src):
            pass
        def can_move_kv_cache(self):
            return self._can

    a = TokenToKVPoolAllocator(
        size=24, dtype=torch.float16, device="cpu",
        kvcache=_StubKV(can_move), need_sort=False,
    )
    a.free_pages = torch.tensor([8, 9, 12, 13, 16, 17, 18, 19], dtype=torch.int64)
    kv_act = types.SimpleNamespace(_tokens_per_page=lambda: 4, n_pages=6)
    sched = types.SimpleNamespace(token_to_kv_pool_allocator=a, tree_cache=None)
    prov = SchedulerOwnerProvider(
        scheduler=sched, kv_actuator=kv_act, mamba_actuator=None,
    )
    prov._cached_kv_slots = lambda: {20, 21, 22, 23}
    return prov


def test_3_walk_refuses_when_pool_cannot_migrate():
    """The Stage-3 walk gate must verify the SOURCE pool can ACTUALLY migrate
    (kvcache.can_move_kv_cache), not just the env var. Else a runtime env flip
    after boot — or an MLA/NPU hybrid whose inner pool has no usable
    move_kv_cache — would emit moves that assert mid-fire (swallowed by
    agent.tick = half-applied corruption). With env ON: a capable pool yields
    moves; an INCAPABLE pool yields [] (fail-closed, zero mutation)."""
    prev = os.environ.get("SGLANG_XPOOL_KV_MIGRATE")
    os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "1"
    try:
        assert _walk_provider(can_move=True)._kv_live_pages_in_cost_order(), (
            "a migration-capable pool must still yield moves"
        )
        assert _walk_provider(can_move=False)._kv_live_pages_in_cost_order() == [], (
            "env ON but pool cannot migrate -> walk must refuse (return []), "
            "not emit moves that assert mid-fire"
        )
    finally:
        if prev is None:
            os.environ.pop("SGLANG_XPOOL_KV_MIGRATE", None)
        else:
            os.environ["SGLANG_XPOOL_KV_MIGRATE"] = prev
    print("  PASS  3  walk gate verifies pool migration capability "
          "(incapable -> [] even with env ON)")


def main() -> int:
    tests = [
        test_1_flag_reads_env_fail_closed,
        test_2_hybrid_forwards_enable_kv_cache_copy_to_inner_pool,
        test_3_walk_refuses_when_pool_cannot_migrate,
    ]
    print(f"\n#294 enable_kv_cache_copy wiring tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\n#294: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
