"""Phase 5 — Budgeter post-fire admission cap update tests.

Uses REAL `MambaPool`, `HybridReqToTokenPool`, `FutureMap` (no stubs).
Tests exercise the real `_maybe_update_admission_cap` against real
grow/shrink semantics, verifying the cascade end-to-end.

A thin `_SchedProxy` wraps the real pools + a server-args namespace so
`BudgetAgent._maybe_update_admission_cap` sees the API it expects.
There is NO mock of MambaPool / ReqToTokenPool / FutureMap.

Tests:
  1. first tick captures init state (mamba_live, ratio, ceiling)
  2. mamba grew (set_capacity_slots N) → req_to_token pool grew
  3. user ceiling honored (cap clipped)
  4. mamba shrank → pool.shrink called
  5. shrink blocked by held slot → RuntimeError caught, retry next tick
  6. KV-only model (kv_pool without mamba_pool attribute) → no-op
"""
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch  # noqa: E402

DEVICE = "cuda:0"
torch.cuda.set_device(0)


def _make_real(*, mamba_init, mamba_max, pool_init, pool_max, user_max):
    """Build real MambaPool + HybridReqToTokenPool + FutureMap + a
    thin scheduler proxy. Returns (sched_proxy, mamba_pool, req_to_token_pool,
    future_map)."""
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import (
        HybridReqToTokenPool,
        MambaPool,
    )
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=128, n_groups=1,
        num_heads=4, head_dim=64, state_size=16, conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=[0, 1])
    pool = HybridReqToTokenPool(
        size=pool_init,
        mamba_size=mamba_init,
        mamba_spec_state_size=pool_init,
        max_context_len=1024,
        device=DEVICE,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        enable_mamba_extra_buffer=False,
        max_size=pool_max,
    )
    # HybridReqToTokenPool builds its own MambaPool via _init_mamba_pool;
    # for this test we want full control over mamba_init / mamba_max
    # independent of the auto-derived `pool.max_size * 3`. Replace it.
    pool.mamba_pool = MambaPool(
        size=mamba_init,
        spec_state_size=mamba_init,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        device=DEVICE,
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=mamba_max,
    )
    future_map = FutureMap(
        max_running_requests=pool_init,
        chunked_prefill_size=8192,
        context_len=16384,
        device=torch.device(DEVICE),
        spec_algo=SpeculativeAlgorithm.NONE,
        max_running_requests_max=pool_max,
    )
    # Scheduler proxy: only the attributes _maybe_update_admission_cap
    # reads. token_to_kv_pool_allocator.get_kvcache() returns the
    # pool's mamba_pool wrapper (a thin object with `.mamba_pool` attr).
    kv_pool = types.SimpleNamespace(mamba_pool=pool.mamba_pool)
    alloc = types.SimpleNamespace(get_kvcache=lambda: kv_pool)
    sched = types.SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        req_to_token_pool=pool,
        future_map=future_map,
        server_args=types.SimpleNamespace(
            max_running_requests=user_max,
            # The boot-derived live admission gate (= max_running // pp_size).
            # Scheduler.get_num_allocatable_reqs reads THIS, not
            # sched.max_running_requests, to cap concurrent prefills.
            pp_max_micro_batch_size=pool_init,
        ),
        max_running_requests=pool_init,
        pp_size=1,
    )
    # The real gate reads get_global_server_args().pp_max_micro_batch_size; bind
    # the global to this proxy's server_args (production binds the global to the
    # scheduler's own server_args instance) so the test drives the REAL gate.
    from sglang.srt.server_args import set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(sched.server_args)
    return sched, pool.mamba_pool, pool, future_map


def _make_agent(scheduler):
    """Construct BudgetAgent without running __init__ (avoids file/thread
    creation). Only the attributes _maybe_update_admission_cap touches."""
    from sglang.srt.budgeter.agent import BudgetAgent
    a = BudgetAgent.__new__(BudgetAgent)
    a.scheduler = scheduler
    a._last_mamba_size = None
    a._mamba_per_req_ratio = None
    a._user_max_running = None
    return a


def test_1_first_tick_captures_init():
    """First tick: capture mamba_live + ratio + ceiling; no grow."""
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=256,
    )
    a = _make_agent(sched)
    init_pool_size = pool.size  # snapshot before any potential grow
    a._maybe_update_admission_cap()
    assert a._last_mamba_size == 99
    assert a._mamba_per_req_ratio == 3  # 99 // 33
    assert a._user_max_running == 256
    assert pool.size == init_pool_size  # no grow on first tick
    print("  PASS  1  first tick captures init (mamba=99, ratio=3, ceiling=256)")


def test_1b_none_user_cap_no_crash():
    """Prod runs with --max-running-requests UNSET, so
    server_args.max_running_requests is None (the resolved cap lives on
    sched.max_running_requests). The init must NOT crash on int(None); the
    growth ceiling falls back to pool.max_size. RED before the fix: int(None)
    raised mid-init, so _mamba_per_req_ratio stayed None and every later tick
    did `current_mamba // None` (observed 624x in the 262k sys run)."""
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=None,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()  # init must not raise
    assert a._mamba_per_req_ratio == 3, f"ratio={a._mamba_per_req_ratio} (None = the bug)"
    assert a._user_max_running == pool.max_size, \
        f"no user cap -> ceiling should be pool.max_size={pool.max_size}, got {a._user_max_running}"
    # A later grow must not crash on // None and must follow.
    mamba.set_capacity_slots(198)
    a._maybe_update_admission_cap()
    assert pool.size == min(pool.max_size, 198 // 3), f"grow failed: pool.size={pool.size}"
    print("  PASS  1b None user-cap: no int(None) crash, ceiling=pool.max_size, grow follows")


def test_2_mamba_grew_triggers_grow():
    """Real MambaPool.set_capacity_slots(N) → Budgeter cascades pool.grow."""
    sched, mamba, pool, future_map = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=256,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()  # init

    # Grow mamba live cap on the real pool. size goes from 99 → 198.
    new = mamba.set_capacity_slots(198)
    assert new == 198, f"mamba grew to {new}, expected 198"
    assert mamba.live_size == 198

    # Now budgeter sees the change.
    a._maybe_update_admission_cap()
    expected_new_cap = 198 // 3  # 66
    assert pool.size == expected_new_cap, \
        f"pool.size = {pool.size}, expected {expected_new_cap}"
    assert sched.max_running_requests == expected_new_cap
    assert future_map.max_running_requests == expected_new_cap
    print("  PASS  2  mamba grew 99→198 → pool grew to 66 + future_map followed")


def test_2c_real_gate_follows_grow():
    """The REAL admission gate (Scheduler.get_num_allocatable_reqs, which reads
    get_global_server_args().pp_max_micro_batch_size) must rise after a k2m
    grow. RED before the fix: the cap update bumps max_running_requests but NOT
    pp_max_micro_batch_size, so the gate stays frozen at the boot value and a
    grown mamba pool admits ZERO extra concurrent requests (the case2 win is a
    no-op). GREEN after: the gate follows the grow."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import get_global_server_args
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=256,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()  # init; gate = boot 33
    # At the boot ceiling the gate is full: no allocatable reqs at running_bs=33.
    assert Scheduler.get_num_allocatable_reqs(sched, 33) <= 0
    mamba.set_capacity_slots(198)   # k2m grow -> new_cap 66
    a._maybe_update_admission_cap()
    gate = Scheduler.get_num_allocatable_reqs(sched, 33)
    assert gate == 33, f"gate did not follow grow: allocatable={gate} (want 66-33)"
    assert get_global_server_args().pp_max_micro_batch_size == 66
    print("  PASS  2c real admission gate follows grow 33->66 (case2 unblocked)")


def test_7_shrink_lowers_gate():
    """Symmetric: an m2k shrink must lower the real gate so the scheduler does
    not admit beyond the shrunk mamba pool."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import get_global_server_args
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=256,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()
    mamba.set_capacity_slots(60)    # shrink -> new_cap 20
    a._maybe_update_admission_cap()
    assert get_global_server_args().pp_max_micro_batch_size == 20
    assert Scheduler.get_num_allocatable_reqs(sched, 20) <= 0  # full at new cap
    print("  PASS  7  m2k shrink lowers the real admission gate 33->20")


def test_3_user_ceiling_honored():
    """user_max=40 caps growth even when mamba could support more."""
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=40,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()
    mamba.set_capacity_slots(198)  # would imply pool=66, but ceiling=40
    a._maybe_update_admission_cap()
    assert pool.size == 40, f"pool.size = {pool.size}, expected 40 (ceiling)"
    assert sched.max_running_requests == 40
    print("  PASS  3  user ceiling honored (cap clipped to user_max=40)")


def test_4_mamba_shrank_triggers_shrink():
    """Mamba live cap shrinks → Budgeter calls pool.shrink with smaller cap.
    Real shrink requires slots in the shrunk range to be free."""
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=256,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()

    # Shrink mamba live cap. size goes 99 → 60.
    new = mamba.set_capacity_slots(60)
    assert new == 60
    a._maybe_update_admission_cap()
    expected_new_cap = 60 // 3  # 20
    assert pool.size == expected_new_cap, \
        f"pool.size = {pool.size}, expected {expected_new_cap}"
    assert sched.max_running_requests == 20
    print("  PASS  4  mamba shrank 99→60 → pool shrank to 20")


def test_5_shrink_blocked_by_held_retried():
    """Held slot blocks shrink (RuntimeError); next tick retries after release."""
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=256,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()  # init

    # Hold slot 25 by removing it from free_slots (simulates an in-flight req).
    pool.free_slots = [s for s in pool.free_slots if s != 25]

    mamba.set_capacity_slots(60)
    a._maybe_update_admission_cap()
    # Shrink to 20 would drop slot 25 → raises, agent logs + defers.
    assert pool.size == 33, f"pool.size should NOT have shrunk: {pool.size}"
    assert a._last_mamba_size == 99, \
        "last_mamba_size should not advance on a failed shrink (will retry)"

    # Release slot 25 + retry.
    pool.free_slots.append(25)
    a._maybe_update_admission_cap()
    assert pool.size == 20, f"after release, pool should shrink to 20: {pool.size}"
    assert a._last_mamba_size == 60
    print("  PASS  5  shrink blocked by held slot; retried after release")


def test_5b_held_slot_does_not_block_grow():
    """P0: grow() MUST succeed even when slots are held. Only shrink()
    should be blocked by held slots in the shrunk range. Otherwise
    admission can deadlock under fire pressure (hold prevents growth
    that would relieve pressure)."""
    sched, mamba, pool, _ = _make_real(
        mamba_init=99, mamba_max=297, pool_init=33, pool_max=128, user_max=256,
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()

    # Hold slot 5 — well inside the live range; doesn't conflict with grow.
    pool.free_slots = [s for s in pool.free_slots if s != 5]

    # Mamba grows; budgeter wants to grow pool to 66.
    mamba.set_capacity_slots(198)
    a._maybe_update_admission_cap()
    assert pool.size == 66, f"grow blocked by held slot? pool.size={pool.size}"
    assert 5 not in pool.free_slots, "held slot 5 should stay held"
    assert sched.max_running_requests == 66
    print("  PASS  5b held slot does NOT block grow — only shrink")


def test_6_kv_only_no_op():
    """KV-only model (kv_pool without mamba_pool attribute) → no-op."""
    # Construct a kv_pool stand-in that GENUINELY lacks mamba_pool —
    # mirrors MHATokenToKVPool (no mamba). hasattr(kv_pool, 'mamba_pool')
    # is the legitimate class-difference branch.
    kv_pool = types.SimpleNamespace()  # NO mamba_pool attr
    alloc = types.SimpleNamespace(get_kvcache=lambda: kv_pool)
    sched = types.SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        # The KV-only branch returns before touching these.
    )
    a = _make_agent(sched)
    a._maybe_update_admission_cap()  # must not raise
    assert a._last_mamba_size is None
    print("  PASS  6  KV-only model (no mamba_pool attr) — no-op")


def main():
    tests = [test_1_first_tick_captures_init, test_1b_none_user_cap_no_crash,
             test_2_mamba_grew_triggers_grow,
             test_2c_real_gate_follows_grow,
             test_3_user_ceiling_honored, test_4_mamba_shrank_triggers_shrink,
             test_5_shrink_blocked_by_held_retried,
             test_5b_held_slot_does_not_block_grow,
             test_7_shrink_lowers_gate,
             test_6_kv_only_no_op]
    print(f"\nBudgetAgent Phase 5 tests (n={len(tests)}, real classes):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 5: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
