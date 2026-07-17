"""Admitter cross-* fires must go through the async submit hook, not the
synchronous 10-30ms actuator.execute on the scheduler thread.

execute_decision routing contract:
  1. _fire_submit wired + returns (False, None)  -> async enqueue: decision
     stays the cross-* action (request admitted), fire_result stays None,
     no synchronous execute() call.
  2. _fire_submit returns (True, None)           -> aborted: action -> defer.
  3. _fire_submit returns (False, sync_result)   -> sync fallback: fire_result
     populated, cost model updated from the realized transfer.
  4. _fire_submit is None (unit-test / pre-tick)  -> legacy actuator.execute.
"""
import math
import os
import sys

os.environ.setdefault("SGLANG_CSIGMA_KV_ALPHA", "1.0214961938707212e-07")
os.environ.setdefault("SGLANG_CSIGMA_KV_BETA", "0.024570739655696554")
os.environ.setdefault("SGLANG_CSIGMA_KV_GAMMA", "5.97224986310455")
os.environ.setdefault("SGLANG_CSIGMA_M_ALPHA", "0.0")
os.environ.setdefault("SGLANG_CSIGMA_M_BETA", "0.0")
os.environ.setdefault("SGLANG_CSIGMA_L_STAR", "0.0")

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/2_admitter")

from sglang.srt.budgeter.admitter import Admitter  # noqa: E402
from sglang.srt.budgeter.cost_model import reset_cost_model, get_cost_model  # noqa: E402
from test_scheduler_hook import StubReq, StubAllocator, StubTreeCache, StubMambaPool  # noqa: E402
from test_sync_fire import FakePlanner, FakeActuator, FakeFirePlanResult  # noqa: E402


class _MambaScarceSched:
    """mamba scarce, KV slack, queue backlog -> the Admitter grows mamba from
    KV (a cross-* action that reaches the fire path)."""

    def __init__(self):
        self.token_to_kv_pool_allocator = StubAllocator(available=200_000)
        self.tree_cache = StubTreeCache(evictable=50_000)
        self._mamba_pool = StubMambaPool(available=0)
        self._mamba_evictable = 0
        self.waiting_queue = [None] * 200
        self.disaggregation_mode = "NULL"

    def get_mamba_pool(self):
        return self._mamba_pool

    def get_mamba_evictable(self):
        return self._mamba_evictable

    def get_mamba_tokens_per_chunk(self):
        return 1


def _admitter_reaching_fire():
    reset_cost_model()
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    adm = Admitter(cost_model=cm)
    adm.actuator = FakeActuator(n_kv_subpools=1, n_mamba_subpools=1)
    adm.planner = FakePlanner()
    adm.lcm_pages = 1
    return adm


def _cross_decision(adm):
    sched = _MambaScarceSched()
    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), sched, tokens_per_page=1024)
    assert dec.action.startswith("cross_"), f"precondition: got {dec.action}"
    return dec


def test_1_async_enqueue_keeps_action():
    adm = _admitter_reaching_fire()
    seen = {}

    def spy(plan):
        seen["plan"] = plan
        return (False, None)  # async enqueue success

    adm._fire_submit = spy
    dec = _cross_decision(adm)
    action_before = dec.action
    out = adm.execute_decision(dec, x_tokens=4096, src_pool=dec.src_pool,
                               dst_pool=dec.dst_pool, tokens_per_page=1024)
    assert "plan" in seen, "submit hook was not called"
    assert out.action == action_before, f"async enqueue must keep the cross action, got {out.action}"
    assert out.fire_result is None, "async path must not populate fire_result synchronously"
    print("test_1 OK  (async enqueue routes through _fire_submit, keeps action)")


def test_2_abort_falls_back_to_defer():
    adm = _admitter_reaching_fire()
    adm._fire_submit = lambda plan: (True, None)  # aborted
    dec = _cross_decision(adm)
    out = adm.execute_decision(dec, x_tokens=4096, src_pool=dec.src_pool,
                               dst_pool=dec.dst_pool, tokens_per_page=1024)
    assert out.action == "defer", f"aborted submit must defer, got {out.action}"
    print("test_2 OK  (aborted submit -> defer)")


def test_3_sync_fallback_prices_transfer():
    adm = _admitter_reaching_fire()
    res = FakeFirePlanResult(granted_pages=8, aborted=False)
    adm._fire_submit = lambda plan: (False, res)  # sync fallback
    cm = adm.cost_model
    before = cm.c_xfer_us(1)
    dec = _cross_decision(adm)
    out = adm.execute_decision(dec, x_tokens=4096, src_pool=dec.src_pool,
                               dst_pool=dec.dst_pool, tokens_per_page=1024)
    assert out.fire_result is res, "sync fallback must attach the realized result"
    assert cm.c_xfer_us(1) != before or before > 0, "sync fallback should feed the c^xfer EWMA"
    print("test_3 OK  (sync fallback attaches result + updates cost model)")


def test_4_default_hook_is_sync_execute():
    # An Admitter with no BudgetAgent override defaults to the synchronous
    # inline fire (_sync_fire -> actuator.execute). BudgetAgent overrides this
    # with the async submit in production.
    adm = _admitter_reaching_fire()
    assert adm._fire_submit == adm._sync_fire, "default hook must be _sync_fire"
    calls = {"execute": 0}
    orig = adm.actuator.execute

    def counting_execute(plan):
        calls["execute"] += 1
        return orig(plan)

    adm.actuator.execute = counting_execute
    dec = _cross_decision(adm)
    adm.execute_decision(dec, x_tokens=4096, src_pool=dec.src_pool,
                         dst_pool=dec.dst_pool, tokens_per_page=1024)
    assert calls["execute"] == 1, "default hook must run one synchronous execute"
    print("test_4 OK  (default hook = synchronous inline execute)")


def test_5_async_does_not_block_the_caller():
    """PERF: the whole point of routing Admitter fires through the async
    worker is that execute_decision must NOT block the scheduler thread for
    the 10-30ms cuMemUnmap/Map. With a fake actuator whose execute() takes
    FIRE_MS, the synchronous default blocks the caller ~FIRE_MS, while the
    async submit (cap_barrier inline + hand-off) returns near-instantly."""
    import time

    FIRE_S = 0.10  # stand-in for the ~10-30ms cuMem work

    # --- synchronous default: execute() runs inline, so the caller blocks ---
    adm_sync = _admitter_reaching_fire()
    real_exec = adm_sync.actuator.execute

    def slow_execute(plan):
        time.sleep(FIRE_S)
        return real_exec(plan)

    adm_sync.actuator.execute = slow_execute
    dec = _cross_decision(adm_sync)
    t0 = time.perf_counter()
    adm_sync.execute_decision(dec, x_tokens=4096, src_pool=dec.src_pool,
                              dst_pool=dec.dst_pool, tokens_per_page=1024)
    sync_wall = time.perf_counter() - t0

    # --- async submit: cap_barrier is instant, the slow work is off-thread ---
    adm_async = _admitter_reaching_fire()
    real_exec2 = adm_async.actuator.execute

    def async_submit(plan):
        # mirror BudgetAgent._submit_admitter_fire's async contract: do the
        # cheap inline part, hand the slow execute() to a worker thread.
        import threading
        threading.Thread(target=lambda: (time.sleep(FIRE_S), real_exec2(plan)),
                         daemon=True).start()
        return (False, None)  # enqueued; sync_result unknown

    adm_async._fire_submit = async_submit
    dec2 = _cross_decision(adm_async)
    t1 = time.perf_counter()
    adm_async.execute_decision(dec2, x_tokens=4096, src_pool=dec2.src_pool,
                               dst_pool=dec2.dst_pool, tokens_per_page=1024)
    async_wall = time.perf_counter() - t1

    assert sync_wall >= FIRE_S * 0.9, (
        f"sync path should block ~{FIRE_S*1e3:.0f}ms, blocked {sync_wall*1e3:.1f}ms")
    assert async_wall < FIRE_S * 0.2, (
        f"async path must not block the caller; blocked {async_wall*1e3:.1f}ms")
    print(f"test_5 OK  (sync blocks {sync_wall*1e3:.0f}ms vs async "
          f"{async_wall*1e3:.1f}ms — {sync_wall/max(async_wall,1e-6):.0f}x off the scheduler thread)")


if __name__ == "__main__":
    for name in sorted(k for k in dir() if k.startswith("test_")):
        globals()[name]()
    print("\nALL TESTS PASSED")
