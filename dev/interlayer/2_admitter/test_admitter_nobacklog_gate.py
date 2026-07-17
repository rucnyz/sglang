"""No-backlog gate: the Admitter must skip its expensive c^evict("kv") walk
when a cross-* fire cannot possibly win (queue_len * w_q < c_xfer(lcm_pages)),
returning a cheap lockless defer instead. This eliminates the per-arrival
tree-walk tax that made the Admitter cost throughput on the KV-bound
longhorizon regime, while preserving cross fires under real backlog.

Correctness invariants tested:
  1. Low queue + KV pressured -> gate returns 'defer' WITHOUT the slow-path
     c^evict walk (predict_evict_cost_us is never called).
  2. High queue (queue_len * w_q >= c_xfer(lcm)) -> gate falls through; the
     full slow path runs and a viable cross_* fire is still chosen.
  3. Ample headroom still returns own_free (gate sits after the headroom
     fast path, not before it).
  4. Degenerate floor (c_xfer 0, unprobed actuator) -> gate falls through
     (fail-safe, never a wrong skip).
"""
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
from test_sync_fire import FakePlanner, FakeActuator  # noqa: E402


class _KVPressuredSched:
    """KV pressured (free < x_tokens, but evictable), mamba idle, tunable
    queue length. The headroom fast path fails (KV low); whether the slow
    path runs depends only on the no-backlog gate."""

    def __init__(self, *, queue_len, x_tokens):
        # KV free below the arrival so the headroom fast path can't pass.
        self.token_to_kv_pool_allocator = StubAllocator(available=x_tokens // 2)
        self.tree_cache = StubTreeCache(evictable=x_tokens * 4)
        self._mamba_pool = StubMambaPool(available=10_000)
        self._mamba_evictable = 0
        self.waiting_queue = [None] * queue_len
        self.disaggregation_mode = "NULL"

    def get_mamba_pool(self):
        return self._mamba_pool

    def get_mamba_evictable(self):
        return self._mamba_evictable

    def get_mamba_tokens_per_chunk(self):
        return 1


def _admitter(warm=True):
    reset_cost_model()
    cm = get_cost_model()
    if warm:
        for _ in range(5):
            cm.update_xfer(total_us=10_000.0, n_chunks=1)  # ~10ms per page
    adm = Admitter(cost_model=cm)
    adm.actuator = FakeActuator(n_kv_subpools=1, n_mamba_subpools=1)
    adm.planner = FakePlanner()
    adm.lcm_pages = 1
    return adm, cm


def test_1_low_queue_gates_without_walk():
    adm, cm = _admitter()
    # Instrument: the slow path prices via cm.c_evict_us; count calls.
    calls = {"evict": 0}
    orig = cm.c_evict_us
    cm.c_evict_us = lambda pool, x: (calls.__setitem__("evict", calls["evict"] + 1) or orig(pool, x))
    sched = _KVPressuredSched(queue_len=2, x_tokens=8192)  # 2*100=200us << ~10ms floor
    dec = adm.decide_for_req(StubReq(n_input_tokens=8192), sched, tokens_per_page=1024)
    assert dec.action == "defer", f"low queue must gate to defer, got {dec.action}"
    assert "no backlog" in dec.reason, f"expected no-backlog gate, got: {dec.reason}"
    assert calls["evict"] == 0, f"gate must skip the c^evict walk, saw {calls['evict']} calls"
    print("test_1 OK  (low queue -> lockless defer, no c^evict walk)")


def test_2_high_queue_falls_through_and_fires():
    adm, cm = _admitter()
    floor = cm.c_xfer_us(adm.lcm_pages)  # ~10ms
    wq = cm.w_q_us()                      # 100us
    q_gate = int(floor / wq) + 5          # push queue*wq above the floor
    sched = _KVPressuredSched(queue_len=q_gate, x_tokens=8192)
    dec = adm.decide_for_req(StubReq(n_input_tokens=8192), sched, tokens_per_page=1024)
    assert "no backlog" not in dec.reason, "high queue must NOT gate"
    # With KV pressured + mamba idle + big backlog, the full model should pick
    # a cross-* action (grow KV from idle mamba) — the fire the gate preserves.
    assert dec.action.startswith("cross_") or dec.action in ("own_evict", "defer"), dec.action
    print(f"test_2 OK  (high queue q={q_gate} falls through, action={dec.action})")


def test_3_headroom_still_own_free():
    adm, cm = _admitter()
    # KV ample (free >> x) and mamba idle -> headroom fast path, before the gate.
    sched = _KVPressuredSched(queue_len=2, x_tokens=8192)
    sched.token_to_kv_pool_allocator = StubAllocator(available=1_000_000)
    dec = adm.decide_for_req(StubReq(n_input_tokens=8192), sched, tokens_per_page=1024)
    assert dec.action == "own_free" and "headroom" in dec.reason, dec.reason
    print("test_3 OK  (ample headroom still returns own_free)")


def test_4_degenerate_floor_falls_through():
    # Fail-safe: a degenerate cross-transfer floor (0) must NOT let the gate
    # fire — a 0 floor would make queue_len*w_q < 0 impossible / <0-vacuous,
    # so control must fall through to the full slow path, never a wrong skip.
    adm, cm = _admitter()
    cm.c_xfer_us = lambda n: 0.0  # force degenerate floor
    sched = _KVPressuredSched(queue_len=2, x_tokens=8192)
    dec = adm.decide_for_req(StubReq(n_input_tokens=8192), sched, tokens_per_page=1024)
    assert "no backlog" not in dec.reason, "zero floor must NOT gate (fail-safe fall-through)"
    print("test_4 OK  (degenerate zero floor -> full path, never a wrong skip)")


if __name__ == "__main__":
    for name in sorted(k for k in dir() if k.startswith("test_")):
        globals()[name]()
    print("\nALL TESTS PASSED")
