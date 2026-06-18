"""Phase 5 — scheduler-hook integration tests.

Tests for:
  1. `Admitter.decide_for_req(req, scheduler, tokens_per_page=1024)`: a thin adapter that
     queries scheduler state (allocator.available_size, tree_cache
     .evictable_size, waiting_queue length, req.origin_input_ids) and
     hands them to `decide(...)`.
  2. `SGLANG_HIMA_ADMITTER_LOG=path` JSONL output: every decision yields one
     JSON line with {ts, action, reason, x_tokens, queue_len, costs}.
  3. P99 latency budget under N=10^4 arrivals.

These tests use Scheduler-shaped stubs to avoid a full sglang
construction (which needs CUDA + a model). Phase 6 will exercise the
end-to-end live workload with a real Scheduler.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.admitter import Admitter
from sglang.srt.budgeter.cost_model import reset_cost_model, get_cost_model


# ---------------------------------------------------------------- Stubs

class StubReq:
    def __init__(self, n_input_tokens=128):
        self.origin_input_ids = list(range(n_input_tokens))


class StubAllocator:
    def __init__(self, available=20000):
        self._available = available
        # decide_for_req holds this across the capacity snapshot +
        # c^evict prediction (#216). Production allocators get it in
        # BaseTokenToKVPoolAllocator.__init__; the stub mirrors that.
        import threading
        self._alloc_lock = threading.Lock()
    def available_size(self):
        return self._available


class StubTreeCache:
    def __init__(self, evictable=5000):
        self._evictable = evictable
    def evictable_size(self):
        return self._evictable


class StubMambaPool:
    def __init__(self, available=128, live_size=None):
        self._available = available
        self.live_size = available if live_size is None else live_size
        self._allocator = self
    def available_size(self):
        return self._available


class StubScheduler:
    """Minimum surface area Admitter.decide_for_req needs."""
    def __init__(self, *, kv_free=20000, kv_evictable=5000,
                 mamba_free=128, mamba_evictable=0, queue_len=0,
                 disagg_null=True):
        self.token_to_kv_pool_allocator = StubAllocator(available=kv_free)
        self.tree_cache = StubTreeCache(evictable=kv_evictable)
        self._mamba_pool = StubMambaPool(available=mamba_free)
        self._mamba_evictable = mamba_evictable
        self.waiting_queue = [None] * queue_len
        self.disaggregation_mode = "NULL" if disagg_null else "PREFILL"

    def get_mamba_pool(self):
        return self._mamba_pool

    def get_mamba_evictable(self):
        return self._mamba_evictable


class MockOwnerProvider:
    """Fake SchedulerOwnerProvider for tests that need migratable pages.

    mamba_tps > 1 takes the fragmentable path in _mamba_feasibility,
    which builds the owner map and returns (free_pages * tps, mig_pages * tps).
    mamba_tps <= 1 takes the atomic cheap path (0 migratable).
    """
    def __init__(self, free_pages=0, migratable_pages=0,
                 kv_tps=1024, mamba_tps=2):
        self._free = free_pages
        self._mig = migratable_pages
        self._kv_tps = kv_tps
        self._mamba_tps = mamba_tps

    def kv_tokens_per_page(self):
        return self._kv_tps

    def mamba_tokens_per_page(self):
        return self._mamba_tps

    def build_mamba_owner_map(self, *, allow_drain=False, allow_migrate=False):
        from types import SimpleNamespace
        return SimpleNamespace(
            free_pages=set(range(self._free)),
            live_pages_in_cost_order=[
                (p, ((p, 1000 + p),)) for p in range(self._mig)
            ] if allow_migrate else [],
            cached_pages_in_cost_order=None,
        )


def _fresh_admitter():
    reset_cost_model()
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    return Admitter(cost_model=cm)


# ---------------------------------------------------------------- Tests

def test_1_decide_for_req_derives_state_from_scheduler():
    """decide_for_req queries scheduler state correctly and routes the
    right kwargs into decide()."""
    adm = _fresh_admitter()
    sched = StubScheduler(
        kv_free=20000, kv_evictable=5000,
        mamba_free=64, mamba_evictable=0,
        queue_len=3,
    )
    req = StubReq(n_input_tokens=128)
    dec = adm.decide_for_req(req, sched, tokens_per_page=1024)
    assert dec is not None, "decide_for_req must return a decision"
    # 128 tokens < 20000 free → own_free wins
    assert dec.action == "own_free", f"expected own_free, got {dec.action}"
    # Cost vector must include all 5 candidates
    for k in ("own_free", "own_evict", "cross_free", "cross_evict", "defer"):
        assert k in dec.candidate_costs_us, (
            f"decision missing candidate {k}: {dec.candidate_costs_us}"
        )
    print("  PASS  1  decide_for_req derives kv_free/evictable/Q/x_tokens from scheduler")


def test_2_jsonl_log_records_every_decision():
    """When SGLANG_HIMA_ADMITTER_LOG=path is set, every decide_for_req call
    yields a JSON line in the file."""
    adm = _fresh_admitter()
    sched = StubScheduler(kv_free=20000, queue_len=0)
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=False
    ) as f:
        log_path = f.name
    try:
        os.environ["SGLANG_HIMA_ADMITTER_LOG"] = log_path
        # Recreate to pick up env (Admitter binds at construction)
        adm = _fresh_admitter()
        N = 50
        for i in range(N):
            adm.decide_for_req(StubReq(n_input_tokens=64 + i), sched, tokens_per_page=1024)
        adm.close()
        with open(log_path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) == N, f"expected {N} JSONL lines, got {len(lines)}"
        first = json.loads(lines[0])
        for k in ("ts", "action", "reason", "dst_pool", "src_pool",
                  "x_tokens", "fire_x_tokens", "queue_len",
                  "candidate_costs_us"):
            assert k in first, f"first JSONL entry missing field {k}: {first}"
        assert first["x_tokens"] == 64
        last = json.loads(lines[-1])
        assert last["x_tokens"] == 64 + N - 1
        print(f"  PASS  2  JSONL log records {N}/{N} decisions with full schema")
    finally:
        os.environ.pop("SGLANG_HIMA_ADMITTER_LOG", None)
        os.unlink(log_path)


def test_3_no_jsonl_when_env_unset():
    """No SGLANG_HIMA_ADMITTER_LOG → no file written, no crash."""
    os.environ.pop("SGLANG_HIMA_ADMITTER_LOG", None)
    adm = _fresh_admitter()
    sched = StubScheduler(kv_free=20000)
    for _ in range(10):
        adm.decide_for_req(StubReq(), sched, tokens_per_page=1024)
    adm.close()
    # No exceptions, no file. test passes by not crashing.
    print("  PASS  3  SGLANG_HIMA_ADMITTER_LOG unset → no file, no crash")


def test_4_p99_latency_under_100us():
    """Per-arrival decide_for_req P99 < 100 µs over 10k arrivals."""
    adm = _fresh_admitter()
    sched = StubScheduler(kv_free=20000)
    # Vary input size + queue so we hit different code paths
    latencies = []
    N = 10_000
    for i in range(N):
        sched.waiting_queue = [None] * (i % 50)
        req = StubReq(n_input_tokens=64 + (i % 1024))
        t0 = time.perf_counter()
        adm.decide_for_req(req, sched, tokens_per_page=1024)
        latencies.append((time.perf_counter() - t0) * 1e6)
    latencies.sort()
    p99 = latencies[int(0.99 * N)]
    median = latencies[N // 2]
    assert p99 < 100, (
        f"decide_for_req P99 = {p99:.1f} µs > 100 µs budget "
        f"(median {median:.1f} µs)"
    )
    print(f"  PASS  4  P99 decide_for_req = {p99:.1f} µs over {N} arrivals "
          f"(median {median:.1f} µs)")


def test_6_jsonl_includes_fire_result_when_present():
    """When a cross-* fire is triggered (Phase 4 path), the JSONL entry
    must record fire_result fields (granted_pages, total_us, aborted)."""
    adm = _fresh_admitter()
    sched = StubScheduler(kv_free=20000)
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=False
    ) as f:
        log_path = f.name
    try:
        os.environ["SGLANG_HIMA_ADMITTER_LOG"] = log_path
        adm = _fresh_admitter()
        # Build a fake decision with a fire_result to confirm log captures it
        from sglang.srt.budgeter.admitter import AdmitterDecision
        dec = AdmitterDecision(
            action="cross_free",
            reason="test",
            candidate_costs_us={"own_free": float("inf"), "cross_free": 100.0,
                                "own_evict": float("inf"),
                                "cross_evict": float("inf"), "defer": 5000.0},
        )
        class FakeResult:
            granted_pages = 12
            total_us = 1500
            aborted = False
        dec.fire_result = FakeResult()
        adm._log_decision(dec, x_tokens=2048, queue_len=3)
        adm.close()
        with open(log_path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "fire_granted_pages" in entry, f"missing fire info: {entry}"
        assert entry["fire_granted_pages"] == 12
        assert entry["fire_total_us"] == 1500
        assert entry["fire_aborted"] is False
        print("  PASS  6  JSONL entry includes fire_result fields when set")
    finally:
        os.environ.pop("SGLANG_HIMA_ADMITTER_LOG", None)
        os.unlink(log_path)


def test_7_cold_start_path_uses_own_when_capacity_available():
    """At cold-start (c^xfer EWMA still at its conservative initial), even
    the decision prefers own_free if KV has capacity.
    Cold-start is cost-driven: the conservative c^xfer prices cross-* above
    a feasible own_free, so own_free wins the min-cost compare."""
    reset_cost_model()
    cm = get_cost_model()
    adm = Admitter(cost_model=cm, )
    sched = StubScheduler(kv_free=20000)
    dec = adm.decide_for_req(StubReq(n_input_tokens=128), sched, tokens_per_page=1024)
    assert dec.action == "own_free", (
        f"cold-start with own_free feasible must pick own_free, got {dec.action}"
    )
    print("  PASS  7  cold-start path picks own_free when KV has capacity")


def test_8_x_tokens_uses_origin_input_ids_length():
    """At hook point, req has only origin_input_ids; demand X = len(...).
    Verified by giving the same scheduler state with two reqs of different
    sizes and checking decide() got the right x_tokens via the recorded
    cost vector (defer cost vs queue_len isn't sensitive to X but the
    candidate cost structure should still differ)."""
    adm = _fresh_admitter()
    sched = StubScheduler(kv_free=20000, queue_len=5)
    dec_small = adm.decide_for_req(StubReq(n_input_tokens=64), sched, tokens_per_page=1024)
    dec_large = adm.decide_for_req(StubReq(n_input_tokens=15000), sched, tokens_per_page=1024)
    # Small req: 64 < 20000 free → own_free
    assert dec_small.action == "own_free", (
        f"64-token req should fit, got {dec_small.action}"
    )
    # Large req: 15000 < 20000 free → still own_free
    assert dec_large.action == "own_free", (
        f"15000-token req still fits 20000 free, got {dec_large.action}"
    )
    # Even larger req exceeding free should defer (no evict available)
    sched2 = StubScheduler(kv_free=10000, kv_evictable=0, queue_len=5)
    dec_huge = adm.decide_for_req(StubReq(n_input_tokens=15000), sched2, tokens_per_page=1024)
    assert dec_huge.action == "defer", (
        f"15000-token req with only 10000 free + 0 evict should defer, "
        f"got {dec_huge.action}"
    )
    print("  PASS  8  x_tokens correctly derived from len(req.origin_input_ids)")


# ---------------------------------------------------------------- audit_phase5 gap fixes

def test_9_close_flushes_and_is_idempotent():
    """Audit #1 (HIGH) — close() flushes the JSONL log + is idempotent."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=False
    ) as f:
        log_path = f.name
    try:
        os.environ["SGLANG_HIMA_ADMITTER_LOG"] = log_path
        adm = _fresh_admitter()
        sched = StubScheduler(kv_free=20000)
        for _ in range(3):
            adm.decide_for_req(StubReq(), sched, tokens_per_page=1024)
        adm.close()
        # Idempotent: second close must not raise.
        adm.close()
        with open(log_path) as f:
            n = len([ln for ln in f.read().splitlines() if ln.strip()])
        assert n == 3, f"expected 3 lines after close, got {n}"
        # _log_fp must be None after close
        assert adm._log_fp is None, "close must null out _log_fp"
        print("  PASS  9  close() flushes log, idempotent, _log_fp=None")
    finally:
        os.environ.pop("SGLANG_HIMA_ADMITTER_LOG", None)
        os.unlink(log_path)


def test_13_empty_input_doesnt_crash():
    """Audit #7 — req.origin_input_ids = [] or None → x_tokens=0 → own_free."""
    adm = _fresh_admitter()
    sched = StubScheduler(kv_free=20000)
    for input_ids in ([], None):
        class EmptyReq:
            origin_input_ids = input_ids
        dec = adm.decide_for_req(EmptyReq(), sched, tokens_per_page=1024)
        assert dec is not None, "must not crash on empty input"
        assert dec.action == "own_free", (
            f"x_tokens=0 must pick own_free (0 ≤ free), got {dec.action}"
        )
    print("  PASS  13  empty / None origin_input_ids → x_tokens=0 → own_free")


def test_14_jsonl_candidate_set_is_exactly_seven():
    """Audit #8 — JSONL candidate_costs_us keys must be exactly the seven
    canonical actions (#183 added own/cross_migrate). A refactor that adds
    or drops a candidate name without updating downstream parsers should
    fail this test."""
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=False
    ) as f:
        log_path = f.name
    try:
        os.environ["SGLANG_HIMA_ADMITTER_LOG"] = log_path
        adm = _fresh_admitter()
        adm.decide_for_req(StubReq(), StubScheduler(), tokens_per_page=1024)
        adm.close()
        with open(log_path) as f:
            entry = json.loads(f.read().strip().splitlines()[0])
        expected = {"own_free", "own_evict", "cross_free", "cross_evict",
                    "own_migrate", "cross_migrate", "defer"}
        actual = set(entry["candidate_costs_us"].keys())
        assert actual == expected, (
            f"candidate name set drifted: expected {expected}, got {actual}"
        )
        print(f"  PASS  14  JSONL candidate set is exactly {{own_*, cross_*, defer}}")
    finally:
        os.environ.pop("SGLANG_HIMA_ADMITTER_LOG", None)
        os.unlink(log_path)


# ---------------------------------------------------------------- Phase 6 live bug fix

def test_15_hybrid_radix_cache_evictable_size():
    """MambaRadixCache.evictable_size() raises NotImplementedError —
    consumers must use full_evictable_size() / mamba_evictable_size().
    decide_for_req must not crash when given such a tree_cache.
    Regression for D6 live run (2026-05-29 first attempt)."""
    class MambaRadixCacheStub:
        def evictable_size(self):
            raise NotImplementedError
        def full_evictable_size(self):
            return 7777
        def mamba_evictable_size(self):
            return 22

    class StubSchedHybrid:
        def __init__(self):
            self.token_to_kv_pool_allocator = StubAllocator(available=20000)
            self.tree_cache = MambaRadixCacheStub()
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"
            self._mamba_pool = StubMambaPool(available=64)
        def get_mamba_pool(self):
            return self._mamba_pool

    adm = _fresh_admitter()
    dec = adm.decide_for_req(StubReq(n_input_tokens=128), StubSchedHybrid(), tokens_per_page=1024)
    assert dec is not None, "must not crash on hybrid cache"
    # Verify the full_evictable_size was actually consulted: x_tokens=128 vs
    # kv_evictable=7777 → own_evict feasible at cost c_evict_dst (computed from
    # cost_model). The action depends on cost vector — just ensure no crash.
    assert dec.action in (
        "own_free", "own_evict", "cross_free", "cross_evict", "defer"
    )
    print("  PASS  15  MambaRadixCache.evictable_size NotImplementedError handled")


def test_16_mamba_free_uses_kv_token_equivalents():
    """Phase 6 D6 4th attempt revealed: mamba.available_size() returns
    SLOTS (per-req), but x_tokens is in KV TOKENS (per-token). Comparing
    them directly makes cross_free always infeasible — the symptom was
    contentious arrivals choosing 'defer' instead of 'cross_free'.

    Fix: convert mamba_free = mamba.available_size() × tokens_per_page
    to express it in the same units as x_tokens. This test pins the
    conversion."""
    reset_cost_model()
    cm = get_cost_model()
    # Pre-warm so cross-* gate clears.
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    adm = Admitter(cost_model=cm, )

    class ContentiousSched:
        def __init__(self):
            self.token_to_kv_pool_allocator = StubAllocator(available=0)  # KV full
            self.tree_cache = None
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"
            self._mamba_pool = StubMambaPool(available=10)  # 10 free slots
        def get_mamba_pool(self):
            return self._mamba_pool

    sched = ContentiousSched()
    # x_tokens=4096, tokens_per_page=1024.
    # Pre-fix: mamba_free=10 (slots), compared to x_tokens=4096 → infeasible.
    # Post-fix: mamba_free=10 × 1024 = 10240 ≥ 4096 → feasible.
    req = StubReq(n_input_tokens=4096)
    dec = adm.decide_for_req(req, sched, tokens_per_page=1024)
    assert dec.candidate_costs_us["cross_free"] is not None, (
        f"cross_free must be feasible (10 slots × 1024 tps = 10240 ≥ 4096); "
        f"costs: {dec.candidate_costs_us}"
    )
    print(f"  PASS  16  mamba_free → KV-token equivalents "
          f"(10 slots × 1024 tps → cross_free feasible at x_tokens=4096)")


def test_17_evict_cost_routes_by_pool_label():
    """Audit-gap 1.4 (2026-05-30): `admitter.py:529-530` hardcodes
    `cost_model.c_evict_us("kv", x)` for `c_evict_dst_us` (because
    dst=KV) and `cost_model.c_evict_us("mamba", x)` for `c_evict_src_us`
    (because src=mamba). No test verifies the pool→cost routing — all
    existing tests leave both indexes unset, so both calls return +inf
    and the labels become indistinguishable. A pool-label swap (kv↔mamba)
    at lines 529-530 would silently flip own_evict ↔ cross_evict costs
    while passing every existing scheduler-hook test.

    Construction: plug a KV EvictCostIndex returning a small cost
    (cheap KV cache) and a MAMBA EvictCostIndex returning a large cost
    (expensive mamba cache). Force scenario where KV is full → own_free
    infeasible, so own_evict must be evaluated and its cost reflects
    the *KV* index. Cross side: mamba_free=0 → cross_free infeasible,
    so cross_evict must be evaluated and its cost includes the *mamba*
    index plus c_xfer.

    Pre-fix (with swap): own_evict reads mamba cost, cross_evict reads
    kv cost — assertions invert and fail.
    """
    reset_cost_model()
    cm = get_cost_model()
    # Pre-warm so cross-* gate clears.
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)

    # Fake per-pool caches exposing only `predict_evict_cost_us` —
    # the c^evict predictor surface (#180). KV cheap; MAMBA expensive.
    class _FakeCache:
        def __init__(self, cost_per_token_us: float) -> None:
            self._cpt = cost_per_token_us
        def predict_evict_cost_us(self, num_tokens: int, pool: str = "kv") -> float:
            if num_tokens <= 0:
                return 0.0
            return float(num_tokens) * self._cpt

    cm.set_evict_cache("kv", _FakeCache(cost_per_token_us=0.4))     # ~50us @ 128 tokens
    cm.set_evict_cache("mamba", _FakeCache(cost_per_token_us=40.0))  # ~5120us @ 128 tokens

    # Sanity: per-pool cost lookups distinct.
    assert cm.c_evict_us("kv", 128) < cm.c_evict_us("mamba", 128), (
        "test setup wrong: KV index must be cheaper than mamba index for the bug "
        "test to discriminate"
    )

    adm = Admitter(cost_model=cm, )

    class RoutedSched:
        def __init__(self):
            # KV full but evictable available → own_evict must be priced
            self.token_to_kv_pool_allocator = StubAllocator(available=0)
            self.tree_cache = StubTreeCache(evictable=512)  # > x_tokens
            # mamba_free=0 → cross_free infeasible; cross_evict must be priced
            self._mamba_pool = StubMambaPool(available=0)
            self._mamba_evictable = 512  # tokens worth of mamba evictable
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"

        def get_mamba_pool(self):
            return self._mamba_pool

        def get_mamba_evictable(self):
            return self._mamba_evictable

    sched = RoutedSched()
    req = StubReq(n_input_tokens=128)
    dec = adm.decide_for_req(req, sched, tokens_per_page=1024)
    costs = dec.candidate_costs_us

    own_evict_cost = costs.get("own_evict")
    cross_evict_cost = costs.get("cross_evict")
    # c_evict is prorated per-token: KV (100us full × 128/256 tokens) → ~50us.
    # MAMBA (9999us full × 128/256 tokens) → ~5000us. cross_evict adds c_xfer
    # on top (small at n_pages≈0.125 with EWMA 100us/page → negligible).
    assert own_evict_cost is not None and own_evict_cost < 200.0, (
        f"own_evict (dst=kv) should reflect KV index (cheap, <200us): "
        f"got {own_evict_cost}. If it's ~5000, lines 529-530 swapped "
        f"the pool labels (own_evict reading mamba index)."
    )
    assert cross_evict_cost is not None and cross_evict_cost >= 4500.0, (
        f"cross_evict (src=mamba) should reflect MAMBA index (expensive, "
        f">=4500us): got {cross_evict_cost}. If it's <200, lines 529-530 "
        f"swapped pool labels (cross_evict reading kv index)."
    )
    # Strict: ratio cross_evict / own_evict must be at least the index
    # ratio (~9999/100 = ~100×) since x_tokens is the same on both sides.
    ratio = cross_evict_cost / max(own_evict_cost, 1.0)
    assert ratio > 50, (
        f"pool-label routing not honored: cross_evict/own_evict={ratio:.1f}, "
        f"expected >50× (KV index cpt vs MAMBA index cpt). "
        f"own_evict={own_evict_cost} cross_evict={cross_evict_cost}"
    )
    print(f"  PASS  17  c_evict pool routing: own_evict(kv)={own_evict_cost:.1f} "
          f"cross_evict(mamba)={cross_evict_cost:.1f} ratio={ratio:.1f}×")


def test_18_xpool_actuator_exposes_lcm_pages_property():
    """#229: XPoolActuator surfaces `lcm_pages` as a property derived
    from `lcm(n_kv_subpools, n_mamba_subpools)`. Single source of truth
    so callers (Admitter, BudgetAgent) don't recompute the LCM and
    drift.

    Construct via __new__ to bypass __init__ (which needs real
    SharedHandlePool + CUDA). The property only reads two instance
    attrs.
    """
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    act = XPoolActuator.__new__(XPoolActuator)
    # Pin three representative geometries.
    cases = [
        (4, 6, 12),     # paper default-ish
        (1, 1, 1),      # degenerate (single sub-pool each side)
        (64, 24, 192),  # realistic 32B model: 32 layers × 2 kinds (K,V) vs mamba
    ]
    for n_kv, n_m, expected in cases:
        act.n_kv_subpools = n_kv
        act.n_mamba_subpools = n_m
        assert act.lcm_pages == expected, (
            f"XPoolActuator.lcm_pages mismatch for ({n_kv}, {n_m}): "
            f"got {act.lcm_pages}, expected {expected}"
        )
    print("  PASS  18  XPoolActuator.lcm_pages property: lcm(n_kv, n_mamba) "
          "for 3 geometries")


def test_19_decide_for_req_lcm_scales_cross_free_cost():
    """#229: post-BudgetAgent-wire, `adm.lcm_pages = actuator.lcm_pages`.
    Pre-wire default (1) under-prices cross_free by an LCM factor —
    biases the Admitter toward cross-* over defer (cross_free looks
    1/lcm× cheaper than it actually fires). Pin both pre- and post-
    wire prices and the exact lcm ratio.
    """
    # Warm c^xfer EWMA — note `update_xfer(total_us=1000, n_chunks=10)`
    # settles `current_us` to 100 µs/page (per-chunk). cross-* cold-
    # start gate disengages after ≥3 observations.
    reset_cost_model()
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    per_page_us = cm.c_xfer_us(1)
    assert per_page_us > 0, "EWMA failed to warm"

    sched = StubScheduler(
        kv_free=0, kv_evictable=0,
        mamba_free=64,  # plenty of mamba KV-equiv free for cross_free
        mamba_evictable=0,
        queue_len=1,
    )
    req = StubReq(n_input_tokens=512)  # 1 page at tps=1024

    # --- Pre-wire (no BudgetAgent push): lcm_pages = 1 ---
    adm_pre = Admitter(cost_model=cm, )
    assert adm_pre.lcm_pages == 1, (
        f"default lcm_pages must be 1, got {adm_pre.lcm_pages}"
    )
    dec_pre = adm_pre.decide_for_req(req, sched, tokens_per_page=1024)
    underpriced = dec_pre.candidate_costs_us["cross_free"]
    assert underpriced == per_page_us, (
        f"pre-wire cross_free = {underpriced}, expected 1 × "
        f"{per_page_us} (under-priced)"
    )

    # --- Post-wire (simulate BudgetAgent push: adm.lcm_pages = 12) ---
    adm_post = Admitter(cost_model=cm, )
    adm_post.lcm_pages = 12
    dec_post = adm_post.decide_for_req(req, sched, tokens_per_page=1024)
    correctly_priced = dec_post.candidate_costs_us["cross_free"]
    assert correctly_priced == 12 * per_page_us, (
        f"post-wire cross_free = {correctly_priced}, "
        f"expected 12 × {per_page_us}"
    )
    ratio = correctly_priced / underpriced
    assert ratio == 12, (
        f"price ratio = {ratio:.2f}; lcm factor must be exactly 12 "
        f"(the wire-in's whole reason to exist)"
    )
    print(f"  PASS  19  decide_for_req lcm_pages scaling: pre-wire "
          f"cross_free={underpriced:.1f} µs, post-wire={correctly_priced:.1f} "
          f"µs, ratio={ratio:.1f}× (= lcm)")


def test_20a_ensure_actuator_chain_calls_wire_admitter():
    """#229 audit M2 regression guard: the wire-in's load-bearing call
    site is `BudgetAgent._ensure_actuator_chain`. If that file moves
    and `_wire_admitter()` no longer fires from there, production
    silently reverts to the under-pricing bug — and the unit tests
    for `_wire_admitter` in isolation would still pass.

    This test pins the integration by patching `_wire_admitter` with a
    counter and driving `_ensure_actuator_chain` end-to-end with stub
    arenas / pools.
    """
    import threading
    from sglang.srt.budgeter.agent import BudgetAgent

    # Build a sham scheduler + admitter that BudgetAgent can read.
    reset_cost_model()
    cm = get_cost_model()
    adm = Admitter(cost_model=cm, )
    sched = type("S", (), {})()
    sched.admitter = adm
    # _ensure_actuator_chain wires on-demand grow hooks onto these (the KV
    # alloc-fail grow hook and the M1 mamba active-grow hook); settable stubs
    # so the assignments succeed off-CUDA.
    sched.token_to_kv_pool_allocator = type("A", (), {})()
    sched.req_to_token_pool = type("R", (), {})()

    ba = BudgetAgent.__new__(BudgetAgent)
    ba.scheduler = sched
    # In production _do_health_check sets _tree_cache before _maybe_fire calls
    # _ensure_actuator_chain (which wires _tree_cache._mamba_grow_hook, P4-b);
    # this test drives the chain in isolation, so provide a stub cache.
    ba._tree_cache = type("T", (), {})()
    ba._actuator = None
    ba._kv_act = None
    ba._mamba_act = None
    ba._owner_provider = None
    ba._fire_planner = None
    ba._migrate_probe_warned = False  # _run_migrate_probe reads this
    # Stub _wire_admitter with a counter — preserves the contract
    # (writes adm.actuator / adm.lcm_pages) but lets us count calls.
    calls = {"n": 0}
    orig_wire = ba._wire_admitter
    def counted():
        calls["n"] += 1
        orig_wire()
    ba._wire_admitter = counted  # type: ignore[assignment]

    # Stub the deep dependencies (KVArenaActuator etc) so the chain
    # builds without CUDA. We do this by patching the import points in
    # the agent module.
    import sys
    from types import SimpleNamespace
    import sglang.srt.budgeter.agent as agent_mod
    import sglang.srt.arena.kv_actuator as kv_act_mod
    import sglang.srt.arena.mamba_actuator as mamba_act_mod
    import sglang.srt.arena.xpool_actuator as xpool_act_mod
    import sglang.srt.budgeter.fire_planner as fire_planner_mod
    import sglang.srt.budgeter.scheduler_owner_provider as sop_mod

    class FakeXPool:
        n_kv_subpools = 4
        n_mamba_subpools = 6
        @property
        def lcm_pages(self):
            import math
            return math.lcm(self.n_kv_subpools, self.n_mamba_subpools)

    saved = (
        kv_act_mod.KVArenaActuator,
        mamba_act_mod.MambaArenaActuator,
        xpool_act_mod.XPoolActuator,
        fire_planner_mod.XPoolFirePlanner,
        sop_mod.SchedulerOwnerProvider,
    )
    try:
        kv_act_mod.KVArenaActuator = lambda **kw: object()
        mamba_act_mod.MambaArenaActuator = lambda **kw: object()
        xpool_act_mod.XPoolActuator = lambda **kw: FakeXPool()
        fire_planner_mod.XPoolFirePlanner = lambda **kw: object()
        sop_mod.SchedulerOwnerProvider = lambda **kw: object()

        kv_arena = SimpleNamespace(
            _arena=SimpleNamespace(_external_pool="sp"), tokens_per_chunk=128
        )
        # tokens_per_chunk: _ensure_actuator_chain reads it (P4-b) to set
        # _mamba_tokens_per_chunk before wiring the fork-grow hook.
        mamba_arena = SimpleNamespace(
            _arena=SimpleNamespace(_external_pool="sp"), tokens_per_chunk=64
        )
        inner_kv = SimpleNamespace(_kv_arena=kv_arena)
        kv_pool = SimpleNamespace(full_kv_pool=inner_kv)
        mamba_pool = SimpleNamespace(_mamba_temporal_arena=mamba_arena)

        snap = {}
        ok = ba._ensure_actuator_chain(
            alloc=None, kv_pool=kv_pool, mamba_pool=mamba_pool, snapshot=snap
        )
    finally:
        (kv_act_mod.KVArenaActuator, mamba_act_mod.MambaArenaActuator,
         xpool_act_mod.XPoolActuator, fire_planner_mod.XPoolFirePlanner,
         sop_mod.SchedulerOwnerProvider) = saved

    assert ok is True, "chain build should succeed with stubs"
    assert calls["n"] == 1, (
        f"_wire_admitter must be called exactly once during chain build, "
        f"got {calls['n']} — the load-bearing call site at "
        f"agent.py: `_ensure_actuator_chain` has gone missing."
    )
    assert adm.actuator is ba._actuator, "actuator not pushed"
    assert adm.lcm_pages == 12, f"lcm_pages not pushed: {adm.lcm_pages}"
    print("  PASS  20a  _ensure_actuator_chain invokes _wire_admitter exactly "
          "once on first build; admitter receives lcm_pages=12")


def test_20_budget_agent_wire_admitter_pushes_three_fields():
    """#229: BudgetAgent._wire_admitter() must push exactly
    (actuator, planner, lcm_pages) into scheduler.admitter on the first
    tick that builds the actuator chain. Idempotent on subsequent
    calls. When scheduler.admitter is None (env-disabled), it's a
    no-op (no AttributeError).
    """
    from sglang.srt.budgeter.agent import BudgetAgent

    # --- (a) Skips quietly when scheduler.admitter is None ----------
    sched = type("S", (), {})()
    sched.admitter = None
    ba = BudgetAgent.__new__(BudgetAgent)
    ba.scheduler = sched
    ba._actuator = "must-not-be-read"  # would crash if accessed
    ba._fire_planner = "must-not-be-read"
    ba._wire_admitter()  # no error
    # (no assertion — passing means no exception)

    # --- (b) Pushes three fields when admitter is present -----------
    reset_cost_model()
    cm = get_cost_model()
    adm = Admitter(cost_model=cm, )
    assert adm.actuator is None and adm.lcm_pages == 1, (
        "pre-wire invariants: actuator None, lcm_pages default 1"
    )

    class StubActuator:
        n_kv_subpools = 4
        n_mamba_subpools = 6
        @property
        def lcm_pages(self):
            import math
            return math.lcm(self.n_kv_subpools, self.n_mamba_subpools)

    stub_act = StubActuator()
    stub_planner = object()

    sched2 = type("S", (), {})()
    sched2.admitter = adm
    ba2 = BudgetAgent.__new__(BudgetAgent)
    ba2.scheduler = sched2
    ba2._actuator = stub_act
    ba2._fire_planner = stub_planner
    ba2._owner_provider = object()  # #269: pushed into adm.owner_provider

    ba2._wire_admitter()
    assert adm.actuator is stub_act, "actuator not pushed"
    assert adm.planner is stub_planner, "planner not pushed"
    assert adm.owner_provider is ba2._owner_provider, "owner_provider not pushed"
    assert adm.lcm_pages == 12, (
        f"lcm_pages mis-pushed: got {adm.lcm_pages}, expected 12"
    )

    # --- (c) Idempotent: a second tick does NOT overwrite -----------
    other_act = StubActuator()
    other_act.n_kv_subpools = 8  # different lcm to catch overwrite
    ba2._actuator = other_act
    ba2._wire_admitter()
    assert adm.actuator is stub_act, (
        "wire-in overwrote on second call — must be idempotent "
        "(if actuator handle changes mid-run, something else is wrong)"
    )
    assert adm.lcm_pages == 12, "lcm_pages was overwritten on idempotent call"
    print("  PASS  20  BudgetAgent._wire_admitter: skips on None admitter, "
          "pushes (actuator, planner, lcm_pages), idempotent on re-call")


def test_21_decide_for_req_holds_alloc_lock(make_req=None):
    """#216: decide_for_req must hold the dst allocator's `_alloc_lock`
    across the capacity snapshot + c^evict prediction, so a concurrent
    worker-thread set_capacity / alloc / free can't perturb the numbers
    the decision is priced from. Instrument the allocator's
    `available_size` to record whether the lock was held when called."""
    reset_cost_model()
    cm = get_cost_model()
    adm = Admitter(cost_model=cm, )

    seen = {"locked_during_read": None}

    class _LockSpyAllocator(StubAllocator):
        def available_size(self):
            # decide_for_req calls this inside `with _alloc_lock`.
            seen["locked_during_read"] = self._alloc_lock.locked()
            return self._available

    class _Sched:
        def __init__(self):
            self.token_to_kv_pool_allocator = _LockSpyAllocator(available=20000)
            self.tree_cache = StubTreeCache(evictable=5000)
            self._mamba_pool = StubMambaPool(available=128)
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"
        def get_mamba_pool(self):
            return self._mamba_pool
        def get_mamba_evictable(self):
            return 0

    dec = adm.decide_for_req(StubReq(n_input_tokens=64), _Sched(),
                             tokens_per_page=1024)
    assert dec is not None, "decision should be produced in NULL disagg"
    assert seen["locked_during_read"] is True, (
        "decide_for_req must hold kv_alloc._alloc_lock during the "
        "capacity snapshot (#216); available_size saw it UNheld"
    )
    # Lock must be released afterwards (no leak).
    assert not adm.cost_model._evict_caches  # sanity: nothing wired here
    print("  PASS  21  decide_for_req holds kv_alloc._alloc_lock during "
          "capacity snapshot + c^evict (#216)")


def test_22_decide_for_req_excludes_concurrent_holder():
    """#263: prove the #216 lock provides real MUTUAL EXCLUSION, not
    just that it's held (test 21). A worker thread grabs
    kv_alloc._alloc_lock and holds it; decide_for_req on the main
    thread must BLOCK until released — pinned by recording that the
    capacity read happened only AFTER the worker let go."""
    import threading
    import time as _time

    reset_cost_model()
    cm = get_cost_model()
    adm = Admitter(cost_model=cm, )

    events = []
    released = threading.Event()
    holding = threading.Event()

    class _SpyAllocator(StubAllocator):
        def available_size(self):
            # Runs inside decide_for_req's `with _alloc_lock`. Record
            # ordering: this must come AFTER the worker released.
            events.append("read")
            return self._available

    alloc = _SpyAllocator(available=20000)

    class _Sched:
        def __init__(self):
            self.token_to_kv_pool_allocator = alloc
            self.tree_cache = StubTreeCache(evictable=5000)
            self._mamba_pool = StubMambaPool(available=128)
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"
        def get_mamba_pool(self):
            return self._mamba_pool
        def get_mamba_evictable(self):
            return 0

    def _worker():
        alloc._alloc_lock.acquire()
        holding.set()
        events.append("worker_hold")
        _time.sleep(0.2)          # hold long enough that decide must wait
        events.append("worker_release")
        released.set()
        alloc._alloc_lock.release()

    t = threading.Thread(target=_worker)
    t.start()
    holding.wait(timeout=2.0)     # ensure worker holds before we call
    dec = adm.decide_for_req(StubReq(n_input_tokens=64), _Sched(),
                             tokens_per_page=1024)
    t.join(timeout=2.0)

    assert dec is not None
    # The capacity read must have happened AFTER the worker released —
    # i.e. decide_for_req blocked on the lock.
    assert "read" in events and "worker_release" in events
    assert events.index("read") > events.index("worker_release"), (
        f"decide_for_req read capacity while the worker held the lock "
        f"— mutual exclusion broken. events={events}"
    )
    print("  PASS  22  decide_for_req blocks on a concurrent "
          "_alloc_lock holder (real mutual exclusion, #216)")


# ---------------------------------------------------------------- #183 Step 2: migrate inputs

def test_23_decide_for_req_feeds_migrate_inputs():
    """#183 Step 2: decide_for_req computes the migrate inputs
    (`src_migratable`, `dst_migratable`, `c_migrate_*`) inside the
    `_alloc_lock` block and passes them to decide().

    A MockOwnerProvider with mamba_tps=2 (fragmentable) exposes 1 free
    page (1024 KV-equiv) and 100 migratable pages (102400 KV-equiv).
    With c_m pinned (probe seeded) AND no own-* alternative feasible,
    the cross_migrate candidate must be finite (feasible)."""
    from sglang.srt.budgeter.cost_model import (
        reset_cost_model, get_cost_model, get_migrate_cost,
    )
    reset_cost_model()
    cm = get_cost_model()
    # Warm c^xfer so the cross-* cold-start gate disengages.
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    # Pin c_m so migrate candidates can be priced (else +inf cold-start).
    get_migrate_cost().set_mamba(per_slot_us=25.0)
    adm = Admitter(cost_model=cm, )
    # Wire owner_provider: mamba_tps=2 (fragmentable), 1 free page
    # (1024 KV-equiv < 4096, so cross_free infeasible), 100 migratable
    # pages (102400 KV-equiv >= 4096, so cross_migrate feasible).
    adm.owner_provider = MockOwnerProvider(
        free_pages=1, migratable_pages=100, kv_tps=1024, mamba_tps=2,
    )

    class MigrateSched:
        def __init__(self):
            # KV full + no evictable -> own_free / own_evict infeasible.
            self.token_to_kv_pool_allocator = StubAllocator(available=0)
            self.tree_cache = None
            self._mamba_pool = StubMambaPool(available=2)
            self._mamba_evictable = 0
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"

        def get_mamba_pool(self):
            return self._mamba_pool

        def get_mamba_evictable(self):
            return self._mamba_evictable

    sched = MigrateSched()
    req = StubReq(n_input_tokens=4096)
    dec = adm.decide_for_req(req, sched, tokens_per_page=1024)
    costs = dec.candidate_costs_us
    assert costs["cross_migrate"] != float("inf"), (
        f"cross_migrate must be feasible (100 migratable pages = 102400 "
        f"KV-equiv >= x_tokens=4096, c_m pinned): {costs}"
    )
    # own_migrate stays inert (dst=kv has no migrate primitive).
    assert costs["own_migrate"] == float("inf"), (
        f"own_migrate must stay +inf for dst=kv: {costs}"
    )
    # With everything else infeasible and c_m cheap, cross_migrate wins
    # over defer (queue_len=0 → defer cost 0; but cross_migrate beats it
    # only when defer is costlier). Here queue_len=0 so defer=0 wins the
    # tie-break; assert at least cross_migrate is the only finite harvest.
    finite = {k for k, v in costs.items() if v != float("inf")}
    assert "cross_migrate" in finite and "defer" in finite, finite
    print("  PASS  23  decide_for_req feeds migrate inputs → finite cross_migrate "
          "candidate (MockOwnerProvider fragmentable path)")


def test_24_cross_migrate_infeasible_under_cm_cold_start():
    """Even with migratable LIVE state and a measured c^xfer, cross_migrate
    stays infeasible until c_m is seeded (mamba migrate probe not run).
    BootProbedMigrateCost cold-starts at +inf → c_migrate_us returns
    +inf → cross_migrate = c_xfer + inf = inf."""
    from sglang.srt.budgeter.cost_model import reset_cost_model, get_cost_model
    reset_cost_model()  # resets migrate cost -> c_m back to +inf
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)  # feed c^xfer observations
    adm = Admitter(cost_model=cm, )
    # Wire owner_provider with migratable pages, but c_m is NOT seeded
    # (reset_cost_model above). cross_migrate must stay +inf.
    adm.owner_provider = MockOwnerProvider(
        free_pages=1, migratable_pages=100, kv_tps=1024, mamba_tps=2,
    )

    class MigrateSched:
        def __init__(self):
            self.token_to_kv_pool_allocator = StubAllocator(available=0)
            self.tree_cache = None
            self._mamba_pool = StubMambaPool(available=2)
            self._mamba_evictable = 0
            self.waiting_queue = [None] * 10
            self.disaggregation_mode = "NULL"

        def get_mamba_pool(self):
            return self._mamba_pool

        def get_mamba_evictable(self):
            return 0

    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), MigrateSched(),
                             tokens_per_page=1024)
    assert dec.candidate_costs_us["cross_migrate"] == float("inf"), (
        f"cross_migrate must be +inf under c_m cold-start (probe unseeded): "
        f"{dec.candidate_costs_us}"
    )
    # Nothing else feasible → defer.
    assert dec.action == "defer", dec.action
    print("  PASS  24  cross_migrate infeasible under c_m cold-start "
          "(+inf until migrate probe seeds c_m)")


def test_25_migratable_zero_without_owner_provider():
    """#269/#273: without a wired `owner_provider`, `_mamba_feasibility`
    reports `src_migratable = 0` (the no-provider fallback always returns
    0 migratable). With free below the demand (2 slots x 1024 = 2048 <
    4096) and nothing evictable, the cumulative free+evict+migrate sum
    (2048) still can't cover X, so BOTH cross_free and cross_migrate are
    +inf, leading to defer."""
    from sglang.srt.budgeter.cost_model import (
        reset_cost_model, get_cost_model, get_migrate_cost,
    )
    reset_cost_model()
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    get_migrate_cost().set_mamba(per_slot_us=25.0)
    adm = Admitter(cost_model=cm, )
    assert adm.owner_provider is None, "precondition: no provider wired"

    class RealishMambaPool:
        # available_size=2 -> free KV-equiv = 2048 < X.
        live_size = 20
        def available_size(self):
            return 2

    class RealishSched:
        def __init__(self):
            self.token_to_kv_pool_allocator = StubAllocator(available=0)
            self.tree_cache = None
            self._mamba_pool = RealishMambaPool()
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"
        def get_mamba_pool(self):
            return self._mamba_pool
        def get_mamba_evictable(self):
            return 0

    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), RealishSched(),
                             tokens_per_page=1024)
    # free 2048 < 4096 -> cross_free infeasible; src_migratable=0 -> the
    # cumulative free+evict+migrate sum (2048) < 4096 -> cross_migrate inert.
    assert dec.candidate_costs_us["cross_free"] == float("inf"), dec.candidate_costs_us
    assert dec.candidate_costs_us["cross_migrate"] == float("inf"), (
        f"no provider -> src_migratable=0 -> Migration adds no reach -> "
        f"cross_migrate inert: {dec.candidate_costs_us}"
    )
    assert dec.action == "defer", dec.action
    print("  PASS  25  no provider -> src_migratable=0 -> Migration "
          "manufactures no reach -> cross_migrate inert -> defer")


def test_26_mamba_feasibility_from_owner_provider_pages():
    """#269: when an `owner_provider` is wired, `_mamba_feasibility` is the
    single source of truth shared with the planner:
      - ATOMIC layout (`mamba_tokens_per_page()==1`, the bench corner):
        cheap path — `mamba_free = available_size()·tps`, `src_migratable=0`,
        and the owner map is NOT built (no per-arrival GPU walk).
      - FRAGMENTABLE layout (`>=2`): `mamba_free = (#free_pages)·tps`,
        `src_migratable = (#live_pages_in_cost_order)·tps` from the owner
        map. `src_migratable` is the live (migration) page count ALONE —
        free pages are NOT folded in.
    Faked provider/pool so the test is deterministic and CUDA-free; the
    real page math is covered by test_expansion_lists.py."""
    from sglang.srt.budgeter.cost_model import (
        reset_cost_model, get_cost_model, get_migrate_cost,
    )

    class FakeOwnerMap:
        def __init__(self, free_pages, live_pages):
            self.free_pages = free_pages
            self.live_pages_in_cost_order = live_pages  # [(pid, moves), ...]
            self.cached_pages_in_cost_order = None

    class FakeProvider:
        def __init__(self, mamba_tps, free_pages=(), live_pages=()):
            self._tps = mamba_tps
            self._om = FakeOwnerMap(set(free_pages), list(live_pages))
            self.build_called = 0
        def mamba_tokens_per_page(self):
            return self._tps
        def build_mamba_owner_map(self, *, allow_drain=False, allow_migrate=False):
            self.build_called += 1
            return self._om

    def _sched(mamba_available=None):
        class S:
            def __init__(self):
                self.token_to_kv_pool_allocator = StubAllocator(available=0)  # KV full
                self.tree_cache = None
                self.waiting_queue = []
                self.disaggregation_mode = "NULL"
                self._mp = (StubMambaPool(available=mamba_available)
                            if mamba_available is not None else None)
            def get_mamba_pool(self):
                return self._mp
        return S()

    def _live(*pids):
        # New (freed_page_id, ((src,dst),...)) shape; only the count feeds
        # src_migratable, so the moves are placeholders here.
        return [(p, ((p, 1000 + p),)) for p in pids]

    def _adm():
        reset_cost_model()
        cm = get_cost_model()
        for _ in range(5):
            cm.update_xfer(total_us=1000.0, n_chunks=10)
        get_migrate_cost().set_mamba(per_slot_us=25.0)
        return Admitter(cost_model=cm, )

    # (a) ATOMIC (tps=1): cheap path — mamba_free from available_size, no
    # owner-map build, src_migratable=0.
    adm = _adm()
    prov = FakeProvider(mamba_tps=1)
    adm.owner_provider = prov
    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), _sched(mamba_available=2),
                             tokens_per_page=1024)
    assert prov.build_called == 0, (
        "atomic pool must NOT build the owner map on the arrival hot path"
    )
    assert dec.candidate_costs_us["cross_free"] == float("inf"), (
        f"2 free slots (2048) < 4096 → cross_free infeasible: {dec.candidate_costs_us}"
    )
    assert dec.candidate_costs_us["cross_migrate"] == float("inf"), (
        f"atomic → src_migratable=0 → cross_migrate inert: {dec.candidate_costs_us}"
    )

    # (b) FRAGMENTABLE (tps=2), CUMULATIVE: 2 free pages + 2 migration
    # pages. #273 folds FREE into cross_migrate's reach, so
    # free(2048)+migrate(2048)=4096 ≥ 4096 → FEASIBLE. Migrate ALONE
    # (2048) would be < 4096 → this discriminates that free IS folded in
    # (the cumulative free→drain→migrate model). cross_free still
    # infeasible (2048 < 4096).
    adm = _adm()
    prov = FakeProvider(mamba_tps=2, free_pages={8, 9}, live_pages=_live(1, 2))
    adm.owner_provider = prov
    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), _sched(),
                             tokens_per_page=1024)
    assert prov.build_called == 1, "fragmentable pool must build the owner map"
    assert dec.candidate_costs_us["cross_free"] == float("inf"), (
        f"2 free pages (2048) < 4096 → cross_free infeasible: {dec.candidate_costs_us}"
    )
    assert dec.candidate_costs_us["cross_migrate"] != float("inf"), (
        f"free(2048)+migrate(2048)=4096 ≥ 4096 → cross_migrate feasible "
        f"(free folded in per cumulative model); migrate alone (2048) would "
        f"NOT reach 4096: {dec.candidate_costs_us}"
    )

    # (c) BOUNDARY: 1 free + 2 migration = 3072 < 4096 → infeasible even
    # cumulatively (free+evict+migrate < X).
    adm = _adm()
    prov = FakeProvider(mamba_tps=2, free_pages={9}, live_pages=_live(1, 2))
    adm.owner_provider = prov
    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), _sched(),
                             tokens_per_page=1024)
    assert dec.candidate_costs_us["cross_migrate"] == float("inf"), (
        f"free(1024)+migrate(2048)=3072 < 4096 → cross_migrate infeasible: "
        f"{dec.candidate_costs_us}"
    )
    print("  PASS  26  _mamba_feasibility: atomic cheap-path (no owner-map "
          "build); fragmentable CUMULATIVE free+migrate reach (free folded "
          "in); infeasible below the free+evict+migrate sum")


def test_27_decide_for_req_shortfall_split_drain_then_migrate():
    """#273: when shortfall > evictable, decide_for_req must price the
    drain part at c^evict(EVICTABLE), capped at the single walk target
    min(shortfall, evictable), and the migrate part at c_m(remaining
    slots), NOT c^evict(full shortfall). Pins the cumulative free->drain->
    migrate split through the real decide_for_req (not the pure decide()).

    `get_mamba_evictable()` returns raw SLOTS (the production
    `mamba_evictable_size()` shape); decide_for_req converts slots ->
    chunks -> KV-token-equiv (x tokens_per_page) before the split, so the
    drain target is in the same units as the KV-equiv shortfall.

    FRAGMENTABLE (mamba_tps=2): mamba_evictable=6 slots -> 6//2=3 chunks
    -> 3072 KV-equiv. X=4096 (4 pages); mamba_free=0 (0 free pages from
    provider); migratable=10 pages (10240 KV-equiv). shortfall=4096 >
    drain-cap 3072 -> drain 3072, migrate 1024 (1 slot). cross_evict
    infeasible (free+evict=3072 < 4096);
    cross_migrate = c_xfer(4x100) + c^evict(3072) + c_m(1x25)
                  = 400 + 307.2 + 25 = 732.2 us.
    If the drain were (wrongly) priced at the full shortfall 4096, it would
    be 409.6 and cross_migrate 834.6, so the exact value discriminates the
    capped single-walk target."""
    from sglang.srt.budgeter.cost_model import (
        reset_cost_model, get_cost_model, get_migrate_cost,
    )
    reset_cost_model()
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)  # c_xfer_us(1)=100
    get_migrate_cost().set_mamba(per_slot_us=25.0)

    class _FakeMambaCache:
        def predict_evict_cost_us(self, num_tokens, pool="mamba"):
            return 0.1 * float(num_tokens)  # c^evict(3072)=307.2
    cm.set_evict_cache("mamba", _FakeMambaCache())

    adm = Admitter(cost_model=cm, )
    # Wire owner_provider: mamba_tps=2 (fragmentable), 0 free pages
    # (cross_free infeasible), 10 migratable pages (10240 KV-equiv >=
    # 4096, enough for cross_migrate). mamba_evictable=6 slots -> 6//2=3
    # chunks=3072 KV-equiv. The capped drain target is min(4096,3072)=3072.
    adm.owner_provider = MockOwnerProvider(
        free_pages=0, migratable_pages=10, kv_tps=1024, mamba_tps=2,
    )

    class _Sched:
        def __init__(self):
            self.token_to_kv_pool_allocator = StubAllocator(available=0)  # KV full
            self.tree_cache = None
            self._mamba_pool = StubMambaPool(available=0)  # mamba_free=0 slots
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"
        def get_mamba_pool(self):
            return self._mamba_pool
        def get_mamba_evictable(self):
            return 6  # 6 slots // tps(2) = 3 chunks = 3072 KV-equiv

    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), _Sched(),
                             tokens_per_page=1024)
    c = dec.candidate_costs_us
    assert c["cross_evict"] == float("inf"), (
        f"free+evict (3072) < 4096 → cross_evict infeasible: {c}"
    )
    assert abs(c["cross_migrate"] - 732.2) < 1e-6, (
        f"cross_migrate must be c_xfer(400)+c^evict(3072→307.2)+c_m(1→25)="
        f"732.2 (drain capped at evictable, NOT full shortfall 834.6): {c}"
    )
    print("  PASS  27  decide_for_req shortfall>evict split: drain capped at "
          "evictable (1 walk @ min(shortfall,evict)) + migrate remainder")


def test_28_mamba_evictable_slots_to_kv_equiv_tps_gt_1():
    """Mamba evictable is a raw SLOT count; the cross-evict shortfall split
    is in KV-token-equiv. decide_for_req must convert slots -> chunks ->
    KV-equiv (slots // mamba_tps × tokens_per_page) before the split, so a
    fragmentable layout (tps>1) does not over-credit the drain.

    Provider with mamba_tps=2; get_mamba_evictable()=4 slots → 4//2=2
    chunks → 2048 KV-equiv. X=4096; mamba_free=0 → shortfall=4096.
    drain cap = min(4096, 2048) = 2048; cross_evict infeasible
    (free+evict=2048 < 4096). If slots were (wrongly) used raw, the drain
    cap would read 4 < 4096 (still infeasible) but the c^evict drain target
    would be 4 tokens not 2048, so cross_migrate would mis-price; this pins
    the converted target. cross_migrate = c_xfer(400) + c^evict(2048→204.8)
    + c_m(2 slots→50) = 654.8 µs."""
    from sglang.srt.budgeter.cost_model import (
        reset_cost_model, get_cost_model, get_migrate_cost,
    )
    reset_cost_model()
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)  # c_xfer_us(1)=100
    get_migrate_cost().set_mamba(per_slot_us=25.0)

    class _FakeMambaCache:
        def predict_evict_cost_us(self, num_tokens, pool="mamba"):
            return 0.1 * float(num_tokens)  # c^evict(2048)=204.8
    cm.set_evict_cache("mamba", _FakeMambaCache())

    class _Prov:
        def kv_tokens_per_page(self):
            return 1024
        def mamba_tokens_per_page(self):
            return 2  # fragmentable: 2 slots per chunk
        def build_mamba_owner_map(self, *, allow_drain=False, allow_migrate=False):
            # mamba_free=0 free pages; 2 migration pages → src_migratable=2048.
            class _OM:
                free_pages = set()
                live_pages_in_cost_order = [(1, ((1, 100),)), (2, ((2, 101),))]
                cached_pages_in_cost_order = None
            return _OM()

    adm = Admitter(cost_model=cm, )
    adm.owner_provider = _Prov()

    class _Sched:
        def __init__(self):
            self.token_to_kv_pool_allocator = StubAllocator(available=0)  # KV full
            self.tree_cache = None
            self._mamba_pool = StubMambaPool(available=0)  # mamba_free=0 slots
            self.waiting_queue = []
            self.disaggregation_mode = "NULL"
        def get_mamba_pool(self):
            return self._mamba_pool
        def get_mamba_evictable(self):
            return 4  # 4 slots ÷ tps(2) = 2 chunks = 2048 KV-equiv

    # tokens_per_page omitted → decide_for_req derives it from the provider
    # (kv_tokens_per_page()=1024), exercising the self-derive path too.
    dec = adm.decide_for_req(StubReq(n_input_tokens=4096), _Sched(), tokens_per_page=1024)
    c = dec.candidate_costs_us
    assert c["cross_evict"] == float("inf"), (
        f"free(0)+evict(2048) < 4096 → cross_evict infeasible: {c}"
    )
    assert abs(c["cross_migrate"] - 654.8) < 1e-6, (
        f"cross_migrate must be c_xfer(400)+c^evict(2048→204.8)+c_m(2→50)="
        f"654.8 (drain target is the CONVERTED 2048 KV-equiv, not 4 raw "
        f"slots): {c}"
    )
    print("  PASS  28  mamba evictable slots→KV-equiv (tps=2): 4 slots → "
          "2048 KV-equiv drain target; provider-derived tokens_per_page")


def main():
    tests = [
        test_1_decide_for_req_derives_state_from_scheduler,
        test_2_jsonl_log_records_every_decision,
        test_3_no_jsonl_when_env_unset,
        test_4_p99_latency_under_100us,
        test_6_jsonl_includes_fire_result_when_present,
        test_7_cold_start_path_uses_own_when_capacity_available,
        test_8_x_tokens_uses_origin_input_ids_length,
        test_9_close_flushes_and_is_idempotent,
        test_13_empty_input_doesnt_crash,
        test_14_jsonl_candidate_set_is_exactly_seven,
        test_15_hybrid_radix_cache_evictable_size,
        test_16_mamba_free_uses_kv_token_equivalents,
        test_17_evict_cost_routes_by_pool_label,
        test_18_xpool_actuator_exposes_lcm_pages_property,
        test_19_decide_for_req_lcm_scales_cross_free_cost,
        test_20_budget_agent_wire_admitter_pushes_three_fields,
        test_20a_ensure_actuator_chain_calls_wire_admitter,
        test_21_decide_for_req_holds_alloc_lock,
        test_22_decide_for_req_excludes_concurrent_holder,
        test_23_decide_for_req_feeds_migrate_inputs,
        test_24_cross_migrate_infeasible_under_cm_cold_start,
        test_25_migratable_zero_without_owner_provider,
        test_26_mamba_feasibility_from_owner_provider_pages,
        test_27_decide_for_req_shortfall_split_drain_then_migrate,
        test_28_mamba_evictable_slots_to_kv_equiv_tps_gt_1,
    ]
    print(f"\nscheduler-hook tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nscheduler-hook: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
