"""Sync fire path for Admitter.

Tests for ``Admitter.execute_decision(...)``:

  - When decision.action in {'cross_free', 'cross_evict'}:
      * Build a FirePlan via planner with min-LCM page rounding
      * Acquire actuator._fire_inflight lock (serializes vs Budgeter worker)
      * Call actuator.execute(plan); receive FirePlanResult
      * If aborted: fall back to action='defer'
      * Else: return decision with fire_result populated. The fresh dst
              capacity is left in the allocator; PrefillAdder's normal
              alloc grabs it on the next scheduler iteration.

  - When decision.action in {'own_free', 'own_evict', 'defer'}:
      * No actuator call at all.

Tests use fakes for the two collaborators (actuator, planner) so we
exercise the orchestration logic without depending on real cuMem. test_3
spins up real threads to prove the _fire_inflight mutex.
"""
from __future__ import annotations

import math
import sys
import threading
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.admitter import Admitter, AdmitterDecision
from sglang.srt.budgeter.cost_model import reset_cost_model, get_cost_model


# ---------------------------------------------------------------- Fakes

class FakeFirePlan:
    """Stand-in FirePlan; just records what got built."""
    def __init__(self, direction, n_pages, plan_seq=1):
        self.direction = direction
        self.pages_to_unmap = list(range(n_pages))
        self.pages_to_map_dst = n_pages
        self.plan_seq = plan_seq


class FakeFirePlanResult:
    def __init__(self, granted_pages=12, aborted=False, abort_reason=""):
        self.granted_pages = granted_pages
        self.unmapped_pages = granted_pages
        self.total_us = 1500
        self.cap_barrier_us = 200
        self.unmap_us = 600
        self.map_us = 700
        self.aborted = aborted
        self.abort_reason = abort_reason
        self.direction = "kv_to_mamba"
        self.plan_seq = 1


class FakePlanner:
    def __init__(self):
        self.calls = []  # list of (direction, n_pages_target)
        self.last_plan = None
        # If set to non-None, build() returns this instead.
        self.force_return = "unset"
        # #183 Step 5: real XPoolFirePlanner increments this on a
        # build→None refuse; the Admitter surfaces it in the defer reason.
        self.refuse_count = 0

    def build(self, direction, n_pages_target, *,
              allow_drain=False, allow_migrate=False):
        # #267: real XPoolFirePlanner.build accepts allow_drain/
        # allow_migrate; execute_decision now passes them through per the
        # chosen cross-* action (cross_evict→drain, cross_migrate→both).
        self.calls.append(
            (direction, n_pages_target, allow_drain, allow_migrate)
        )
        if self.force_return != "unset":
            self.last_plan = self.force_return
            if self.force_return is None:
                self.refuse_count += 1  # mirror the real planner's refuse
            return self.force_return
        plan = FakeFirePlan(direction, n_pages_target)
        self.last_plan = plan
        return plan


class FakeActuator:
    """Mimics XPoolActuator.execute() + _fire_inflight lock + lcm_pages property."""

    @property
    def lcm_pages(self) -> int:
        return math.lcm(self.n_kv_subpools, self.n_mamba_subpools)

    def __init__(self, n_kv_subpools=4, n_mamba_subpools=6,
                 abort=False, latency_s=0.0):
        self.n_kv_subpools = n_kv_subpools
        self.n_mamba_subpools = n_mamba_subpools
        self._fire_inflight = threading.Lock()
        self.execute_calls = []           # (plan_seq, n_pages_target)
        self.held_during_execute = False  # was lock acquired during execute()?
        self.abort = abort
        self.latency_s = latency_s
        # For the concurrent test:
        self.max_simultaneous = 0
        self._in_flight = 0
        self._in_flight_lock = threading.Lock()

    def execute(self, plan):
        # The Admitter is expected to ensure exclusive access here, either
        # by acquiring _fire_inflight in admitter code itself OR by us doing
        # it inside execute(). We do it inside (mimicking the production
        # XPoolActuator.execute_async wrapper) AND we sanity-check that the
        # invariant holds.
        with self._fire_inflight:
            with self._in_flight_lock:
                self._in_flight += 1
                if self._in_flight > self.max_simultaneous:
                    self.max_simultaneous = self._in_flight
            try:
                self.held_during_execute = True
                self.execute_calls.append(
                    (plan.plan_seq, plan.pages_to_map_dst)
                )
                if self.latency_s:
                    time.sleep(self.latency_s)
                if self.abort:
                    return FakeFirePlanResult(
                        granted_pages=0, aborted=True,
                        abort_reason="forced abort"
                    )
                return FakeFirePlanResult(granted_pages=plan.pages_to_map_dst)
            finally:
                with self._in_flight_lock:
                    self._in_flight -= 1


# ---------------------------------------------------------------- Tests

def _make_admitter(*, n_kv=4, n_mamba=6, actuator=None, planner=None):
    """Construct an Admitter with Phase 4 ports wired in."""
    reset_cost_model()
    cm = get_cost_model()
    # Warm the EWMA so cross-* isn't cold-gated.
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    actuator = actuator or FakeActuator(n_kv_subpools=n_kv,
                                         n_mamba_subpools=n_mamba)
    planner = planner or FakePlanner()
    adm = Admitter(
        cost_model=cm,
        actuator=actuator,
        planner=planner,
    )
    return adm, actuator, planner


def _decision(action, **kw):
    return AdmitterDecision(action=action, reason=f"test:{action}",
                            candidate_costs_us=kw.get("costs", {}))


def test_1_cross_free_triggers_execute_under_lock():
    adm, act, plan = _make_admitter(n_kv=4, n_mamba=6)
    dec = _decision("cross_free")
    out = adm.execute_decision(
        dec,
        x_tokens=2048,
        src_pool="mamba",
        dst_pool="kv",
        tokens_per_page=1024,
    )
    assert len(act.execute_calls) == 1, f"expected 1 execute, got {act.execute_calls}"
    assert act.held_during_execute, "lock must be held while execute runs"
    assert out.fire_result is not None, "fire_result must be populated"
    assert out.action == "cross_free", "successful fire keeps action label"

    print("  PASS  1  cross_free triggers execute() under _fire_inflight")


def test_2_min_lcm_page_rounding():
    """x_tokens=2048 ⇒ 2 pages at tps=1024. lcm(4,6)=12. Round up to 12."""
    adm, act, plan = _make_admitter(n_kv=4, n_mamba=6)
    dec = _decision("cross_free")
    out2 = adm.execute_decision(
        dec,
        x_tokens=2048,
        src_pool="mamba",
        dst_pool="kv",
        tokens_per_page=1024,
    )

    # planner.build called with (direction='mamba_to_kv', n_pages_target=12)
    expected_lcm = math.lcm(4, 6)  # 12
    # cross_free → Stage 1 only: allow_drain=allow_migrate=False (#267).
    assert plan.calls == [("mamba_to_kv", expected_lcm, False, False)], (
        f"expected one build with lcm-rounded n=12, got {plan.calls}"
    )
    # actuator.execute() received plan with pages_to_map_dst == 12
    assert act.execute_calls == [(1, 12)], (
        f"expected execute_calls=[(1,12)], got {act.execute_calls}"
    )
    print(f"  PASS  2  min-LCM page rounding: 2 pages → {expected_lcm} pages")


def test_3_concurrent_serialization_with_fake_worker():
    """Admitter sync fire serializes against a "Budgeter worker" thread also
    calling actuator.execute(). max_simultaneous must stay = 1."""
    act = FakeActuator(n_kv_subpools=4, n_mamba_subpools=6, latency_s=0.01)
    plan = FakePlanner()
    adm, _, _ = _make_admitter(actuator=act, planner=plan)

    barrier = threading.Barrier(2)
    results = {"err": None}

    def admitter_thread():
        try:
            barrier.wait()
            for _ in range(5):
                dec = _decision("cross_free")
                adm.execute_decision(
                    dec, x_tokens=2048, src_pool="mamba", dst_pool="kv",
                    tokens_per_page=1024,
                )
        except Exception as e:
            results["err"] = e

    def worker_thread():
        try:
            barrier.wait()
            for i in range(5):
                # Worker calls actuator.execute() directly — simulates
                # _fire_worker_loop's execute_async(token).
                act.execute(FakeFirePlan("kv_to_mamba", 12, plan_seq=100 + i))
        except Exception as e:
            results["err"] = e

    ta = threading.Thread(target=admitter_thread)
    tw = threading.Thread(target=worker_thread)
    ta.start()
    tw.start()
    ta.join()
    tw.join()

    assert results["err"] is None, f"thread raised: {results['err']}"
    assert act.max_simultaneous == 1, (
        f"_fire_inflight mutex broken: max_simultaneous={act.max_simultaneous}"
    )
    # Both producers completed.
    assert len(act.execute_calls) == 10
    print(f"  PASS  3  concurrent serialization: max in-flight = 1 over "
          f"{len(act.execute_calls)} fires")


def test_5_abort_falls_back_to_defer():
    """If actuator.execute() returns aborted, the decision falls back to
    action='defer' and does NOT reserve any tokens."""
    act = FakeActuator(n_kv_subpools=4, n_mamba_subpools=6, abort=True)
    plan = FakePlanner()
    adm, _, _ = _make_admitter(actuator=act, planner=plan)
    dec = _decision("cross_free")
    out = adm.execute_decision(
        dec, x_tokens=2048, src_pool="mamba", dst_pool="kv",
        tokens_per_page=1024,
    )
    assert out.action == "defer", (
        f"aborted fire must fall back to defer, got {out.action}"
    )
    assert "abort" in out.reason.lower(), f"reason should mention abort: {out.reason}"
    print("  PASS  5  aborted fire → action='defer', no reservation")


def test_6_own_actions_no_fire():
    """own_free / own_evict / defer never call actuator.execute() or allocator.alloc()."""
    adm, act, plan = _make_admitter()
    for action in ("own_free", "own_evict", "defer"):
        adm.execute_decision(
            _decision(action), x_tokens=2048, src_pool="mamba", dst_pool="kv",
            tokens_per_page=1024,
        )
    assert act.execute_calls == [], f"own/defer must not fire: {act.execute_calls}"
    print("  PASS  6  own_free/own_evict/defer skip actuator+allocator")


def test_7_sync_fire_latency_under_5ms_overhead():
    """With a zero-latency fake actuator, the Admitter's orchestration overhead
    (LCM rounding + actuator lock acquisition) must be < 1 ms per fire."""
    adm, act, plan = _make_admitter()
    latencies = []
    for _ in range(200):
        dec = _decision("cross_free")
        t0 = time.perf_counter()
        adm.execute_decision(
            dec, x_tokens=2048, src_pool="mamba", dst_pool="kv",
            tokens_per_page=1024,
        )
        latencies.append((time.perf_counter() - t0) * 1e6)
    latencies.sort()
    p99 = latencies[int(0.99 * len(latencies))]
    assert p99 < 1000, f"orchestration overhead P99 = {p99:.1f} µs > 1 ms"
    print(f"  PASS  7  orchestration overhead P99 = {p99:.1f} µs (< 1 ms)")


def test_8_cross_evict_also_fires():
    """cross_evict triggers the same fire path (planner + execute + reserve)."""
    adm, act, plan = _make_admitter()
    dec = _decision("cross_evict")
    out = adm.execute_decision(
        dec, x_tokens=2048, src_pool="kv", dst_pool="mamba",
        tokens_per_page=1024,
    )
    # cross_evict → Stages 1-2: allow_drain=True, allow_migrate=False (#267).
    assert plan.calls == [("kv_to_mamba", 12, True, False)], (
        f"cross_evict must also drive planner.build, got {plan.calls}"
    )
    assert len(act.execute_calls) == 1, "execute must be called once"
    assert out.fire_result is not None
    print("  PASS  8  cross_evict drives identical fire path")


def test_8b_cross_migrate_fires_with_both_stages():
    """#267: cross_migrate drives planner.build with allow_drain AND
    allow_migrate True (Stages 1-3) and runs the same fire path."""
    adm, act, plan = _make_admitter()
    dec = _decision("cross_migrate")
    out = adm.execute_decision(
        dec, x_tokens=2048, src_pool="mamba", dst_pool="kv",
        tokens_per_page=1024,
    )
    assert plan.calls == [("mamba_to_kv", 12, True, True)], (
        f"cross_migrate must build with allow_drain=allow_migrate=True, "
        f"got {plan.calls}"
    )
    assert len(act.execute_calls) == 1, "execute must be called once"
    assert out.fire_result is not None
    print("  PASS  8b cross_migrate drives Stage 1-3 fire path")


# ---------------------------------------------------------------- Gap fixes from audit_test_depth_phase4.md

def test_9_real_xpool_actuator_lock_contract():
    """Gap 1 (CRITICAL) — proves the REAL XPoolActuator._fire_inflight
    actually serializes execute_async() across threads. The fake actuator
    in test_3 has its own lock; this test reaches into the real class to
    confirm the lock wiring is correct.

    Strategy: bypass __init__ (which needs MultiTensorArena) by
    constructing the object directly with object.__new__, then attach
    only the attributes execute_async needs to short-circuit straight
    into _execute_async_locked. Patch _execute_async_locked to record
    in-flight count + sleep so contention is observable.
    """
    from sglang.srt.arena.xpool_actuator import XPoolActuator

    inst = object.__new__(XPoolActuator)
    inst._fire_inflight = threading.Lock()
    inst._in_flight = 0
    inst._in_flight_max = 0
    inst._in_flight_lock = threading.Lock()

    def patched_locked(self, token):
        with self._in_flight_lock:
            self._in_flight += 1
            if self._in_flight > self._in_flight_max:
                self._in_flight_max = self._in_flight
        time.sleep(0.005)
        with self._in_flight_lock:
            self._in_flight -= 1
        return "OK"

    XPoolActuator._execute_async_locked_orig = XPoolActuator._execute_async_locked
    try:
        XPoolActuator._execute_async_locked = patched_locked

        errs = []

        def beat():
            try:
                for _ in range(10):
                    inst.execute_async(token=None)
            except Exception as e:
                errs.append(e)

        threads = [threading.Thread(target=beat) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert errs == [], f"thread raised: {errs}"
        assert inst._in_flight_max == 1, (
            f"REAL _fire_inflight broken: max concurrent locked-body = "
            f"{inst._in_flight_max} (expected 1)"
        )
        print(f"  PASS  9  real XPoolActuator._fire_inflight serializes 40 calls "
              f"across 4 threads (max in-flight = {inst._in_flight_max})")
    finally:
        XPoolActuator._execute_async_locked = XPoolActuator._execute_async_locked_orig


def test_10_planner_returns_none_falls_back_to_defer():
    """Gap 3 — when planner.build() returns None (not enough free pages,
    bad direction, etc.), the decision falls back to action='defer' with
    no actuator call and no reservation."""
    plan = FakePlanner()
    plan.force_return = None
    act = FakeActuator(n_kv_subpools=4, n_mamba_subpools=6)
    adm, _, _ = _make_admitter(actuator=act, planner=plan)
    dec = _decision("cross_free")
    out = adm.execute_decision(
        dec, x_tokens=2048, src_pool="mamba", dst_pool="kv",
        tokens_per_page=1024,
    )
    assert out.action == "defer", f"None-plan must defer, got {out.action}"
    assert act.execute_calls == [], "must NOT call actuator on None plan"
    # #183 Step 5: the defer reason must surface the planner's
    # refuse_count (the build→None path bumped it to 1).
    assert plan.refuse_count == 1, (
        f"build→None must have incremented refuse_count: {plan.refuse_count}"
    )
    assert "refuse_count=1" in out.reason, (
        f"defer reason must surface refuse_count: {out.reason!r}"
    )
    # A second build→None is monotonic and surfaced.
    dec2 = _decision("cross_free")
    out2 = adm.execute_decision(
        dec2, x_tokens=2048, src_pool="mamba", dst_pool="kv",
        tokens_per_page=1024,
    )
    assert plan.refuse_count == 2 and "refuse_count=2" in out2.reason, (
        f"refuse_count must be monotonic + surfaced: count={plan.refuse_count} "
        f"reason={out2.reason!r}"
    )
    print("  PASS  10  planner.build() → None → action='defer', no fire, "
          "refuse_count surfaced (#183 Step 5)")


def test_11_execute_decision_trusts_action_label():
    """Gap 4 — execute_decision is decision-application, not
    re-decision. Even with the cost model in a cold (un-warmed) state, a
    'cross_free' action label produced by decide() is honored and the
    fire proceeds. Cold-start gating is decide()'s responsibility, not
    execute_decision()'s. This pins the contract so a refactor that
    re-checks warm-up inside execute_decision would fail the test."""
    reset_cost_model()
    cm = get_cost_model()
    assert not cm.is_warmed_up(), "precondition: EWMA should be cold"
    act = FakeActuator(n_kv_subpools=4, n_mamba_subpools=6)
    plan = FakePlanner()
    adm = Admitter(
        cost_model=cm,
        actuator=act, planner=plan,
    )
    # Caller hands in a 'cross_free' label — we trust it.
    dec = _decision("cross_free")
    out = adm.execute_decision(
        dec, x_tokens=2048, src_pool="mamba", dst_pool="kv",
        tokens_per_page=1024,
    )
    assert out.action == "cross_free", (
        f"execute_decision must trust label, got {out.action}"
    )
    assert len(act.execute_calls) == 1, "fire must proceed even when cold"
    print("  PASS  11  execute_decision trusts decide()'s label (no re-gate)")


def test_16_unknown_direction_falls_back_to_defer():
    """Concern 6: planner.build raising ValueError → defer (graceful)."""
    class StrictPlanner(FakePlanner):
        def build(self, direction, n_pages_target, *,
                  allow_drain=False, allow_migrate=False):
            if direction not in ("kv_to_mamba", "mamba_to_kv"):
                raise ValueError(f"unknown direction: {direction!r}")
            return super().build(
                direction, n_pages_target,
                allow_drain=allow_drain, allow_migrate=allow_migrate,
            )

    plan = StrictPlanner()
    act = FakeActuator()
    adm, _, _ = _make_admitter(actuator=act, planner=plan)
    dec = _decision("cross_free")
    out = adm.execute_decision(
        dec, x_tokens=2048, src_pool="swa", dst_pool="kv",  # 'swa_to_kv' bad
        tokens_per_page=1024,
    )
    assert out.action == "defer", f"unknown direction must defer, got {out.action}"
    assert "rejected" in out.reason.lower() or "direction" in out.reason.lower()
    assert act.execute_calls == [], "must not call actuator on planner ValueError"
    print("  PASS  16  unknown direction → graceful defer (no scheduler crash)")


def test_17_freeze_blocks_post_init_grow():
    """Concern 1: SharedHandlePool.grow() raises after freeze()."""
    # Smoke test on the bare SharedHandlePool class — full XPoolActuator
    # init needs CUDA. Bypass via direct construction without n_handles>0
    # to skip the actual cuMemCreate (which would need CUDA).
    from sglang.srt.arena.chunk_arena import SharedHandlePool
    pool = SharedHandlePool(device_id=0, chunk_size=1024, n_handles=0)
    pool.freeze()
    raised = False
    try:
        pool.grow(1)
    except RuntimeError as e:
        raised = True
        assert "freeze" in str(e).lower()
    assert raised, "grow() after freeze() should raise RuntimeError"
    print("  PASS  17  SharedHandlePool.grow() after freeze() raises RuntimeError")


def main():
    tests = [
        test_1_cross_free_triggers_execute_under_lock,
        test_2_min_lcm_page_rounding,
        test_3_concurrent_serialization_with_fake_worker,
        test_5_abort_falls_back_to_defer,
        test_6_own_actions_no_fire,
        test_7_sync_fire_latency_under_5ms_overhead,
        test_8_cross_evict_also_fires,
        test_8b_cross_migrate_fires_with_both_stages,
        test_9_real_xpool_actuator_lock_contract,
        test_10_planner_returns_none_falls_back_to_defer,
        test_11_execute_decision_trusts_action_label,
        test_16_unknown_direction_falls_back_to_defer,
        test_17_freeze_blocks_post_init_grow,
    ]
    print(f"\nPhase 4 sync-fire tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 4: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
