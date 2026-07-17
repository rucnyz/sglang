"""Phase 1 — CostModel facade + c^xfer producer wiring tests.

Acceptance per plan.md Phase 1:
  1. EWMA convergence: 10 samples of 1000 µs each → c_xfer_us(1) ≈ 1000 µs
  2. Warm-up gate: returns False until 3 obs, True after — exact boundary
  3. c_recompute_us reads from CostCurves (KV vs mamba)
  4. w_q_us reads env override
  5. Producer wiring: simulated fire completion → EWMA updated correctly

Run: .venv/bin/python dev/interlayer/2_admitter/test_cost_model_facade.py

Note: this is TDD — tests are written BEFORE the implementation. They
will fail until CostModel facade is added to cost_model.py.
"""
import math
import os
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _reset():
    """Reset singletons between tests."""
    from sglang.srt.budgeter.cost_model import (
        reset_cost_curves,
        reset_runtime_actuator_cost,
    )
    reset_cost_curves()
    reset_runtime_actuator_cost()


def test_1_xfer_ewma_convergence():
    """Feed 10 samples of 1000 µs/page each → c_xfer_us(1) ≈ 1000."""
    _reset()
    os.environ["SGLANG_XPOOL_NB_CHUNK_COST_INIT_US"] = "3000"
    os.environ["SGLANG_XPOOL_NB_CHUNK_COST_EWMA_ALPHA"] = "0.3"
    from sglang.srt.budgeter.cost_model import CostModel
    cm = CostModel()
    # Feed 10 observations of 1000 µs/page (i.e., n_pages=1, total=1000)
    for _ in range(10):
        cm.update_xfer(total_us=1000.0, n_chunks=1)
    val = cm.c_xfer_us(n_pages=1)
    assert abs(val - 1000.0) < 50.0, (
        f"c_xfer_us(1) = {val:.0f} after 10 × 1000us obs; expected ≈ 1000 ± 50"
    )
    # c_xfer_us(N) should scale linearly
    val_5 = cm.c_xfer_us(n_pages=5)
    assert abs(val_5 - 5000.0) < 250.0, (
        f"c_xfer_us(5) = {val_5:.0f}; expected ≈ 5000 ± 250 (5 × per-page)"
    )
    print(f"  PASS  1  EWMA convergence: c_xfer_us(1)={val:.0f}us, c_xfer_us(5)={val_5:.0f}us")


def test_2_warmup_gate_boundary():
    """Warmup gate: False until 3 obs, True after; never flips back."""
    _reset()
    from sglang.srt.budgeter.cost_model import CostModel
    cm = CostModel()
    assert not cm.is_warmed_up(), "cold start: must be unwarmed"
    cm.update_xfer(1000.0, 1)
    assert not cm.is_warmed_up(), f"after 1 obs: still unwarmed"
    cm.update_xfer(1000.0, 1)
    assert not cm.is_warmed_up(), f"after 2 obs: still unwarmed"
    cm.update_xfer(1000.0, 1)
    assert cm.is_warmed_up(), f"after 3 obs: must be warmed (BOUNDARY)"
    cm.update_xfer(1000.0, 1)
    assert cm.is_warmed_up(), f"after 4 obs: still warmed"
    print("  PASS  2  warmup gate transitions at exactly N=3")


def test_3_c_recompute_reads_curves_per_pool():
    """c_recompute_us(pool='kv', s) ≈ CostCurves.c_kv_ms(s) × 1000.
    c_recompute_us(pool='mamba', s) ≈ c_m_ms(s) × 1000.
    """
    _reset()
    # Override curves via env so the test is deterministic.
    # Env var names per cost_model.py:_try_load_env: SGLANG_CSIGMA_{KV,M}_{ALPHA,BETA,GAMMA}.
    os.environ["SGLANG_CSIGMA_KV_ALPHA"]   = "0.0"
    os.environ["SGLANG_CSIGMA_KV_BETA"]    = "0.5"   # 0.5 ms/token
    os.environ["SGLANG_CSIGMA_KV_GAMMA"]   = "10.0"  # 10 ms constant
    os.environ["SGLANG_CSIGMA_M_ALPHA"]    = "0.1"   # 0.1 ms/token (linear, no quadratic)
    os.environ["SGLANG_CSIGMA_M_BETA"]     = "5.0"   # 5 ms constant
    os.environ["SGLANG_CSIGMA_LSTAR"]      = "1024"
    from sglang.srt.budgeter.cost_model import CostModel
    cm = CostModel()
    # s=100: c_kv = 0.0 × 100² + 0.5 × 100 + 10 = 60 ms → 60000 µs
    # s=100: c_mamba = 0.1 × 100 + 5 = 15 ms → 15000 µs
    kv_us = cm.c_recompute_us(pool="kv", s_tokens=100)
    mamba_us = cm.c_recompute_us(pool="mamba", s_tokens=100)
    assert abs(kv_us - 60000.0) < 1.0, f"kv: got {kv_us}, expected 60000"
    assert abs(mamba_us - 15000.0) < 1.0, f"mamba: got {mamba_us}, expected 15000"
    # Cleanup env (other tests don't want overrides)
    for k in ["SGLANG_CSIGMA_KV_ALPHA","SGLANG_CSIGMA_KV_BETA","SGLANG_CSIGMA_KV_GAMMA",
              "SGLANG_CSIGMA_M_ALPHA","SGLANG_CSIGMA_M_BETA","SGLANG_CSIGMA_LSTAR"]:
        os.environ.pop(k, None)
    print(f"  PASS  3  c_recompute_us reads CostCurves: kv@100={kv_us:.0f}us, mamba@100={mamba_us:.0f}us")


def test_4_w_q_reads_env_override():
    """w_q_us reads SGLANG_XPOOL_QUEUE_WAIT_US env override."""
    _reset()
    from sglang.srt.budgeter.cost_model import CostModel
    # Default (no env): should be 100 (per audit_cost_model.md §4)
    os.environ.pop("SGLANG_XPOOL_QUEUE_WAIT_US", None)
    cm = CostModel()
    default_val = cm.w_q_us()
    assert default_val == 100.0, f"default w_q: got {default_val}, expected 100"
    # Override
    os.environ["SGLANG_XPOOL_QUEUE_WAIT_US"] = "500"
    cm2 = CostModel()
    override_val = cm2.w_q_us()
    assert override_val == 500.0, f"override w_q: got {override_val}, expected 500"
    os.environ.pop("SGLANG_XPOOL_QUEUE_WAIT_US", None)
    print(f"  PASS  4  w_q_us: default=100us, env override=500us")


def test_5_producer_wiring_from_fire_result():
    """The Admitter (or fire_worker_loop) calls cost_model.update_xfer
    with (result.total_us, result.granted_pages). Simulate this and
    verify the EWMA reflects observations.
    """
    _reset()
    os.environ["SGLANG_XPOOL_NB_CHUNK_COST_INIT_US"] = "3000"
    from sglang.srt.budgeter.cost_model import CostModel, get_runtime_actuator_cost
    cm = CostModel()
    # Simulate a series of fire completion records with realistic timings.
    # Each fire: 48 pages, ~50ms total → ~1042 µs/page
    fires = [
        (50_000, 48),  # 50ms / 48 pages = 1042 us/page
        (52_000, 48),  # similar
        (48_000, 48),
        (55_000, 48),
        (51_000, 48),
    ]
    for total_us, n_pages in fires:
        cm.update_xfer(total_us=total_us, n_chunks=n_pages)
    # After 5 obs, EWMA should be near 1042 us/page
    ewma = get_runtime_actuator_cost().current_us
    assert 900 < ewma < 1200, (
        f"EWMA after 5 fires ≈ 1042 us/page expected; got {ewma:.0f}"
    )
    assert cm.is_warmed_up(), "after 5 obs: must be warmed"
    print(f"  PASS  5  producer wiring: 5 fires → EWMA={ewma:.0f}us/page, warmed=True")


def test_6_singleton_shared_across_facade_instances():
    """CostModel is a thin facade over module-level singletons; multiple
    facade instances see the SAME EWMA state.
    """
    _reset()
    from sglang.srt.budgeter.cost_model import CostModel
    cm1 = CostModel()
    cm2 = CostModel()
    assert not cm1.is_warmed_up()
    cm1.update_xfer(1000.0, 1)
    cm1.update_xfer(1000.0, 1)
    cm1.update_xfer(1000.0, 1)
    # cm2 should also be warmed (shared singleton)
    assert cm2.is_warmed_up(), (
        "cm2.is_warmed_up() != cm1.is_warmed_up() — facade is NOT a singleton "
        "view of the underlying state. Must read get_runtime_actuator_cost()."
    )
    assert abs(cm1.c_xfer_us(1) - cm2.c_xfer_us(1)) < 0.01
    print(f"  PASS  6  singleton sharing: cm1 == cm2 facade views of same state")


def test_7_update_xfer_concurrent_producers_no_lost_observations():
    """Audit Category B3 (HIGH) — Phase 6 introduced a second writer
    to `RuntimeActuatorCost.update()`: the Admitter's sync fire path
    now calls update_xfer alongside the Budgeter worker thread. Without
    a lock, `_n_observations += 1` is LOAD/INC/STORE (3 bytecodes, GIL
    can switch) → lost increments. Likewise `_current = α·x + (1-α)·c`
    is read-modify-write; two threads can lose one of their inputs.

    Test: 2 threads × 500 updates each → expect _n_observations==1000.
    Without the lock, this regularly comes back ~990-998 on CPython 3.12.
    With the lock, it's deterministic.
    """
    import threading
    from sglang.srt.budgeter.cost_model import (
        get_runtime_actuator_cost,
        reset_cost_model,
    )

    reset_cost_model()
    cost = get_runtime_actuator_cost()
    N_PER_THREAD = 500
    N_THREADS = 2

    def producer(thread_id):
        # Use unique values so we'd also detect mixed values.
        for i in range(N_PER_THREAD):
            cost.update(total_us=1000.0 + thread_id * 10 + i, n_chunks=1)

    threads = [
        threading.Thread(target=producer, args=(t,))
        for t in range(N_THREADS)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]

    expected = N_THREADS * N_PER_THREAD
    actual = cost.n_observations
    assert actual == expected, (
        f"Concurrent update_xfer lost observations: expected {expected}, "
        f"got {actual} (delta {expected - actual}). EWMA producer "
        f"thread-safety regressed. See audit Category B3."
    )
    assert cost._current > 0, "EWMA current must be finite + positive"
    print(f"  PASS  7  EWMA concurrent producers: {N_THREADS} threads × "
          f"{N_PER_THREAD} updates → {actual}/{expected} observations "
          f"(no lost increments)")


def main():
    tests = [
        test_1_xfer_ewma_convergence,
        test_2_warmup_gate_boundary,
        test_3_c_recompute_reads_curves_per_pool,
        test_4_w_q_reads_env_override,
        test_5_producer_wiring_from_fire_result,
        test_6_singleton_shared_across_facade_instances,
        test_7_update_xfer_concurrent_producers_no_lost_observations,
    ]
    print(f"\nCostModel facade tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 1: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
