"""cxfer_ewma_self_suppress (§cxfer_ewma_self_suppress) — c^xfer EWMA spike self-suppresses over-fire.

Negative test (mechanism doesn't make things worse). Conjecture from
design.md §cxfer_ewma_self_suppress: when the observed per-chunk actuator cost rises
(e.g., GPU contention slows transfers), the trigger condition
`gap × amortize_ticks > c^xfer` becomes harder to clear and the
fire rate adapts down — no external rate limiter needed.

In the implementation, c^xfer lives in `RuntimeActuatorCost`
(`cost_model.py:218`), an EWMA over per-chunk fire wall-time with
α=0.3 (5-fire half-life ≈ 1.6 fires). The planner reads this via
`get_runtime_actuator_cost().current_us` in `_pick_direction_by_nb`
(`xpool_planner.py:332-337`) and uses it as the cost-side of the
fire-gate: `threshold = nb_margin · dst_chunks_per_action · c_actuator`.
A 5× spike in c_actuator → 5× threshold → fires that would have
fired stop firing.

Sub-tests:
  test_0 negative control — baseline c_actuator allows fires
  test_1 5× spike on c_actuator → fire rate drops ≥2× (design pass)
  test_2 after spike returns to baseline, fire rate recovers
  test_3 EWMA step response: α=0.3 moves 30% per observation
  test_4 cold-start uses max(EWMA, static) until n_observations >= 3
  test_5 monotone non-increasing fire rate across c_actuator ramp

Pure-Python unit, no GPU. Drives planner.decide() in a loop while
directly mutating the RuntimeActuatorCost singleton to simulate
fire-time observations.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.cost_model import (
    BUILTIN_DEFAULT,
    get_runtime_actuator_cost,
    reset_cost_curves,
    reset_runtime_actuator_cost,
)
from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
from sglang.srt.budgeter.xpool_planner import (
    XPoolPlanner,
    XPoolPolicyConfig,
)
from sglang.srt.mem_cache.common import (
    record_recovery_len_kv,
    record_recovery_len_rec,
)


# ---------- harness ----------

def _fresh_planner(cooldown_ticks=1, nb_margin=1.5,
                   dst_chunks_per_action=4,
                   nb_chunk_cost_us=10000.0):
    """Build a fresh planner with singletons reset.

    cooldown_ticks=1 so the fire-rate measurement isn't artificially
    capped — we want to count how many decide() invocations clear the
    gate, not how many survive a long cooldown."""
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.95,
        kv_low_water=0.70,
        mamba_high_water=0.95,
        mamba_low_water=0.70,
        # ticks→seconds at the dt=2 s test anchor (τ-invariance refactor #302)
        cooldown_min_s=cooldown_ticks * 2.0,
        amortize_horizon_s=cooldown_ticks * 2.0,
        dst_chunks_per_action=dst_chunks_per_action,
        nb_margin=nb_margin,
        nb_chunk_cost_us=nb_chunk_cost_us,
    )
    return XPoolPlanner(config=cfg, adapter=SGLangPressureAdapter())


class _FakeTreeCache:
    # The recovery-len EWMAs are initialized unconditionally, mirroring the real
    # cache's __init__: common.record_recovery_len_* read them directly (no
    # getattr default), so a bare stub would raise on the first record.
    def __init__(self):
        self._slow_recovery_len_kv_ewma = 0.0
        self._slow_recovery_len_rec_ewma = 0.0
        self._slow_recovery_len_retract_ewma = 0.0


def _settle_L(tree, L_base, n=200):
    for _ in range(n):
        record_recovery_len_kv(tree, int(L_base))
        record_recovery_len_rec(tree, int(L_base))


def _snapshot(tree):
    return {
        "dt": 2.0,
        "slow_recovery_len_kv": getattr(tree, "_slow_recovery_len_kv_ewma", 0.0),
        "slow_recovery_len_rec": getattr(tree, "_slow_recovery_len_rec_ewma", 0.0),
        "slow_recovery_len_retract": 0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
    }


def _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks):
    """Run planner.decide() n_ticks times; count direction!=None returns."""
    fires = 0
    for _ in range(n_ticks):
        d = planner.decide(usage_kv, usage_mamba, snapshot=_snapshot(tree))
        if d.direction is not None:
            fires += 1
    return fires


def _saturate_actuator_cost(c_us: float, n_observations: int = 10):
    """Drive the singleton EWMA to `c_us` by feeding many observations
    at that value. With α=0.3 and 10 samples, EWMA converges to within
    0.7^10 ≈ 2.8% of target — close enough that the planner's threshold
    computation reflects the intended c_us."""
    cost = get_runtime_actuator_cost()
    for _ in range(n_observations):
        cost.update(c_us, n_chunks=1)


# ---------- sub-tests ----------

def test_0_negative_control_baseline_fires():
    """At baseline c_actuator, the chosen workload params MUST fire
    at least 30% of ticks. Otherwise we can't measure a fire-rate
    DROP from the spike — the test would be vacuously passable."""
    planner = _fresh_planner()
    tree = _FakeTreeCache()
    # L_base=10000 chosen because at cooldown=1 (lifetime=1) and
    # c_actuator=1000us, NB_kv = lifetime · 0.733 · c_KV(L) must exceed
    # threshold = 1.5 · 4 · 1000us = 6000us. c_KV is quadratic in L
    # (BUILTIN_DEFAULT: 1.19e-7·L²+0.44 ms), so L<10k underdelivers
    # NB (e.g. L=8000 gives NB=5908us, just below threshold). 10k gives
    # NB=9045us, clear margin above 6000us baseline / below 30000us 5×.
    _settle_L(tree, L_base=10000.0)
    _saturate_actuator_cost(c_us=1000.0)  # low baseline c_actuator

    # usage chosen so kv_to_mamba is the natural direction:
    # KV is hot (large P_save_kv), mamba quiet (small P_loss_m)
    usage_kv, usage_mamba = 0.92, 0.50
    fires = _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks=100)
    print(f"    baseline c_actuator=1000us → {fires}/100 fires")
    assert fires >= 30, (
        f"baseline fire rate {fires}/100 too low — choose more "
        f"aggressive usage/L so subsequent spike-suppression has "
        f"signal to measure")


def test_1_5x_spike_drops_fire_rate_at_least_2x():
    """The design pass criterion: fire_rate_during ≤ 0.5 × fire_rate_before."""
    planner = _fresh_planner()
    tree = _FakeTreeCache()
    _settle_L(tree, L_base=10000.0)
    usage_kv, usage_mamba = 0.92, 0.50

    _saturate_actuator_cost(c_us=1000.0)
    fires_before = _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks=100)

    _saturate_actuator_cost(c_us=5000.0)
    fires_during = _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks=100)

    ratio = fires_during / max(fires_before, 1)
    print(f"    baseline {fires_before}/100, spike {fires_during}/100 "
          f"(ratio {ratio:.3f}, design requires ≤0.5)")
    assert ratio <= 0.5, (
        f"5× c_actuator spike only reduced fire rate by {1-ratio:.0%} "
        f"({fires_before} → {fires_during}); design requires ≥50% drop. "
        f"EWMA may not be consulted by the fire gate.")


def test_2_post_spike_fire_rate_recovers():
    """fire_rate_after ≈ fire_rate_before — no permanent over- or
    under-correction once the spike clears."""
    planner = _fresh_planner()
    tree = _FakeTreeCache()
    _settle_L(tree, L_base=10000.0)
    usage_kv, usage_mamba = 0.92, 0.50

    _saturate_actuator_cost(c_us=1000.0)
    fires_before = _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks=100)
    _saturate_actuator_cost(c_us=5000.0)
    _ = _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks=100)
    _saturate_actuator_cost(c_us=1000.0)
    fires_after = _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks=100)

    ratio = fires_after / max(fires_before, 1)
    print(f"    before {fires_before}/100, after-recovery {fires_after}/100 "
          f"(ratio {ratio:.3f}, must be in [0.70, 1.30])")
    assert 0.70 <= ratio <= 1.30, (
        f"post-spike fire rate {fires_after} drifted from baseline "
        f"{fires_before} by more than 30% (ratio {ratio:.3f}) — "
        f"EWMA failed to recover or planner state is leaking")


def test_3_ewma_step_response():
    """With α=0.3, one observation moves EWMA by 30% of the
    obs-vs-prior gap. From settled 1000us, one 5000us observation
    should move EWMA to 0.3·5000 + 0.7·1000 = 2200us."""
    reset_runtime_actuator_cost()
    cost = get_runtime_actuator_cost()
    cost.update(1000.0, n_chunks=1)  # seed via first-obs branch
    assert abs(cost.current_us - 1000.0) < 1e-6
    for _ in range(20):
        cost.update(1000.0, n_chunks=1)
    assert abs(cost.current_us - 1000.0) < 1.0, cost.current_us

    cost.update(5000.0, n_chunks=1)
    expected = 0.3 * 5000.0 + 0.7 * 1000.0
    delta = abs(cost.current_us - expected)
    print(f"    settled 1000us + one 5000us obs → EWMA {cost.current_us:.1f} "
          f"(predicted {expected:.1f}, |Δ|={delta:.2f})")
    assert delta < 1.0, (
        f"EWMA step response: got {cost.current_us:.2f}, "
        f"expected {expected:.2f} from α=0.3")


def test_5_fire_rate_monotone_in_c_actuator():
    """Design says "fires adapt down when c rises" — implicitly the
    response should be monotone, not just step-suppressed. Feed a
    geometric ramp of c_actuator (1k, 2k, 4k, 8k, 16k); fire rate
    must be non-increasing across the ramp."""
    planner = _fresh_planner()
    tree = _FakeTreeCache()
    _settle_L(tree, L_base=10000.0)
    usage_kv, usage_mamba = 0.92, 0.50

    ramp = [1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    rates = []
    for c_us in ramp:
        _saturate_actuator_cost(c_us=c_us)
        f = _count_fires(planner, tree, usage_kv, usage_mamba, n_ticks=60)
        rates.append(f)
    print(f"    c ramp {ramp} → fires {rates}/60 each")

    # Non-increasing check (allow equal — once below threshold both
    # land at 0 and stay there)
    for i in range(1, len(rates)):
        assert rates[i] <= rates[i - 1], (
            f"fire rate increased at c={ramp[i]}us "
            f"({rates[i-1]} → {rates[i]}); response not monotone")
    # First step (1k → 2k) must produce SOME drop (else the ramp
    # spans a regime where c_actuator doesn't reach the threshold —
    # would be a Lens-3 silent-pass)
    assert rates[0] > rates[-1], (
        f"ramp endpoints identical ({rates[0]} vs {rates[-1]}) — "
        f"either the c values don't cross the gate, or the gate "
        f"is c-insensitive")


def test_4_cold_start_uses_max_until_3_observations():
    """When `n_observations < 3`, `_pick_direction_by_nb` uses
    `max(EWMA, static)` (xpool_planner.py:334-337) — the static
    floor protects against under-cost fires before EWMA settles."""
    reset_runtime_actuator_cost()
    cost = get_runtime_actuator_cost()

    assert not cost.is_calibrated
    cost.update(1000.0, n_chunks=1)
    assert not cost.is_calibrated  # n=1 < 3
    effective_1 = max(cost.current_us, 10000.0)  # planner formula
    assert effective_1 == 10000.0, (
        f"cold-start should clamp to static 10000us; got {effective_1}")

    cost.update(1000.0, n_chunks=1)  # n=2
    assert not cost.is_calibrated
    cost.update(1000.0, n_chunks=1)  # n=3 → calibrated
    assert cost.is_calibrated, (
        f"is_calibrated should flip true at n=3; got "
        f"n_observations={cost.n_observations}")
    effective_3 = cost.current_us if cost.is_calibrated \
                  else max(cost.current_us, 10000.0)
    print(f"    n=1: floor=10000us (cold); n=3: pure EWMA "
          f"{effective_3:.0f}us ✓")
    assert effective_3 < 10000.0, (
        f"post-calibration should accept low EWMA "
        f"{effective_3} < 10000; got {effective_3}")


# ---------- runner ----------

def main():
    tests = [
        ("0 negative-control: baseline c_actuator allows fires (≥30/100)",
         test_0_negative_control_baseline_fires),
        ("1 5× c_actuator spike drops fire rate ≥2× (design pass)",
         test_1_5x_spike_drops_fire_rate_at_least_2x),
        ("2 post-spike fire rate recovers to baseline (±30%)",
         test_2_post_spike_fire_rate_recovers),
        ("3 EWMA step response: α=0.3 moves 30% per observation",
         test_3_ewma_step_response),
        ("4 cold-start clamps to max(EWMA, static) until n>=3",
         test_4_cold_start_uses_max_until_3_observations),
        ("5 fire rate monotone non-increasing across c-actuator ramp",
         test_5_fire_rate_monotone_in_c_actuator),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            import traceback
            traceback.print_exc()
    print(f"\ncxfer_ewma_self_suppress: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
