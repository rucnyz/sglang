"""Reproducing test for #302 — the Budgeter control loop is tau-coupled.

The Budgeter re-decides every tick of length tau = SGLANG_HIMA_TICK_S
seconds. Its net-benefit (NB) gate should depend on the REAL workload
pressure (rates per second, payback horizons in seconds), NOT on the tick
granularity tau, which is an implementation detail of how often we poll.
Today it does. Two coupled defects:

  (1) FLOW signals are raw per-tick counts with no /dt conversion. Over a
      30 s tick `num_evicted_tokens_recent` accumulates ~15x what it does
      over a 2 s tick for the SAME real eviction rate, so the per-tick NB
      benefit scales with tau.

  (2) The amortization horizon is `lifetime = cooldown_ticks` (a tick
      count) and the cooldown gate decrements once per decide() call, so
      both the payback window and the inter-fire lockout are tau-blind tick
      counts whose wall-clock meaning silently scales with tau.

Net effect: the SAME real workload observed at two different tau produces
different fire decisions. tau should set polling frequency only.

Test-first protocol (bug-workflow):
  1. On current code BOTH tests below FAIL (tau-coupled).
  2. After the tau-invariant reparameterization (#302) BOTH PASS.

Contract the fix must satisfy:
  - The planner consumes snapshot["dt"] (wall seconds since the last tick).
  - FLOW signals (num_evicted_tokens_recent, retract/evict deltas) are
    priced as per-second rates (count / dt).
  - The payback horizon is `amortize_horizon_s` (seconds) and the inter-fire
    lockout is `cooldown_min_s` (seconds); both are wall-clock, tau-derived.
  - Result: identical real pressure -> identical NB and identical fire
    decision regardless of tau.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.cost_model import (
    BUILTIN_DEFAULT,
    reset_cost_curves,
    reset_runtime_actuator_cost,
)
from sglang.srt.budgeter.agent import _grow_priced_us
from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
from sglang.srt.budgeter.xpool_planner import XPoolPlanner, XPoolPolicyConfig

# Two tick granularities that bracket the harness (2 s) and production
# default (30 s). The bug shows as a ~15x divergence between them.
TAU_FAST = 2.0
TAU_SLOW = 30.0
EVICT_RATE_TOK_PER_S = 1000.0   # the SAME real eviction rate in both cells


def _fresh_planner():
    """Cold planner on the builtin curves, scalar evict path (no calibrated
    c_sigma), k2m-reachable config. Mirrors no_spike/_fresh_planner."""
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.95, kv_low_water=0.70,
        mamba_high_water=0.95, mamba_low_water=0.70,
        cooldown_min_s=32.0, amortize_horizon_s=32.0,
        dst_chunks_per_action=4,
        nb_margin=1.5, nb_chunk_cost_us=10000.0,
    )
    return XPoolPlanner(config=cfg, adapter=SGLangPressureAdapter())


def _nb_k2m_for_tau(tau: float) -> float:
    """Drive ONE decide() that models a steady real eviction rate observed
    over a tick of length `tau`, and return the planner's NB[k2m] in us.

    The only live signal is tree-cache eviction (queue/paused/retract = 0,
    L = 0 so the c_sigma terms drop out). mamba active is above low-water so
    the eviction pressure is attributed to growing mamba (k2m); kv is slack.
    Over a tau-second tick the engine evicts EVICT_RATE_TOK_PER_S * tau
    tokens, exactly as agent._snapshot's `num_evicted_tokens_recent` delta
    would report. `dt` carries the tick length to the planner.
    """
    planner = _fresh_planner()
    evicted_this_tick = int(round(EVICT_RATE_TOK_PER_S * tau))
    snap = {
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": evicted_this_tick,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
        "usage_kv_active": 0.50,     # slack donor
        "usage_mamba_active": 0.90,  # above low-water -> pressure lands here
        "dt": tau,                   # wall seconds since last tick (fix consumes this)
    }
    dec = planner.decide(
        usage_kv=0.50, usage_mamba=0.90, queue_depth=0, snapshot=snap,
    )
    m = re.search(r"NB\[k2m\]=([-0-9.einf]+)us", dec.reason)
    assert m, f"could not parse NB[k2m] from: {dec.reason!r}"
    return float(m.group(1))


def test_flow_signal_nb_is_tau_invariant():
    """Same real eviction rate, two tick lengths -> NB[k2m] must match.

    Pre-fix: the planner prices the raw per-tick token count, so NB at
    tau=30 is ~15x NB at tau=2 for IDENTICAL real pressure -> FAIL.
    Post-fix: the flow is converted to a per-second rate (/dt) and the
    horizon is in seconds, so NB is identical -> PASS.
    """
    nb_fast = _nb_k2m_for_tau(TAU_FAST)
    nb_slow = _nb_k2m_for_tau(TAU_SLOW)
    # Guard against a degenerate zero read masking the comparison.
    assert nb_fast > 0.0, f"setup error: NB[k2m] should be positive, got {nb_fast}"
    ratio = nb_slow / nb_fast
    print(f"    NB[k2m] tau={TAU_FAST}s -> {nb_fast:.0f}us ; "
          f"tau={TAU_SLOW}s -> {nb_slow:.0f}us ; ratio={ratio:.2f} "
          f"(tau ratio={TAU_SLOW / TAU_FAST:.1f})")
    assert abs(ratio - 1.0) < 0.05, (
        f"TAU-COUPLED (#302): identical real eviction rate "
        f"({EVICT_RATE_TOK_PER_S:.0f} tok/s) priced {ratio:.1f}x differently "
        f"at tau={TAU_SLOW}s vs tau={TAU_FAST}s (NB {nb_slow:.0f} vs "
        f"{nb_fast:.0f} us). The flow signal is a raw per-tick count with no "
        f"/dt conversion, so the NB benefit scales with the tick granularity. "
        f"Fix: price flows as per-second rates and the horizon in seconds so "
        f"NB is invariant to tau."
    )


def test_cooldown_is_wall_time_bounded_not_tick_bounded():
    """After a fire, the inter-fire lockout must be a fixed WALL-CLOCK
    duration regardless of tau.

    We fire once, then advance wall time in `dt`-sized steps and count how
    many decide() calls stay in cooldown. Pre-fix the cooldown is
    `cooldown_ticks` decide() calls regardless of dt, so the wall-clock
    lockout = cooldown_ticks * tau differs by 15x between tau=2 and tau=30.
    Post-fix the lockout is `cooldown_min_s` seconds, so the two taus reach
    the same wall-clock lockout (the call count scales as cooldown_min_s/dt).
    """
    def _wall_cooldown_seconds(tau: float) -> float:
        planner = _fresh_planner()
        # A scenario that fires k2m on the first decision: strong eviction
        # pressure attributed to mamba.
        fire_snap = {
            "slow_recovery_len_kv": 0.0,
            "slow_recovery_len_rec": 0.0,
            "num_evicted_tokens_recent": int(round(EVICT_RATE_TOK_PER_S * tau)),
            "num_retracted_reqs": 0,
            "num_paused_reqs": 0,
            "num_queue_reqs": 200,
            "usage_kv_active": 0.50,
            "usage_mamba_active": 0.90,
            "dt": tau,
        }
        first = planner.decide(0.50, 0.90, queue_depth=200, snapshot=fire_snap)
        assert first.direction == "kv_to_mamba", (
            f"setup: expected a k2m fire at tau={tau}s to start the cooldown, "
            f"got {first.direction!r} ({first.reason[:160]})"
        )
        # Now keep deciding under the SAME pressure; count ticks suppressed
        # by cooldown until a fire is allowed again.
        suppressed_ticks = 0
        for _ in range(1000):
            d = planner.decide(0.50, 0.90, queue_depth=200, snapshot=dict(fire_snap))
            if d.direction is not None:
                break
            suppressed_ticks += 1
        return suppressed_ticks * tau

    wall_fast = _wall_cooldown_seconds(TAU_FAST)
    wall_slow = _wall_cooldown_seconds(TAU_SLOW)
    print(f"    wall cooldown tau={TAU_FAST}s -> {wall_fast:.0f}s ; "
          f"tau={TAU_SLOW}s -> {wall_slow:.0f}s")
    # The wall-clock lockout should be the same physical duration at both
    # taus (within one tick of quantization at the coarser tau).
    assert abs(wall_slow - wall_fast) <= TAU_SLOW + 1e-6, (
        f"TAU-COUPLED cooldown (#302): inter-fire lockout is {wall_fast:.0f}s "
        f"at tau={TAU_FAST}s but {wall_slow:.0f}s at tau={TAU_SLOW}s for the "
        f"same workload. The cooldown is counted in decide() calls "
        f"(cooldown_ticks), so its wall-clock duration scales with tau. Fix: "
        f"express the lockout as cooldown_min_s seconds."
    )


def test_grow_signal_rate_is_tau_invariant():
    """The reuse-aware GROW signal (agent._grow_priced_us) is a per-tick
    eviction FLOW the planner divides by dt. For the SAME real eviction rate,
    the priced grow RATE (result / dt) must be identical across tau.

    Regression guard for the saturated tiny-grant regime (audit F-1): when one
    fire's grant is small (e.g. an atomic mamba layout, grant=1 slot), an
    integer per-tick cap floored to 1 (max(1, int(grant*dt/cool_s))) becomes a
    tau-independent constant, so result/dt scales as 1/dt — a tau leak. The
    fractional-cap fix prices min(recent, grant*dt/cool_s) victims so the rate
    stays tau-invariant.

    Pre-fix (floor-to-1 cap) this FAILS; post-fix (fractional cap) it PASSES.
    """
    price = lambda n: n * 7.0          # linear stub for predict_evict_cost_us
    cool_s = 32.0
    grant = 1                          # tiny grant — the floor-leak regime
    real_rate = 25.0                   # victims/s, the SAME real pressure
    rates = []
    for tau in (1.0, 2.0, 30.0):
        recent = int(round(real_rate * tau))   # per-tick flow ∝ dt
        grow_us = _grow_priced_us(recent, grant, tau, cool_s, price)
        assert grow_us is not None, f"setup: expected a grow signal at tau={tau}"
        rates.append(grow_us / tau)            # the planner's ÷dt → rate
    print(f"    grow rate(÷dt) across tau={[1.0, 2.0, 30.0]}: "
          f"{[round(r, 4) for r in rates]}")
    spread = max(rates) - min(rates)
    assert spread < 1e-6, (
        f"TAU-COUPLED grow signal (#302 audit F-1): identical real eviction "
        f"rate ({real_rate} victims/s) priced to different grow RATES across "
        f"tau ({[round(r, 4) for r in rates]}) — the per-tick cap is not "
        f"proportional to dt (a floor-to-1 makes it a tau-constant, so ÷dt "
        f"leaks 1/dt). Fix: price min(recent, grant*dt/cool_s) victims "
        f"(fractional cap, no floor)."
    )


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
