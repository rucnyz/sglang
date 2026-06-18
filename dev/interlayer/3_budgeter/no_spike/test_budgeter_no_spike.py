"""no_spike — single-tick pressure spike does NOT trigger Budgeter fire.

Negative test (mechanism doesn't make things worse). Conjecture from
design.md §no_spike: "EWMA on raw pressure signals smooths the spike,
keeping the composed pressure_i near baseline."

In the current implementation, the EWMA lives in
`mem_cache/common.py:record_recovery_len_kv` (α=0.05, ~14-event
half-life). The recovery-length `L` is the dominant driver of
`c_KV(L)` (quadratic) and `c_M(L)` (linear); a 10× spike in L
without smoothing would scale `c_KV` by up to 100× and cross the
fire threshold immediately. With α=0.05, a single 10×L event
shifts the EWMA to 1.45×L_base — `c_KV` scales by ~2×, which
must NOT cross the threshold when baseline NB was below it.

This test is structured so that:
  - test_0 verifies the negative control (NO smoothing → spike fires)
  - test_1 verifies the positive guarantee (WITH smoothing → no fire)
  - test_2 verifies sustained spikes eventually fire (smoothing
    doesn't permanently suppress real load)
  - test_3 verifies post-spike state recovers within ~1 half-life

If test_0 doesn't fire we don't know the threshold was reachable —
the test is meaningless. If test_1 fires we've found a real anti-
thrash bug. If test_2 never fires the EWMA is over-suppressing
real load.

Pure-Python unit; no GPU. Uses an in-process XPoolPlanner +
SGLangPressureAdapter wired to a mock tree_cache.
"""
from __future__ import annotations

import os
import sys

# Make sglang imports work from this script
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.cost_model import (
    BUILTIN_DEFAULT,
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

def _fresh_planner(cooldown_ticks=10, nb_margin=1.5,
                   dst_chunks_per_action=4):
    """Build a fresh planner with the actuator-cost + cost-curves
    singletons reset so prior tests don't leak EWMA / curve state.
    Pin curves to BUILTIN_DEFAULT for reproducibility."""
    reset_runtime_actuator_cost()
    reset_cost_curves()
    # Force the singleton to BUILTIN_DEFAULT — no env curves should
    # affect this unit test's behavior
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
        nb_chunk_cost_us=10000.0,
    )
    adapter = SGLangPressureAdapter()
    return XPoolPlanner(config=cfg, adapter=adapter)


class _FakeTreeCache:
    """Minimal stand-in carrying the recovery-length EWMA counters the
    record_recovery_len_* functions read/write. Real caches
    (RadixCache/MambaRadixCache/ChunkCache/SWARadixCache) init these to
    0.0 unconditionally in __init__; the stub mirrors that contract."""

    def __init__(self):
        self._slow_recovery_len_kv_ewma = 0.0
        self._slow_recovery_len_rec_ewma = 0.0
        self._slow_recovery_len_retract_ewma = 0.0


def _settle_ewma(tree, L_base, n_events=200):
    """Feed many baseline events so the EWMA fully converges to L_base."""
    for _ in range(n_events):
        record_recovery_len_kv(tree, int(L_base))
        record_recovery_len_rec(tree, int(L_base))


def _snapshot_from(tree, evicted_tokens=0):
    """Build a snapshot dict the planner expects."""
    return {
        "dt": 2.0,
        "slow_recovery_len_kv": getattr(tree, "_slow_recovery_len_kv_ewma", 0.0),
        "slow_recovery_len_rec": getattr(tree, "_slow_recovery_len_rec_ewma", 0.0),
        "slow_recovery_len_retract": 0,
        "num_evicted_tokens_recent": evicted_tokens,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
    }


# ---------- sub-tests ----------

def test_0_negative_control_no_ewma_spike_fires():
    """Without smoothing, a 10× L spike must cross the fire threshold.
    If this doesn't fire, the test setup can't actually exercise the
    smoothing — the rest of no_spike would be meaningless."""
    planner = _fresh_planner()
    # Pick a usage point that's hot enough to make P_save meaningful
    # but below high-water (otherwise saturation guard kicks in).
    usage_kv = 0.92    # below kv_high_water=0.95
    usage_mamba = 0.50  # well below mamba's low_water, so kv_to_mamba favored
    L_base = 1000.0
    L_spike = L_base * 10.0  # 10× spike

    # Feed the spike L directly (no EWMA path).
    snap_spike = {
        "dt": 2.0,
        "slow_recovery_len_kv": L_spike,
        "slow_recovery_len_rec": L_spike,
        "slow_recovery_len_retract": 0,
        "num_evicted_tokens_recent": int(L_spike),
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
    }
    decision = planner.decide(
        usage_kv=usage_kv, usage_mamba=usage_mamba,
        queue_depth=0, snapshot=snap_spike,
    )
    print(f"    raw-spike L={L_spike:.0f} usage_kv={usage_kv:.2f} → "
          f"direction={decision.direction} reason={decision.reason[:90]}")
    assert decision.direction is not None, (
        f"negative control FAILED: even a raw 10× spike (L={L_spike:.0f}, "
        f"usage_kv={usage_kv:.2f}) didn't fire. The rest of no_spike can't "
        f"verify smoothing is what suppresses the fire — adjust the "
        f"baseline params until this fires.")


def test_1_single_event_spike_through_ewma_no_fire():
    """With α=0.05 EWMA, ONE 10×L event shifts EWMA to 1.45×L_base.
    With the same usage point as test_0, this must NOT fire."""
    planner = _fresh_planner()
    tree = _FakeTreeCache()
    usage_kv = 0.92
    usage_mamba = 0.50
    L_base = 1000.0

    # Settle EWMA at L_base
    _settle_ewma(tree, L_base, n_events=200)
    assert abs(tree._slow_recovery_len_kv_ewma - L_base) < 1.0, (
        f"settle failed: EWMA={tree._slow_recovery_len_kv_ewma:.2f} "
        f"!= L_base={L_base}")

    # Verify baseline (no spike) does NOT fire
    snap = _snapshot_from(tree)
    d0 = planner.decide(usage_kv, usage_mamba, snapshot=snap)
    assert d0.direction is None, (
        f"baseline fired unexpectedly — choose a usage point that's "
        f"below-threshold at L_base. Got direction={d0.direction}, "
        f"reason={d0.reason[:120]}")

    # Inject ONE spike event of 10× L_base
    record_recovery_len_kv(tree, int(L_base * 10))
    record_recovery_len_rec(tree, int(L_base * 10))
    L_post = tree._slow_recovery_len_kv_ewma
    # α=0.05 → expected = 0.05 * 10000 + 0.95 * 1000 = 1450
    expected_post = 0.05 * (L_base * 10) + 0.95 * L_base
    assert abs(L_post - expected_post) < 1.0, (
        f"EWMA post-spike: got {L_post:.2f}, expected ~{expected_post:.2f}")
    print(f"    after 1 spike event: L_EWMA {L_base:.0f} → {L_post:.2f} "
          f"(α=0.05 → 1.45× as predicted)")

    # Critical assertion: planner does NOT fire from the spike-perturbed L
    snap_post = _snapshot_from(tree)
    d1 = planner.decide(usage_kv, usage_mamba, snapshot=snap_post)
    assert d1.direction is None, (
        f"SPIKE BUG: single 10×L event triggered a fire — EWMA didn't "
        f"sufficiently smooth. Got direction={d1.direction}, "
        f"L_post={L_post:.2f}, reason={d1.reason[:120]}")
    print(f"    no fire under smoothed spike (L={L_post:.0f}, "
          f"usage_kv={usage_kv:.2f}) ✓")


def test_2_sustained_high_pressure_eventually_fires():
    """EWMA should suppress single spikes but NOT permanently
    suppress real sustained load. Feed many high-L events; planner
    should eventually fire.

    Predicted fire point: nb_m2k = lifetime · p_save_kv · c_kv(L) must
    cross threshold = nb_margin · dst_chunks_per_action · c_actuator
    = 1.5 · 4 · 10000 = 60000us. With lifetime=10, p_save_kv=0.733,
    need c_kv(L) > 8189us → L > 8068. L_EWMA reaches 8068 after
    log(1 - 8068/9000) / log(0.95) ≈ 30 events. Bound at 80 (gives
    headroom for cost-curve coefficient variation)."""
    planner = _fresh_planner(cooldown_ticks=10)
    tree = _FakeTreeCache()
    usage_kv = 0.92
    usage_mamba = 0.50
    L_base = 1000.0
    _settle_ewma(tree, L_base, n_events=200)

    fired_tick = -1
    for tick in range(200):
        record_recovery_len_kv(tree, int(L_base * 10))
        record_recovery_len_rec(tree, int(L_base * 10))
        snap = _snapshot_from(tree)
        d = planner.decide(usage_kv, usage_mamba, snapshot=snap)
        if d.direction is not None:
            fired_tick = tick
            print(f"    fired at tick {tick} (L_EWMA="
                  f"{tree._slow_recovery_len_kv_ewma:.0f}) ✓")
            break

    assert fired_tick >= 0, (
        "sustained high load NEVER fired in 200 ticks — EWMA is "
        "permanently over-suppressing real load.")
    # Closed-form predicts ~30 ticks; bound at 45 to catch an α
    # regression (e.g., halving α → ~60 ticks) without spurious failure
    # from cost-curve coefficient drift (worst-case ~10-tick shift).
    assert fired_tick < 45, (
        f"fire took {fired_tick} ticks to land — convergence too slow "
        f"to be useful in practice (closed-form ≈30, bound 45)")


def test_3_post_spike_ewma_recovers_to_baseline():
    """After 1 spike event, returning to baseline events should
    re-converge geometrically. Half-life at α=0.05 is ln(0.5)/ln(0.95)
    ≈ 13.5 events. After 1 half-life the spike's contribution to drift
    must halve; after 3 half-lives (≈40 events) drift must be <15% of
    the initial perturbation; after ≥150 events drift must be <1%."""
    tree = _FakeTreeCache()
    L_base = 1000.0
    _settle_ewma(tree, L_base, n_events=200)
    record_recovery_len_kv(tree, int(L_base * 10))  # 1 spike
    initial_drift = tree._slow_recovery_len_kv_ewma - L_base  # ≈450

    # Feed 14 baseline events ≈ 1 half-life
    for _ in range(14):
        record_recovery_len_kv(tree, int(L_base))
    drift_1hl = tree._slow_recovery_len_kv_ewma - L_base
    ratio_1hl = drift_1hl / initial_drift
    print(f"    after 14 events (~1 half-life): drift "
          f"{initial_drift:.1f} → {drift_1hl:.1f} (ratio {ratio_1hl:.3f})")
    assert 0.40 < ratio_1hl < 0.60, (
        f"half-life ratio off: {ratio_1hl:.3f} not in [0.40, 0.60]")

    # Continue to 150 events total → drift should be sub-1%
    for _ in range(150 - 14):
        record_recovery_len_kv(tree, int(L_base))
    drift_final = tree._slow_recovery_len_kv_ewma - L_base
    drift_pct = abs(drift_final) / L_base * 100
    print(f"    after 150 events total: L_EWMA = "
          f"{tree._slow_recovery_len_kv_ewma:.2f} (drift "
          f"{drift_pct:.4f}%)")
    assert drift_pct < 1.0, (
        f"after 150 baseline events, EWMA still {drift_pct:.2f}% off "
        f"L_base — convergence broken")


def test_4_mamba_side_ewma_smooths_independently():
    """Coverage gap caught by review: tests 1-3 spike both
    record_recovery_len_kv AND record_recovery_len_rec in lockstep,
    so a bug where ONLY the kv EWMA wires correctly would pass. Here
    we spike only the rec (mamba-side) EWMA and verify the parallel
    smoothing applies."""
    tree = _FakeTreeCache()
    L_base = 1000.0
    # Settle rec-side only
    for _ in range(200):
        record_recovery_len_rec(tree, int(L_base))
    pre = tree._slow_recovery_len_rec_ewma
    assert abs(pre - L_base) < 1.0, f"rec settle failed: {pre:.2f}"

    record_recovery_len_rec(tree, int(L_base * 10))  # spike rec only
    post = tree._slow_recovery_len_rec_ewma
    expected = 0.05 * 10000 + 0.95 * 1000
    assert abs(post - expected) < 1.0, (
        f"rec-side EWMA differs from kv-side formula: got {post:.2f}, "
        f"expected {expected:.2f} — rec wiring may use different α")
    print(f"    mamba-side rec EWMA: {pre:.0f} → {post:.2f} "
          f"(same α=0.05 as kv) ✓")


def test_5_cold_start_first_event_seeds_directly_no_alpha_dilution():
    """Cold-start branch (common.py:322 `prev <= 0` → set L directly).
    Without this branch, the first event would be diluted against the
    initial 0.0 prev (effective L = 0.05 · L), which would silently
    suppress the very first observation. Verify the cold-start path
    correctly seeds to L_first."""
    tree = _FakeTreeCache()
    L_first = 5000.0
    record_recovery_len_kv(tree, int(L_first))
    assert abs(tree._slow_recovery_len_kv_ewma - L_first) < 1e-9, (
        f"cold-start: first call should seed L_EWMA = L_first; "
        f"got {tree._slow_recovery_len_kv_ewma:.4f}, expected {L_first}. "
        f"This means α-smoothing was applied against initial=0 — fix "
        f"the prev<=0 branch in record_recovery_len_kv.")
    print(f"    cold-start: 1st event seeded L_EWMA = "
          f"{tree._slow_recovery_len_kv_ewma:.0f} (no dilution) ✓")


# ---------- runner ----------

def main():
    tests = [
        ("0 negative-control: raw 10× spike DOES fire (threshold reachable)",
         test_0_negative_control_no_ewma_spike_fires),
        ("1 EWMA absorbs single 10× spike — no fire",
         test_1_single_event_spike_through_ewma_no_fire),
        ("2 sustained 10× load eventually fires (EWMA not over-suppressing)",
         test_2_sustained_high_pressure_eventually_fires),
        ("3 EWMA recovers to baseline within ~1 half-life",
         test_3_post_spike_ewma_recovers_to_baseline),
        ("4 mamba-side EWMA smooths independently of kv-side",
         test_4_mamba_side_ewma_smooths_independently),
        ("5 cold-start first event seeds directly (no α dilution)",
         test_5_cold_start_first_event_seeds_directly_no_alpha_dilution),
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
    print(f"\nD6c: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
