"""Unit tests for SGLANG_XPOOL_EDGE_TRIGGER edge-triggered planner.

Verifies:
  T1: edge_trigger=0 (default) — legacy level-triggered path unchanged
  T2: steady-state above high (no transitions) → 0 transfers
  T3: rising edge IN_BAND→ABOVE_HIGH on mamba → fires kv_to_mamba once
  T4: hysteresis — usage drops back into IN_BAND, no fire; then back to
      ABOVE_HIGH → fires again (single-direction edge re-trigger is OK)
  T5: reverse trigger — mamba ABOVE_HIGH→BELOW_LOW while kv ABOVE_HIGH
      → fires mamba_to_kv
  T6: bouncy steady-state (usage oscillates within IN_BAND) → 0 transfers
  T7: PlanDecision.queue_depth still propagated under edge_trigger
"""
from __future__ import annotations
import os
import sys


def setup_planner(edge: bool, **kwargs):
    os.environ["SGLANG_XPOOL_EDGE_TRIGGER"] = "1" if edge else "0"
    os.environ["SGLANG_XPOOL_KV_HIGH"] = str(kwargs.get("kv_high", 0.85))
    os.environ["SGLANG_XPOOL_KV_LOW"] = str(kwargs.get("kv_low", 0.50))
    os.environ["SGLANG_XPOOL_MAMBA_HIGH"] = str(kwargs.get("mb_high", 0.80))
    os.environ["SGLANG_XPOOL_MAMBA_LOW"] = str(kwargs.get("mb_low", 0.40))
    os.environ["SGLANG_XPOOL_COOLDOWN"] = str(kwargs.get("cooldown", 0))
    os.environ.pop("SGLANG_XPOOL_QDEPTH_TRIGGER", None)
    from importlib import reload
    import sglang.srt.budgeter.cross_pool_planner as cpp
    reload(cpp)
    return cpp.CrossPoolPlanner()


def test_legacy_default():
    print("== T1: edge_trigger=0 → legacy level-triggered preserved ==")
    p = setup_planner(edge=False)
    # Repeated above-threshold reads should keep firing (legacy behavior).
    fires = 0
    for _ in range(10):
        d = p.decide(usage_kv=0.30, usage_mamba=0.95)
        if d.direction == "kv_to_mamba":
            fires += 1
    print(f"  10 reads at mamba=0.95/kv=0.30: {fires} legacy transfers")
    assert fires >= 5, f"legacy should fire repeatedly (got {fires})"
    print("  → PASS")
    print()


def test_steady_state_above_high():
    print("== T2: edge_trigger=1, steady mamba=0.95 → 0 transfers after first crossing ==")
    p = setup_planner(edge=True)
    # First tick: IN_BAND → ABOVE_HIGH (rising edge), should fire once.
    d0 = p.decide(usage_kv=0.30, usage_mamba=0.95)
    print(f"  tick 0: {d0.direction} ({d0.reason})")
    assert d0.direction == "kv_to_mamba", f"first crossing should fire (got {d0.direction})"
    # Subsequent ticks at the same level: state stays ABOVE_HIGH, no fires.
    fires_after = 0
    for i in range(1, 30):
        d = p.decide(usage_kv=0.30, usage_mamba=0.95)
        if d.direction is not None:
            fires_after += 1
    print(f"  29 follow-up ticks at same usage: {fires_after} transfers")
    assert fires_after == 0, f"steady state should not fire (got {fires_after})"
    print("  → PASS (the property that makes regressions impossible at steady state)")
    print()


def test_rising_edge_kv():
    print("== T3: rising edge on KV → mamba_to_kv ==")
    p = setup_planner(edge=True)
    # Establish baseline (both IN_BAND).
    p.decide(usage_kv=0.60, usage_mamba=0.50)
    d = p.decide(usage_kv=0.90, usage_mamba=0.50)
    print(f"  kv 0.60→0.90 mamba=0.50: {d.direction} ({d.reason})")
    assert d.direction == "mamba_to_kv"
    print("  → PASS")
    print()


def test_hysteresis_recrossing():
    print("== T4: drop back to IN_BAND, then re-cross → fires again ==")
    p = setup_planner(edge=True)
    p.decide(usage_kv=0.30, usage_mamba=0.95)  # initial crossing → fire
    # Drop into IN_BAND.
    d_drop = p.decide(usage_kv=0.30, usage_mamba=0.60)
    print(f"  drop to mamba=0.60: {d_drop.direction or 'none'}")
    assert d_drop.direction is None, "drop into IN_BAND should not fire"
    # Re-cross into ABOVE_HIGH.
    d_recross = p.decide(usage_kv=0.30, usage_mamba=0.92)
    print(f"  re-cross to mamba=0.92: {d_recross.direction}")
    assert d_recross.direction == "kv_to_mamba", "re-crossing should fire again"
    print("  → PASS (responds to genuine phase shifts, ignores noise)")
    print()


def test_reverse_trigger():
    print("== T5: mamba drops to BELOW_LOW while kv is ABOVE_HIGH → mamba_to_kv ==")
    p = setup_planner(edge=True)
    # Both pools enter ABOVE_HIGH initially. Establish state via two ticks.
    p.decide(usage_kv=0.50, usage_mamba=0.50)  # IN_BAND/IN_BAND
    p.decide(usage_kv=0.92, usage_mamba=0.92)  # both ABOVE_HIGH; KV crosses, fires mamba_to_kv
    # Now mamba drops to BELOW_LOW while kv stays ABOVE_HIGH.
    d = p.decide(usage_kv=0.92, usage_mamba=0.30)
    print(f"  mamba ABOVE→BELOW with kv still ABOVE: {d.direction} ({d.reason})")
    assert d.direction == "mamba_to_kv", \
        f"reverse trigger should fire (got {d.direction})"
    print("  → PASS")
    print()


def test_bouncy_in_band():
    print("== T6: usage oscillates within IN_BAND → 0 transfers ==")
    p = setup_planner(edge=True)
    fires = 0
    for usage in [0.55, 0.65, 0.55, 0.70, 0.60, 0.65, 0.55, 0.70]:
        d = p.decide(usage_kv=0.30, usage_mamba=usage)
        if d.direction is not None:
            fires += 1
    print(f"  8 ticks oscillating mamba in [0.55, 0.70]: {fires} transfers")
    assert fires == 0, f"bouncing within IN_BAND should not fire (got {fires})"
    print("  → PASS (noise-immune)")
    print()


def test_qdepth_propagated():
    print("== T7: PlanDecision.queue_depth still populated under edge_trigger ==")
    p = setup_planner(edge=True)
    d = p.decide(usage_kv=0.5, usage_mamba=0.5, queue_depth=11)
    assert d.queue_depth == 11, f"queue_depth field broken (got {d.queue_depth})"
    print(f"  queue_depth = {d.queue_depth} ✓")
    print("  → PASS")
    print()


def main():
    test_legacy_default()
    test_steady_state_above_high()
    test_rising_edge_kv()
    test_hysteresis_recrossing()
    test_reverse_trigger()
    test_bouncy_in_band()
    test_qdepth_propagated()
    print("== ALL PASS: edge-triggered planner ready ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
