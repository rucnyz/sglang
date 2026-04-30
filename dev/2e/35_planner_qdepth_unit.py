"""Unit test for the cross-pool planner's saturation+queue rule
(Setting 4 follow-up). Verifies:

  - With SGLANG_XPOOL_QDEPTH_TRIGGER=0 (default), the planner's
    behavior is unchanged: at saturation+saturation neither rule fires.
  - With SGLANG_XPOOL_QDEPTH_TRIGGER=4, a saturated mamba pool with
    queue_depth >= 4 fires kv_to_mamba even when KV usage > kv_low.

Run:
  PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
    .venv/bin/python -u dev/2e/35_planner_qdepth_unit.py
"""
from __future__ import annotations
import os
import sys


def setup_planner(qdepth_trigger: int):
    if qdepth_trigger > 0:
        os.environ["SGLANG_XPOOL_QDEPTH_TRIGGER"] = str(qdepth_trigger)
    else:
        os.environ.pop("SGLANG_XPOOL_QDEPTH_TRIGGER", None)
    os.environ["SGLANG_XPOOL_KV_HIGH"] = "0.85"
    os.environ["SGLANG_XPOOL_KV_LOW"] = "0.50"
    os.environ["SGLANG_XPOOL_MAMBA_HIGH"] = "0.80"
    os.environ["SGLANG_XPOOL_MAMBA_LOW"] = "0.40"
    os.environ["SGLANG_XPOOL_COOLDOWN"] = "0"
    from importlib import reload
    import sglang.srt.budgeter.cross_pool_planner as cpp
    reload(cpp)
    return cpp.CrossPoolPlanner()


def test_qdepth_disabled():
    print("== Test 1: qdepth_trigger=0 (default) — saturation+queue ignored ==")
    p = setup_planner(qdepth_trigger=0)
    # Mamba saturated at 0.66, KV at mid-band (0.7), queue 10.
    d = p.decide(usage_kv=0.7, usage_mamba=0.85, queue_depth=10)
    print(f"  decision: {d.direction} ({d.reason})")
    assert d.direction is None, \
        f"with qdepth_trigger=0, saturated mamba + mid-band kv should NOT fire (got {d.direction})"
    print("  → PASS (legacy behavior preserved)")
    print()


def test_qdepth_fires_kv_to_mamba():
    print("== Test 2: qdepth_trigger=4 — saturated mamba + queue ⇒ kv_to_mamba ==")
    p = setup_planner(qdepth_trigger=4)
    d = p.decide(usage_kv=0.6, usage_mamba=0.85, queue_depth=10)
    print(f"  decision: {d.direction} ({d.reason})")
    assert d.direction == "kv_to_mamba", \
        f"saturated mamba + queue=10 should fire kv_to_mamba (got {d.direction})"
    assert "saturation+queue" in d.reason, \
        f"reason should mention saturation+queue (got {d.reason!r})"
    print("  → PASS (new saturation+queue rule fires correctly)")
    print()


def test_qdepth_below_threshold():
    print("== Test 3: qdepth_trigger=4 but queue=2 — should NOT fire ==")
    p = setup_planner(qdepth_trigger=4)
    d = p.decide(usage_kv=0.6, usage_mamba=0.85, queue_depth=2)
    print(f"  decision: {d.direction} ({d.reason})")
    assert d.direction is None, \
        f"queue below trigger should not fire (got {d.direction})"
    print("  → PASS")
    print()


def test_legacy_kv_to_mamba_still_works():
    print("== Test 4: legacy KV-low + mamba-high rule still fires when applicable ==")
    p = setup_planner(qdepth_trigger=4)
    d = p.decide(usage_kv=0.40, usage_mamba=0.85, queue_depth=0)
    print(f"  decision: {d.direction} ({d.reason})")
    assert d.direction == "kv_to_mamba", \
        f"legacy mamba_high & kv_low rule should fire (got {d.direction})"
    print("  → PASS")
    print()


def test_qdepth_field_in_decision():
    print("== Test 5: PlanDecision.queue_depth field is populated ==")
    p = setup_planner(qdepth_trigger=4)
    d = p.decide(usage_kv=0.5, usage_mamba=0.5, queue_depth=7)
    assert d.queue_depth == 7, f"queue_depth field not set (got {d.queue_depth})"
    print(f"  queue_depth = {d.queue_depth} ✓")
    print("  → PASS")
    print()


def main():
    test_qdepth_disabled()
    test_qdepth_fires_kv_to_mamba()
    test_qdepth_below_threshold()
    test_legacy_kv_to_mamba_still_works()
    test_qdepth_field_in_decision()
    print("== ALL PASS: planner saturation+queue rule ready ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
