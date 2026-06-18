"""#217 — c^xfer persistent-invalid-measurement detector.

`RuntimeActuatorCost.update(total_us, n_chunks)` feeds the EWMA from observed
fire wall-times. A single invalid sample (non-finite, <=0, or n_chunks<=0) is a
transient and is skipped — the EWMA just doesn't update that tick. But a
SUSTAINED stream of invalid measurements is not noise: the actuator's
wall-time path is broken, the EWMA can never update, and the Budgeter is stuck
forever on the conservative cold-start default (silently disabling cross-pool
rebalancing). That is a bug, so `update()` RAISES after
`_INVALID_STREAK_LIMIT` consecutive invalid samples rather than swallowing them.

Normal c^xfer drift (the live value moving with GPU contention) is NOT what
this guards — that is expected and the EWMA tracks it. Only IMPOSSIBLE values
in a row trip it.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.cost_model import RuntimeActuatorCost  # noqa: E402


def test_1_single_invalid_is_skipped_not_raised():
    c = RuntimeActuatorCost(initial_us=3000.0)
    c.update(float("nan"), 4)      # transient bad sample
    c.update(-5.0, 4)              # not consecutive-enough yet (limit default 3)
    assert c.current_us == 3000.0, "a few invalids must not move the EWMA"
    assert c.n_observations == 0, "invalid samples must not count as observations"
    print("  PASS  1  isolated invalid samples skipped, EWMA untouched, no raise")


def test_2_good_sample_resets_the_streak():
    c = RuntimeActuatorCost(initial_us=3000.0)
    for _ in range(c._INVALID_STREAK_LIMIT - 1):
        c.update(float("inf"), 4)  # one short of the limit
    c.update(1200.0, 1)            # GOOD → resets the consecutive counter
    assert c.current_us == 1200.0
    # now another (limit-1) invalids must STILL not raise (streak was reset)
    for _ in range(c._INVALID_STREAK_LIMIT - 1):
        c.update(0.0, 4)
    print("  PASS  2  a valid sample resets the invalid streak (no false raise)")


def test_3_persistent_invalid_stream_raises():
    c = RuntimeActuatorCost(initial_us=3000.0)
    raised = False
    try:
        for _ in range(c._INVALID_STREAK_LIMIT):
            c.update(float("nan"), 4)
    except ValueError as e:
        raised = True
        msg = str(e)
        assert "c^xfer" in msg or "xfer" in msg.lower(), msg
        assert str(c._INVALID_STREAK_LIMIT) in msg, msg
    assert raised, (
        f"{c._INVALID_STREAK_LIMIT} consecutive invalid measurements must raise "
        "(the actuator wall-time path is broken); current code silently swallows "
        "them (#217)"
    )
    print(f"  PASS  3  {c._INVALID_STREAK_LIMIT} consecutive invalid → ValueError")


def test_4_valid_stream_unchanged_regression():
    # Pin that the normal path is unaffected: EWMA seeds + smooths as before.
    c = RuntimeActuatorCost(initial_us=3000.0, alpha=0.3)
    c.update(1000.0, 1)            # first obs seeds directly
    assert abs(c.current_us - 1000.0) < 1e-6
    c.update(2000.0, 1)            # 0.3*2000 + 0.7*1000 = 1300
    assert abs(c.current_us - 1300.0) < 1e-6
    assert c.n_observations == 2
    print("  PASS  4  valid stream: EWMA seed + smoothing unchanged (regression)")


if __name__ == "__main__":
    test_1_single_invalid_is_skipped_not_raised()
    test_2_good_sample_resets_the_streak()
    test_3_persistent_invalid_stream_raises()
    test_4_valid_stream_unchanged_regression()
    print("ALL PASS (#217 c^xfer invalid-stream detector)")
