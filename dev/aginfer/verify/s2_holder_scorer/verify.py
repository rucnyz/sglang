"""verify/s2_holder_scorer — drift guard for the holder-count term in the
LIVE eviction scorer (``baselines.sglang_adapter._v_u_from_unit``, used by
``hint_v_u`` and ``ours_greedy_score``).

WHY THIS EXISTS (the bug it guards):
The DESIGN §2 fact-1 holder-count term ("a unit shared by N programs is
worth N× keeping") was implemented in ``OursGreedyPolicy._value`` (the
daemon's migrate value) but NOT in ``_v_u_from_unit`` (the cache's
eviction heap key).  The two V_u implementations drifted: the daemon
would *migrate* a fleet-shared prefix to keep it, but the cache eviction
heap scored it as if n_holders=1 and evicted it anyway.  Result: the S2
holder-count lever was INERT in every live eviction run (N=3 ties),
because a shared prefix (p_hat=1.0, n_holders=8) and an active bg leaf
(p_hat=1.0, n_holders=1) of equal size got IDENTICAL scores → recency
tie-break = LRU.  ``kv_scheduler_value_rule`` only tested ``_value``, so
nothing caught it.

This verify pins the contract that ``_v_u_from_unit`` actually applies
n_holders, and that it agrees in DIRECTION with ``_value``.

Usage:
    PYTHONPATH=dev/aginfer python dev/aginfer/verify/s2_holder_scorer/verify.py
"""
from __future__ import annotations

import sys

from baselines.base import ReuseUnit, Scope, Tier, UnitType
import baselines.sglang_adapter as A


class StageFail(Exception):
    pass


def _mk(n_tokens: int, p_hat: float, lam: float, n_holders: int) -> ReuseUnit:
    b = n_tokens * A._BYTES_PER_TOKEN
    u = ReuseUnit(
        id="x", type=UnitType.SESSION, scope=Scope.SESSION, n_tokens=n_tokens,
        n_bytes_by_tier={Tier.HBM: {"full": b}}, residence=[Tier.HBM],
        age_seconds=1.0, p_hat=p_hat, lambda_rate=lam, holders=[],
    )
    u.n_holders = n_holders
    return u


def stage_a_holder_count_changes_score() -> None:
    """n_holders MUST change _v_u_from_unit (the bug: it was ignored)."""
    base = _mk(24000, 1.0, 1.0, 1)
    boosted = _mk(24000, 1.0, 1.0, 8)
    v1 = A._v_u_from_unit(base)
    v8 = A._v_u_from_unit(boosted)
    if not (v8 > v1):
        raise StageFail(
            f"A: n_holders=8 must score ABOVE n_holders=1 (live scorer), "
            f"got v8={v8:.4g} !> v1={v1:.4g} — holder-count INERT in _v_u_from_unit"
        )


def stage_b_shared_outranks_active_bg() -> None:
    """The S2 scenario: a fleet-shared prefix (n_holders=8) must outrank a
    same-size active bg leaf (n_holders=1) at the daemon's any-alive
    p_hat=1.0 — else eviction falls to the recency tie-break (== LRU)."""
    shared = _mk(24000, 1.0, 1.0, 8)
    active_bg = _mk(24000, 1.0, 1.0, 1)
    vs = A._v_u_from_unit(shared)
    vb = A._v_u_from_unit(active_bg)
    if not (vs > vb):
        raise StageFail(
            f"B: shared prefix (n_holders=8) must outrank active bg "
            f"(n_holders=1) at p_hat=1.0, got shared={vs:.4g} !> bg={vb:.4g}"
        )


def stage_c_single_holder_unchanged() -> None:
    """n_holders in {0,1} must give the SAME score (the boost only applies
    to genuinely-shared units; single-program units are untouched)."""
    u0 = _mk(16000, 0.7, 0.5, 0)
    u1 = _mk(16000, 0.7, 0.5, 1)
    v0, v1 = A._v_u_from_unit(u0), A._v_u_from_unit(u1)
    if abs(v0 - v1) > 1e-9:
        raise StageFail(f"C: n_holders 0 vs 1 must match, got {v0:.6g} vs {v1:.6g}")


def stage_d_agrees_with_value_direction() -> None:
    """_v_u_from_unit and OursGreedyPolicy._value must agree that the
    holder-count RAISES value (no drift in direction)."""
    from baselines.ours_greedy import reload_cost

    def value_with_holders(n: int) -> float:
        u = _mk(24000, 1.0, 1.0, n)
        pi = A._PI_U
        tier = u.authoritative_tier
        nh = max(1, len(u.holders), int(getattr(u, "n_holders", 0)))
        sp = nh * u.p_hat * (
            reload_cost(u, Tier.DROP, A._COSTS, pi)
            - reload_cost(u, tier, A._COSTS, pi)
        )
        h = A._COSTS.h_base[tier]
        eff = nh * u.lambda_rate
        ht = 1.0 / eff if eff > 0 else 1e6
        return float(sp - h * u.n_bytes * ht)

    # both formulas: 8 holders strictly above 1 holder
    live_1, live_8 = A._v_u_from_unit(_mk(24000, 1.0, 1.0, 1)), A._v_u_from_unit(_mk(24000, 1.0, 1.0, 8))
    val_1, val_8 = value_with_holders(1), value_with_holders(8)
    if not (live_8 > live_1 and val_8 > val_1):
        raise StageFail(
            f"D: both formulas must boost with holders; "
            f"live({live_1:.3g}->{live_8:.3g}) value({val_1:.3g}->{val_8:.3g})"
        )
    # and the live scorer == the hand-rolled mirror (they share semantics)
    if abs(live_8 - val_8) > 1e-6:
        raise StageFail(
            f"D: _v_u_from_unit must MATCH the _value holder-count mirror, "
            f"got {live_8:.6g} vs {val_8:.6g} (drift!)"
        )


def main() -> int:
    stages = [
        ("A holder-count changes score", stage_a_holder_count_changes_score),
        ("B shared outranks active bg", stage_b_shared_outranks_active_bg),
        ("C single-holder unchanged", stage_c_single_holder_unchanged),
        ("D agrees with _value direction", stage_d_agrees_with_value_direction),
    ]
    print("=== verify/s2_holder_scorer (holder-count live-scorer drift guard) ===")
    failed = 0
    for name, fn in stages:
        try:
            fn()
            print(f"  PASS  {name}")
        except StageFail as e:
            print(f"  FAIL  {e}")
            failed += 1
    print("=" * 60)
    if failed:
        print(f"RESULT: FAIL ({failed} stage(s))")
        return 1
    print("s2_holder_scorer PASS — holder-count live in _v_u_from_unit, no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
