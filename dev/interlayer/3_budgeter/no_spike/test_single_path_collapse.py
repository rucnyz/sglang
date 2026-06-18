"""Behavior-preservation guard for collapsing XPoolPlanner to its single
arg-max NB decision path.

The planner historically carried two decision paths gated on config flags:
the default arg-max net-benefit path (`_pick_direction_by_nb`) and a dormant
edge-trigger / persist-reeval fallback. The fallback never ran in production
(the default flag preempted it) and its `_net_benefit_ok` helper carried a
dead TypeError, so it was confirmed dead. This test pins TWO things across the
collapse to the single path:

  1. The arg-max fire decisions are unchanged: a representative sequence of
     snapshot dicts driven through the REAL `XPoolPlanner` + real
     `SGLangPressureAdapter` (no GPU) yields exactly the directions the
     arg-max path produces (cold-start L=0 fires from queue/persist, the
     negative controls do not fire, the reuse-aware drain/grow guards hold).
  2. The deleted config knobs are GONE: `XPoolPolicyConfig()` exposes none of
     `edge_trigger`, `nb_direction_aware`, `net_benefit_enabled`,
     `nb_persist_eval_period`. A field silently re-appearing (e.g. re-added as
     an unused default) would let the dead fallback resurface, so the absence
     is asserted directly.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.cost_model import (
    BUILTIN_DEFAULT,
    reset_cost_curves,
    reset_runtime_actuator_cost,
)
from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
from sglang.srt.budgeter.xpool_planner import XPoolPlanner, XPoolPolicyConfig


def _fresh_planner(cooldown_min_s=20.0, amortize_horizon_s=20.0):
    """Cold planner on the builtin curves, k2m/m2k-reachable config.
    Mirrors no_spike/_fresh_planner but constructs the config with ONLY the
    surviving fields (the collapse removed the path-selecting knobs)."""
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.95, kv_low_water=0.70,
        mamba_high_water=0.95, mamba_low_water=0.70,
        cooldown_min_s=cooldown_min_s, amortize_horizon_s=amortize_horizon_s,
        dst_chunks_per_action=4,
        nb_margin=1.5, nb_chunk_cost_us=10000.0,
    )
    return XPoolPlanner(config=cfg, adapter=SGLangPressureAdapter())


def _seed_consec(planner, usage_kv, usage_mamba, ticks=15):
    """Seed the persist consec + dwell-seconds counters the arg-max path
    reads, without eating the cooldown by calling decide() repeatedly."""
    if usage_mamba >= planner.config.mamba_high_water:
        planner._mamba_above_high_consec = ticks
        planner._mamba_dwell_s = ticks * 2.0
    if usage_kv >= planner.config.kv_high_water:
        planner._kv_above_high_consec = ticks
        planner._kv_dwell_s = ticks * 2.0


def _base_snap(**overrides):
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
    }
    snap.update(overrides)
    return snap


# Each case: (name, usage_kv, usage_mamba, seed_kv, seed_mamba, snapshot,
#             expected_direction). The expected direction is what the arg-max
# NB path produces; the collapse must not change any of them.
CASES = [
    # cold-start L=0, mamba saturated + queue → grow mamba
    ("L0_mamba_sat_queue_k2m", 0.75, 0.97, 0.75, 0.97,
     _base_snap(num_queue_reqs=200), "kv_to_mamba"),
    # cold-start L=0, kv saturated + queue → grow kv
    ("L0_kv_sat_queue_m2k", 0.97, 0.75, 0.97, 0.75,
     _base_snap(num_queue_reqs=200), "mamba_to_kv"),
    # cold-start L=0, no signals → no fire
    ("L0_no_signal_none", 0.80, 0.80, 0.80, 0.80,
     _base_snap(num_queue_reqs=0), None),
    # both pools slack + queue → stall is not memory-bound → no fire
    ("both_slack_queue_none", 0.30, 0.10, 0.30, 0.10,
     _base_snap(num_queue_reqs=200,
                usage_kv_active=0.30, usage_mamba_active=0.10), None),
    # hot mamba cache, both active sides below low-water → not memory-bound,
    # so the arg-max path fires neither direction (and certainly not k2m)
    ("hot_mamba_cache_not_k2m", 0.60, 0.95, 0.60, 0.20,
     _base_snap(num_queue_reqs=50,
                usage_kv_active=0.60, usage_mamba_active=0.20), None),
    # kv genuinely saturating + mamba cache full → grow kv
    ("kv_sat_mamba_cache_m2k", 0.92, 0.95, 0.92, 0.20,
     _base_snap(num_queue_reqs=200,
                usage_kv_active=0.92, usage_mamba_active=0.20), "mamba_to_kv"),
    # hot mamba cache (large drain cost) must block m2k
    ("hot_mamba_drain_blocks_m2k", 0.92, 0.95, 0.92, 0.20,
     _base_snap(num_queue_reqs=200,
                usage_kv_active=0.92, usage_mamba_active=0.20,
                mamba_drain_cost_us=10_000_000.0), None),
]


def test_argmax_decisions_preserved_across_collapse():
    """Drive the representative sequence and assert each direction equals the
    arg-max-path expectation. A regression in the surviving path (or the dead
    fallback leaking back) would flip one of these."""
    for (name, ukv, um, skv, sm, snap, expected) in CASES:
        planner = _fresh_planner()
        _seed_consec(planner, skv, sm, ticks=15)
        dec = planner.decide(
            usage_kv=ukv, usage_mamba=um,
            queue_depth=int(snap.get("num_queue_reqs", 0)), snapshot=snap,
        )
        assert dec.direction == expected, (
            f"case {name!r}: arg-max path should decide {expected!r} but got "
            f"{dec.direction!r} (reason: {dec.reason[:200]})"
        )


def test_deleted_config_knobs_are_gone():
    """The path-selecting / dead knobs must not exist on the default config.
    A re-appearance (even unused) would reopen the dead fallback path."""
    cfg = XPoolPolicyConfig()
    for field in (
        "edge_trigger",
        "nb_direction_aware",
        "net_benefit_enabled",
        "nb_persist_eval_period",
    ):
        assert not hasattr(cfg, field), (
            f"XPoolPolicyConfig still carries deleted field {field!r}; the "
            f"single-path collapse must remove it so the dead edge-trigger "
            f"fallback cannot silently return."
        )


def test_net_benefit_ok_method_is_gone():
    """`_net_benefit_ok` was only called from the deleted fallback and carried
    a dead TypeError. The collapse deletes the method outright."""
    planner = _fresh_planner()
    assert not hasattr(planner, "_net_benefit_ok"), (
        "XPoolPlanner still defines _net_benefit_ok; it was only reachable "
        "from the removed edge-trigger path and must be deleted."
    )


def test_decide_rejects_edge_active_kwarg():
    """`edge_active` was read only by the deleted fallback; `decide()` no
    longer accepts it. Passing it must raise (no silent swallow)."""
    planner = _fresh_planner()
    snap = _base_snap(num_queue_reqs=0)
    try:
        planner.decide(usage_kv=0.5, usage_mamba=0.5, snapshot=snap,
                       edge_active=True)
    except TypeError:
        return
    raise AssertionError(
        "decide() accepted edge_active=; the parameter should be removed "
        "with the dead fallback that read it."
    )


def main():
    tests = [
        ("arg-max decisions preserved across collapse",
         test_argmax_decisions_preserved_across_collapse),
        ("deleted config knobs are gone",
         test_deleted_config_knobs_are_gone),
        ("_net_benefit_ok method is gone",
         test_net_benefit_ok_method_is_gone),
        ("decide() rejects edge_active kwarg",
         test_decide_rejects_edge_active_kwarg),
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
    print(f"\nsingle_path_collapse: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
