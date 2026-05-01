"""Unit tests for the cross-pool planner's net-benefit gate (栓3).

Architecture (post sglang `87360b2c7`): the gate is engine-agnostic
(B ≥ C × margin where B is the sum of `EnginePressureAdapter` signals
in microseconds). This test exercises the gate against a synthetic
snapshot dict that the adapter (`SGLangPressureAdapter`) translates
through fixed coefficients:

  prefill_save_us_per_token = 12.5    # ~80 K tok/s prefill on H200
  full_prefill_us           = 75000   # ~6K-token req re-prefill
  pause_penalty_us          = 1000
  queue_wait_us             = 100
  persist_tick_us           = 5000

Tests use cost = 50,000 us and margin = 1.5 → cost*margin = 75,000 us
so the threshold math is easy to reason about.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, "/data/yuzhou/projects/sglang/python")
from sglang.srt.budgeter.cross_pool_planner import (  # noqa: E402
    CrossPoolPlanner, CrossPoolPolicyConfig,
)


def cfg(**kw):
    base = dict(
        kv_high_water=0.85, kv_low_water=0.40,
        mamba_high_water=0.80, mamba_low_water=0.40,
        cooldown_ticks=0, dst_chunks_per_action=1,
        edge_trigger=True,
        # Tests run against unit chunk cost so threshold math is clean:
        # cost = 50_000 us, margin = 1.5 → cost*margin = 75_000 us.
        nb_chunk_cost_us=50_000.0,
    )
    base.update(kw)
    return CrossPoolPolicyConfig(**base)


def snap(**kw):
    """Build a snapshot dict with adapter-relevant keys."""
    base = dict(num_evicted_tokens_recent=0, num_retracted_reqs=0,
                num_paused_reqs=0, num_queue_reqs=0)
    base.update(kw)
    return base


print("== T1: nb=off → existing edge-trigger behavior preserved ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=False))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())
d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap())
assert d.direction == "kv_to_mamba", f"want kv_to_mamba, got {d.direction}"
assert "nb=off" in d.reason, f"reason should mark nb=off: {d.reason}"
print(f"  fire on edge: {d.direction}")
print("  → PASS")


print("\n== T2: nb=on, no signals at all → block fire ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())
# B_persist = 1 × 5000us = 5000us, < cost 50000 × margin 1.5 = 75000us
d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap())
assert d.direction is None, f"should block, got {d.direction}"
assert "5000us" in d.reason or "no pressure" in d.reason, d.reason
print(f"  blocked: {d.reason}")
print("  → PASS (mamba just transitioned, persist=1, B too small)")


print("\n== T3: nb=on, 1 retracted req → benefit > cost → allow fire ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())
# benefit = 75000 (retract) + 5000 (persist) = 80000 > 75000 → fire
d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap(num_retracted_reqs=1))
assert d.direction == "kv_to_mamba", f"should fire, got {d.direction}"
print(f"  fired: {d.direction}")
print("  → PASS")


print("\n== T4: nb=on, only 1 paused → benefit << cost → block ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())
# benefit = 1000 (paused) + 5000 (persist) = 6000 < 75000 → block
d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap(num_paused_reqs=1))
assert d.direction is None, f"should block, got {d.direction}"
print(f"  blocked: {d.reason}")
print("  → PASS")


print("\n== T5: nb=on, lots of paused → benefit > cost → allow ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())
# benefit = 100*1000 + 5000 = 105000 > 75000 → fire
d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap(num_paused_reqs=100))
assert d.direction == "kv_to_mamba", f"should fire, got {d.direction}"
print(f"  fired: {d.direction}")
print("  → PASS")


print("\n== T6: SGLang signal — eviction-only pressure crosses threshold ==")
# SGLang's primary pressure signal: tree-cache eviction. At 12.5 us/token
# saved, 6000 evicted tokens = 75000 us benefit.
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())
# benefit = 6000 * 12.5 + 5000 (persist) = 75000 + 5000 = 80000 > 75000 → fire
d = p.decide(usage_kv=0.20, usage_mamba=0.92,
             snapshot=snap(num_evicted_tokens_recent=6000))
assert d.direction == "kv_to_mamba", f"should fire on evict pressure, got {d.direction}"
print(f"  fired on evict signal: {d.direction}")
print("  → PASS (SGLang adapter's primary signal works)")


print("\n== T7: nb=on, simulate v9-auto Phase A — repeated edge crossings, no pressure → 0 fires ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.30, usage_mamba=0.55, snapshot=snap())  # IN_BAND
fires = 0
for i in range(15):
    p.decide(usage_kv=0.30, usage_mamba=0.55, snapshot=snap())  # back IN_BAND
    d = p.decide(usage_kv=0.30, usage_mamba=0.92, snapshot=snap())  # cross
    if d.direction is not None:
        fires += 1
assert fires == 0, f"expected 0 fires under no-pressure, got {fires}"
print(f"  15 crossings × 0 pressure → fires = {fires}")
print("  → PASS (suppresses v9-auto-style false fires)")


print("\n== T8: nb=on, reverse trigger also gated ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.30, usage_mamba=0.50, snapshot=snap())
p.decide(usage_kv=0.92, usage_mamba=0.50, snapshot=snap())  # kv ABOVE → would fire mamba_to_kv
d = p.decide(usage_kv=0.92, usage_mamba=0.30, snapshot=snap())  # mamba drops
assert d.direction is None, f"reverse trigger should be gated, got {d}"
print(f"  reverse trigger gated: {d.reason[:80]}")
print("  → PASS")


print("\n== T9: persist accumulator — sustained ABOVE_HIGH eventually fires ==")
# At cost=50_000, margin=1.5 → threshold = 75_000. persist_tick=5000.
# Need persist_consec >= 15 ticks for benefit ≥ threshold.
p = CrossPoolPlanner(cfg(net_benefit_enabled=True, nb_persist_eval_period=10))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())  # IN_BAND
# First crossing: persist=1, benefit too low
d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap())
assert d.direction is None
# Tick 8 stable: persist=8, no period re-eval (period=10)
for i in range(8):
    d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap())
    assert d.direction is None
# Tick 10: re-eval at consec=10. benefit = 10 × 5000 = 50000 < 75000. Still block.
d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap())
assert d.direction is None, f"consec=10 still < threshold, got {d.direction}"
# Tick 20: re-eval at consec=20. benefit = 20 × 5000 = 100000 > 75000. Fire.
fired_at = -1
for i in range(11, 25):
    d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap())
    if d.direction == "kv_to_mamba":
        fired_at = i + 1
        break
assert fired_at > 0, "persist accumulator should eventually fire"
print(f"  fired at consec≈{fired_at} (B_persist crossed threshold)")
print("  → PASS")


print("\n== T10: B_persist disabled (nb_persist_eval_period=0) → never fires under no pressure ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True, nb_persist_eval_period=0))
p.decide(usage_kv=0.20, usage_mamba=0.50, snapshot=snap())
fires = 0
for i in range(50):
    d = p.decide(usage_kv=0.20, usage_mamba=0.92, snapshot=snap())
    if d.direction is not None:
        fires += 1
assert fires == 0, f"with period=0, no re-eval, no fires under no pressure, got {fires}"
print(f"  50 ticks at sustained ABOVE_HIGH, no admission pressure, persist disabled → fires={fires}")
print("  → PASS")


print("\n== ALL PASS: net-benefit gate ready (栓3 + B_persist + adapter) ==")
