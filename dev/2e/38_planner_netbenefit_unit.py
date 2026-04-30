"""Unit tests for the cross-pool planner's net-benefit gate (栓3).

Motivation: v9-auto FULL 4-cell showed L1+L2 fires 15 transfers under
L1's mamba_usage signal even though L1's HPB-LRU snapshot retention had
already absorbed the binding shift, making L1+L2 a +42% Phase C P99
regression vs L1-only. The gate refuses to fire when no admission
pressure (paused/retracted reqs) is present, OR when the estimated
re-prefill savings won't cover the cuMemUnmap+cuMemMap actuator cost.
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
        # Tests run against the unit chunk cost (50 ms) to keep numbers
        # easy to reason about. Production default is 3,000,000 us to
        # absorb the lcm-aware actuator unit (60 chunks for Qwen3.5-A3B).
        nb_chunk_cost_us=50_000.0,
    )
    base.update(kw)
    return CrossPoolPolicyConfig(**base)


print("== T1: nb=off → existing edge-trigger behavior preserved ==")
p = CrossPoolPlanner(cfg())
p.decide(usage_kv=0.20, usage_mamba=0.50)
d = p.decide(usage_kv=0.20, usage_mamba=0.92)
assert d.direction == "kv_to_mamba", f"want kv_to_mamba, got {d.direction}"
assert "nb=off" in d.reason, f"reason should mark nb=off: {d.reason}"
print(f"  fire on edge: {d.direction} ({d.reason})")
print("  → PASS")

print("\n== T2: nb=on, no paused/retracted, persist=1 → block fire ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50)
d = p.decide(usage_kv=0.20, usage_mamba=0.92,
             num_paused_reqs=0, num_retracted_reqs=0)
# B_persist = 1*5000us = 5000us, still < cost 50000us × margin 1.5
assert d.direction is None, f"should block, got {d.direction}"
assert "benefit 5000us" in d.reason or "no pressure" in d.reason, d.reason
print(f"  blocked: {d.reason}")
print("  → PASS (mamba just transitioned, persist=1, B_lb too small)")

print("\n== T3: nb=on, 1 retracted req + persist=1 → benefit > cost → allow fire ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50)
# benefit_retracted = 1 * 4096 * 1e6 / 50000 = 81920 us
# benefit_persist   = (0+1) * 5000 = 5000 us
# total benefit = 86920 us; cost*margin = 75000 us → allow
d = p.decide(usage_kv=0.20, usage_mamba=0.92,
             num_paused_reqs=0, num_retracted_reqs=1)
assert d.direction == "kv_to_mamba", f"should fire, got {d.direction}"
assert "86920us" in d.reason, d.reason
print(f"  fired: {d.direction}")
print("  → PASS")

print("\n== T4: nb=on, only 1 paused + persist=1 → benefit << cost → block ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50)
# benefit = 1*1000 paused + 1*5000 persist = 6000 us
# cost*margin = 75000 us → block
d = p.decide(usage_kv=0.20, usage_mamba=0.92,
             num_paused_reqs=1, num_retracted_reqs=0)
assert d.direction is None, f"should block, got {d.direction}"
assert "6000us < cost 50000us" in d.reason, d.reason
print(f"  blocked: {d.reason}")
print("  → PASS")

print("\n== T5: nb=on, lots of paused → benefit > cost → allow ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.20, usage_mamba=0.50)
# benefit = 100 * 1000 = 100000 us
# cost = 50000 → 100000 > 75000 → allow
d = p.decide(usage_kv=0.20, usage_mamba=0.92,
             num_paused_reqs=100, num_retracted_reqs=0)
assert d.direction == "kv_to_mamba", f"should fire, got {d.direction}"
print(f"  fired: {d.direction}")
print("  → PASS")

print("\n== T6: nb=on, simulate v9-auto Phase C — 15 high-mamba ticks no blocking → 0 fires ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
# Establish IN_BAND first
p.decide(usage_kv=0.30, usage_mamba=0.55)
fires = 0
# Crossings: each tick crosses ABOVE_HIGH but no admission pressure
for i in range(15):
    # alternate to force re-edges
    p.decide(usage_kv=0.30, usage_mamba=0.55)  # back to IN_BAND
    d = p.decide(usage_kv=0.30, usage_mamba=0.92,  # cross ABOVE_HIGH
                 num_paused_reqs=0, num_retracted_reqs=0)
    if d.direction is not None:
        fires += 1
assert fires == 0, f"expected 0 fires under no-pressure, got {fires}"
print(f"  15 ABOVE_HIGH crossings × 0 admission pressure → fires = {fires}")
print("  → PASS (this is exactly what the v9-auto fix needs)")

print("\n== T7: nb=on, margin tunable via env-style override ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True, nb_margin=0.5))
p.decide(usage_kv=0.20, usage_mamba=0.50)
# With margin=0.5, even 1 paused (1000us) >= cost (50000) * 0.5 = 25000? No, 1000 < 25000.
# Need 25001 us benefit. 50 paused = 50000 us > 25000 → allow.
d = p.decide(usage_kv=0.20, usage_mamba=0.92,
             num_paused_reqs=50, num_retracted_reqs=0)
assert d.direction == "kv_to_mamba", f"with margin=0.5, 50 paused should fire, got {d.direction}"
print(f"  margin=0.5, 50 paused → fired: {d.direction}")
print("  → PASS")

print("\n== T8: nb=on, reverse trigger (mamba drops while kv ABOVE) also gated ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True))
p.decide(usage_kv=0.30, usage_mamba=0.50)
p.decide(usage_kv=0.92, usage_mamba=0.50)  # kv crosses ABOVE_HIGH, would fire mamba_to_kv if benefit
# now drop mamba to BELOW_LOW with kv still ABOVE — reverse trigger path
d = p.decide(usage_kv=0.92, usage_mamba=0.30,
             num_paused_reqs=0, num_retracted_reqs=0)
# Even reverse trigger should be gated when no admission pressure
assert d.direction is None or "no admission pressure" in d.reason, \
    f"reverse trigger should also be gated, got {d}"
print(f"  reverse trigger gated: {d.reason}")
print("  → PASS")

print("\n== T9: B_persist — sustained ABOVE_HIGH triggers fire even with no paused/retracted ==")
# v9-auto v2 failure mode: stock cache hides cost behind 0 paused/0 retracted,
# but mamba_usage stays 1.0. With nb_persist_eval_period=10 and persist_tick_us=5000,
# at consec=15 → B_persist=15*5000=75000us = cost*margin (50000*1.5). Should fire.
p = CrossPoolPlanner(cfg(net_benefit_enabled=True, nb_persist_eval_period=10))
p.decide(usage_kv=0.20, usage_mamba=0.50)  # establish IN_BAND
# First tick at ABOVE_HIGH = edge transition; B_persist=1*5000=5000 → block
d = p.decide(usage_kv=0.20, usage_mamba=0.92, num_paused_reqs=0, num_retracted_reqs=0)
assert d.direction is None, f"first tick should still block, got {d.direction}"
# Next 8 ticks: stable above_high, no re-eval (consec < 10).
for i in range(8):
    d = p.decide(usage_kv=0.20, usage_mamba=0.92, num_paused_reqs=0, num_retracted_reqs=0)
    assert d.direction is None, f"tick {i+2}: should be stable, got {d.direction}"
# Tick 10: stable, consec=10, period eval, B_persist=10*5000=50000 < 75000 → still block
d = p.decide(usage_kv=0.20, usage_mamba=0.92, num_paused_reqs=0, num_retracted_reqs=0)
assert d.direction is None, f"consec=10 still < margin, got {d.direction}"
# Continue another 10 ticks; at consec=20, B_persist = 20*5000 = 100000 ≥ 75000 → fire
fired = False
for i in range(11, 25):
    d = p.decide(usage_kv=0.20, usage_mamba=0.92, num_paused_reqs=0, num_retracted_reqs=0)
    if d.direction == "kv_to_mamba":
        fired = True
        print(f"  fired at tick consec ~{i+1}: {d.reason}")
        break
assert fired, "should fire by consec=20"
print("  → PASS (persist re-eval kicks in once B_persist clears margin)")

print("\n== T10: B_persist disabled (nb_persist_eval_period=0) → never fires under no pressure ==")
p = CrossPoolPlanner(cfg(net_benefit_enabled=True, nb_persist_eval_period=0))
p.decide(usage_kv=0.20, usage_mamba=0.50)
fired = 0
for i in range(50):
    d = p.decide(usage_kv=0.20, usage_mamba=0.92, num_paused_reqs=0, num_retracted_reqs=0)
    if d.direction is not None:
        fired += 1
assert fired == 0, f"with persist_eval_period=0, no fires expected, got {fired}"
print(f"  50 ticks at sustained ABOVE_HIGH, no admission pressure, persist disabled → fires={fired}")
print("  → PASS")

print("\n== ALL PASS: net-benefit gate ready (栓3 + B_persist) ==")
