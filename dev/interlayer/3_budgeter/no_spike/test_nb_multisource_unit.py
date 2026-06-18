"""Dverify — multi-source NB direction attribution (verify-gap-3).

Locks in the 0e4051b988 fix at
`python/sglang/srt/budgeter/xpool_planner.py:_pick_direction_by_nb`.

Pre-fix: NB used only `c_KV(L) * P_save` (eviction-cost) term.
Post-fix: NB also sums queue_us / paused_us / retract_us attributed
to the more-saturated pool, and persist_us split per-pool via
kv_consec / mamba_consec. Paper §appendix-trigger:557-566 +
design.md §"Budgeter — steady-state pressure rebalance" specifies this multi-source aggregate as the
PRIMARY signal (not a fallback).

The pre-fix gate refused to fire when L=0 (no eviction history),
even when the pool was obviously saturated. After the fix, fires
trigger from queue/persist signals even at L=0 — which is the
common case for hybrid mamba models where mamba live-state saturates
via back-pressure WITHOUT producing eviction events (byte_transfer's scenario).

Test-first protocol:
  1. `git checkout 0e4051b988~1 -- python/sglang/srt/budgeter/xpool_planner.py`
  2. Run this test → cases A, B, C MUST FAIL (pre-fix returns None
     with "no recovery_len observed" when L=0)
  3. `git checkout 0e4051b988 -- python/sglang/srt/budgeter/xpool_planner.py`
  4. Run this test → all 5 cases MUST PASS
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.cost_model import (
    BUILTIN_DEFAULT,
    reset_cost_curves,
    reset_runtime_actuator_cost,
)
from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
from sglang.srt.budgeter.xpool_planner import XPoolPlanner, XPoolPolicyConfig


def _fresh_planner(cooldown_min_s=20.0, amortize_horizon_s=20.0, nb_margin=1.5,
                   dst_chunks_per_action=4, nb_chunk_cost_us=10000.0,
                   both_full_guard=True):
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.95, kv_low_water=0.70,
        mamba_high_water=0.95, mamba_low_water=0.70,
        cooldown_min_s=cooldown_min_s, amortize_horizon_s=amortize_horizon_s,
        dst_chunks_per_action=dst_chunks_per_action,
        nb_margin=nb_margin, nb_chunk_cost_us=nb_chunk_cost_us,
        both_full_guard=both_full_guard,
    )
    return XPoolPlanner(config=cfg, adapter=SGLangPressureAdapter())


class _Tree:
    """Minimal tree_cache stub carrying the recovery-length EWMA counters
    the record_recovery_len_* functions read/write. Real caches init these
    to 0.0 unconditionally in __init__; the stub mirrors that contract."""

    def __init__(self):
        self._slow_recovery_len_kv_ewma = 0.0
        self._slow_recovery_len_rec_ewma = 0.0
        self._slow_recovery_len_retract_ewma = 0.0


# ---------- sub-tests ----------

def _seed_consec(planner, usage_kv, usage_mamba, ticks=15):
    """Directly bump the planner's persist consec counters without
    calling decide() repeatedly (avoid eating the cooldown on an
    auto-fire). The post-fix `_pick_direction_by_nb` reads
    `self._kv_above_high_consec` / `self._mamba_above_high_consec`
    to compute persist_us per direction."""
    # persist now reads dwell SECONDS (τ-invariant); seed both the legacy
    # consec counter and the dwell clock (dt=2 s anchor → ticks × 2 s).
    if usage_mamba >= planner.config.mamba_high_water:
        planner._mamba_above_high_consec = ticks
        planner._mamba_dwell_s = ticks * 2.0
    if usage_kv >= planner.config.kv_high_water:
        planner._kv_above_high_consec = ticks
        planner._kv_dwell_s = ticks * 2.0


def test_A_L0_queue_pressure_mamba_saturated_fires_k2m():
    """L=0 (no eviction history), substantial queue + sustained
    mamba-above-high persist → fire toward mamba (kv_to_mamba).

    PRE-FIX BUG: returns None ("no recovery_len observed") because
    L=0 short-circuits before consulting queue/persist signals.
    Post-fix: queue_us attributed to mamba (more saturated) +
    mamba_persist_us together cross the 60000us threshold."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,  # ~200 × 100us × 10 lifetime ≈ 200K us benefit
    }
    _seed_consec(planner, 0.75, 0.97, ticks=15)
    decision = planner.decide(
        usage_kv=0.75, usage_mamba=0.97, queue_depth=200, snapshot=snap,
    )
    assert decision.direction == "kv_to_mamba", (
        f"BUG: L=0 + queue/persist pressure on mamba should fire k2m "
        f"but planner returned direction={decision.direction!r} "
        f"(reason: {decision.reason[:200]}). Pre-fix: NB only consults "
        f"c_KV(L)·P_save; with L=0 the c_KV term is 0 → NB=0 → no fire. "
        f"Post-fix should aggregate queue_us + persist_us into NB and fire."
    )


def test_B_L0_queue_pressure_kv_saturated_fires_m2k():
    """Mirror of test_A: kv-side saturated → fire mamba_to_kv."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
    }
    _seed_consec(planner, 0.97, 0.75, ticks=15)
    decision = planner.decide(
        usage_kv=0.97, usage_mamba=0.75, queue_depth=200, snapshot=snap,
    )
    assert decision.direction == "mamba_to_kv", (
        f"BUG: kv-saturated + queue pressure should fire m2k but got "
        f"direction={decision.direction!r} (reason: "
        f"{decision.reason[:200]})"
    )


def test_C_L0_no_signals_does_not_fire():
    """L=0 AND zero queue/pause/retract/evict → no signal → no fire.
    Verifies the multi-source change didn't accidentally make the
    planner over-eager. This is a NEGATIVE control."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
    }
    decision = planner.decide(
        usage_kv=0.80, usage_mamba=0.80,
        queue_depth=0, snapshot=snap,
    )
    assert decision.direction is None, (
        f"OVER-EAGER: no eviction + no admission signal should not fire "
        f"but got direction={decision.direction!r}. Multi-source fix "
        f"may have introduced phantom signal."
    )


def test_D_L_observed_no_queue_uses_evict_cost_only():
    """When L is observed (evictions have happened) and no admission
    pressure, NB should be driven by c(L)·P_save — matches old behavior.
    Regression guard that the multi-source change preserved the
    eviction-cost path."""
    # Seed L via fake tree_cache attribute
    tree = _Tree()
    from sglang.srt.mem_cache.common import (record_recovery_len_kv,
                                              record_recovery_len_rec)
    for _ in range(200):
        record_recovery_len_kv(tree, 10000)
        record_recovery_len_rec(tree, 10000)
    L_kv = tree._slow_recovery_len_kv_ewma
    L_rec = tree._slow_recovery_len_rec_ewma
    assert abs(L_kv - 10000) < 1, f"L EWMA seeding failed: {L_kv}"

    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": L_kv,
        "slow_recovery_len_rec": L_rec,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
    }
    decision = planner.decide(
        usage_kv=0.92, usage_mamba=0.50,
        queue_depth=0, snapshot=snap,
    )
    # At L=10000, c_KV(L) is large enough to fire (no_spike established this)
    assert decision.direction == "mamba_to_kv", (
        f"L-only fire path regressed: usage_kv=0.92, L=10000 should fire "
        f"m2k via c_KV·P_save; got {decision.direction!r}"
    )


def test_E_L_observed_AND_queue_pressure_compose():
    """Both L > 0 AND queue > 0: NB should be larger than L-only case
    (multi-source contributions ADD)."""
    tree = _Tree()
    from sglang.srt.mem_cache.common import (record_recovery_len_kv,
                                              record_recovery_len_rec)
    for _ in range(200):
        record_recovery_len_kv(tree, 10000)
        record_recovery_len_rec(tree, 10000)
    L_kv = tree._slow_recovery_len_kv_ewma
    L_rec = tree._slow_recovery_len_rec_ewma

    # L-only baseline NB (no queue)
    planner1 = _fresh_planner()
    snap1 = {
        "dt": 2.0,
        "slow_recovery_len_kv": L_kv, "slow_recovery_len_rec": L_rec,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
    }
    d1 = planner1.decide(0.92, 0.50, snapshot=snap1)
    # Extract NB from reason string
    import re
    m1 = re.search(r"NB\[m2k\]=([0-9.-]+)us", d1.reason)
    assert m1, f"could not parse NB from: {d1.reason[:200]}"
    nb_l_only = float(m1.group(1))

    # L + queue
    planner2 = _fresh_planner()
    snap2 = dict(snap1, num_queue_reqs=20)
    d2 = planner2.decide(0.92, 0.50, queue_depth=20, snapshot=snap2)
    m2 = re.search(r"NB\[m2k\]=([0-9.-]+)us", d2.reason)
    assert m2
    nb_l_plus_queue = float(m2.group(1))

    print(f"    NB[L-only]={nb_l_only:.0f}us, "
          f"NB[L+queue]={nb_l_plus_queue:.0f}us")
    assert nb_l_plus_queue > nb_l_only, (
        f"Multi-source compose: NB[L+queue]={nb_l_plus_queue} should "
        f"exceed NB[L-only]={nb_l_only}. queue pressure should ADD, "
        f"not replace, the eviction-cost term."
    )


def test_G_both_full_guard_keys_on_active_not_cache_285():
    """#285: the both-full guard must suppress only when BOTH pools are ACTIVE-
    saturated (no reclaimable slack), NOT when a pool is merely CACHE-full.

    Dynamic Case-3 mamba phase: mamba active-bound (0.97), KV active-LIGHT (0.10)
    but KV radix CACHE-full (pool_occupancy_kv 0.97) from the prior long-context
    phase. KV's cold cache is reclaimable slack k2m SHOULD borrow to grow mamba;
    the reuse-aware KV drain cost in nb_k2m prices the cache-eviction harm. The
    cache-inclusive both-full guard wrongly sets nb_k2m=-inf → k2m never fires →
    the mamba phase starves (the #318 dynamic-workload blocker).

    PRE-FIX: direction none (both-full -inf on pool_occupancy). POST-FIX: the
    guard keys on usage_*_active (kv active 0.10 < high) → k2m fires."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0, "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0, "num_retracted_reqs": 0,
        "num_paused_reqs": 0, "num_queue_reqs": 200,
        "usage_mamba_active": 0.97,     # mamba ACTIVE-bound
        "usage_kv_active": 0.10,        # KV ACTIVE-light (short reqs this phase)
        "pool_occupancy_kv": 0.97,      # KV CACHE-full (cold long-ctx residue)
        "pool_occupancy_mamba": 0.97,   # mamba full
    }
    _seed_consec(planner, 0.10, 0.97, ticks=15)
    decision = planner.decide(
        usage_kv=0.10, usage_mamba=0.97, queue_depth=200, snapshot=snap,
    )
    assert decision.direction == "kv_to_mamba", (
        f"#285: both-full guard blocked k2m on CACHE-full KV (pool_occ_kv 0.97) "
        f"despite KV active-light (0.10) — got direction={decision.direction!r} "
        f"(reason: {decision.reason[:220]}). The guard must key on "
        f"usage_*_active, not cache-inclusive pool_occupancy_*."
    )


def test_F_usage_mamba_active_zero_not_falsy_swap():
    """Regression guard for Python falsy-zero bug. The planner now
    classifies on `usage_mamba_active` (admission ceiling, design.md
    §"Budgeter — steady-state pressure rebalance"), falling back to
    total `usage_mamba` if the snapshot doesn't
    carry the active field. The fallback MUST be triggered only when
    the key is absent — NOT when the value is 0.0 (which is the legit
    value on idle workloads where all mamba slots are radix-cached).

    Pre-fix the fallback used `snap.get(k, fb) or fb` — Python treats
    0.0 as falsy → swapped in total → consec incremented on cached
    saturation → phantom fires. Post-fix uses explicit `in` check.

    This test FAILs against the pre-fix-of-fix version (`or` pattern)
    and PASSes against the explicit `in` check version."""
    planner = _fresh_planner()
    # Total mamba 99% (cache full) but ACTIVE 0% (no running req holds a
    # slot). Persist consec should stay 0; no phantom fire.
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
        "usage_mamba_active": 0.0,   # ← legit zero
        "usage_kv_active": 0.0,
    }
    # Drive 20 ticks at high TOTAL usage; persist counter must stay 0
    # because ACTIVE is 0 (cached only).
    for _ in range(20):
        planner.decide(usage_kv=0.10, usage_mamba=0.99, snapshot=snap)
    assert planner._mamba_above_high_consec == 0, (
        f"BUG (falsy-zero): mamba_above_high_consec="
        f"{planner._mamba_above_high_consec} after 20 ticks at "
        f"usage_mamba_active=0.0. Should be 0 because active is 0 — "
        f"only total (which IS 0.99) being misread as the classify input."
    )
    assert planner._kv_above_high_consec == 0, (
        f"same check on kv side: kv_above_high_consec="
        f"{planner._kv_above_high_consec}, should be 0"
    )


def test_G_hot_mamba_cache_must_not_trigger_k2m():
    """CC-trace bug (cc_traces_headline 2026-05-30): mamba pool occupancy 95%+ but
    *active* (admission ceiling) only ~20%; KV active mean 0.60 with
    burst spikes. Pre-fix `_pick_direction_by_nb` was called with
    raw `usage_kv` / `usage_mamba` — so P_save_m read the cache fill
    as pressure → planner fires k2m → shrinks the *real* bottleneck
    (KV) to grow a pool that is already mostly cold cache.

    Active-vs-cache distinction MUST flow into the NB calculation,
    not just the high-water guard. The fix swaps the input at the
    call site to use `usage_*_active` (already classified above).

    With queue pressure in play, the planner is allowed to fire — but
    NOT toward mamba. Either no-fire or fire m2k is acceptable:
    NB[k2m] must NOT exceed NB[m2k]. Equivalently: direction != k2m.
    """
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 50,           # some queue pressure
        "usage_kv_active": 0.60,        # KV is the real workload
        "usage_mamba_active": 0.20,     # mamba is mostly cache (95% raw)
    }
    _seed_consec(planner, usage_kv=0.60, usage_mamba=0.20, ticks=0)
    decision = planner.decide(
        usage_kv=0.60, usage_mamba=0.95,
        queue_depth=50, snapshot=snap,
    )
    assert decision.direction != "kv_to_mamba", (
        f"BUG: hot mamba CACHE (raw=0.95) misread as pressure → "
        f"planner picks k2m, shrinking real-bottleneck KV. "
        f"got direction={decision.direction!r}, reason={decision.reason!r}. "
        f"Active says mamba_active=0.20 (slack), kv_active=0.60 (mild). "
        f"k2m here actively HURTS — KV must evict radix cache to give "
        f"capacity to a pool that already has 75%+ slack."
    )


def test_H_kv_active_high_mamba_cache_high_fires_m2k():
    """Symmetric to G but with KV active at the persist-cooldown
    boundary (kv_active >= high_water). The correct direction is
    m2k — shrink mamba (mostly cold cache) to grow KV (real working
    set under pressure). Pre-fix: NB[k2m] dominates because
    P_save_m(raw=0.95) >> P_save_kv(0.90) is computed but the kv
    guard blocks k2m... wait kv_active=0.90 < high_water=0.95 so
    guard does not block. Pre-fix can still misfire k2m here.
    """
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.92,        # KV genuinely saturating
        "usage_mamba_active": 0.20,     # mamba slack
    }
    _seed_consec(planner, usage_kv=0.92, usage_mamba=0.20, ticks=15)
    decision = planner.decide(
        usage_kv=0.92, usage_mamba=0.95,
        queue_depth=200, snapshot=snap,
    )
    assert decision.direction == "mamba_to_kv", (
        f"BUG: KV active 0.92 saturating + mamba 95% but only 20% active "
        f"should fire m2k (give mamba's nominal capacity to KV). "
        f"got direction={decision.direction!r}, reason={decision.reason!r}"
    )


def test_R_hot_mamba_cache_blocks_m2k():
    """REPRODUCING TEST for the cross-fire regression (task #275),
    RED until the reuse-aware drain cost (option C / task #270) lands.

    Same setup as test_H (KV genuinely saturating, mamba active-slots
    slack, mamba cache full) EXCEPT the mamba cache is HOT: the snapshot
    a drain would force out has high reuse value. test_H fires m2k to
    drain a COLD mamba cache (cheap to re-prefill) and grow KV — correct.
    But when the cache is HOT, draining it evicts high-reuse snapshots
    whose re-prefill is expensive — the m2k is NET-NEGATIVE (this is the
    cc_traces_headline mamba-starve regression: TTFT +23%, cache_hit
    -5.8pp on an 89%-reuse workload).

    The bug: nb_m2k's drain penalty is `c_m × p_loss_m`, and BOTH terms
    collapse on a hot-but-active-low cache — c_m is the re-prefill curve
    (0 at L=0) and p_loss_m = P_save_m is ACTIVE-utilization-based (≈0
    when active slots are slack). So the drain reads as free regardless
    of how hot the cache is, and m2k fires. The active-utilization
    pressure model is blind to cache reuse value.

    The fix (option C, task #275 / #270): price the drain by the
    REUSE-AWARE (hit-weighted) cost of evicting the snapshots the m2k
    would force out — `predict_evict_cost_us('mamba', n)` summed over
    the LRU+LPB eviction victims with their hit counts. The agent
    supplies this per-action cost via snapshot['mamba_drain_cost_us'];
    nb_m2k subtracts it directly (it IS the realized loss — NOT scaled
    by the active-based p_loss_m, which is what zeroes it today). A hot
    cache → large mamba_drain_cost_us → nb_m2k negative → no m2k. A cold
    cache (test_H) → small cost → m2k still fires.

    Pre-C: snapshot['mamba_drain_cost_us'] is IGNORED → m2k fires (like
    test_H) → this assertion FAILS (RED). Post-C: the cost is consumed →
    m2k suppressed → GREEN. The correct direction here is no-fire (KV is
    the saturated pool, so we cannot grow mamba either) — assert the
    core regression claim: do NOT drain the hot mamba cache.
    """
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.92,        # KV genuinely saturating (== test_H)
        "usage_mamba_active": 0.20,     # mamba active-slots slack (== test_H)
        # HOT cache: reuse-aware (hit-weighted) cost to evict the
        # snapshots an m2k drain of dst_chunks_per_action chunks would
        # force out. Large → draining destroys high-reuse prefixes.
        "mamba_drain_cost_us": 10_000_000.0,
    }
    _seed_consec(planner, usage_kv=0.92, usage_mamba=0.20, ticks=15)
    decision = planner.decide(
        usage_kv=0.92, usage_mamba=0.95,
        queue_depth=200, snapshot=snap,
    )
    assert decision.direction != "mamba_to_kv", (
        f"REGRESSION (#275): a HOT mamba cache (mamba_drain_cost_us=10M) "
        f"must NOT be drained — evicting high-reuse snapshots is "
        f"net-negative. The active-utilization drain penalty (c_m × "
        f"p_loss_m) is blind to cache reuse and fires m2k anyway. "
        f"Fix: consume the reuse-aware drain cost (option C / #270). "
        f"got direction={decision.direction!r}, reason={decision.reason!r}"
    )


def test_U_hot_kv_cache_blocks_k2m():
    """#271a — the symmetric image of test_R for the k2m direction. mamba is
    genuinely saturating (so the candidate is k2m: grow mamba by draining KV)
    and KV active-slots are slack, but the KV cache is HOT: the reuse-aware
    cost to evict the KV snapshots a k2m drain would force out is large
    (snapshot['kv_drain_cost_us']). nb_k2m must subtract it once per fire so
    a hot KV cache resists the drain → k2m suppressed. Pre-#271a the KV drain
    was priced only by the active `c_kv × p_loss_kv` (blind to reuse), so k2m
    would fire and evict high-reuse KV prefixes."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.20,        # KV active-slots slack (drainable side)
        "usage_mamba_active": 0.92,     # mamba genuinely saturating → grow mamba
        # HOT KV cache: large reuse-aware cost to evict the KV snapshots a
        # k2m drain would force out.
        "kv_drain_cost_us": 10_000_000.0,
    }
    _seed_consec(planner, usage_kv=0.20, usage_mamba=0.92, ticks=15)
    decision = planner.decide(
        usage_kv=0.95, usage_mamba=0.92,
        queue_depth=200, snapshot=snap,
    )
    assert decision.direction != "kv_to_mamba", (
        f"#271a: a HOT KV cache (kv_drain_cost_us=10M) must NOT be drained — "
        f"evicting high-reuse KV prefixes is net-negative. nb_k2m must "
        f"subtract the reuse-aware KV drain cost once per fire. got "
        f"direction={decision.direction!r}, reason={decision.reason!r}"
    )


def test_S_mamba_evicting_hot_cache_fires_k2m():
    """GROW-side fix (task #277, symmetric to #275's drain fix). When
    mamba is occ-full and actively SHEDDING hot snapshots (cache_hit
    bleeding) but its ACTIVE slot use is moderate, the planner must GROW
    it (fire k2m). Pre-#277 the grow benefit `c_m × P_save_m` was gated by
    the active P_save_m (≈0 at moderate active use), so a 97%-occupied
    pool evicting hot data read as "not pressured" and the planner never
    grew it (the cc_traces_headline mamba-starve regime: NB[k2m]=0).

    The fix adds a reuse-aware eviction-rate term, supplied by the agent
    as snapshot["mamba_evict_grow_us"] = predict_evict_cost_us(mamba, recent
    eviction count) — non-zero exactly when mamba is shedding hot cache,
    and NOT gated by active utilization.

    Pre-fix: the field is ignored → NB[k2m]=0 → no fire → assertion FAILS.
    Post-fix: the grow term clears the gate → fire k2m.
    """
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 2000.0,   # mamba evicting → L_rec observed
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,               # NO queue: pure cache-eviction pressure
        "usage_kv_active": 0.42,           # KV slack (donor)
        "usage_mamba_active": 0.406,       # mamba active-slots moderate (< low_water)
        "pool_occupancy_mamba": 0.969,     # but 97% OCCUPIED
        "pool_occupancy_kv": 0.476,
        # mamba shedding hot snapshots — reuse-aware eviction cost it pays:
        "mamba_evict_grow_us": 20000.0,
    }
    decision = planner.decide(
        usage_kv=0.476, usage_mamba=0.969, queue_depth=0, snapshot=snap,
    )
    assert decision.direction == "kv_to_mamba", (
        f"GROW-side bug (#277): mamba 97% occupied + shedding hot cache "
        f"(mamba_evict_grow_us=20000) but active-slack must fire k2m to "
        f"GROW mamba. The active-gated grow term misses cache-eviction "
        f"pressure. got direction={decision.direction!r}, "
        f"reason={decision.reason!r}"
    )


def test_T_full_but_quiescent_mamba_does_not_grow():
    """Control for test_S (and a guard mirroring test_G): a mamba pool that
    is occ-full but NOT evicting (mamba_evict_grow_us absent/0) is
    reclaimable slack, not pressure — the planner must NOT grow it. This
    pins that the #277 grow term is gated by ACTUAL eviction, not mere
    occupancy."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,      # not evicting
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
        "usage_kv_active": 0.42,
        "usage_mamba_active": 0.406,
        "pool_occupancy_mamba": 0.969,     # full...
        "pool_occupancy_kv": 0.476,
        # ...but NOT shedding (no eviction this tick) → no grow benefit.
    }
    decision = planner.decide(
        usage_kv=0.476, usage_mamba=0.969, queue_depth=0, snapshot=snap,
    )
    assert decision.direction != "kv_to_mamba", (
        f"a full-but-quiescent mamba cache is reclaimable slack, not "
        f"pressure — must NOT grow it. got direction={decision.direction!r}"
    )


def test_V_both_pools_full_suppresses_fire():
    """No-slack guard (cc traces, conc 22, LPB regression). When BOTH pools
    are occupancy-saturated (cache-inclusive), neither has donatable slack —
    cross-fire can only evict cached entries from the 'source' to grow the
    peer, which under coupled prefix-cache demand (a hit needs both the KV
    tokens AND the paired mamba snapshot) orphans the peer's paired entries
    and craters cache_hit. Empirically m2k fired 27x at occ_kv≈occ_mamba≈
    0.99 and dropped cache_hit ~40pp. The NB misses it: grow benefit priced
    per-token (huge) vs drain cost per-slot on the victim's own reuse (tiny).
    Guard: both occupancy ≥ high_water → suppress BOTH directions.

    Here a large kv grow signal + queue would otherwise fire m2k; with both
    occupancies full the guard must return direction None. Pre-fix this
    FAILS (m2k fires)."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 2000.0,
        "slow_recovery_len_rec": 2000.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.60,
        "usage_mamba_active": 0.50,
        "pool_occupancy_kv": 0.99,      # KV cache-full
        "pool_occupancy_mamba": 0.98,   # mamba cache-full → no slack either side
        "cache_hit_rate": 0.50,         # COUPLED prefix-cache (the regime that craters)
        "kv_evict_grow_us": 150000.0,   # KV shedding hot prefixes (per-token, huge)
        "mamba_drain_cost_us": 0.0,     # drained mamba snapshots read cold in isolation
    }
    decision = planner.decide(
        usage_kv=0.99, usage_mamba=0.98, queue_depth=200, snapshot=snap,
    )
    assert decision.direction is None, (
        f"BOTH pools occupancy-full → no donatable slack → must NOT fire "
        f"(rebalancing only shuffles coupled paired entries → cache_hit "
        f"crater). got direction={decision.direction!r}, "
        f"reason={decision.reason!r}"
    )


def test_V2_both_full_guard_off_allows_fire():
    """Toggle (#282 A1 follow-up): the both-full guard exists because m2k
    grow-KV was inert (granted chunks didn't expand effective KV cache) — so
    the only effect of firing both-full was to evict coupled paired entries.
    A1 made KV genuinely growable, so on an arena-backed deployment the
    operator can disable the guard (SGLANG_XPOOL_BOTH_FULL_GUARD=0) to let
    m2k harvest the peer's donatable cold cache. This pins that the toggle
    actually re-opens the path: the SAME both-full scenario that test_V
    suppresses must now fire when the guard is off. (Per-token KV grow
    benefit dominates the per-slot mamba drain cost, so the NB picks m2k.)"""
    planner = _fresh_planner(both_full_guard=False)
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 2000.0,
        "slow_recovery_len_rec": 2000.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.60,
        "usage_mamba_active": 0.50,
        "pool_occupancy_kv": 0.99,
        "pool_occupancy_mamba": 0.98,
        "kv_evict_grow_us": 150000.0,
        "mamba_drain_cost_us": 0.0,
    }
    decision = planner.decide(
        usage_kv=0.99, usage_mamba=0.98, queue_depth=200, snapshot=snap,
    )
    assert decision.direction == "mamba_to_kv", (
        f"with both_full_guard=False the SAME both-full scenario as test_V "
        f"must fire m2k (KV grow benefit per-token dominates mamba drain "
        f"per-slot) — the toggle re-opens the A1 grow-KV path. got "
        f"direction={decision.direction!r}, reason={decision.reason!r}"
    )


def test_W_one_pool_slack_still_fires():
    """Control for test_V: when the SOURCE pool has genuine occupancy slack
    (the starve-win shape: mamba bound + KV slack → k2m to grow mamba), the
    both-full guard must NOT suppress — there is real slack to move. Pins
    that test_V's guard keys on BOTH-full, not either-full."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 2000.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
        "usage_kv_active": 0.30,
        "usage_mamba_active": 0.50,
        "pool_occupancy_kv": 0.40,       # KV has genuine slack (donor)
        "pool_occupancy_mamba": 0.98,    # mamba bound + shedding hot cache
        "mamba_evict_grow_us": 20000.0,
    }
    decision = planner.decide(
        usage_kv=0.40, usage_mamba=0.98, queue_depth=0, snapshot=snap,
    )
    assert decision.direction == "kv_to_mamba", (
        f"KV has occupancy slack (0.40) + mamba shedding hot cache → must "
        f"still fire k2m to grow mamba; the both-full guard must not "
        f"suppress when one pool has slack. got {decision.direction!r}"
    )


def test_N_marginal_saturation_does_not_dominate_pressure():
    """ARCHITECTURAL fix (task #171, 2026-05-30, surfaced by cc_traces_headline@C=56):
    pressure attribution must mirror paper's c_σ(L) × P_save form. A pool
    barely above low_water (P_save ≈ 0.03) carries marginal saturation —
    queue pressure attributed to it should ALSO be marginal, not the full
    100% the binary excess-share split gives.

    Live cc_traces_headline @ C=56 (post-#165+#170) reproduced this with kv_active=0.18,
    mamba_active=0.41 (just 0.01 above mamba_low_water=0.40, with
    mamba_high_water=0.80 so the saturation index = 0.01/0.60 ≈ 0.017).
    Pre-fix (binary excess-share):
      `total_excess = 0.01, m_share = 1.0`
      → `pressure_to_m = queue_us × 1.0` → NB[k2m] ≈ 6M >> threshold 7.5K → fires k2m
    Even though kv has 82% headroom AND mamba has 59% headroom — neither
    is close to saturated.

    Architecturally clean fix: `pressure_to_σ = admit × P_save_σ` (no
    share-split). With P_save_m ≈ 0.017, pressure_to_m is only ~1.7% of
    the queue signal, NB[k2m] doesn't clear threshold, no fire.

    Test mirrors live with _fresh_planner's low_water=0.70 / high_water=0.95:
      kv=0.30 (deeply slack), mamba=0.71 (just 0.01 above low_water).
    Saturation index P_save_m = 0.01/0.30 ≈ 0.033. Pre-fix fires k2m
    (m_share=1.0 → pressure_to_m × lifetime > 60K threshold). Post-fix
    no-fires (pressure_to_m ≈ 3% of queue, NB < threshold).
    """
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,           # significant queue, mirrors live
        "usage_kv_active": 0.30,         # KV 70% headroom
        "usage_mamba_active": 0.71,      # mamba barely above low_water (0.01)
    }
    dec = planner.decide(
        usage_kv=0.30, usage_mamba=0.71,
        queue_depth=200, snapshot=snap,
    )
    assert dec.direction is None, (
        f"BUG (binary excess attribution at C=56): mamba_active=0.71 is "
        f"0.01 above low_water=0.70 but kv has 70% headroom. Pre-fix "
        f"attributes 100% of queue pressure to mamba → fires k2m even "
        f"though neither pool is near saturation. Post-fix should weight "
        f"by P_save_m ≈ 0.033 → marginal NB contribution → no-fire. "
        f"got direction={dec.direction!r} reason={dec.reason!r}"
    )


def test_O_pressure_proportional_to_saturation_index():
    """ARCHITECTURAL invariant (task #171): pressure_to_σ must scale
    LINEARLY with P_save_σ (the paper-faithful saturation ramp). Compare
    NB[m2k] at two saturation levels with same queue:
      A) kv_active = 0.85 → P_save_kv = (0.85-0.70)/0.30 = 0.500
      B) kv_active = 0.95 → P_save_kv = (0.95-0.70)/0.30 = 0.833
    Ratio: 0.833/0.500 = 1.667. NB[m2k] difference must reflect this.

    Pre-fix (binary excess-share): for BOTH cases mamba is below
    low_water → m_share=0, kv_share=1.0 → pressure_to_kv = full queue
    in both cases → NB[m2k] roughly EQUAL (only c_kv × P_save_kv term
    differentiates, but c_kv=0 at L=0). So pre-fix ratio ≈ 1.0 → FAIL.

    Post-fix: NB[m2k]_high / NB[m2k]_mid ≈ 1.67.
    """
    import re
    def _nb_m2k(kv_a):
        planner = _fresh_planner()
        snap = {
            "dt": 2.0,
            "slow_recovery_len_kv": 0.0,
            "slow_recovery_len_rec": 0.0,
            "num_evicted_tokens_recent": 0,
            "num_retracted_reqs": 0,
            "num_paused_reqs": 0,
            "num_queue_reqs": 100,
            "usage_kv_active": kv_a,
            "usage_mamba_active": 0.20,   # well below low_water
        }
        dec = planner.decide(kv_a, 0.20, queue_depth=100, snapshot=snap)
        m = re.search(r"NB\[m2k\]=([-0-9.einf]+)us", dec.reason)
        assert m, f"could not parse NB[m2k] from: {dec.reason!r}"
        return float(m.group(1))

    nb_mid = _nb_m2k(0.85)     # P_save_kv = 0.500
    nb_high = _nb_m2k(0.949)   # P_save_kv ≈ 0.830 (stay under high_water guard 0.95)
    expected_ratio = 0.830 / 0.500
    actual_ratio = nb_high / max(nb_mid, 1.0)
    print(f"    NB[m2k @ kv=0.85]={nb_mid:.0f}us  NB[m2k @ kv=0.949]={nb_high:.0f}us  "
          f"ratio={actual_ratio:.2f} (expected ~{expected_ratio:.2f})")
    # Pre-fix would have ratio ≈ 1.0 (binary share gives equal pressure).
    # Post-fix should be ≈ 1.66. Bound generously to allow persist/cost
    # second-order effects.
    assert actual_ratio > 1.30, (
        f"Attribution NOT proportional to P_save: ratio={actual_ratio:.2f}; "
        f"expected ~1.66 (P_save_kv: 0.500 → 0.830). Pre-fix binary "
        f"excess-share gives equal pressure regardless of saturation magnitude."
    )


def test_M_both_slack_queue_pressure_does_not_fire():
    """Live cc_traces_headline post-#165 (2026-05-30) surfaced THIS bug: with planner
    correctly using `usage_*_active`, the CC trace produced 3 wrong
    k2m fires at kv_active=0.26..0.42, mamba_active=0.09..0.10. Both
    pools FAR below low_water (0.85), no memory pressure at all.

    Reason from the live log:
      `NB[k2m]=1000000us NB[m2k]=1000000us threshold=9382us
       (c_kv=0us@L=0, c_m=0us@L=0, P_save: kv=0.00 m=0.00,
        pressure_to: kv=100000us m=100000us)`

    Both NB are EXACTLY equal because at total_excess=0,
    `_pick_direction_by_nb` (lines 304-311) splits queue pressure
    50/50 between the two pools — but a queue with both pools slack
    means the stall is NOT memory-bound (it's compute/batch). Splitting
    a non-memory signal across memory pools then tie-breaks via
    `nb_k2m >= nb_m2k` → k2m wins by default → wastes a ~300ms
    unmap/map cycle that helps nothing.

    Test: both pools at active << low_water + non-zero queue → planner
    MUST return direction=None ("no candidate cleared gate"), NOT a
    k2m fire by tie-break.
    """
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,           # significant queue
        "usage_kv_active": 0.30,         # both far below low_water 0.85
        "usage_mamba_active": 0.10,
    }
    dec = planner.decide(
        usage_kv=0.30, usage_mamba=0.10,
        queue_depth=200, snapshot=snap,
    )
    assert dec.direction is None, (
        f"BUG (tie-break-on-both-slack): both pools far below low_water "
        f"(kv_active=0.30, mamba_active=0.10), no memory pressure. Queue "
        f"stall is NOT memory-bound — pressure must NOT be attributed to "
        f"either pool. got direction={dec.direction!r}, reason={dec.reason!r}"
    )


def test_M2_single_side_pressed_still_fires():
    """Positive control: when ONE pool exceeds low_water, queue pressure
    SHOULD be attributed to that side. This guards against over-fix
    where we zero out pressure entirely whenever total_excess=0 but
    forget that "one side > low_water" still warrants attribution."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.92,         # above low_water
        "usage_mamba_active": 0.10,      # slack
    }
    dec = planner.decide(
        usage_kv=0.92, usage_mamba=0.10,
        queue_depth=200, snapshot=snap,
    )
    assert dec.direction == "mamba_to_kv", (
        f"OVER-FIX: kv_active=0.92 > low_water=0.85, queue pressure "
        f"belongs to KV side → m2k should fire. got direction={dec.direction!r} "
        f"reason={dec.reason!r}"
    )


def test_P_low_max_psave_kills_admit_pressure():
    """COST-MODEL FIX (task #173, 2026-05-30): N=3 cc_traces_headline@C=56 confirmed
    the planner fires 5/8 WRONG k2m at deep KV slack. Per-run mean_ttft:
    +3.86%, +4.56%, -11.68%; p99: -5%, -5%, -0.4% (all LOSS); tps:
    -36%, -39%, -42% (consistent LOSS).

    The wrong fires all have shape kv_active=0.18..0.45 (deep slack)
    + mamba_active=0.42 (just above mamba_low_water=0.40). The cost
    model computes pressure_to_m = admit × P_save_m ≈ admit × 0.03 ≈ 9K
    per tick; × lifetime ≈ 90-145K; > threshold 7.5K → fires.

    But the queue with both pools deeply slack is NOT memory-bound.
    Neither pool can credibly absorb the stall. The cost model is
    missing the Bayesian credibility factor: P(stall is memory-bound)
    ≈ max(P_save_kv, P_save_m). At max≈0.03, only ~3% of queue stall
    is plausibly memory-bound. Net benefit ~3% × original → falls
    below threshold → no fire.

    Same P_save ramp as the eviction-cost term (c_σ × P_save_σ) and
    the directional attribution (#171). Architecturally symmetric.

    Test uses PRODUCTION config (low_kv=0.50, low_m=0.40) to mirror
    the live workload that surfaced the bug.
    """
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.85, kv_low_water=0.50,
        mamba_high_water=0.80, mamba_low_water=0.40,
        cooldown_min_s=20.0, amortize_horizon_s=20.0,
        dst_chunks_per_action=4,
        nb_margin=1.5, nb_chunk_cost_us=10000.0,
    )
    # Production-matched queue_wait_us (run_cc.sh sets
    # SGLANG_XPOOL_QUEUE_WAIT_US=125000). Live cc_traces_headline reasons reverse-engineer
    # to admit_pressure ≈ 228K us/tick with queue ≈ 2 reqs, so we mirror.
    planner = XPoolPlanner(
        config=cfg,
        adapter=SGLangPressureAdapter(queue_wait_us=125000.0),
    )
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 2,           # production live had ~2 reqs queued
        "usage_kv_active": 0.20,      # deep KV slack (P_save_kv = 0)
        "usage_mamba_active": 0.42,   # marginal (P_save_m = 0.033)
    }
    dec = planner.decide(0.20, 0.42, queue_depth=2, snapshot=snap)
    assert dec.direction is None, (
        f"BUG (#173 missing): kv_active=0.20 (80% headroom) + "
        f"mamba_active=0.42 (just 0.02 above low_water 0.40) is NOT "
        f"memory-bound. max(P_save) ≈ 0.03 → queue stall is compute/"
        f"batch-bound. Cost model must credit admit_pressure by "
        f"max(P_save) so this case does not fire. "
        f"got direction={dec.direction!r}, reason={dec.reason!r}"
    )


def test_Q_genuine_kv_pressure_still_fires_m2k():
    """Regression guard for #173: when KV is climbing toward saturation
    (P_save_kv >= 0.4), the credibility gate must still let m2k fire.
    Mirror of run 2 fire 3 (live cc_traces_headline@C=56): kv_active=0.73,
    mamba_active=0.43. max(P_save) = 0.46. m2k must still fire.
    """
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.85, kv_low_water=0.50,
        mamba_high_water=0.80, mamba_low_water=0.40,
        cooldown_min_s=20.0, amortize_horizon_s=20.0,
        dst_chunks_per_action=4,
        nb_margin=1.5, nb_chunk_cost_us=10000.0,
    )
    # Production-matched queue_wait_us (run_cc.sh sets
    # SGLANG_XPOOL_QUEUE_WAIT_US=125000). Live cc_traces_headline reasons reverse-engineer
    # to admit_pressure ≈ 228K us/tick with queue ≈ 2 reqs, so we mirror.
    planner = XPoolPlanner(
        config=cfg,
        adapter=SGLangPressureAdapter(queue_wait_us=125000.0),
    )
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 2,
        "usage_kv_active": 0.73,
        "usage_mamba_active": 0.43,
    }
    dec = planner.decide(0.73, 0.43, queue_depth=2, snapshot=snap)
    assert dec.direction == "mamba_to_kv", (
        f"OVER-FIX (#173): kv_active=0.73 → P_save_kv=0.46 is real "
        f"saturation pressure; m2k MUST still fire. Live run 2 fire 3 "
        f"was correct. got {dec.direction!r}, reason={dec.reason!r}"
    )


def test_J_distinct_L_kv_L_m_route_into_distinct_c_sigma_inputs():
    """The planner routes `slow_recovery_len_kv` into the c_kv curve and
    `slow_recovery_len_rec` into the c_m curve, NOT lockstep. All other
    no_spike tests either zero L_rec (→ fallback to L_kv) or set them
    equal (D, E), so a future "half-fix" that wires only one side (e.g.
    `c_m_us(L_kv)` typo) would slip through.

    This locks the per-side L routing: with distinct L_kv vs L_rec, the
    planner's reason must report `c_kv@L=L_kv` and `c_m@L=L_rec`. We do
    NOT assert the c_σ magnitudes: the BUILTIN_DEFAULT curve is
    single-curve (c_M ≡ 0, every miss re-prefills the whole bound prefix
    folded into c_KV), so c_m is 0 by design regardless of L_rec. The
    routing of the L INPUT is what this guards.
    """
    import re
    from sglang.srt.mem_cache.common import (record_recovery_len_kv,
                                              record_recovery_len_rec)

    def _drive(L_kv_target: float, L_rec_target: float):
        # Seed BOTH EWMAs to their distinct targets via the per-side
        # producers (NOT lockstep). Use enough events that the EWMA
        # converges to the target value.
        tree = _Tree()
        for _ in range(200):
            record_recovery_len_kv(tree, L_kv_target)
            record_recovery_len_rec(tree, L_rec_target)
        L_kv = tree._slow_recovery_len_kv_ewma
        L_rec = tree._slow_recovery_len_rec_ewma
        # Sanity: EWMA settled to per-side targets, not a blend.
        assert abs(L_kv - L_kv_target) < 1.0, (
            f"L_kv EWMA did not converge: {L_kv} vs target {L_kv_target}")
        assert abs(L_rec - L_rec_target) < 1.0, (
            f"L_rec EWMA did not converge: {L_rec} vs target {L_rec_target}")

        planner = _fresh_planner()
        snap = {
            "dt": 2.0,
            "slow_recovery_len_kv": L_kv,
            "slow_recovery_len_rec": L_rec,
            "num_evicted_tokens_recent": 0,
            "num_retracted_reqs": 0,
            "num_paused_reqs": 0,
            "num_queue_reqs": 0,
        }
        dec = planner.decide(0.92, 0.92, snapshot=snap)
        m_kv = re.search(r"c_kv=([0-9.]+)us@L=([0-9.]+)", dec.reason)
        m_m  = re.search(r"c_m=([0-9.]+)us@L=([0-9.]+)", dec.reason)
        assert m_kv and m_m, (
            f"could not parse c_kv/c_m from: {dec.reason!r}")
        return (float(m_kv.group(1)), float(m_kv.group(2)),
                float(m_m.group(1)), float(m_m.group(2)))

    # Regime 1: L_kv=2000 vs L_rec=16000; each must reach its own curve.
    c_kv_1, L_kv_seen_1, c_m_1, L_m_seen_1 = _drive(2000.0, 16000.0)
    assert abs(L_kv_seen_1 - 2000.0) < 100, (
        f"planner used wrong L for c_kv path: saw L={L_kv_seen_1}, "
        f"expected ~2000. Bug: c_kv computed from L_rec instead of L_kv.")
    assert abs(L_m_seen_1 - 16000.0) < 100, (
        f"planner used wrong L for c_m path: saw L={L_m_seen_1}, "
        f"expected ~16000. Bug: c_m computed from L_kv instead of L_rec.")

    # Regime 2: swap. L_kv=16000 vs L_rec=2000; routing must follow.
    c_kv_2, L_kv_seen_2, c_m_2, L_m_seen_2 = _drive(16000.0, 2000.0)
    assert abs(L_kv_seen_2 - 16000.0) < 100, (
        f"swap regime: c_kv used L={L_kv_seen_2}, expected ~16000")
    assert abs(L_m_seen_2 - 2000.0) < 100, (
        f"swap regime: c_m used L={L_m_seen_2}, expected ~2000")

    print(f"    regime1 (L_kv=2k, L_rec=16k): c_kv@L={L_kv_seen_1:.0f} "
          f"c_m@L={L_m_seen_1:.0f}")
    print(f"    regime2 (L_kv=16k, L_rec=2k): c_kv@L={L_kv_seen_2:.0f} "
          f"c_m@L={L_m_seen_2:.0f}")


def test_K_classify_must_use_active():
    """The arg-max planner classifies the per-pool saturation state on ACTIVE
    usage, so a hot mamba CACHE (raw 0.99) with active-slack (0.20) must
    classify BELOW_LOW and must NOT fire k2m: the cache fill is cheaply
    LRU-evictable, not pressure.

    classify(mamba_active=0.20) → BELOW_LOW → no mamba pressure → no k2m.
    """
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.95, kv_low_water=0.70,
        mamba_high_water=0.95, mamba_low_water=0.70,
        cooldown_min_s=20.0, amortize_horizon_s=20.0,
        dst_chunks_per_action=4,
        nb_margin=1.5, nb_chunk_cost_us=10000.0,
    )
    planner = XPoolPlanner(config=cfg, adapter=SGLangPressureAdapter())

    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,           # plenty of pressure
        "usage_kv_active": 0.60,         # KV mild
        "usage_mamba_active": 0.20,      # mamba truly slack (cache only)
    }
    # Raw mamba=0.99 but active 0.20: classify must read active → BELOW_LOW.
    dec = planner.decide(
        usage_kv=0.60, usage_mamba=0.99,
        queue_depth=200, snapshot=snap,
    )
    assert dec.direction != "kv_to_mamba", (
        f"hot mamba CACHE (raw=0.99, active=0.20) must NOT fire k2m: the "
        f"cache fill is reclaimable slack, not pressure. got "
        f"direction={dec.direction!r}, reason={dec.reason!r}."
    )
    # Stronger: classify state must reflect active.
    assert planner._mamba_state == planner.BELOW_LOW, (
        f"_mamba_state={planner._mamba_state!r} after tick; expected "
        f"BELOW_LOW because mamba_active=0.20 < low_water=0.70. classify "
        f"must consult active usage, not raw."
    )


def test_K2_active_genuinely_pressured_still_fires():
    """Positive control mirror of test_I: when mamba IS actually saturated
    (active high, not just cache), classify must cross to ABOVE_HIGH and the
    arg-max planner must fire k2m. Guards against over-correction.
    """
    reset_runtime_actuator_cost()
    reset_cost_curves()
    import sglang.srt.budgeter.cost_model as cm
    cm._cost_curves = BUILTIN_DEFAULT
    cfg = XPoolPolicyConfig(
        kv_high_water=0.95, kv_low_water=0.70,
        mamba_high_water=0.95, mamba_low_water=0.70,
        cooldown_min_s=20.0, amortize_horizon_s=20.0,
        dst_chunks_per_action=4,
        nb_margin=1.5, nb_chunk_cost_us=10000.0,
    )
    planner = XPoolPlanner(config=cfg, adapter=SGLangPressureAdapter())
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.40,
        "usage_mamba_active": 0.99,      # genuinely saturated
    }
    dec = planner.decide(
        usage_kv=0.40, usage_mamba=0.99,
        queue_depth=200, snapshot=snap,
    )
    assert dec.direction == "kv_to_mamba", (
        f"OVER-CORRECTION: mamba_active=0.99 is real pressure; classify must "
        f"cross ABOVE_HIGH → fire k2m. "
        f"got direction={dec.direction!r} reason={dec.reason!r}"
    )


def test_L_consec_increments_naturally_from_active():
    """Audit-gap 1.3 (2026-05-30): no_spike sub-tests A/B/D/E/G/H/I all use
    `_seed_consec` to write `_kv_above_high_consec` directly, bypassing
    the `_classify → consec` wiring at lines 655-668. Test F runs 20
    ticks naturally but only covers the falsy-zero edge case
    (`usage_*_active = 0.0`). No test verifies "classify reads
    `_active` and increments consec accordingly" with `_active`
    actually nonzero.

    Two phases:
      Phase 1: 5 ticks with raw=0.99 but active=0.50 (BELOW high)
               → consec MUST stay 0 (because active is the input).
      Phase 2: 5 more ticks with active=0.99 → consec MUST reach 5
               (because active is the input).

    Pre-active-fix (#113): Phase 1 would see consec=5 (bug, raw used).
    Current code: should pass both phases.

    A future "half-fix" that wires _active into some sub-paths but
    not `_classify` falls here.
    """
    planner = _fresh_planner()
    snap_raw_only = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
        "usage_kv_active": 0.50,         # explicit non-zero, non-trigger
        "usage_mamba_active": 0.50,
    }
    for _ in range(5):
        planner.decide(usage_kv=0.99, usage_mamba=0.99, snapshot=snap_raw_only)
    assert planner._kv_above_high_consec == 0, (
        f"Phase 1 (raw=0.99 active=0.50): _kv_above_high_consec="
        f"{planner._kv_above_high_consec} after 5 ticks; expected 0 "
        f"because active=0.50 < high_water=0.95. _classify is reading "
        f"raw, not active."
    )
    assert planner._mamba_above_high_consec == 0, (
        f"Phase 1 mamba: _mamba_above_high_consec="
        f"{planner._mamba_above_high_consec}, expected 0"
    )

    # Phase 2: bump active high; raw also high. consec must climb.
    snap_active_high = dict(snap_raw_only,
                             usage_kv_active=0.99,
                             usage_mamba_active=0.99)
    for _ in range(5):
        planner.decide(usage_kv=0.99, usage_mamba=0.99, snapshot=snap_active_high)
    # We may have fired and reset the counters mid-loop; allow >= 1 but
    # require monotone climb to be possible. Stronger: at least one tick
    # of accumulation happened from active=0.99.
    assert planner._kv_above_high_consec >= 1 or planner._cooldown_remaining > 0, (
        f"Phase 2: _kv_above_high_consec={planner._kv_above_high_consec} "
        f"and cooldown={planner._cooldown_remaining}. Expected either "
        f"consec climbed (no fire) or planner fired (cooldown engaged). "
        f"Both zero means _classify still ignores active."
    )


def test_I_mamba_genuinely_pressured_fires_k2m():
    """Positive control: when mamba is ACTUALLY pressured (active
    high, not just cache), the planner SHOULD still pick k2m. This
    guards against an over-correction where the fix completely
    ignores raw usage and never fires k2m anymore."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 0.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 200,
        "usage_kv_active": 0.40,        # KV slack
        "usage_mamba_active": 0.92,     # mamba GENUINELY pressured
    }
    _seed_consec(planner, usage_kv=0.40, usage_mamba=0.92, ticks=15)
    decision = planner.decide(
        usage_kv=0.40, usage_mamba=0.92,
        queue_depth=200, snapshot=snap,
    )
    assert decision.direction == "kv_to_mamba", (
        f"OVER-CORRECTION: mamba_active=0.92 is real pressure; "
        f"planner should still fire k2m. got {decision.direction!r}, "
        f"reason={decision.reason!r}"
    )


# ---------- runner ----------

def main():
    tests = [
        ("A L=0 + mamba-saturated + queue → fire kv_to_mamba",
         test_A_L0_queue_pressure_mamba_saturated_fires_k2m),
        ("B L=0 + kv-saturated + queue → fire mamba_to_kv",
         test_B_L0_queue_pressure_kv_saturated_fires_m2k),
        ("C L=0 + zero signals → no fire (no over-eager)",
         test_C_L0_no_signals_does_not_fire),
        ("D L observed + no queue → c(L)·P_save still works (regression guard)",
         test_D_L_observed_no_queue_uses_evict_cost_only),
        ("E L + queue together → NB[L+queue] > NB[L-only] (compose ADD)",
         test_E_L_observed_AND_queue_pressure_compose),
        ("F usage_*_active=0.0 must NOT be falsy-swapped to total",
         test_F_usage_mamba_active_zero_not_falsy_swap),
        ("G hot mamba CACHE (raw 0.95, active 0.20) must NOT trigger k2m",
         test_G_hot_mamba_cache_must_not_trigger_k2m),
        ("H kv_active 0.92 + mamba 95% cache → fire m2k",
         test_H_kv_active_high_mamba_cache_high_fires_m2k),
        ("I mamba_active genuinely 0.92 → still fire k2m (regression guard)",
         test_I_mamba_genuinely_pressured_fires_k2m),
        ("J distinct L_kv vs L_rec route into distinct c_kv vs c_m inputs",
         test_J_distinct_L_kv_L_m_route_into_distinct_c_sigma_inputs),
        ("K hot mamba CACHE must classify BELOW_LOW (active, not raw)",
         test_K_classify_must_use_active),
        ("K2 genuine mamba_active 0.99 → still fire k2m",
         test_K2_active_genuinely_pressured_still_fires),
        ("L _classify+consec naturally reads active, not raw",
         test_L_consec_increments_naturally_from_active),
        ("M both-slack + queue: NOT memory pressure → no-fire (live cc_traces_headline)",
         test_M_both_slack_queue_pressure_does_not_fire),
        ("M2 one-side pressed + queue → still fires (over-fix guard)",
         test_M2_single_side_pressed_still_fires),
        ("N marginal saturation does NOT dominate pressure (architectural)",
         test_N_marginal_saturation_does_not_dominate_pressure),
        ("O pressure attribution proportional to P_save (architectural)",
         test_O_pressure_proportional_to_saturation_index),
        ("P low max(P_save): admit_pressure killed by credibility gate",
         test_P_low_max_psave_kills_admit_pressure),
        ("Q genuine KV pressure → m2k still fires (regression guard)",
         test_Q_genuine_kv_pressure_still_fires_m2k),
        ("R hot mamba cache blocks m2k (reuse-aware drain cost, #275/#270)",
         test_R_hot_mamba_cache_blocks_m2k),
        ("S mamba evicting hot cache → fire k2m (#277 grow-side)",
         test_S_mamba_evicting_hot_cache_fires_k2m),
        ("T full-but-quiescent mamba does not grow (#277 gate)",
         test_T_full_but_quiescent_mamba_does_not_grow),
        ("U hot KV cache blocks k2m (reuse-aware KV drain cost, #271a)",
         test_U_hot_kv_cache_blocks_k2m),
        ("V both pools full → no fire (both-full no-slack guard)",
         test_V_both_pools_full_suppresses_fire),
        ("V2 both-full guard off → same scenario fires m2k (#282 toggle)",
         test_V2_both_full_guard_off_allows_fire),
        ("W one pool has slack → still fires (both-full guard control)",
         test_W_one_pool_slack_still_fires),
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
    print(f"\nDverify NB multi-source: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
