"""#270 — the Budgeter steady-state m2k fire must DRAIN cold cache, not
just harvest genuinely-free pages.

Reproduces the cc_traces_headline 3b finding (2026-06-07): at conc22 the
Budgeter fired m2k 12x with `free=4 drain=0` while mamba sat cache-full /
active-empty (occ 0.945, usage_mamba_active 0.175) — ~77% donatable cold
cache that the fire refused to evict, so KV grew only a trickle (+~9K
tokens) and cache_hit didn't move.

Root cause: `BudgetAgent._maybe_fire` called `XPoolFirePlanner.build(dir,
n)` WITHOUT `allow_drain=True`, so Stage-2 (Drain-expansion) was skipped
and the plan was free-only. This contradicts design.md §"Budgeter —
steady-state pressure rebalance" (the Budgeter exists precisely to catch
"mamba sits half-empty holding cold cache ... cache hit rate slowly
bleeds") and §"Grow benefit and drain cost are both reuse-aware": the
planner's `nb_m2k` already SUBTRACTS the reuse-aware `mamba_drain_cost_us`
once per fire, i.e. it priced a drain it then never executed.

Two layers pinned:
  - test_A/B (planner contract): mamba-direction Stage-2 gating — free-only
    refuses when free<n; allow_drain harvests cold cached pages.
  - test_C (the fix site): the BudgetAgent steady-state fire passes
    allow_drain=True to the planner.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.arena.owner_provider import OwnerMap
from sglang.srt.budgeter.fire_planner import XPoolFirePlanner


# ---------------------------------------------------------------------------
# Planner-contract layer: a canned OwnerProvider whose mamba pool has only a
# FEW free pages but MANY cold cached pages (the 3b shape).
# ---------------------------------------------------------------------------
class _ColdMambaProvider:
    """mamba source: n_pages total, `free` genuinely-free, the rest cold
    cached (drainable only when allow_drain=True — mirrors the real
    SchedulerOwnerProvider, which only walks the radix cache for drain
    victims when allow_drain is set)."""

    def __init__(self, *, n_pages, free, cached_in_cost_order):
        self.n_pages = n_pages
        self.free = list(free)
        self.cached = list(cached_in_cost_order)
        self.last_allow_drain = None

    def build_mamba_owner_map(self, *, allow_drain=False, allow_migrate=False,
                              max_drain_pages=None):
        self.last_allow_drain = allow_drain
        self.last_max_drain_pages = max_drain_pages
        cached = None
        if allow_drain:
            cached = list(self.cached)
            if max_drain_pages is not None:
                cached = cached[:max_drain_pages]
        return OwnerMap(
            pool_name="mamba",
            n_pages=self.n_pages,
            free_pages=list(self.free),
            cached_pages_in_cost_order=cached,
            live_pages_in_cost_order=None,
        )

    def build_kv_owner_map(self, *, allow_drain=False, allow_migrate=False,
                           max_drain_pages=None):
        raise AssertionError("test only fires mamba_to_kv")


def _planner(provider):
    return XPoolFirePlanner(
        kv_actuator=types.SimpleNamespace(),
        mamba_actuator=types.SimpleNamespace(),
        owner_provider=provider,
    )


def test_A_free_only_refuses_when_free_short():
    """The Budgeter's OLD call shape (no allow_drain): mamba has 2 free but
    we need 6 — Stage-1 alone can't reach n, and with drain gated off the
    plan REFUSES (returns None). This is the inert 3b behavior."""
    prov = _ColdMambaProvider(n_pages=64, free=[60, 61],
                              cached_in_cost_order=[10, 11, 12, 13, 14, 15])
    pl = _planner(prov)
    plan = pl.build("mamba_to_kv", 6)  # no allow_drain → free-only
    assert prov.last_allow_drain is False, "default must be free-only"
    assert plan is None, (
        "free-only (drain gated off) with free<n must refuse — this is the "
        "drain=0 inertia the 3b run hit"
    )


def test_B_allow_drain_harvests_cold_cache():
    """Same pool, allow_drain=True: Stage-2 fills the remaining 4 pages from
    the cold cached set in cost order → plan built with drain=4."""
    prov = _ColdMambaProvider(n_pages=64, free=[60, 61],
                              cached_in_cost_order=[10, 11, 12, 13, 14, 15])
    pl = _planner(prov)
    plan = pl.build("mamba_to_kv", 6, allow_drain=True)
    assert plan is not None, "drain-expansion must reach n=6"
    assert len(plan.drains) == 4, (
        f"need 4 cold pages drained to top up 2 free → 6; got "
        f"{len(plan.drains)} ({plan.drains})"
    )
    # cheapest-first cost order preserved
    assert set(plan.drains) == {10, 11, 12, 13}, (
        f"drains must be the 4 cheapest cached pages in cost order; got "
        f"{plan.drains}"
    )


# ---------------------------------------------------------------------------
# Fix-site layer: the BudgetAgent steady-state fire must pass allow_drain.
# ---------------------------------------------------------------------------
class _SpyFirePlanner:
    """Records the (allow_drain, allow_migrate) the agent passes, then
    refuses (returns None) so _maybe_fire returns right after build with no
    actuator execution."""

    def __init__(self):
        self.calls = []

    def build(self, direction, n_pages_target, *,
              allow_drain=False, allow_migrate=False):
        self.calls.append({
            "direction": direction,
            "n_pages_target": n_pages_target,
            "allow_drain": allow_drain,
            "allow_migrate": allow_migrate,
        })
        return None  # refuse → _maybe_fire returns at the plan-None branch


class _SpyPlanner:
    """Decides a fixed direction so the agent reaches the build call."""

    config = types.SimpleNamespace(
        dst_chunks_per_action=1, cooldown_min_s=32.0, amortize_horizon_s=32.0,
    )

    def __init__(self, direction="mamba_to_kv"):
        self._dir = direction

    def decide(self, usage_kv, usage_mamba, *, queue_depth, snapshot):
        from sglang.srt.budgeter.xpool_planner import PlanDecision
        return PlanDecision(
            direction=self._dir, reason="spy: force fire",
            usage_kv=usage_kv, usage_mamba=usage_mamba, queue_depth=queue_depth,
        )


class _FakeTree:
    """Minimal tree_cache for the drain gate / cost enrichment. Records the
    slot count passed to predict_evict_cost_us so the drain-volume fix can
    be pinned."""

    def __init__(self, lpb=True):
        self._lpb = lpb
        self.evict_cost_calls = []

    def _should_use_lpb(self):
        return self._lpb

    def predict_evict_cost_us(self, n, pool):
        self.evict_cost_calls.append((int(n), pool))
        return 1000.0

    def full_evictable_size(self):
        return 0

    def mamba_evictable_size(self):
        return 0

    def mamba_protected_size(self):
        # Locked mamba cache the m2k drain cannot reclaim (#312 floor input).
        return 0

    # Fork-failure grow hook (P4-b): None here = fork grow not wired, so the
    # adaptive floor (P4.5) stays the full max_running+protected+fork_headroom.
    _mamba_grow_hook = None


def _make_drain_agent(direction="mamba_to_kv", lpb=True, kv_headroom=True,
                      tokens_per_page=2, n_free=0):
    """Construct a BudgetAgent wired with spies, with the drain gate's deps
    present: an LPB/LRU tree_cache, an owner_provider exposing tokens/page,
    and a KV allocator with (or without) grow headroom."""
    from sglang.srt.budgeter.agent import BudgetAgent
    # arena-backed pool: tokens_per_chunk=1 so the #312 working-set floor caps
    # in plain page units (live_size 256 − floor 80 = 176 >= _n_pages_per_fire,
    # so the m2k drain volume is unchanged by the floor here).
    mamba_pool = types.SimpleNamespace(
        live_size=256, max_size=512, size=256, available_size=lambda: 200,
        _mamba_temporal_arena=types.SimpleNamespace(tokens_per_chunk=1),
    )
    kv_pool = types.SimpleNamespace(mamba_pool=mamba_pool)
    live = 1000 if kv_headroom else 2000
    alloc = types.SimpleNamespace(
        live_size=live, size=2000, max_size=2000, available_size=lambda: 50,
        get_kvcache=lambda: kv_pool,
    )
    scheduler = types.SimpleNamespace(
        token_to_kv_pool_allocator=alloc, tree_cache=None,
        max_running_requests=48,
        # The tick k2m free-slack bound reads one prefill chunk of KV headroom
        # (`server_args.chunked_prefill_size`) so an incoming prefill still fits.
        server_args=types.SimpleNamespace(chunked_prefill_size=2048),
    )
    agent = BudgetAgent(scheduler)
    spy_fp = _SpyFirePlanner()
    tree = _FakeTree(lpb=lpb)
    agent._planner = _SpyPlanner(direction)
    agent._fire_planner = spy_fp
    agent._tree_cache = tree
    agent._owner_provider = types.SimpleNamespace(
        mamba_tokens_per_page=lambda: tokens_per_page,
        kv_tokens_per_page=lambda: tokens_per_page,
        # Single free-supply source the drain pricing nets against (#316).
        # Default 0 → no free pages → fire drains the full magnitude.
        n_free_source_pages=lambda direction: n_free,
    )
    agent._ensure_actuator_chain = lambda *a, **k: True
    return agent, spy_fp, tree


def test_C_m2k_drains_when_lpb_and_curve_ok():
    """The fix: a m2k fire passes allow_drain=True ONLY in the safe regime —
    LPB + a non-degenerate mamba cost curve. Here both hold → drain enabled,
    migrate off. (Pre-#270 it passed allow_drain=False unconditionally →
    free-only → the 3b drain=0 inertia.)"""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()  # builtin default = non-degenerate κ_M
    agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True)
    agent._maybe_fire({"dt": 1.0})
    assert len(spy_fp.calls) == 1, f"expected one build; got {spy_fp.calls}"
    call = spy_fp.calls[0]
    assert call["direction"] == "mamba_to_kv"
    assert call["allow_drain"] is True, (
        "m2k under LPB + non-degenerate c_M must enable drain (harvest cold "
        "mamba cache; design §Budgeter steady-state rebalance)"
    )
    assert call["allow_migrate"] is True, (
        "#271 step 5: the agent now always requests Stage-3 migration; the "
        "OwnerProvider walk self-gates it (SGLANG_XPOOL_KV_MIGRATE + "
        "can_migrate_slot for KV; mamba migration is atomic-inert), so True is "
        "safe and fail-closed-off by default"
    )


def test_D_m2k_refused_when_kv_at_ceiling():
    """#282 H1: a grow fire must be REFUSED when the destination pool is
    already at its page-id ceiling — growing past it makes the arena
    cuMemMap chunks the allocator can't represent (unmark fail-fast /
    orphaned handles). KV at ceiling (live_size == size) → abort before
    build."""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()
    agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True,
                                         kv_headroom=False)
    agent._maybe_fire({"dt": 1.0})
    assert len(spy_fp.calls) == 0, (
        f"m2k must be refused at the KV max_size ceiling (#282 H1); "
        f"got {spy_fp.calls}"
    )


def test_E_drain_priced_for_drained_volume_no_free():
    """#270 H1 + #316: the reuse-aware drain cost is priced for the volume the
    fire ACTUALLY evicts = `max(0, _n_pages_per_fire - n_free) × tpp`, not the
    planner's `dst_chunks_per_action` unit (#270, would under-count) and not
    the full magnitude regardless of free supply (#316, would over-count a
    free-harvest fire). With NO free pages (n_free=0) the fire drains the full
    magnitude → priced at `_n_pages_per_fire × mamba_tpp`."""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()
    agent, _, tree = _make_drain_agent("mamba_to_kv", lpb=True, tokens_per_page=2,
                                       n_free=0)
    agent._maybe_fire({"dt": 1.0})
    mamba_calls = [n for (n, pool) in tree.evict_cost_calls if pool == "mamba"]
    assert mamba_calls, "drain cost must be priced for m2k when free=0"
    expected = agent._n_pages_per_fire * 2  # (magnitude - 0) * mamba_tpp
    assert mamba_calls[0] == expected, (
        f"with no free supply the m2k fire drains the full magnitude → priced "
        f"for {expected} slots; got {mamba_calls[0]}."
    )


def test_E2_drain_priced_zero_when_free_covers():
    """#316 (the cc no-win root cause): when the source pool has ≥
    `_n_pages_per_fire` FREE pages, the fire free-harvests and drains NOTHING,
    so the drain cost must be 0 and predict_evict_cost_us must NOT be invoked
    for the drain. Pre-#316 the agent priced the full magnitude regardless of
    free supply, charging a phantom eviction that drove NB negative and
    suppressed the fire."""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()
    agent, _, tree = _make_drain_agent("mamba_to_kv", lpb=True, tokens_per_page=2,
                                       n_free=999)  # free covers any fire
    snapshot = {"dt": 1.0}
    agent._maybe_fire(snapshot)
    # The drain pricing must short-circuit to 0 without charging an eviction.
    # (The #277 grow term may still price mamba evictions, but the DRAIN
    # snapshot field must be 0.)
    assert snapshot.get("mamba_drain_cost_us") == 0.0, (
        f"free-harvest m2k (n_free≥magnitude) must price drain at 0; got "
        f"{snapshot.get('mamba_drain_cost_us')}"
    )
    assert snapshot.get("kv_drain_cost_us") == 0.0, (
        f"free-harvest k2m must price drain at 0; got "
        f"{snapshot.get('kv_drain_cost_us')}"
    )


def test_F_kappaM_zero_with_healthy_kv_allows_drain():
    """#276 ROOT-CAUSE FIX: κ_M=0 is the EXPECTED post-calibration state —
    `calibrate_kappa.py` cannot split a hybrid prefill into per-stack costs,
    so it folds the TOTAL recompute into the KV curve and sets m_alpha=m_beta=0
    by design. The coupled hybrid eviction is priced `c_kv + c_m = total + 0`,
    so a NON-degenerate κ_KV already prices a hot cache as expensive; κ_M=0
    must NOT fail the drain closed (the old gate did, which disabled drain
    after every real calibration). Drain is allowed iff LPB + κ_KV
    non-degenerate."""
    from sglang.srt.budgeter.cost_model import (
        CostCurves, set_cost_curves, reset_cost_curves,
    )
    try:
        set_cost_curves(CostCurves(
            kv_alpha=1.1e-7, kv_beta=2.0e-3, kv_gamma=16.0,  # real-shaped κ_KV
            m_alpha=0.0, m_beta=0.0, L_star=0.0,             # folded → κ_M=0
            source="test-calibrated-hybrid",
        ))
        agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True)
        agent._maybe_fire({"dt": 1.0})
        assert len(spy_fp.calls) == 1
        assert spy_fp.calls[0]["allow_drain"] is True, (
            "κ_M=0 with a healthy κ_KV is the calibrated hybrid state (total "
            "recompute folded into κ_KV) — drain MUST be allowed; the coupled "
            "c_kv+c_m=total prices hot cache (#276 fix)"
        )
    finally:
        reset_cost_curves()


def test_G_lru_fails_closed():
    """#270 C1 / #280: under LRU (n_b≡1) the drain cost can't distinguish
    hot from cold cache, so the drain must fail closed to free-only — same
    reuse-awareness gate the grow benefit uses."""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()
    agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=False)
    agent._maybe_fire({"dt": 1.0})
    assert len(spy_fp.calls) == 1
    assert spy_fp.calls[0]["allow_drain"] is False, (
        "under LRU the drain must fail closed to free-only (n_b≡1 can't "
        "price hot vs cold; #280)"
    )


def test_H_k2m_drains_when_lpb_and_kv_curve_ok():
    """#271a (symmetric to test_C): now that the reuse-aware KV drain cost is
    wired (snapshot["kv_drain_cost_us"] + nb_k2m subtracts it), k2m drains its
    cold KV cache under LPB + a non-degenerate KV curve — the symmetric image
    of the m2k drain. (Pre-#271a this was gated free-only.)"""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()  # builtin KV curve non-degenerate
    agent, spy_fp, _ = _make_drain_agent("kv_to_mamba", lpb=True)
    agent._maybe_fire({"dt": 1.0})
    assert len(spy_fp.calls) == 1
    assert spy_fp.calls[0]["allow_drain"] is True, (
        "k2m under LPB + non-degenerate c_KV must drain cold KV cache "
        "(#271a, symmetric to m2k)"
    )
    assert spy_fp.calls[0]["allow_migrate"] is True, (
        "#271 step 5: k2m now requests Stage-3 migration too; the OwnerProvider "
        "KV walk self-gates on SGLANG_XPOOL_KV_MIGRATE + can_migrate_slot "
        "(fail-closed off by default), so requesting it here is safe"
    )


def test_I_k2m_degenerate_kv_curve_fails_closed():
    """#271a fail-closed: a degenerate KV recompute curve
    (kv_alpha=kv_beta=kv_gamma=0) collapses the k2m drain cost to ~0 — it
    can't gate hot-KV eviction, so k2m must fall back to free-only (symmetric
    to test_F's κ_M=0 case for m2k)."""
    from sglang.srt.budgeter.cost_model import (
        CostCurves, set_cost_curves, reset_cost_curves,
    )
    try:
        set_cost_curves(CostCurves(
            kv_alpha=0.0, kv_beta=0.0, kv_gamma=0.0,
            m_alpha=2e-3, m_beta=7.0, L_star=0.0, source="test-degenerate-kv",
        ))
        agent, spy_fp, _ = _make_drain_agent("kv_to_mamba", lpb=True)
        agent._maybe_fire({"dt": 1.0})
        assert len(spy_fp.calls) == 1
        assert spy_fp.calls[0]["allow_drain"] is False, (
            "degenerate c_KV (kv_alpha=kv_beta=kv_gamma=0) must fail closed "
            "to free-only for k2m"
        )
    finally:
        reset_cost_curves()


def test_J_k2m_lru_fails_closed():
    """#271a / #280: under LRU the KV drain cost can't tell hot from cold
    (n_b≡1), so k2m must fail closed to free-only — same gate as m2k."""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()
    agent, spy_fp, _ = _make_drain_agent("kv_to_mamba", lpb=False)
    agent._maybe_fire({"dt": 1.0})
    assert len(spy_fp.calls) == 1
    assert spy_fp.calls[0]["allow_drain"] is False, (
        "k2m under LRU must fail closed to free-only (n_b≡1)"
    )


def test_K_k2m_drain_priced_for_drained_volume_no_free():
    """#271a + #316 (symmetric to test_E): the KV drain cost is priced for the
    volume k2m actually evicts = `max(0, _n_pages_per_fire - n_free) × kv_tpp`.
    With no free KV pages (n_free=0) the fire drains the full magnitude →
    `_n_pages_per_fire × kv_tpp` KV tokens. (test_E2 covers the free-harvest
    n_free≥magnitude → 0 case for both directions.)"""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()
    agent, _, tree = _make_drain_agent("kv_to_mamba", lpb=True, tokens_per_page=2,
                                       n_free=0)
    agent._maybe_fire({"dt": 1.0})
    kv_calls = [n for (n, pool) in tree.evict_cost_calls if pool == "kv"]
    assert kv_calls, "k2m drain cost must be priced via predict_evict_cost_us(pool='kv')"
    expected = agent._n_pages_per_fire * 2  # (magnitude - 0) * kv_tpp
    assert kv_calls[0] == expected, (
        f"KV drain cost priced for {kv_calls[0]} tokens but with no free supply "
        f"the fire drains _n_pages_per_fire×kv_tpp = {expected}"
    )


def test_L_k2m_kappaM_zero_allows_drain():
    """#276 fix, k2m direction (symmetric to test_F). A k2m drain evicts
    KV-side leaves via the hybrid cache's full eviction; re-prefilling that
    leaf recomputes the whole prefix = total wall, which the calibration folds
    into κ_KV (κ_M=0 by design). So κ_M=0 with a healthy κ_KV must ALLOW the
    k2m drain — the coupled c_kv+c_m=total prices the hot KV+mamba prefix."""
    from sglang.srt.budgeter.cost_model import (
        CostCurves, set_cost_curves, reset_cost_curves,
    )
    try:
        set_cost_curves(CostCurves(
            kv_alpha=1.1e-7, kv_beta=2.0e-3, kv_gamma=16.0,  # healthy κ_KV
            m_alpha=0.0, m_beta=0.0, L_star=0.0,             # folded → κ_M=0
            source="test-calibrated-hybrid",
        ))
        agent, spy_fp, _ = _make_drain_agent("kv_to_mamba", lpb=True)
        agent._maybe_fire({"dt": 1.0})
        assert len(spy_fp.calls) == 1
        assert spy_fp.calls[0]["allow_drain"] is True, (
            "k2m with κ_M=0 + healthy κ_KV is the calibrated hybrid state — "
            "drain MUST be allowed (#276 fix; coupled c_kv+c_m=total)"
        )
    finally:
        reset_cost_curves()


def test_M_fully_degenerate_curve_fails_closed():
    """The genuine degeneracy the gate must still catch: κ_KV all-zero (no
    calibration at all) means the folded total recompute cost is ~0, so the
    drain cost can't price a hot cache → fail closed (both directions).
    Distinct from #276's κ_M=0 (which is the EXPECTED calibrated state)."""
    from sglang.srt.budgeter.cost_model import (
        CostCurves, set_cost_curves, reset_cost_curves,
    )
    for direction in ("mamba_to_kv", "kv_to_mamba"):
        try:
            set_cost_curves(CostCurves(
                kv_alpha=0.0, kv_beta=0.0, kv_gamma=0.0,
                m_alpha=0.0, m_beta=0.0, L_star=0.0, source="test-all-zero",
            ))
            agent, spy_fp, _ = _make_drain_agent(direction, lpb=True)
            agent._maybe_fire({"dt": 1.0})
            assert len(spy_fp.calls) == 1
            assert spy_fp.calls[0]["allow_drain"] is False, (
                f"{direction}: an all-zero (uncalibrated) κ_KV has no cost "
                f"signal — drain must fail closed"
            )
        finally:
            reset_cost_curves()


def main() -> int:
    tests = [
        test_A_free_only_refuses_when_free_short,
        test_B_allow_drain_harvests_cold_cache,
        test_C_m2k_drains_when_lpb_and_curve_ok,
        test_D_m2k_refused_when_kv_at_ceiling,
        test_E_drain_priced_for_drained_volume_no_free,
        test_E2_drain_priced_zero_when_free_covers,
        test_F_kappaM_zero_with_healthy_kv_allows_drain,
        test_G_lru_fails_closed,
        test_H_k2m_drains_when_lpb_and_kv_curve_ok,
        test_I_k2m_degenerate_kv_curve_fails_closed,
        test_J_k2m_lru_fails_closed,
        test_K_k2m_drain_priced_for_drained_volume_no_free,
        test_L_k2m_kappaM_zero_allows_drain,
        test_M_fully_degenerate_curve_fails_closed,
    ]
    print(f"\n#270 Budgeter-drain fire tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\n#270: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
