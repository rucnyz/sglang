"""Reproducing test for #297 — the m2k mamba floor reserves the NOMINAL
concurrency cap (`max_running_requests`), not the LIVE working set, so in a
KV-bound long-context regime it refuses the cross-pool donate that IS the
inter-layer win.

Ground truth (agentreplay 262k natural run, ar_natural_sys/budgeter.jsonl,
2026-06-15): 59 of 72 m2k fires aborted with
  "mamba working-set floor (#312): live_size=194,
   floor=200 (max_running=147 + protected=53 + headroom=32)".
`max_running=147` is CONSTANT (the configured cap); `live_size` is only ~190.
KV binds concurrency at ~8 long requests, so reserving 147 active mamba slots
reserves for a concurrency the KV pool can never reach. That nominal-cap
reserve IS the "static floor" design.md §"Allocator floor" says to remove.

Working-set decomposition (all in mamba SLOTS):
  m_used     = live_size − available
             = active_running + evictable_cached + protected_locked
  evictable  = mamba_evictable_size()   (unlocked → DONATABLE)
  protected  = mamba_protected_size()   (locked   → irreducible)
  active     = m_used − evictable − protected
Irreducible reservation = active + protected = m_used − evictable. The floor
is that plus a fixed burst headroom; the `_mamba_active_grow_hook` recovers
larger bursts from idle KV (the safety net the on-demand m2k path already
declares — so headroom is thrash-avoidance, not the safety mechanism, and
stays fixed). Donatable = live_size − floor = available + evictable − headroom,
which unblocks m2k exactly where the nominal-cap floor blocked it.

Test-first: each test bakes the RED (nominal-cap floor refuses / over-drains)
next to the GREEN (working-set floor) so it demonstrably catches the bug.
"""
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/3_budgeter/no_spike")

from sglang.srt.budgeter.cost_model import reset_cost_curves  # noqa: E402
from test_budgeter_drain_fire import _make_drain_agent  # noqa: E402


def _kv_bound_regime(agent, *, max_running, live_size, available,
                     evictable, protected):
    """Pose the ar_natural_sys KV-bound regime on a _make_drain_agent: a large
    nominal cap, a small mamba pool mostly holding EVICTABLE cached snapshots
    (tiny active set), arena tps=1 so slots==pages. Returns the irreducible
    working set (active+protected) for invariant checks."""
    mamba_pool = agent.scheduler.token_to_kv_pool_allocator.get_kvcache().mamba_pool
    mamba_pool.live_size = live_size
    mamba_pool.size = live_size
    mamba_pool.available_size = lambda: available
    mamba_pool._mamba_temporal_arena = types.SimpleNamespace(tokens_per_chunk=1)
    agent.scheduler.max_running_requests = max_running
    agent._tree_cache.mamba_evictable_size = lambda: evictable
    agent._tree_cache.mamba_protected_size = lambda: protected
    agent._tree_cache._mamba_grow_hook = lambda n: False  # wired (self-heals)
    m_used = live_size - available
    return m_used - evictable  # active + protected (irreducible)


def test_tick_m2k_fires_when_pool_holds_idle_evictable_cache():
    """The headline #297 case. live_size=194 < max_running(147)+protected(53)
    =200, but active is only 8 (KV-bound) and 100 slots are EVICTABLE cached
    snapshots. The nominal-cap floor refuses (self-defeat); the working-set
    floor (irreducible 61) leaves donatable headroom → the fire is NOT
    refused."""
    reset_cost_curves()
    agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True)
    _kv_bound_regime(
        agent, max_running=147, live_size=194, available=33,
        evictable=100, protected=53,  # active = 161 − 100 − 53 = 8
    )
    assert 194 < 147 + 53, "setup: nominal-cap floor must exceed live_size (RED witness)"

    agent._maybe_fire({"dt": 1.0})

    assert len(spy_fp.calls) == 1, (
        "working-set floor must let m2k fire: 8 active + 53 protected are "
        "irreducible (61 slots); 100 evictable cached snapshots are donatable. "
        f"The nominal-cap floor (#297) refused instead. calls={spy_fp.calls}"
    )
    assert spy_fp.calls[0]["n_pages_target"] > 0


def test_drain_never_breaches_active_plus_protected():
    """#312 safety, exact: across a grid, the fired drain never shrinks
    live_size below the irreducible working set (active+protected), so a future
    active-slot alloc / fork can never be stranded — regardless of the nominal
    cap. (Draining genuinely-free + evictable slots is always safe; draining
    into active/protected is the crash.)"""
    reset_cost_curves()
    live, max_running = 194, 147
    for available in (0, 4, 40):
        for evictable in (0, 20, 120):
            for protected in (0, 53):
                active = live - available - evictable - protected
                if active < 0 or active > max_running:
                    continue  # physically impossible (can't exceed the cap)
                agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True)
                irreducible = _kv_bound_regime(
                    agent, max_running=max_running, live_size=live,
                    available=available, evictable=evictable, protected=protected,
                )
                agent._maybe_fire({"dt": 1.0})
                drained = spy_fp.calls[0]["n_pages_target"] if spy_fp.calls else 0
                assert live - drained >= irreducible, (
                    f"#312 BREACH: avail={available} evict={evictable} "
                    f"prot={protected} drained={drained} → live {live - drained} "
                    f"< working set {irreducible}"
                )


def test_genuinely_full_pool_refuses():
    """No free and no evictable slack (every used slot is active/protected) →
    nothing donatable → refuse (the fire can't manufacture capacity)."""
    reset_cost_curves()
    agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True)
    # live=194, active=147 (= cap), protected=47, no free, no evictable.
    _kv_bound_regime(
        agent, max_running=147, live_size=194, available=0,
        evictable=0, protected=47,
    )
    agent._maybe_fire({"dt": 1.0})
    assert len(spy_fp.calls) == 0, (
        f"genuinely-full mamba (no free, no evictable) must refuse; got {spy_fp.calls}"
    )


def test_working_set_floor_independent_of_nominal_cap():
    """The floor must NOT scale with max_running_requests (the nominal cap):
    holding the live working set fixed, varying the cap leaves the donatable
    volume unchanged. Pins design §501 'working-set only, no static floor'."""
    reset_cost_curves()
    grants = []
    for cap in (64, 147, 512):
        agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True)
        _kv_bound_regime(
            agent, max_running=cap, live_size=194, available=33,
            evictable=100, protected=53,
        )
        agent._maybe_fire({"dt": 1.0})
        grants.append(spy_fp.calls[0]["n_pages_target"] if spy_fp.calls else 0)
    assert len(set(grants)) == 1 and grants[0] > 0, (
        f"donatable volume must be invariant to the nominal cap; got {grants} "
        f"for caps (64,147,512)"
    )


def main() -> int:
    tests = [
        test_tick_m2k_fires_when_pool_holds_idle_evictable_cache,
        test_drain_never_breaches_active_plus_protected,
        test_genuinely_full_pool_refuses,
        test_working_set_floor_independent_of_nominal_cap,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
    print(f"\n#297 working-set floor: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
