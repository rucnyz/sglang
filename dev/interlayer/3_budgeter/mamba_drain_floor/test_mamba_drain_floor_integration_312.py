"""Integration coverage for the #312 m2k working-set floor (audit F1 + F2).

The unit tests in `test_mamba_drain_floor_312.py` pin the pure
`_mamba_drain_floor` helper. These pin the two pieces the helper CANNOT see —
the production wiring in `BudgetAgent`:

  F1 (fail-loud, not fail-soft): the m2k floor reads
      `self._tree_cache.mamba_evictable_size()` and the pool's `available_size`
      DIRECTLY (no getattr default). `tick()` wraps `_maybe_fire` in
      `try/except → logger.warning`, so a tree cache lacking that method would
      make the m2k fire silently skip EVERY tick. That contradicts
      `_do_health_check`'s own contract ("hard-disable on schema drift so the
      error surfaces as one log line instead of per-tick AttributeError"). The
      fix extends the health check to require the m2k floor's API whenever a
      mamba pool is present.

  F2 (the load-bearing slot↔page conversion, end-to-end): the only other
      _maybe_fire test pins `tokens_per_chunk=1`, so the production wiring
      `slots_per_page = arena.tokens_per_chunk` and the working-set
      `floor_slots = (m_used − evictable) + headroom` are never exercised at
      tps>1. A regression that hardcoded slots_per_page=1 would pass the whole
      suite. This drives the real `_maybe_fire` m2k path at tps=12 and asserts
      the FirePlan it builds keeps the post-drain live_size above the slot
      floor.
"""
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.agent import BudgetAgent

# Reuse the drain-fire harness (spies + arena-backed mamba pool stub).
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/3_budgeter/no_spike")
from test_budgeter_drain_fire import _make_drain_agent  # noqa: E402


# ---------------------------------------------------------------------------
# F1 — health check must require the m2k floor's API (fail loud at startup,
# not fail soft per-tick).
# ---------------------------------------------------------------------------
_ALL_STATS_FIELDS = (
    "max_total_num_tokens", "kv_used_tokens", "kv_evictable_tokens",
    "kv_available_tokens", "token_usage", "full_token_usage",
    "swa_token_usage", "mamba_usage", "cache_hit_rate", "num_running_reqs",
    "num_queue_reqs", "num_paused_reqs", "num_retracted_reqs", "gen_throughput",
)


def _full_stats():
    return types.SimpleNamespace(**{f: 0 for f in _ALL_STATS_FIELDS})


def _healthcheck_agent(*, tree_cache, mamba_pool):
    """A BudgetAgent (init bypassed) wired only for `_do_health_check`: a
    complete stats stub (so the stats schema check passes and we isolate the
    mamba-API check) and an allocator whose kvcache carries `mamba_pool`."""
    agent = BudgetAgent.__new__(BudgetAgent)
    kvcache = types.SimpleNamespace(mamba_pool=mamba_pool)
    agent.scheduler = types.SimpleNamespace(
        stats=_full_stats(),
        token_to_kv_pool_allocator=types.SimpleNamespace(
            get_kvcache=lambda: kvcache,
        ),
        max_running_requests=48,
    )
    return agent


def _arena_pool():
    return types.SimpleNamespace(
        live_size=256, size=256, max_size=512, available_size=lambda: 200,
        _mamba_temporal_arena=types.SimpleNamespace(tokens_per_chunk=1),
    )


def test_F1_healthcheck_requires_mamba_evictable_size_when_mamba_present():
    """RED (pre-fix): a tree cache without `mamba_evictable_size` + a mamba
    pool present passes _do_health_check (it only validates stats fields), so
    the per-tick AttributeError is later swallowed by tick()'s try/except —
    the m2k fire silently never fires. GREEN: the health check hard-disables
    (returns False) so the failure is one log line, matching the schema-drift
    contract. (#297 working-set floor reads mamba_evictable_size + the pool's
    available_size directly.)"""
    tc_missing = types.SimpleNamespace()  # NO mamba_evictable_size
    agent = _healthcheck_agent(tree_cache=tc_missing, mamba_pool=_arena_pool())
    agent.scheduler.tree_cache = tc_missing
    ok = agent._do_health_check()
    assert ok is False, (
        "health check must FAIL LOUD when a mamba pool is present but the "
        "tree cache lacks mamba_evictable_size (the m2k working-set floor "
        "reads it directly); otherwise tick() swallows the per-tick "
        "AttributeError and m2k silently never fires (#297)."
    )


def test_F1_healthcheck_passes_with_full_mamba_api():
    """Positive: a tree cache exposing mamba_evictable_size + an arena-backed
    mamba pool (with available_size) passes the health check (no false
    hard-disable)."""
    tc = types.SimpleNamespace(mamba_evictable_size=lambda: 0)
    agent = _healthcheck_agent(tree_cache=tc, mamba_pool=_arena_pool())
    agent.scheduler.tree_cache = tc
    assert agent._do_health_check() is True


def test_F1_healthcheck_passes_when_no_mamba_pool():
    """A pure-transformer config (mamba_pool is None) never takes the m2k
    floor path, so the mamba API is NOT required — health check still passes."""
    tc = types.SimpleNamespace()  # no mamba API, but no mamba pool either
    agent = _healthcheck_agent(tree_cache=tc, mamba_pool=None)
    agent.scheduler.tree_cache = tc
    assert agent._do_health_check() is True


# ---------------------------------------------------------------------------
# F2 — the m2k floor wiring must convert pages→slots at tps>1, end-to-end.
# ---------------------------------------------------------------------------
def _drive_m2k_floor(*, live_size, available, tokens_per_chunk, n_pages_per_fire):
    """Drive the real _maybe_fire m2k branch and return the n_pages_target it
    passed to the FirePlanner (the spy records it). The working-set floor reads
    m_used = live − available and evictable (0 from the _FakeTree stub), so the
    floor here is `(live − available) + headroom`."""
    from sglang.srt.budgeter.cost_model import reset_cost_curves
    reset_cost_curves()
    agent, spy_fp, _ = _make_drain_agent("mamba_to_kv", lpb=True)
    # Override the stub pool to the F2 regime: arena tps>1, live near floor.
    mamba_pool = agent.scheduler.token_to_kv_pool_allocator.get_kvcache().mamba_pool
    mamba_pool.live_size = live_size
    mamba_pool.available_size = lambda: available
    mamba_pool._mamba_temporal_arena = types.SimpleNamespace(
        tokens_per_chunk=tokens_per_chunk
    )
    agent._n_pages_per_fire = n_pages_per_fire
    agent._maybe_fire({"dt": 1.0})
    return agent, spy_fp


def test_F2_m2k_floor_converts_pages_to_slots_at_tps_gt1():
    """At tps=12 the production caller must cap the drain in PAGES so that
    pages×tps slots keep live_size above the working-set slot floor.

    Config: live=256, available=100 → m_used=156, evictable=0, headroom=32 →
    floor = 156 + 32 = 188.
      Page-unit (buggy) cap would drain min(8, 256-188)=8 pages × 12 = 96
        slots → live 256-96=160 < floor 188 (BREACH — re-opens #312).
      tps-aware cap drains (256-188)//12 = 5 pages × 12 = 60 → live 196 >= 188.
    """
    HEADROOM = 32  # SGLANG_XPOOL_MAMBA_FLOOR_SLOTS default
    live, available, tps, ppf = 256, 100, 12, 8
    floor = (live - available) + HEADROOM
    agent, spy_fp = _drive_m2k_floor(
        live_size=live, available=available, tokens_per_chunk=tps,
        n_pages_per_fire=ppf,
    )
    assert agent._mamba_fork_headroom_slots == HEADROOM, (
        f"setup assumes default headroom {HEADROOM}, got "
        f"{agent._mamba_fork_headroom_slots}")
    # The page-unit cap would breach — proves the regime actually tests F2.
    buggy = min(ppf, max(0, live - floor))
    assert live - buggy * tps < floor, (
        f"setup: page-unit cap should breach (left {live - buggy*tps} "
        f">= {floor} proves nothing)")

    assert len(spy_fp.calls) == 1, f"expected one fire; got {spy_fp.calls}"
    n_pages = spy_fp.calls[0]["n_pages_target"]
    assert live - n_pages * tps >= floor, (
        f"F2 BREACH: _maybe_fire drained {n_pages}p×{tps}={n_pages*tps} slots, "
        f"live {live} -> {live - n_pages*tps} < floor {floor}. The caller did "
        f"not convert pages→slots via tokens_per_chunk.")
    assert n_pages == (live - floor) // tps, (
        f"expected {(live-floor)//tps} pages, got {n_pages}")


def test_F2_m2k_floor_refuses_when_below_floor_at_tps_gt1():
    """When live_size <= floor (here live=170, available=0 → m_used=170,
    floor=202) the caller must refuse the m2k fire (no FirePlanner.build call),
    regardless of tps."""
    agent, spy_fp = _drive_m2k_floor(
        live_size=170, available=0, tokens_per_chunk=12, n_pages_per_fire=8,
    )
    assert len(spy_fp.calls) == 0, (
        f"m2k must be refused when live_size <= floor; got {spy_fp.calls}")


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError:
                failures += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failures else 0)
