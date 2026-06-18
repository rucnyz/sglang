"""#183 Step 3 — XPoolFirePlanner three-stage knapsack page selection.

design.md §"Page selection: anywhere-free, Drain-expansion,
Migration-expansion": the planner picks `n` pages from src, expanding
the candidate set in increasing cost order until `n` is met:

  Stage 1 (anywhere-free):   FREE pages, descending page-id.
  Stage 2 (Drain-expansion): CACHED pages in active-eviction order.
  Stage 3 (Migration-exp.):  LIVE pages by ascending per-page c_m.

When all three exhaust, the planner increments `refuse_count` and
returns None.

Pins (CPU-only — fake OwnerProvider with canned free/cached/live sets):
  1. Stage-1 still wins when free >= n (drains / migrations empty; plan
     byte-identical to today's free-only plan).
  2. Stage-2 fills the remainder from cached (populates `drains`,
     cheapest-first); free pages still anywhere-free.
  3. Stage-3 fills the remainder from live (populates `migrations`, by
     ascending c_m order).
  4. refuse_count increments when free + cached + live all exhausted
     and the planner returns None.
  5. An empty-expansion plan (allow_drain/allow_migrate default False)
     is byte-identical to today's free-only plan.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.arena.fire_plan import FirePlan
from sglang.srt.arena.owner_provider import OwnerMap
from sglang.srt.budgeter.fire_planner import XPoolFirePlanner


# ---------------------------------------------------------------- Fakes

class _FakeOwnerProvider:
    """Canned per-pool OwnerMaps. `cached_in_cost_order` /
    `live_in_cost_order` are returned only when the planner requests
    expansion (the planner passes allow_drain / allow_migrate)."""

    def __init__(self, *, n_pages, free, cached_in_cost_order=None,
                 live_in_cost_order=None):
        self.n_pages = n_pages
        self.free = set(free)
        self.cached = list(cached_in_cost_order or [])
        self.live = list(live_in_cost_order or [])
        self.last_allow_drain = None
        self.last_allow_migrate = None

    def _map(self, name, *, allow_drain, allow_migrate, max_drain_pages=None):
        self.last_allow_drain = allow_drain
        self.last_allow_migrate = allow_migrate
        self.last_max_drain_pages = max_drain_pages
        # Mirror the real provider's #284 bound: cap the cost-order cached
        # list to max_drain_pages when set.
        cached = None
        if allow_drain:
            cached = list(self.cached)
            if max_drain_pages is not None:
                cached = cached[:max_drain_pages]
        return OwnerMap(
            pool_name=name,
            n_pages=self.n_pages,
            free_pages=set(self.free),
            cached_pages_in_cost_order=cached,
            live_pages_in_cost_order=(list(self.live) if allow_migrate
                                      else None),
        )

    # The planner calls build_*_owner_map with the expansion flags.
    def build_kv_owner_map(self, *, allow_drain=False, allow_migrate=False,
                           max_drain_pages=None):
        return self._map("kv", allow_drain=allow_drain,
                         allow_migrate=allow_migrate,
                         max_drain_pages=max_drain_pages)

    def build_mamba_owner_map(self, *, allow_drain=False, allow_migrate=False,
                              max_drain_pages=None):
        return self._map("mamba", allow_drain=allow_drain,
                         allow_migrate=allow_migrate,
                         max_drain_pages=max_drain_pages)


def _planner(provider):
    # kv/mamba actuators unused by build(); pass sentinels.
    return XPoolFirePlanner(kv_actuator=object(), mamba_actuator=object(),
                            owner_provider=provider)


# ---------------------------------------------------------------- Tests

def test_1_stage1_wins_when_free_enough():
    """free >= n → Stage-1 plan; drains/migrations empty; refuse_count 0."""
    prov = _FakeOwnerProvider(n_pages=100, free=[90, 91, 92, 93, 94])
    pl = _planner(prov)
    plan = pl.build("kv_to_mamba", 3, allow_drain=True, allow_migrate=True)
    assert plan is not None
    assert plan.drains == () and plan.migrations == (), (
        f"Stage-1 plan must leave drains/migrations empty: "
        f"drains={plan.drains} migrations={plan.migrations}"
    )
    assert len(plan.pages_to_unmap) == 3
    assert plan.pages_to_map_dst == 3
    # Highest-id free pages first (descending), then sorted ascending.
    assert plan.pages_to_unmap == [92, 93, 94], plan.pages_to_unmap
    assert pl.refuse_count == 0
    print("  PASS  1  Stage-1 anywhere-free wins when free>=n (drains/migrations empty)")


def test_2_stage2_fills_from_cached_cheapest_first():
    """free < n, free + cached >= n → Stage-2 populates drains,
    cheapest-first (the provider's cost order)."""
    # 2 free, need 5 → 3 from cached. Cached cost order = [10, 11, 12, 13].
    prov = _FakeOwnerProvider(
        n_pages=100, free=[98, 99],
        cached_in_cost_order=[10, 11, 12, 13],
    )
    pl = _planner(prov)
    plan = pl.build("kv_to_mamba", 5, allow_drain=True, allow_migrate=True)
    assert plan is not None
    # Drains: the 3 cheapest cached pages, in cost order.
    assert plan.drains == (10, 11, 12), plan.drains
    assert plan.migrations == (), plan.migrations
    # All 5 selected pages end up unmapped (free 2 + drained 3).
    assert set(plan.pages_to_unmap) == {98, 99, 10, 11, 12}, plan.pages_to_unmap
    assert plan.pages_to_map_dst == 5
    assert pl.refuse_count == 0
    print("  PASS  2  Stage-2 fills remainder from cached (drains cheapest-first)")


def test_3_stage3_fills_from_live_by_cm():
    """free + cached < n, + live >= n → Stage-3 populates migrations,
    ascending-c_m order (the provider's live cost order)."""
    # 1 free + 1 cached = 2, need 4 → 2 from live. live_pages_in_cost_order
    # is the new (freed_page_id, ((src,dst),...)) shape: each page carries
    # its concrete relocation moves; the planner flattens the taken pages'
    # moves into plan.migrations.
    prov = _FakeOwnerProvider(
        n_pages=100, free=[99],
        cached_in_cost_order=[10],
        live_in_cost_order=[
            (20, ((200, 900),)),
            (21, ((201, 901),)),
            (22, ((202, 902),)),
        ],
    )
    pl = _planner(prov)
    plan = pl.build("kv_to_mamba", 4, allow_drain=True, allow_migrate=True)
    assert plan is not None
    assert plan.drains == (10,), plan.drains
    # Two migration PAGES taken (20, 21); their moves flattened.
    assert plan.migrations == ((200, 900), (201, 901)), plan.migrations
    assert set(plan.pages_to_unmap) == {99, 10, 20, 21}, plan.pages_to_unmap
    assert plan.pages_to_map_dst == 4
    assert pl.refuse_count == 0
    print("  PASS  3  Stage-3 fills remainder from live (migration moves, "
          "ascending c_m page order)")


def test_4_refuse_increments_when_all_exhausted():
    """free + cached + live < n → return None + refuse_count++."""
    prov = _FakeOwnerProvider(
        n_pages=100, free=[99],
        cached_in_cost_order=[10],
        live_in_cost_order=[(20, ((200, 900),))],
    )
    pl = _planner(prov)
    assert pl.refuse_count == 0
    plan = pl.build("kv_to_mamba", 10, allow_drain=True, allow_migrate=True)
    assert plan is None, "must refuse when all three stages exhaust"
    assert pl.refuse_count == 1, f"refuse_count must increment: {pl.refuse_count}"
    # Monotonic on a second refuse.
    pl.build("kv_to_mamba", 10, allow_drain=True, allow_migrate=True)
    assert pl.refuse_count == 2, pl.refuse_count
    print("  PASS  4  refuse_count increments (monotonic) when all stages exhausted")


def test_5_empty_expansion_byte_identical_to_today():
    """allow_drain/allow_migrate default False → free-only plan, drains
    and migrations empty, byte-identical to the pre-#183 Stage-1 plan."""
    prov = _FakeOwnerProvider(n_pages=100, free=[90, 91, 92, 93, 94])
    pl = _planner(prov)
    # No expansion flags → defaults False.
    plan = pl.build("kv_to_mamba", 3)
    assert plan is not None
    assert plan.drains == () and plan.migrations == ()
    assert plan.pages_to_unmap == [92, 93, 94]
    assert plan.pages_to_map_dst == 3
    # The provider must NOT have been asked for cost-order lists.
    assert prov.last_allow_drain is False and prov.last_allow_migrate is False, (
        "Stage-1-only callers must not request expansion (zero-cost path)"
    )
    # Structural: a hand-built free-only FirePlan with the same fields
    # compares equal on the load-bearing attributes.
    ref = FirePlan(direction="kv_to_mamba", pages_to_unmap=[92, 93, 94],
                   pages_to_map_dst=3, plan_seq=plan.plan_seq)
    assert (plan.direction == ref.direction
            and plan.pages_to_unmap == ref.pages_to_unmap
            and plan.pages_to_map_dst == ref.pages_to_map_dst
            and plan.drains == ref.drains
            and plan.migrations == ref.migrations), (
        "empty-expansion plan must be byte-identical to a free-only plan"
    )
    print("  PASS  5  empty-expansion plan byte-identical to today's free-only plan")


def test_6_agent_snapshot_surfaces_refuse_count():
    """#183 Step 5: BudgetAgent._snapshot reads
    `self._fire_planner.refuse_count` into the per-tick JSONL dict
    (`fire_refuse_count`). None before the chain builds; tracks the
    planner's counter once wired."""
    from types import SimpleNamespace
    from sglang.srt.budgeter.agent import BudgetAgent

    # Minimal stats + scheduler surface _snapshot reads.
    stats = SimpleNamespace(
        max_total_num_tokens=1000, kv_used_tokens=0, kv_evictable_tokens=0,
        kv_available_tokens=1000, token_usage=0.0, full_token_usage=0.0,
        swa_token_usage=0.0, mamba_usage=0.0, cache_hit_rate=0.0,
        num_paused_reqs=0, num_retracted_reqs=0, gen_throughput=0.0,
    )

    class _Alloc:
        size = 1000
        def available_size(self):
            return 1000
        def get_kvcache(self):
            return SimpleNamespace(mamba_pool=None)

    sched = SimpleNamespace(
        stats=stats, running_batch=None, waiting_queue=[],
        token_to_kv_pool_allocator=_Alloc(),
    )

    ba = BudgetAgent.__new__(BudgetAgent)
    ba.scheduler = sched
    ba._tick_count = 0
    # _tree_cache is always a wired cache post-_do_health_check (which
    # pre-inits the eviction counters / recovery EWMAs the snapshot reads);
    # mirror that minimal surface here instead of None.
    ba._tree_cache = SimpleNamespace(
        _admission_cumulative_evicted_tokens=0,
        _slow_recovery_len_kv_ewma=0.0,
        _slow_recovery_len_rec_ewma=0.0,
        _cumulative_evicted_mamba_slots=0,
        _cumulative_evicted_kv_tokens=0,
    )
    ba._last_evicted_cumulative = 0
    ba._last_evicted_mamba_slots = 0
    ba._last_evicted_kv_tokens = 0
    ba._fire_planner = None

    # Pre-chain: planner None → key absent (don't fabricate a value).
    snap = ba._snapshot(now=1.0)
    assert "fire_refuse_count" not in snap, (
        "fire_refuse_count must be absent before the planner is built"
    )

    # Wire a planner with a non-zero refuse_count → surfaced.
    prov = _FakeOwnerProvider(n_pages=100, free=[99])
    pl = _planner(prov)
    pl.build("kv_to_mamba", 10, allow_drain=True, allow_migrate=True)  # refuse → 1
    ba._fire_planner = pl
    snap2 = ba._snapshot(now=2.0)
    assert snap2.get("fire_refuse_count") == 1, (
        f"snapshot must surface planner.refuse_count: {snap2.get('fire_refuse_count')}"
    )
    print("  PASS  6  BudgetAgent snapshot surfaces fire_refuse_count "
          "(absent pre-chain, tracks planner once wired)")


def main():
    tests = [
        test_1_stage1_wins_when_free_enough,
        test_2_stage2_fills_from_cached_cheapest_first,
        test_3_stage3_fills_from_live_by_cm,
        test_4_refuse_increments_when_all_exhausted,
        test_5_empty_expansion_byte_identical_to_today,
        test_6_agent_snapshot_surfaces_refuse_count,
    ]
    print(f"\n#183 Step 3 fire-planner three-stage tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}"); traceback.print_exc()
    print(f"#183 Step 3: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
