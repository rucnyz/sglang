"""Budgeter snapshot must plumb the slow-recovery-length EWMAs
(reproducing test for the #156/#159 root-cause finding, 2026-06-03).

ROOT CAUSE found via the cc_traces_headline mamba-starve re-run: the
planner's eviction-cost term `c_σ(L) × P_save_σ` (design.md §"Budgeter —
steady-state pressure rebalance", a PRIMARY signal) was DEAD in
production. `record_recovery_len_kv` / `_rec` (mem_cache/common.py) write
`_slow_recovery_len_kv_ewma` / `_rec_ewma` onto the tree_cache on every
KV / mamba eviction, and `XPoolPlanner._pick_direction_by_nb` reads
`snapshot["slow_recovery_len_kv" / "_rec"]` — but `BudgetAgent._snapshot`
NEVER read those EWMAs into the snapshot. So `snap.get(...)` always
defaulted to 0 → `L=0` → `c_kv=c_m=0` every tick (confirmed: 0 across all
297 ticks of the starve run despite heavy cache thrashing). With the
eviction-cost term dead, the planner could only ever fire on
queue/persist — and in particular could never grow the mamba pool in
response to mamba evicting hot snapshots.

This pins the fix: `_snapshot` reads `_slow_recovery_len_kv_ewma` /
`_rec_ewma` (pre-init'd in `_do_health_check`) into the snapshot.

Pre-fix: the keys are ABSENT → assertion FAILS (KeyError-equivalent).
Post-fix: present and equal to the tree_cache EWMAs → PASS.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.agent import BudgetAgent


def _stub_stats():
    return types.SimpleNamespace(
        max_total_num_tokens=10000,
        kv_used_tokens=0,
        kv_evictable_tokens=0,
        kv_available_tokens=10000,
        token_usage=0.0,
        full_token_usage=0.0,
        swa_token_usage=0.0,
        mamba_usage=0.0,
        cache_hit_rate=0.0,
        num_paused_reqs=0,
        num_retracted_reqs=0,
        gen_throughput=0.0,
    )


def _stub_allocator():
    kvcache = types.SimpleNamespace(mamba_pool=None)
    return types.SimpleNamespace(
        size=10000,
        available_size=lambda: 10000,
        get_kvcache=lambda: kvcache,
    )


def _make_agent(rec_ewma: float, kv_ewma: float) -> BudgetAgent:
    """A BudgetAgent with __init__ bypassed — wire only the attributes
    `_snapshot` reads, plus a tree_cache carrying the recovery EWMAs as
    the eviction sites would have left them."""
    agent = BudgetAgent.__new__(BudgetAgent)
    tree_cache = types.SimpleNamespace(
        _slow_recovery_len_kv_ewma=kv_ewma,
        _slow_recovery_len_rec_ewma=rec_ewma,
        _slow_recovery_len_retract_ewma=0.0,
        _admission_cumulative_evicted_tokens=0,
        _cumulative_evicted_mamba_slots=0,
        _cumulative_evicted_kv_tokens=0,
    )
    agent.scheduler = types.SimpleNamespace(
        stats=_stub_stats(),
        running_batch=None,
        waiting_queue=[],
        token_to_kv_pool_allocator=_stub_allocator(),
    )
    agent._tree_cache = tree_cache
    agent._fire_planner = None
    agent._last_evicted_cumulative = 0
    agent._last_evicted_mamba_slots = 0
    agent._last_evicted_kv_tokens = 0
    agent._tick_count = 0
    # `_snapshot` also computes the per-fire admission yield (fire_admit_{kv,
    # mamba}) for the marginal-fire cap; wire the page/chunk sizing it reads.
    agent._n_pages_per_fire = 4
    agent._kv_tokens_per_chunk = 1024
    agent._mamba_tokens_per_chunk = 1
    return agent


def test_snapshot_plumbs_recovery_lengths():
    agent = _make_agent(rec_ewma=1234.0, kv_ewma=567.0)
    snap = agent._snapshot(now=0.0)

    assert "slow_recovery_len_rec" in snap, (
        "BUG: _snapshot does not plumb the mamba recovery-length EWMA — "
        "the planner's eviction-cost term c_m(L) is dead (L=0 always), so "
        "it can never grow mamba in response to mamba eviction pressure."
    )
    assert "slow_recovery_len_kv" in snap, (
        "BUG: _snapshot does not plumb the KV recovery-length EWMA — "
        "the planner's eviction-cost term c_kv(L) is dead."
    )
    assert snap["slow_recovery_len_rec"] == 1234.0, (
        f"mamba recovery length must equal the tree_cache EWMA, got "
        f"{snap['slow_recovery_len_rec']}"
    )
    assert snap["slow_recovery_len_kv"] == 567.0, (
        f"KV recovery length must equal the tree_cache EWMA, got "
        f"{snap['slow_recovery_len_kv']}"
    )
    print(
        f"  PASS  recovery lengths plumbed: kv={snap['slow_recovery_len_kv']} "
        f"rec={snap['slow_recovery_len_rec']}"
    )


def test_snapshot_recovery_lengths_zero_when_no_eviction():
    """Negative control: a fresh cache (no evictions) → EWMAs 0 → the
    snapshot carries 0 (not absent), so the planner reads a real L=0."""
    agent = _make_agent(rec_ewma=0.0, kv_ewma=0.0)
    snap = agent._snapshot(now=0.0)
    assert snap.get("slow_recovery_len_rec") == 0.0
    assert snap.get("slow_recovery_len_kv") == 0.0
    print("  PASS  no-eviction → recovery lengths present and 0")


if __name__ == "__main__":
    fails = 0
    for t in (test_snapshot_plumbs_recovery_lengths,
              test_snapshot_recovery_lengths_zero_when_no_eviction):
        try:
            t()
        except Exception as e:
            fails += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{'ALL GREEN' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)
