"""The paper-calibrated L-aware retract cost must be LIVE end-to-end.

ROOT CAUSE (companion to test_snapshot_recovery_len.py's #156/#159 finding):
`pressure_adapter.py` reads `snapshot.get("slow_recovery_len_retract", 0.0)`
to price the retract-pressure term at the paper-calibrated k_retract (75 ms)
cost. But `BudgetAgent._snapshot` emitted only `slow_recovery_len_kv` /
`_rec` — never `_retract`. So the adapter ALWAYS defaulted to L=0 and fell
back to the un-calibrated full-prefill cost; the calibrated retract cost was
silently dead.

The writer `record_recovery_len_retract` (mem_cache/common.py) sets
`tree_cache._slow_recovery_len_retract_ewma`. That counter (plus its `_kv` /
`_rec` siblings) used to be lazy-created via getattr-default in the writers
plus a hasattr pre-init in `_do_health_check` — a no-getattr-none-state
violation. They are now init'd UNCONDITIONALLY at every concrete cache's
`__init__`.

This pins the whole chain:
  (a) a freshly-constructed REAL cache has all three EWMAs == 0.0 right
      after __init__ (unconditional init, not lazy);
  (b) record_recovery_len_retract(cache, 100) sets the EWMA to 100.0
      (first observation = the value, no blending against a 0 prior);
  (c) _snapshot emits 'slow_recovery_len_retract' equal to the cache EWMA
      (behavioral pin), backed by a source-level pin so the key can never
      silently disappear from _snapshot again.

Standalone: `CUDA_VISIBLE_DEVICES=7 python test_retract_cost_live.py`.
pytest is optional (guarded import); the __main__ runner reports PASS/FAIL.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

try:
    import pytest  # noqa: F401

    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

from sglang.srt.budgeter.agent import BudgetAgent
from sglang.srt.mem_cache.common import record_recovery_len_retract
from sglang.srt.mem_cache.radix_cache import RadixCache

_AGENT_SRC = Path(BudgetAgent.__init__.__globals__["__file__"]).read_text()


def test_counters_unconditionally_initialized():
    """(a) A freshly-constructed REAL cache carries all three recovery-length
    EWMAs at 0.0 right after __init__ — unconditionally init'd, NOT lazy."""
    cache = RadixCache.create_simulated()
    assert cache._slow_recovery_len_kv_ewma == 0.0, (
        "BUG: _slow_recovery_len_kv_ewma not init'd at __init__ "
        "(lazy getattr-default state is forbidden)."
    )
    assert cache._slow_recovery_len_rec_ewma == 0.0, (
        "BUG: _slow_recovery_len_rec_ewma not init'd at __init__."
    )
    assert cache._slow_recovery_len_retract_ewma == 0.0, (
        "BUG: _slow_recovery_len_retract_ewma not init'd at __init__ — "
        "the retract cost writer would have to lazy-create it (forbidden)."
    )
    print("  PASS  all three recovery EWMAs init'd to 0.0 at cache __init__")


def test_record_recovery_len_retract_first_observation():
    """(b) The first retract observation is taken verbatim (no EWMA blend
    against the 0 prior), so the calibrated retract cost engages immediately."""
    cache = RadixCache.create_simulated()
    record_recovery_len_retract(cache, 100)
    assert cache._slow_recovery_len_retract_ewma == 100.0, (
        f"first retract observation must set the EWMA to the value (100.0), "
        f"got {cache._slow_recovery_len_retract_ewma}"
    )
    # Guard unchanged: a non-positive L is a no-op.
    record_recovery_len_retract(cache, 0)
    assert cache._slow_recovery_len_retract_ewma == 100.0, (
        "L<=0 must be a no-op (guard unchanged)."
    )
    print("  PASS  record_recovery_len_retract(cache, 100) -> EWMA == 100.0")


def _make_agent(retract_ewma: float) -> BudgetAgent:
    """A BudgetAgent with __init__ bypassed — wire only the attributes
    `_snapshot` reads, plus a tree_cache carrying the three recovery EWMAs
    as the eviction / retraction sites would have left them."""
    agent = BudgetAgent.__new__(BudgetAgent)
    tree_cache = types.SimpleNamespace(
        _slow_recovery_len_kv_ewma=0.0,
        _slow_recovery_len_rec_ewma=0.0,
        _slow_recovery_len_retract_ewma=retract_ewma,
        _admission_cumulative_evicted_tokens=0,
        _cumulative_evicted_mamba_slots=0,
        _cumulative_evicted_kv_tokens=0,
    )
    stats = types.SimpleNamespace(
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
    kvcache = types.SimpleNamespace(mamba_pool=None)
    allocator = types.SimpleNamespace(
        size=10000,
        available_size=lambda: 10000,
        get_kvcache=lambda: kvcache,
    )
    agent.scheduler = types.SimpleNamespace(
        stats=stats,
        running_batch=None,
        waiting_queue=[],
        token_to_kv_pool_allocator=allocator,
    )
    agent._tree_cache = tree_cache
    agent._fire_planner = None
    agent._last_evicted_cumulative = 0
    agent._last_evicted_mamba_slots = 0
    agent._last_evicted_kv_tokens = 0
    agent._tick_count = 0
    agent._n_pages_per_fire = 4
    agent._kv_tokens_per_chunk = 1024
    agent._mamba_tokens_per_chunk = 1
    return agent


def test_snapshot_emits_retract_recovery_len():
    """(c) _snapshot plumbs the retract EWMA into the snapshot the pressure
    adapter reads — behavioral pin."""
    agent = _make_agent(retract_ewma=4096.0)
    snap = agent._snapshot(now=0.0)
    assert "slow_recovery_len_retract" in snap, (
        "BUG: _snapshot does not emit 'slow_recovery_len_retract' — the "
        "pressure adapter's L-aware retract cost (k_retract) is dead "
        "(L=0 always) and it falls back to the full-prefill cost."
    )
    assert snap["slow_recovery_len_retract"] == 4096.0, (
        f"retract recovery length must equal the tree_cache EWMA, got "
        f"{snap['slow_recovery_len_retract']}"
    )
    print(
        f"  PASS  _snapshot emits slow_recovery_len_retract="
        f"{snap['slow_recovery_len_retract']}"
    )


def test_snapshot_retract_key_present_in_source():
    """(c-source) Belt-and-braces: the literal key lives in agent.py's
    _snapshot, so it can never silently disappear again (the original bug)."""
    assert 'snap["slow_recovery_len_retract"]' in _AGENT_SRC, (
        "BUG: agent.py no longer assigns snap['slow_recovery_len_retract'] — "
        "the retract recovery-length signal has been dropped from _snapshot."
    )
    print("  PASS  agent.py _snapshot source assigns slow_recovery_len_retract")


_TESTS = (
    test_counters_unconditionally_initialized,
    test_record_recovery_len_retract_first_observation,
    test_snapshot_emits_retract_recovery_len,
    test_snapshot_retract_key_present_in_source,
)


if __name__ == "__main__":
    fails = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{'ALL GREEN' if not fails else str(fails) + ' FAILED'}")
    raise SystemExit(1 if fails else 0)
