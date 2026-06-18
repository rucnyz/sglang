"""Bound-pinned tests: the KV–mamba coupling bound + the design's crash-safety
at the extremes it implies (Phase 4).

`kv_mamba_ratio.py` derived, from sglang's own constants, that KV and mamba
fill proportionally (each cached node / running req consumes both), so the
reachable (KV_used, mamba_used) region is a band, not a square. These tests
PIN that bound's load-bearing numbers (so a sglang config change is caught) and
assert the shipped design (working-set floor + fork grow) stays crash-safe at
the two extremes of the band.

Constants pulled from sglang (not hardcoded) so the test tracks the real code:
  - FLA_CHUNK_SIZE (mamba cache token alignment)
  - MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO (the pool/max_running ratio)
The cc-config capacities (M, K, context_len) are the bench parameters.
"""
import math
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO as MAMBA_RATIO,
)
from sglang.srt.budgeter.agent import BudgetAgent

# cc bench config (the regime kv_mamba_ratio.py analyzed).
M = 64                  # max_mamba_cache_size (slots)
K = 1_827_295           # max_total_num_tokens (boot-observed)
CONTEXT_LEN = 262_144   # mc.context_len


def test_kv_full_cannot_starve_mamba():
    """Coupling lower bound: KV full ⟹ mamba ≥ ceil(K/context_len) slots,
    because the fewest mamba slots that can carry K KV tokens is achieved by
    maximal-context active reqs (1 slot : context_len tokens). For cc that is
    ~7 slots ≈ 10.9% — mamba can NOT be compressed to ~0 while KV is full (the
    user's coupling argument, pinned)."""
    mamba_floor_slots = math.ceil(K / CONTEXT_LEN)
    assert mamba_floor_slots == 7, (
        f"KV-full mamba floor = ceil(K/context_len) should be 7 slots; "
        f"got {mamba_floor_slots} (K={K}, context_len={CONTEXT_LEN})")
    occ = mamba_floor_slots / M
    assert occ > 0.10, (
        f"KV-full forces mamba occupancy >10% ({occ:.3f}); 'KV full, mamba "
        f"empty' is structurally unreachable")


def test_mamba_is_the_binding_pool_for_cc():
    """mamba saturates before KV: its M slots are exhausted ~K/FLA_CHUNK_SIZE/M
    times sooner (even at the minimum FLA_CHUNK_SIZE-token cached span). So
    mamba is the lead pressure signal for cc, and the useful cross-pool
    direction is grow-mamba-from-KV (k2m / the fork grow), not m2k."""
    kv_unit_capacity_min_span = K / FLA_CHUNK_SIZE   # most KV "units" possible
    assert M < kv_unit_capacity_min_span, (
        f"mamba ({M} slots) must bind before KV ({kv_unit_capacity_min_span:.0f} "
        f"units at min span); else m2k, not k2m, would be the cc direction")
    factor = kv_unit_capacity_min_span / M
    assert factor > 100, (
        f"mamba should bind >>100x sooner than KV for cc; got {factor:.0f}x")


def test_floor_reserves_live_working_set_not_nominal_cap():
    """#297: the floor reserves the LIVE active+protected working set
    (`m_used − evictable`) so a running req's slot is never drained, but does
    NOT statically reserve the nominal `max_running` cap. At the KV-full extreme
    the coupling bound forces ceil(K/context_len) LIVE mamba slots (7 for cc),
    far below the nominal cap (M//MAMBA_RATIO = 21); the floor tracks the live 7,
    and a burst beyond it is recovered by the active-slot grow hook, not a
    static cap reserve."""
    a = BudgetAgent.__new__(BudgetAgent)
    a._mamba_fork_headroom_slots = 4
    nominal_cap = M // MAMBA_RATIO                 # 21 — must NOT be reserved
    active = math.ceil(K / CONTEXT_LEN)            # 7 live at the KV-full extreme
    protected, evictable = 0, 40                   # rest of pool is idle cache
    m_used = active + evictable + protected
    floor = a._mamba_working_set_floor_slots(m_used, evictable)
    assert floor == active + protected + a._mamba_fork_headroom_slots, (
        f"floor must reserve the live working set active({active})+protected"
        f"({protected})+headroom; got {floor}")
    assert floor < nominal_cap + protected, (
        f"floor {floor} must NOT reserve the nominal cap {nominal_cap} (#297 "
        f"over-reservation that blocks m2k in the KV-bound regime)")


def test_mamba_full_extreme_fork_self_heals():
    """At the mamba-full extreme (the binding case), a new caching fork finds
    no slot and no unlocked cold cache → the fork grow must recover from KV's
    slack (which the coupling guarantees exists, since mamba binds first while
    KV has headroom), NOT crash. Mirrors the #312 regime at the bound."""
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
    import types

    c = MambaRadixCache.__new__(MambaRadixCache)

    class _FullPool:
        def __init__(self): self._free = 0
        def fork_from(self, v):
            if self._free <= 0:
                return None
            self._free -= 1
            return ("slot",)
    pool = _FullPool()
    c.req_to_token_pool = types.SimpleNamespace(mamba_pool=pool)
    c.evict = lambda params: None                       # no unlocked cold cache
    c._mamba_grow_hook = lambda n: (setattr(pool, "_free", pool._free + n) or True)
    forked = c._fork_mamba_with_recovery(mamba_value=("src",))
    assert forked is not None, (
        "at the mamba-full bound the fork must self-heal via the k2m grow "
        "(KV has slack by the coupling bound), not crash (#312)")


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError:
                failures += 1; print("FAIL", name); traceback.print_exc()
            except Exception:
                failures += 1; print("ERROR", name); traceback.print_exc()
    sys.exit(1 if failures else 0)
