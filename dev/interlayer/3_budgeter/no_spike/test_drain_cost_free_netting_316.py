"""#316 — drain-cost free-netting: the planner must price a cross-fire's KV/
mamba drain on the volume the fire ACTUALLY evicts, not the full fire
magnitude.

Root cause (verified on d290_win2 rep1 + code): `BudgetAgent` priced
`kv_drain_cost_us = predict_evict_cost_us(_n_pages_per_fire * kv_tpp)` — the
WORST case, blind to the free supply. But `XPoolFirePlanner.build` sources
free-first (Stage-1 harvests `owner_map.free_pages`) and drains cached pages
only for the shortfall, so a KV-slack fire is `free=8 drain=0` and evicts
NOTHING. Charging the full-magnitude eviction drove `NB[k2m]` negative on
1128/1241 ticks → k2m fired only twice all run → cache_hit only +0.57pp.

Fix: both the agent's drain pricing and the actuator's Stage-1 read ONE free-
supply signal — `SchedulerOwnerProvider.n_free_source_pages(direction)` — so
the priced drain volume `max(0, magnitude - n_free)` equals the executed
drain. A free-harvest fire is priced as the zero-drain it runs.

Two layers, both reuse production components (no hand-rolled imitations):
  - `test_n_free_source_pages_*`: the single-source free count, computed by the
    REAL `SchedulerOwnerProvider._compute_fully_free_pages` reduction.
  - `test_planner_free_harvest_fires_k2m`: the production `XPoolPlanner` — the
    complement of test_U (`test_nb_multisource_unit`): when the KV drain cost
    is 0 (free-harvest), a mamba-shedding-hot + KV-slack regime fires k2m.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider
from sglang.srt.budgeter.xpool_planner import XPoolPlanner, XPoolPolicyConfig
from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
from sglang.srt.budgeter.cost_model import (
    BUILTIN_DEFAULT,
    reset_cost_curves,
    reset_runtime_actuator_cost,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _kv_provider(n_pages, tps, free_slot_ids):
    """Real SchedulerOwnerProvider over a KV allocator carrying `free_slot_ids`
    as its free list. The allocator/scheduler/actuator are pure attribute
    carriers; the behaviour under test (`_compute_fully_free_pages`) is the
    real production reduction."""
    free = (
        torch.tensor(free_slot_ids, dtype=torch.int64, device=_DEVICE)
        if free_slot_ids else torch.empty(0, dtype=torch.int64, device=_DEVICE)
    )
    alloc = SimpleNamespace(
        free_pages=free,
        release_pages=None,
        _capped_pages=torch.empty(0, dtype=torch.int64, device=_DEVICE),
    )
    scheduler = SimpleNamespace(token_to_kv_pool_allocator=alloc)
    kv_act = SimpleNamespace(n_pages=n_pages, _tokens_per_page=lambda: tps)
    return SchedulerOwnerProvider(scheduler, kv_act, mamba_actuator=None)


def test_n_free_source_pages_kv_counts_fully_free():
    """All non-padded slots free → pages [1..n_pages) fully free (page 0 is
    excluded by the #226 padded-slot-0 invariant). The k2m source is KV."""
    n_pages, tps = 10, 4
    all_slots = list(range(1, n_pages * tps))  # slots 1..39 free
    prov = _kv_provider(n_pages, tps, all_slots)
    n_free = prov.n_free_source_pages("kv_to_mamba")
    assert n_free == n_pages - 1, (
        f"all slots free → {n_pages - 1} fully-free pages (page 0 excluded); "
        f"got {n_free}"
    )
    # A default fire (_n_pages_per_fire=8) is fully covered free → drain 0.
    assert n_free >= 8, f"expected free supply to cover an 8-page fire; got {n_free}"


def test_n_free_source_pages_kv_partial_free():
    """Only page 1's slots free (slots 4..7) → exactly 1 fully-free page.
    A fire larger than 1 page would need to DRAIN the shortfall → nonzero cost."""
    n_pages, tps = 10, 4
    page1_slots = list(range(1 * tps, 2 * tps))  # slots 4,5,6,7 → page 1 free
    prov = _kv_provider(n_pages, tps, page1_slots)
    assert prov.n_free_source_pages("kv_to_mamba") == 1


def test_n_free_source_pages_no_free():
    n_pages, tps = 8, 4
    prov = _kv_provider(n_pages, tps, [])
    assert prov.n_free_source_pages("kv_to_mamba") == 0


def test_n_free_source_pages_mamba_absent_is_zero():
    """No mamba actuator → the m2k source pool is absent → 0 (fail-closed: an
    m2k fire can harvest no free mamba pages, so it is priced as full drain)."""
    prov = _kv_provider(8, 4, list(range(1, 32)))
    assert prov.n_free_source_pages("mamba_to_kv") == 0


def test_n_free_source_pages_bad_direction_raises():
    prov = _kv_provider(8, 4, [])
    try:
        prov.n_free_source_pages("sideways")
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown direction")


# ---------------------------------------------------------------------------
# Planner consumption (complement of test_U in test_nb_multisource_unit):
# with kv_drain_cost_us == 0 (a free-harvest k2m), the cc-win regime fires k2m.

def _fresh_planner():
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
    return XPoolPlanner(config=cfg, adapter=SGLangPressureAdapter())


def test_planner_free_harvest_fires_k2m():
    """The win condition the misprice broke. Same mamba-shedding-hot + KV-slack
    shape as test_S (mamba 97% occupied, shedding hot snapshots, active-slack;
    KV donor) — but now the k2m fire FREE-HARVESTS KV (kv_drain_cost_us=0.0,
    the post-fix value of a free=8 drain=0 fire). The planner MUST fire k2m to
    grow mamba. (test_U is the mirror: a HOT KV cache that must be DRAINED
    → kv_drain_cost_us large → k2m suppressed.)"""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 2000.0,    # mamba evicting → L_rec observed
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,                 # pure cache-eviction pressure
        "usage_kv_active": 0.42,             # KV slack (donor, has free pages)
        "usage_mamba_active": 0.406,         # mamba active moderate (< low_water)
        "pool_occupancy_mamba": 0.969,       # but 97% OCCUPIED
        "pool_occupancy_kv": 0.476,
        "mamba_evict_grow_us": 20000.0,      # mamba shedding hot snapshots
        "kv_drain_cost_us": 0.0,             # free-harvest k2m → drains nothing
    }
    decision = planner.decide(
        usage_kv=0.476, usage_mamba=0.969, queue_depth=0, snapshot=snap,
    )
    assert decision.direction == "kv_to_mamba", (
        f"free-harvest k2m (kv_drain_cost_us=0) in a mamba-shedding-hot + "
        f"KV-slack regime must fire to grow mamba; got "
        f"direction={decision.direction!r} reason={decision.reason!r}"
    )


def test_planner_nonzero_kv_drain_still_suppresses_k2m():
    """Guard the other side: if the SAME regime would have to DRAIN a hot KV
    cache (kv_drain_cost_us large — a fire whose free supply did NOT cover it),
    k2m is suppressed. Pins that free-netting changes the INPUT cost, not the
    planner's correct response to a real drain cost (this is test_U's claim,
    re-checked here so the two tests move together)."""
    planner = _fresh_planner()
    snap = {
        "dt": 2.0,
        "slow_recovery_len_kv": 0.0,
        "slow_recovery_len_rec": 2000.0,
        "num_evicted_tokens_recent": 0,
        "num_retracted_reqs": 0,
        "num_paused_reqs": 0,
        "num_queue_reqs": 0,
        "usage_kv_active": 0.42,
        "usage_mamba_active": 0.406,
        "pool_occupancy_mamba": 0.969,
        "pool_occupancy_kv": 0.476,
        "mamba_evict_grow_us": 20000.0,
        "kv_drain_cost_us": 10_000_000.0,    # fire must drain a HOT KV cache
    }
    decision = planner.decide(
        usage_kv=0.476, usage_mamba=0.969, queue_depth=0, snapshot=snap,
    )
    assert decision.direction != "kv_to_mamba", (
        f"a k2m that must drain a HOT KV cache (kv_drain_cost_us=10M) must be "
        f"suppressed; got direction={decision.direction!r}"
    )


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
