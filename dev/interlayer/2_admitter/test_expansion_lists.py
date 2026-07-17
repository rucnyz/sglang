"""#267 — REAL `_expansion_lists` + Stage-0 handler over a real
`MambaRadixCache` / `MambaPool` (GPU).

The mechanism (planner three-stage build, actuator Stage-0 migrate) was
verified earlier with FAKE providers/handlers. This file closes the gap
that #267 fills: the REAL `SchedulerOwnerProvider._expansion_lists` and
the REAL `SchedulerStage0Handler`, exercised against a real
`MambaRadixCache` + `MambaPool` on cuda:0.

Asserts THEORETICAL targets (per design.md §"Page selection"):

  (1) Migration-expansion: `_live_pages_in_cost_order('mamba')` returns
      exactly the LIVE (running-req-owned) mamba slots that are NOT held
      as a cached `mamba_value` snapshot, in ascending-slot-id (c_m
      constant ⇒ stable) order, bounded by the free-dst budget.
  (2) Drain-expansion: `_cached_pages_in_cost_order('mamba')` returns the
      cached-snapshot pages in the SAME cost order the cache's own
      `_plan_full_eviction` pops them (byte-identical to a real evict).
  (3) Stage-0 handler `rewrite_ssm_state_indices` moves BOTH the per-Req
      `mamba_pool_idx` AND the mirrored `req_index_to_mamba_index_mapping`
      the attention backend reads — and `evict_pages` actually frees the
      cached slots back to the pool.
  (4) Free-only request (allow_*=False) stays zero-cost: returns
      (None, None).

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
    dev/interlayer/2_admitter/test_expansion_lists.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch

torch.cuda.set_device(0)
DEVICE = "cuda:0"


def _make_real_cache(policy="lru", size=64, kv_size=256):
    from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
    from sglang.srt.mem_cache.memory_pool import (
        HybridReqToTokenPool,
        MHATokenToKVPool,
    )

    shape = Mamba2StateShape.create(
        tp_world_size=1, intermediate_size=128, n_groups=1, num_heads=4,
        head_dim=64, state_size=16, conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=[0, 1])
    pool = HybridReqToTokenPool(
        size=size, mamba_size=size, mamba_spec_state_size=size,
        max_context_len=1024, device=DEVICE, enable_memory_saver=False,
        cache_params=cache_params, mamba_layer_ids=[0, 1],
        enable_mamba_extra_buffer=False, max_size=None,
    )
    kv = MHATokenToKVPool(
        size=kv_size, page_size=1, dtype=torch.float16, head_num=4,
        head_dim=64, layer_num=2, device=DEVICE, enable_memory_saver=False,
    )
    alloc = TokenToKVPoolAllocator(
        size=kv_size, dtype=torch.float16, device=DEVICE, kvcache=kv,
        need_sort=False,
    )
    params = CacheInitParams(
        disable=False, req_to_token_pool=pool,
        token_to_kv_pool_allocator=alloc, page_size=1, eviction_policy=policy,
    )
    cache = MambaRadixCache(params)
    return cache, pool, alloc


def _insert_cached(cache, pool, alloc, tokens):
    """Insert a node carrying a real KV value + real mamba snapshot
    (a CACHED slot, the Drain harvest target)."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    value = alloc.alloc(len(tokens))
    mamba_value = pool.mamba_pool.alloc(1)
    cache.insert(InsertParams(
        key=RadixKey(tokens, None), value=value,
        mamba_value=mamba_value, prev_prefix_len=0,
    ))
    return int(mamba_value[0].item())


class _Req:
    def __init__(self, rid, mamba_slot, req_pool_idx):
        self.rid = rid
        self.mamba_pool_idx = torch.tensor(
            mamba_slot, dtype=torch.int32, device=DEVICE
        )
        self.req_pool_idx = req_pool_idx


class _RunningBatch:
    def __init__(self, reqs):
        self.reqs = reqs


class _Scheduler:
    def __init__(self, cache, pool, alloc, reqs):
        self.tree_cache = cache
        self.token_to_kv_pool_allocator = alloc
        self.req_to_token_pool = pool
        self.running_batch = _RunningBatch(reqs)


def _provider(sched, pool):
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator
    from sglang.srt.budgeter.scheduler_owner_provider import (
        SchedulerOwnerProvider,
    )

    # Test pool isn't arena-backed (`_make_real_cache` builds a plain
    # HybridReqToTokenPool), so stub `_tokens_per_page` to the real
    # mamba page granularity (1 slot/page under page-grain VMM). The
    # production MambaArenaActuator reads it from `_mamba_temporal_arena`.
    mamba_act = MambaArenaActuator.__new__(MambaArenaActuator)
    mamba_act.pool = pool.mamba_pool
    mamba_act._tokens_per_page = lambda: 1
    prov = SchedulerOwnerProvider(
        scheduler=sched, kv_actuator=None, mamba_actuator=mamba_act,
    )
    return prov, mamba_act


def test_1_migration_atomic_pool_yields_no_pages():
    """#269: on an ATOMIC mamba pool (tokens_per_chunk == 1 — the tp=1/fp32
    layout where one SSM slot fills a whole 2 MiB VMM chunk) Migration can
    NEVER free a net chunk: relocating a LIVE slot into a FREE slot just
    swaps which whole-free page is free. The corrected scattered-free-slot
    budget makes this fall out automatically — every free slot is its OWN
    whole-free page, so the scattered set is empty, the dst budget is 0,
    and `_live_pages_in_cost_order` returns []. (Supersedes the pre-#269
    behavior that listed every live slot as a migration page using the
    available_size() budget, which double-counted Stage-1's whole-free
    transfer payload.)"""
    cache, pool, alloc = _make_real_cache(size=64)
    c1 = _insert_cached(cache, pool, alloc, [1, 2, 3, 4])
    c2 = _insert_cached(cache, pool, alloc, [5, 6, 7, 8])
    # LIVE slots owned by running reqs — would have been migration pages
    # under the old budget, but an atomic pool has no scattered free dst.
    live_a = pool.mamba_pool.alloc(1)
    live_b = pool.mamba_pool.alloc(1)
    sa, sb = int(live_a[0].item()), int(live_b[0].item())
    reqs = [_Req("ra", sa, 10), _Req("rb", sb, 11)]
    sched = _Scheduler(cache, pool, alloc, reqs)
    prov, mamba_act = _provider(sched, pool)

    tps = mamba_act._tokens_per_page()
    assert tps == 1, f"fixture expects mamba tps=1, got {tps}"

    cached, live = prov._expansion_lists(
        "mamba", allow_drain=False, allow_migrate=True
    )
    assert cached is None, "allow_drain=False must keep cached None"
    assert live == [], (
        f"atomic pool (tps=1) must yield NO migration pages — every free "
        f"slot is its own whole-free page, scattered budget = 0: got {live}"
    )
    print("  PASS  1  Migration: atomic pool (tps=1) → [] (no scattered "
          "free dst; Migration cannot consolidate a net free chunk)")


def test_1b_migration_fragmentable_moves_and_budget():
    """#269: on a FRAGMENTABLE pool (tps ≥ 2) Migration CONSOLIDATES — it
    relocates a FULLY-LIVE page's slots into SCATTERED free slots on OTHER
    (kept) partially-live pages. This pins:
      (a) sources are FULLY-LIVE pages; destinations are donor slots on
          DIFFERENT pages → no self-destination (the freed page's own
          slots are never used as its own dst);
      (b) destinations exclude WHOLE-free pages (Stage-1's transfer
          payload);
      (c) the donor budget caps how many source pages a single fire frees;
      (d) the returned shape is `(freed_page_id, ((src,dst),...))` so
          Stage-0 runs the exact relocation (no page-id-as-slot-id, no dst
          guessing).

    Deterministic CPU fake-pool (tps=2, 5 pages over slots 0..9):
      free_slots = {4, 6}
        → page 2 (4-free / 5-live)  → donor slot 4
        → page 3 (6-free / 7-live)  → donor slot 6
      fully-live pages: page 1 (2,3) and page 4 (8,9)
    Donor budget = 2 slots → frees ONE fully-live page (page 1, needs 2),
    page 4 left unfunded. Expected: [(1, ((2,4),(3,6)))] — dsts 4,6 are on
    pages 2,3 (kept), never on page 1 (no self-dst)."""
    import torch
    from sglang.srt.budgeter.scheduler_owner_provider import (
        SchedulerOwnerProvider,
    )
    cpu = torch.device("cpu")

    class FakeMambaPool:
        size = 10
        def __init__(self):
            self.free_slots = torch.tensor([4, 6], dtype=torch.int64, device=cpu)
            self._capped_slots = torch.empty(0, dtype=torch.int64, device=cpu)
        def available_size(self):
            return int(self.free_slots.numel())

    class FakeMambaAct:
        def __init__(self, pool):
            self.pool = pool
            self.n_pages = 5  # size 10 / tps 2
        def _tokens_per_page(self):
            return 2

    class FakeSched:
        tree_cache = None  # no cached snapshots

    pool = FakeMambaPool()
    prov = SchedulerOwnerProvider(
        scheduler=FakeSched(), kv_actuator=None,
        mamba_actuator=FakeMambaAct(pool),
    )
    moves = prov._live_pages_in_cost_order("mamba")
    assert moves == [(1, ((2, 4), (3, 6)))], (
        f"expected one freed page (1) with its two relocations into donor "
        f"slots 4,6 on OTHER pages; page 4 unfunded (only 2 donors): got "
        f"{moves}"
    )
    # No self-destination: every dst is OUTSIDE its move's freed page.
    for pid, mv in moves:
        for src, dst in mv:
            assert dst // 2 != pid, (
                f"dst {dst} lands on the page {pid} being freed (self-dst)"
            )
    print("  PASS  1b Migration: fragmentable (tps=2) → [(1,((2,4),(3,6)))] "
          "(fully-live source, donors on other pages, budget-capped)")


def test_1c_migration_excludes_capped_chunk_donors():
    """#269: a CAPPED chunk (mid-fire, #174) contributes NO donor slot —
    its free slots must not be used as a migration destination. Same layout
    as test_1b but slot 7 is capped, so page 3 is excluded entirely: the
    lone remaining donor (slot 4) can't fund page 1's 2-slot move → []. If
    capped exclusion regressed, slot 6 would re-enter the donor pool and
    page 1 would be (wrongly) freed → [(1,((2,4),(3,6)))]."""
    import torch
    from sglang.srt.budgeter.scheduler_owner_provider import (
        SchedulerOwnerProvider,
    )
    cpu = torch.device("cpu")

    class FakeMambaPool:
        size = 10
        def __init__(self):
            self.free_slots = torch.tensor([4, 6], dtype=torch.int64, device=cpu)
            # slot 7 mid-fire → page 3 capped (its free slot 6 is NOT a donor)
            self._capped_slots = torch.tensor([7], dtype=torch.int64, device=cpu)
        def available_size(self):
            return int(self.free_slots.numel())

    class FakeMambaAct:
        def __init__(self, pool):
            self.pool = pool
            self.n_pages = 5
        def _tokens_per_page(self):
            return 2

    class FakeSched:
        tree_cache = None

    prov = SchedulerOwnerProvider(
        scheduler=FakeSched(), kv_actuator=None,
        mamba_actuator=FakeMambaAct(FakeMambaPool()),
    )
    moves = prov._live_pages_in_cost_order("mamba")
    assert moves == [], (
        f"capped chunk's free slot must not be a donor → only 1 donor (slot "
        f"4) → page 1's 2-slot move unfunded → []: got {moves}"
    )
    print("  PASS  1c Migration: capped chunk donor excluded → [] "
          "(mid-fire page 3's free slot not used as a dst)")


def test_1d_migration_mixed_cached_chunk_classification():
    """#269: a chunk with a CACHED slot is neither whole-free nor
    fully-live-uncached. Pin that (a) its cached slot is excluded from the
    live/source set (cached snapshots are Drain's job, not Migration's),
    and (b) its sibling FREE slot is still a valid donor.

    tps=2, 5 pages, free={4,6}, cached={5}:
      page 1 (2,3)  fully-LIVE  → source
      page 2 (4,5)  4 free / 5 CACHED → donor slot 4 (NOT a source; 5 not live)
      page 3 (6,7)  6 free / 7 live   → donor slot 6
      page 4 (8,9)  fully-LIVE  → source
    donors {4,6} fund ONE source (page 1, needs 2); page 4 unfunded.
    Expected [(1,((2,4),(3,6)))]; slot 5 never appears as src or dst."""
    import torch
    from sglang.srt.budgeter.scheduler_owner_provider import (
        SchedulerOwnerProvider,
    )
    cpu = torch.device("cpu")

    class FakeMambaPool:
        size = 10
        def __init__(self):
            self.free_slots = torch.tensor([4, 6], dtype=torch.int64, device=cpu)
            self._capped_slots = torch.empty(0, dtype=torch.int64, device=cpu)
        def available_size(self):
            return int(self.free_slots.numel())

    class FakeMambaAct:
        def __init__(self, pool):
            self.pool = pool
            self.n_pages = 5
        def _tokens_per_page(self):
            return 2

    # tree_cache exposing one CACHED mamba snapshot (slot 5) via the
    # `mamba_lru_list.cache[*].mamba_value` shape `_cached_mamba_slots` reads.
    class _Node:
        def __init__(self, mv):
            self.mamba_value = mv
    class _LRU:
        def __init__(self):
            self.cache = {0: _Node(torch.tensor([5], dtype=torch.int64, device=cpu))}
    class FakeSched:
        tree_cache = type("TC", (), {"mamba_lru_list": _LRU()})()

    prov = SchedulerOwnerProvider(
        scheduler=FakeSched(), kv_actuator=None,
        mamba_actuator=FakeMambaAct(FakeMambaPool()),
    )
    moves = prov._live_pages_in_cost_order("mamba")
    assert moves == [(1, ((2, 4), (3, 6)))], (
        f"cached slot 5 must be excluded from sources, its free sibling 4 "
        f"still a donor: got {moves}"
    )
    flat = [s for _, mv in moves for pair in mv for s in pair]
    assert 5 not in flat, (
        f"cached slot 5 must never be a migration src or dst: {flat}"
    )
    print("  PASS  1d Migration: mixed [free,cached] chunk → cached slot "
          "excluded from source, free sibling still a donor")


def test_2_drain_cached_in_eviction_cost_order():
    cache, pool, alloc = _make_real_cache(size=64)
    c1 = _insert_cached(cache, pool, alloc, [1, 2, 3, 4])
    c2 = _insert_cached(cache, pool, alloc, [5, 6, 7, 8])
    c3 = _insert_cached(cache, pool, alloc, [9, 10, 11, 12])
    sched = _Scheduler(cache, pool, alloc, [])
    prov, mamba_act = _provider(sched, pool)

    cached, live = prov._expansion_lists(
        "mamba", allow_drain=True, allow_migrate=False
    )
    assert live is None
    # The cost order must match what _plan_full_eviction pops (the SAME
    # selector). Recompute it independently and compare.
    victims, _swept = cache._plan_full_eviction(
        cache.full_evictable_size() + 1
    )
    expected_slots = []
    for v in victims:
        if v.mamba_value is not None and v.mamba_value.numel() > 0:
            expected_slots.extend(int(x) for x in v.mamba_value.cpu().tolist())
    # tps==1 ⇒ page id == slot id; dedupe preserving order.
    seen = set()
    expected_pages = []
    for s in expected_slots:
        if s == 0 or s in seen:
            continue
        seen.add(s)
        expected_pages.append(s)
    assert cached == expected_pages, (
        f"Drain page order must equal _plan_full_eviction victim order: "
        f"got {cached}, expected {expected_pages}"
    )
    assert set(cached) == {c1, c2, c3}, (cached, c1, c2, c3)
    print(f"  PASS  2  Drain: cached pages {cached} in _plan_full_eviction "
          f"cost order (byte-identical to a real evict)")


def test_3_handler_rewrite_and_evict():
    from sglang.srt.budgeter.scheduler_stage0_handler import (
        SchedulerStage0Handler,
    )

    cache, pool, alloc = _make_real_cache(size=64)
    c1 = _insert_cached(cache, pool, alloc, [1, 2, 3, 4])
    live_a = pool.mamba_pool.alloc(1)
    sa = int(live_a[0].item())
    req = _Req("ra", sa, 10)
    # Seed the mirrored mapping the attention backend reads.
    pool.req_index_to_mamba_index_mapping[10] = sa
    sched = _Scheduler(cache, pool, alloc, [req])

    mamba_stub = type(
        "A", (), {"pool": pool.mamba_pool, "_tokens_per_page": lambda self: 1}
    )()
    handler = SchedulerStage0Handler(
        scheduler=sched, kv_actuator=None, mamba_actuator=mamba_stub,
    )
    # rewrite: pick any free dst slot (dst selection is the planner's job
    # now; the handler only rewrites the req pointer), then rewrite it.
    free = pool.mamba_pool.free_slots
    dst = int(free[free != 0][0].item())
    assert dst != sa and dst != 0
    handler.rewrite_ssm_state_indices(sa, dst)
    assert int(req.mamba_pool_idx.item()) == dst, (
        "per-Req mamba_pool_idx must move to dst"
    )
    assert int(pool.req_index_to_mamba_index_mapping[10].item()) == dst, (
        "mirrored req_index_to_mamba_index_mapping must move to dst"
    )

    # evict_pages on the cached page c1 must free its slot back to pool.
    before_free = int(pool.mamba_pool.available_size())
    handler.evict_pages("mamba_to_kv", (c1,))
    after_free = int(pool.mamba_pool.available_size())
    assert after_free > before_free, (
        f"evict_pages must return cached slot to free list: "
        f"{before_free} -> {after_free}"
    )
    print(f"  PASS  3  handler: ssm rewrite moved both pointers "
          f"{sa}->{dst}; evict_pages freed cached page {c1} "
          f"(free {before_free}->{after_free})")


def test_3b_real_handler_through_xpool_actuator_stage0():
    """End-to-end Stage-0 with the PRODUCTION handler: build a real
    XPoolActuator (via __new__, Stage-0 only needs stage0_handler +
    src_act.pool), wire the REAL SchedulerStage0Handler, and run a real
    one-migration plan through `_run_stage0`. Asserts byte-exact slot
    relocation + the production handler's BOTH-pointer rewrite + the src
    slot left in `_capped_slots` (FREE-but-held)."""
    from sglang.srt.arena.fire_plan import FirePlan
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    from sglang.srt.budgeter.scheduler_stage0_handler import (
        SchedulerStage0Handler,
    )

    cache, pool, alloc = _make_real_cache(size=64)
    mamba = pool.mamba_pool
    live_a = mamba.alloc(1)
    src_slot = int(live_a[0].item())
    SENTINEL = 0.375
    mc = mamba.mamba_cache
    for t in mc.conv:
        t[:, src_slot, ...].fill_(SENTINEL)
    if isinstance(mc.temporal, list):
        for t in mc.temporal:
            t[src_slot, ...].fill_(SENTINEL)
    else:
        mc.temporal[:, src_slot, ...].fill_(SENTINEL)
    torch.cuda.synchronize()
    pre = [t[:, src_slot, ...].clone() for t in mc.conv]

    req = _Req("ra", src_slot, 10)
    pool.req_index_to_mamba_index_mapping[10] = src_slot
    sched = _Scheduler(cache, pool, alloc, [req])
    mamba_stub = type(
        "A", (), {"pool": mamba, "_tokens_per_page": lambda self: 1}
    )()
    handler = SchedulerStage0Handler(
        scheduler=sched, kv_actuator=None, mamba_actuator=mamba_stub,
    )

    # Planner-assigned destination: pick a free slot (≠ src, ≠ padded 0)
    # and pass the explicit (src, dst) move — Stage-0 no longer guesses.
    free = mamba.free_slots
    dst_slot = int(free[free != 0][0].item())
    assert dst_slot != src_slot and dst_slot != 0

    act = XPoolActuator.__new__(XPoolActuator)
    act.stage0_handler = handler
    # Uniform Stage-0 surface (#271): _run_stage0 calls src_act.migrate_slot
    # (pool-agnostic), which for mamba delegates to pool.migrate_slot.
    src_act = type(
        "S", (), {
            "pool": mamba,
            "migrate_slot": lambda self, s, d: self.pool.migrate_slot(s, d),
        },
    )()
    plan = FirePlan(
        direction="mamba_to_kv", pages_to_unmap=[src_slot],
        pages_to_map_dst=1, plan_seq=99, migrations=((src_slot, dst_slot),),
    )
    act._run_stage0(plan, src_act)
    torch.cuda.synchronize()

    dst = int(req.mamba_pool_idx.item())
    assert dst == dst_slot, (
        f"req pointer must move to the planner-assigned dst {dst_slot}, "
        f"got {dst}"
    )
    # byte-exact relocation
    for i, t in enumerate(mc.conv):
        assert torch.equal(t[:, dst, ...], pre[i]), (
            f"conv[{i}] dst not byte-exact to pre-migration src"
        )
    # BOTH pointers moved (per-Req + mirrored mapping)
    assert int(pool.req_index_to_mamba_index_mapping[10].item()) == dst
    # src capped (FREE-but-held), not in free_slots
    assert bool((mamba._capped_slots == src_slot).any().item())
    assert not bool((mamba.free_slots == src_slot).any().item())
    print(f"  PASS  3b REAL handler thru XPoolActuator._run_stage0: "
          f"slot {src_slot}->{dst} byte-exact, both pointers rewritten, "
          f"src capped/unmappable")


def test_4_free_only_zero_cost():
    cache, pool, alloc = _make_real_cache(size=64)
    sched = _Scheduler(cache, pool, alloc, [])
    prov, _ = _provider(sched, pool)
    cached, live = prov._expansion_lists(
        "mamba", allow_drain=False, allow_migrate=False
    )
    assert cached is None and live is None, (cached, live)
    print("  PASS  4  free-only path (allow_*=False) returns (None,None)")


def test_5_drain_walk_bounded_by_max_pages():
    """#284 (audit 2026-06-07 #270 M1 perf): the Drain-expansion victim
    materialization (the per-victim `.cpu().tolist()` + per-slot coverage
    loop in `_slots_to_fully_covered_pages`) must be bounded by the fire
    magnitude, not the whole evictable set. With `max_drain_pages=k` the
    walk early-breaks after emitting k fully-covered pages — the cost-order
    PREFIX the planner will consume — instead of materializing all of them
    on the scheduler thread every fire."""
    cache, pool, alloc = _make_real_cache(size=64)
    pages = [_insert_cached(cache, pool, alloc, list(range(i * 4, i * 4 + 4)))
             for i in range(8)]
    sched = _Scheduler(cache, pool, alloc, [])
    prov, _ = _provider(sched, pool)

    # Unbounded: full cost-ordered list (all 8 cached pages).
    full = prov.build_mamba_owner_map(allow_drain=True)
    assert full is not None
    assert len(full.cached_pages_in_cost_order) == 8, (
        f"unbounded drain must list all cached pages; got "
        f"{full.cached_pages_in_cost_order}"
    )

    # Bounded: exactly the first 3 in the SAME cost order (a prefix).
    bounded = prov.build_mamba_owner_map(allow_drain=True, max_drain_pages=3)
    assert bounded is not None
    got = bounded.cached_pages_in_cost_order
    assert len(got) == 3, (
        f"max_drain_pages=3 must early-break at 3 fully-covered pages; "
        f"got {len(got)}: {got}"
    )
    assert got == full.cached_pages_in_cost_order[:3], (
        f"the bounded set must be the cost-order PREFIX (same victims the "
        f"planner would consume), got {got} vs prefix "
        f"{full.cached_pages_in_cost_order[:3]}"
    )
    print("  PASS  5  Drain victim walk bounded by max_drain_pages "
          "(cost-order prefix, no full materialization)")


def main() -> int:
    tests = [
        test_1_migration_atomic_pool_yields_no_pages,
        test_1b_migration_fragmentable_moves_and_budget,
        test_1c_migration_excludes_capped_chunk_donors,
        test_1d_migration_mixed_cached_chunk_classification,
        test_2_drain_cached_in_eviction_cost_order,
        test_3_handler_rewrite_and_evict,
        test_3b_real_handler_through_xpool_actuator_stage0,
        test_4_free_only_zero_cost,
        test_5_drain_walk_bounded_by_max_pages,
    ]
    print(f"\n#267 _expansion_lists + Stage-0 handler real-pool tests "
          f"(n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}"); traceback.print_exc()
    print(f"#267: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
