"""#2 / #262 — REAL-pool MambaRadixCache coverage (GPU).

Every prior Phase-2 mamba test builds the cache via
`MambaRadixCache.__new__(...)` with stubs and hand-fabricated
tombstones. That leaves three things UNTESTED:

  * the real `MambaRadixCache.__init__` byte-denominator probe
    (`TreeNode.lpb_bytes_per_mamba_slot`, design.md §"Shared cost
    model" `B_b`);
  * the real `MambaPool` byte denominator
    (`mamba_cache.mem_usage_bytes() // (max_size+1)`);
  * the real `evict_mamba` / `evict_full` victim-selection ORDER over
    nodes inserted through the real `insert` path with real
    `mamba_pool.alloc` snapshots.

This file builds a REAL `HybridReqToTokenPool` (real `MambaPool`) + a
REAL `TokenToKVPoolAllocator` over a real `MHATokenToKVPool` and a REAL
`MambaRadixCache(params)`, then asserts THEORETICAL targets:

  (a) #2 fix / real bytes: after real `__init__`,
      `TreeNode.lpb_bytes_per_mamba_slot == mem_usage_bytes() // (max_size+1)`
      and is NOT the 1024 placeholder.
  (b) #2 fail-loud (reproducer): a mamba_pool that zeroes the byte
      API must make `__init__` RAISE, not silently keep 1024. Before
      the fix this FAILS (the try/except swallows → 1024).
  (c) evict_mamba real selection ORDER: victims pop in ascending
      `eviction_priority()` under LPB, and LRU vs LPB pick DIFFERENT
      victims (the hit-count / last_access orderings are crossed so a
      silent LRU fallback would surface).
  (d) byte-exactness: `predict_evict_cost_us("kv", x)` equals the cost
      recomputed over what `evict_full` actually frees (incl. the
      tombstone cascade).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch

torch.cuda.set_device(0)

DEVICE = "cuda:0"

PLACEHOLDER_BYTES_PER_MAMBA_SLOT = 1024


def _make_real_pool(size=64, kv_size=256, max_size=None):
    """Real `HybridReqToTokenPool` (real `MambaPool` exposing `size` +
    `mamba_cache.mem_usage_bytes()`), pattern from
    dev/interlayer/1_dyn_admission_cap/test_phase3.py `_make_pool`.
    `max_size > size` exercises dynamic-cap mode (State tensors allocated
    at max_size+1 rows while the live cap is `size`)."""
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=128,
        n_groups=1,
        num_heads=4,
        head_dim=64,
        state_size=16,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=[0, 1])
    return HybridReqToTokenPool(
        size=size,
        mamba_size=size,
        mamba_spec_state_size=size,
        max_context_len=1024,
        device=DEVICE,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        enable_mamba_extra_buffer=False,
        max_size=max_size,
    )


def _make_kv_allocator(kv_size=256):
    """Real `TokenToKVPoolAllocator` over a real `MHATokenToKVPool` —
    satisfies `MambaRadixCache.__init__`'s `isinstance` assertion and
    gives `insert`/`evict_full` a real allocator to `alloc`/`free`."""
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    kv = MHATokenToKVPool(
        size=kv_size,
        page_size=1,
        dtype=torch.float16,
        head_num=4,
        head_dim=64,
        layer_num=2,
        device=DEVICE,
        enable_memory_saver=False,
    )
    return TokenToKVPoolAllocator(
        size=kv_size,
        dtype=torch.float16,
        device=DEVICE,
        kvcache=kv,
        need_sort=False,
    )


def _make_real_cache(policy="lru", size=64, kv_size=256, max_size=None):
    """Real `MambaRadixCache(params)` over the real pool + allocator."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

    pool = _make_real_pool(size=size, kv_size=kv_size, max_size=max_size)
    alloc = _make_kv_allocator(kv_size=kv_size)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=alloc,
        page_size=1,
        eviction_policy=policy,
    )
    cache = MambaRadixCache(params)
    return cache, pool, alloc


def _insert_real(cache, pool, alloc, tokens, n_hits=0):
    """Insert through the real `insert` path with a real KV-index value
    (from the KV allocator) and a real mamba snapshot (from
    `mamba_pool.alloc`). Returns the inserted leaf TreeNode."""
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    value = alloc.alloc(len(tokens))
    assert value is not None, "KV allocator exhausted in fixture"
    mamba_value = pool.mamba_allocator.alloc(1)
    assert mamba_value is not None, "mamba pool exhausted in fixture"
    cache.insert(
        InsertParams(
            key=RadixKey(tokens, None),
            value=value,
            mamba_value=mamba_value,
            prev_prefix_len=0,
        )
    )
    # Recover the leaf node we just inserted to apply hits / assert on it.
    node = cache.root_node.children[RadixKey(tokens, None).child_key(cache.page_size)]
    for _ in range(n_hits):
        node.record_hit()
    return node


def _all_full_nodes(cache):
    out = {}
    stack = list(cache.root_node.children.values())
    while stack:
        n = stack.pop()
        out[tuple(n.key.token_ids)] = n
        stack.extend(n.children.values())
    return out


# --------------------------------------------------------------------- #
# (a) #2 fix / real byte denominator
# --------------------------------------------------------------------- #
def test_1_real_init_sets_ground_truth_bytes():
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode

    cache, pool, alloc = _make_real_cache(policy="lru", size=64)
    mp = pool.mamba_pool
    # B_b denominator is the PHYSICAL allocated slot count = max_size+1
    # (design.md "Padded slot 0": State tensors are allocated at
    # alloc_size+1 == max_size+1 rows). Dividing by size over-estimates
    # B_b by (size+1)/size and skews the LPB KV-vs-mamba weighting.
    expected = mp.mamba_cache.mem_usage_bytes() // (mp.max_size + 1)
    assert TreeNode.lpb_bytes_per_mamba_slot == expected, (
        f"__init__ must set B_b to the real per-slot byte ratio "
        f"{expected}, got {TreeNode.lpb_bytes_per_mamba_slot}"
    )
    assert TreeNode.lpb_bytes_per_mamba_slot != PLACEHOLDER_BYTES_PER_MAMBA_SLOT, (
        "B_b must NOT be the 1024 placeholder after a real __init__"
    )
    print(
        f"  PASS  1  real __init__ B_b = {TreeNode.lpb_bytes_per_mamba_slot} "
        f"= mem_usage_bytes({mp.mamba_cache.mem_usage_bytes()}) // "
        f"(max_size+1)({mp.max_size + 1})"
    )


# --------------------------------------------------------------------- #
# (b) #2 fail-loud reproducer
# --------------------------------------------------------------------- #
def test_2_zero_byte_api_must_raise_not_swallow():
    """The reproducing test for #2. A mamba_pool whose byte API returns
    0 (or a `size` of 0) corrupts the `B_b` denominator. The real
    `__init__` must RAISE rather than silently shipping the 1024
    placeholder. Pre-fix: the `try/except` swallows the bad value and
    `__init__` keeps 1024 → this assertion FAILS. Post-fix: `assert
    mp.size > 0` (or the bare division) makes `__init__` raise."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode

    pool = _make_real_pool(size=64)
    alloc = _make_kv_allocator()

    # Corrupt the byte API: mem_usage_bytes() -> 0 makes the per-slot
    # ratio 0, which is exactly the silent-corruption case #2 guards.
    class _ZeroBytesMambaCache:
        def mem_usage_bytes(self):
            return 0

    pool.mamba_pool.mamba_cache = _ZeroBytesMambaCache()

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=alloc,
        page_size=1,
        eviction_policy="lpb",
    )

    # Reset to the placeholder so we can detect a silent fallback.
    TreeNode.lpb_bytes_per_mamba_slot = PLACEHOLDER_BYTES_PER_MAMBA_SLOT
    raised = False
    msg = ""
    try:
        MambaRadixCache(params)
    except AssertionError as e:
        raised = True
        msg = str(e)
    assert raised, (
        "MambaRadixCache.__init__ must RAISE (AssertionError) when the "
        "mamba byte denominator is unavailable/zero, not silently keep "
        "the 1024 placeholder (the #2 defensive-fallback violation)"
    )
    # Tighten: the raise must be the B_b denominator guard, not some
    # unrelated earlier failure that would pass this test vacuously.
    assert "B_b" in msg or "denominator" in msg, (
        f"__init__ must raise specifically on the B_b/denominator guard, "
        f"got AssertionError: {msg!r}"
    )
    assert TreeNode.lpb_bytes_per_mamba_slot == PLACEHOLDER_BYTES_PER_MAMBA_SLOT, (
        "after a failed __init__, B_b must not have been written to a "
        "bogus value"
    )
    print("  PASS  2  zero byte API -> __init__ RAISES (no silent 1024 fallback)")


# --------------------------------------------------------------------- #
# (c) evict_mamba real selection ORDER (LPB vs LRU diverge)
# --------------------------------------------------------------------- #
def test_3_evict_mamba_order_lpb_vs_lru():
    """Real `evict_mamba` victim ORDER. Three real entries inserted
    through the real `insert` path, with hit counts / access recency
    CROSSED so LRU and LPB disagree:

      * Insert order stamps `last_access_time` ascending (A oldest,
        C newest), so LRU evicts A first.
      * Hits are assigned A=high, C=zero, so LPB's `ℓ(b) = n_b·c_i/B_b`
        is lowest for C (coldest) — LPB evicts C first.

    A silent LRU fallback in the LPB path would evict A first and FAIL
    the LPB assertion."""
    # ---- LRU policy: oldest (A) first ----
    cache, pool, alloc = _make_real_cache(policy="lru", size=64)
    a = _insert_real(cache, pool, alloc, [1, 2], n_hits=20)   # oldest, hot
    _insert_real(cache, pool, alloc, [3, 4], n_hits=5)
    c = _insert_real(cache, pool, alloc, [5, 6], n_hits=0)    # newest, cold
    # Eviction priority under LPB: c is the LOWEST (n_b=0 -> ℓ=0).
    assert c.eviction_priority() < a.eviction_priority(), (
        f"crossing failed: c.ℓ={c.eviction_priority()} must be < "
        f"a.ℓ={a.eviction_priority()}"
    )
    cache.evict_mamba(1)
    after = set(_all_full_nodes(cache).keys())
    # Under LRU, the oldest entry A=(1,2) is evicted first (it is a
    # childless leaf so the whole node disappears).
    assert (1, 2) not in after, (
        f"LRU evict_mamba must drop the OLDEST entry (1,2) first; "
        f"surviving={after}"
    )

    # ---- LPB policy: coldest (C) first ----
    cache2, pool2, alloc2 = _make_real_cache(policy="lpb", size=64)
    _insert_real(cache2, pool2, alloc2, [1, 2], n_hits=20)   # oldest, hot
    _insert_real(cache2, pool2, alloc2, [3, 4], n_hits=5)
    _insert_real(cache2, pool2, alloc2, [5, 6], n_hits=0)    # newest, cold
    before2 = set(_all_full_nodes(cache2).keys())
    cache2.evict_mamba(1)
    after2 = set(_all_full_nodes(cache2).keys())
    assert (5, 6) not in after2, (
        f"LPB evict_mamba must drop the COLDEST entry (5,6) first "
        f"(lowest hits-per-byte); surviving={after2}"
    )
    assert (1, 2) in after2, (
        f"LPB must NOT evict the hot oldest entry (1,2) first — a silent "
        f"LRU fallback would. surviving={after2}"
    )
    print(
        "  PASS  3  evict_mamba order: LRU drops oldest (1,2); LPB drops "
        "coldest (5,6) — policies diverge on real pool"
    )


# --------------------------------------------------------------------- #
# (d) byte-exact predict_evict_cost_us over the real evict_full set
# --------------------------------------------------------------------- #
def test_4_predict_equals_real_evict_full_with_cascade():
    """`predict_evict_cost_us("kv", x)` must equal the cost recomputed
    over what `evict_full` actually frees — including a tombstone
    cascade. Build root -> mid(hybrid) -> leaf via the real insert path,
    then `evict_mamba` mid into a tombstone, so a subsequent
    `evict_full(leaf)` sweeps the tombstone cascade."""
    from sglang.srt.budgeter.cost_model import get_cost_curves

    curves = get_cost_curves()

    def _build():
        cache, pool, alloc = _make_real_cache(policy="lru", size=64)
        # root -> mid=[1,2] (hybrid internal) -> leaf=[1,2,3,4] (child).
        _insert_real(cache, pool, alloc, [1, 2], n_hits=0)
        # Insert a deeper key sharing the [1,2] prefix so mid gets a child.
        _insert_real(cache, pool, alloc, [1, 2, 3, 4], n_hits=0)
        # Tombstone mid via a real evict_mamba: it picks the LRU/coldest
        # mamba node. Force mid to be the mamba victim by evicting until
        # mid is a tombstone (mid is the internal node with children).
        # evict_mamba(1) drops the oldest evictable mamba snapshot; mid
        # was inserted first so it is the oldest.
        cache.evict_mamba(1)
        return cache, pool, alloc

    cache, pool, alloc = _build()
    nodes = _all_full_nodes(cache)
    # Confirm we produced a tombstone internal node (mamba gone, KV kept).
    mid = nodes.get((1, 2))
    assert mid is not None and mid.mamba_value is None and len(mid.children) > 0, (
        f"fixture must leave (1,2) as a tombstone internal node; "
        f"got mid={mid} mamba={None if mid is None else mid.mamba_value}"
    )

    # Demand all remaining KV tokens: leaf (2 tok: the [3,4] tail) +
    # swept tombstone (1,2) (2 tok). predict then real evict_full.
    demand = cache.full_evictable_size()
    predicted = cache.predict_evict_cost_us(demand, pool="kv")

    # Recompute the cost over the EXACT set evict_full frees.
    before = dict(_all_full_nodes(cache))
    cost_by_key = {}
    use_lpb = cache._should_use_lpb()
    for k, n in before.items():
        s_b = len(n.key)
        n_b = n.hits_in_window() if use_lpb else 1
        c_i = curves.c_kv_ms(s_b)
        if n.mamba_value is not None:
            c_i += curves.c_m_ms(s_b)
        cost_by_key[k] = n_b * c_i

    freed = cache.evict_full(demand)
    after = set(_all_full_nodes(cache).keys())
    evicted = set(before.keys()) - after
    actual = sum(cost_by_key[k] for k in evicted) * 1000.0

    assert freed == demand, f"evict_full freed {freed} != demand {demand}"
    assert abs(predicted - actual) < 1e-6, (
        f"predict {predicted:.4f} != real evict_full set cost {actual:.4f} "
        f"over {sorted(evicted)}"
    )
    print(
        f"  PASS  4  predict == real evict_full set cost over cascade: "
        f"{predicted:.1f} us, freed {freed} tok over {len(evicted)} nodes"
    )


def test_5_dynamic_cap_bb_divides_by_max_size_not_size():
    """Dynamic-cap (max_size > size): State tensors are allocated at
    max_size+1 rows, so B_b must divide by max_size+1, NOT size+1. The
    default fixture has max_size==size (so max_size+1==size+1 and the
    distinction is invisible) — this case crosses them to pin that the
    divisor tracks PHYSICAL allocation. The actuator drives the live cap
    up to max_size via set_capacity_slots; B_b is a fixed per-row constant,
    so a revert to `// (size+1)` (or `// size`) would under/over-count here
    and this test would catch it."""
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode

    cache, pool, alloc = _make_real_cache(policy="lru", size=64, max_size=128)
    mp = pool.mamba_pool
    # HybridReqToTokenPool scales the req-level max_size to a larger
    # mamba-pool max_size internally; the exact value doesn't matter, only
    # that the pool is genuinely dynamic-cap (max_size > size) so
    # max_size+1 and size+1 differ.
    assert mp.max_size > mp.size, (
        f"fixture must be dynamic-cap (max_size > size): "
        f"max_size={mp.max_size} size={mp.size}"
    )
    by_max = mp.mamba_cache.mem_usage_bytes() // (mp.max_size + 1)
    by_size = mp.mamba_cache.mem_usage_bytes() // (mp.size + 1)
    assert by_max != by_size, "fixture must cross max_size+1 vs size+1"
    assert TreeNode.lpb_bytes_per_mamba_slot == by_max, (
        f"dynamic-cap B_b must divide by max_size+1={mp.max_size + 1} "
        f"(={by_max}), not size+1={mp.size + 1} (={by_size}); got "
        f"{TreeNode.lpb_bytes_per_mamba_slot}"
    )
    print(f"  PASS  5  dynamic-cap B_b = {by_max} = mem//{mp.max_size + 1} "
          f"(NOT mem//{mp.size + 1}={by_size}); divisor tracks physical alloc")


# --------------------------------------------------------------------- #
# (e) byte-exact predict_evict_cost_us("mamba") over the real evict_mamba
#     set — the #275 / #270 reuse-aware drain cost.
# --------------------------------------------------------------------- #
def test_6_predict_mamba_equals_real_evict_mamba_with_cascade():
    """`predict_evict_cost_us("mamba", n)` must equal the cost recomputed
    over what `evict_mamba` actually frees — an internal tombstone
    (`c_m`), a leaf (`c_kv + c_m`), and the tombstone-leaf cascade
    (`c_kv`). Build root -> mid=[1,2] (hybrid internal) -> leaf=[1,2,3,4];
    draining 2 mamba slots LRU-first tombstones mid (frees mamba), then
    leaf-evicts the child (frees KV+mamba) whose cascade sweeps the mid
    tombstone (frees KV). Parallels test_4 for the mamba side."""
    from sglang.srt.budgeter.cost_model import get_cost_curves

    curves = get_cost_curves()

    cache, pool, alloc = _make_real_cache(policy="lru", size=64)
    _insert_real(cache, pool, alloc, [1, 2], n_hits=0)        # mid (oldest)
    _insert_real(cache, pool, alloc, [1, 2, 3, 4], n_hits=0)  # child leaf (newest)

    before = dict(_all_full_nodes(cache))
    use_lpb = cache._should_use_lpb()
    snap = {
        k: (len(n.key), n.mamba_value is not None,
            n.hits_in_window() if use_lpb else 1)
        for k, n in before.items()
    }

    demand_slots = 2  # mid + leaf each hold one mamba slot
    predicted = cache.predict_evict_cost_us(demand_slots, pool="mamba")

    freed = cache.evict_mamba(demand_slots)
    after = _all_full_nodes(cache)

    # Recompute from the before/after delta (path-independent total):
    #   node fully gone    → lost KV (+ mamba if it had one)
    #   node now tombstone → lost mamba only (c_m)
    actual_ms = 0.0
    for k, (s_b, had_mamba, n_b) in snap.items():
        c_kv = curves.c_kv_ms(s_b)
        c_m = curves.c_m_ms(s_b)
        if k not in after:
            actual_ms += n_b * (c_kv + (c_m if had_mamba else 0.0))
        elif had_mamba and after[k].mamba_value is None:
            actual_ms += n_b * c_m
    actual = actual_ms * 1000.0

    assert freed == demand_slots, f"evict_mamba freed {freed} != {demand_slots}"
    assert abs(predicted - actual) < 1e-6, (
        f"predict('mamba') {predicted:.4f} != real evict_mamba set cost "
        f"{actual:.4f}"
    )
    print(
        f"  PASS  6  predict('mamba') == real evict_mamba cost incl cascade: "
        f"{predicted:.1f} us, freed {freed} slots"
    )


# --------------------------------------------------------------------- #
# (f) LPB reuse-aware drain cost: hot >> cold (the #275 signal)
# --------------------------------------------------------------------- #
def test_7_lpb_drain_cost_hot_gt_cold():
    """Under LPB, draining a HOT mamba snapshot (`hits_in_window > 0`)
    costs strictly more than a COLD one (`n_b = 0` → ~0). This is the
    reuse-aware signal #275 relies on so a hot cache resists the m2k
    drain — the regression was the active-utilization estimate pricing
    both at ~0."""
    from sglang.srt.budgeter.cost_model import get_cost_curves

    curves = get_cost_curves()

    c_cold, p1, a1 = _make_real_cache(policy="lpb", size=64)
    _insert_real(c_cold, p1, a1, [1, 2], n_hits=0)
    cost_cold = c_cold.predict_evict_cost_us(1, pool="mamba")

    c_hot, p2, a2 = _make_real_cache(policy="lpb", size=64)
    node = _insert_real(c_hot, p2, a2, [1, 2], n_hits=10)
    cost_hot = c_hot.predict_evict_cost_us(1, pool="mamba")
    # predict is pure-read → node intact; read its windowed hits directly.
    n_b = node.hits_in_window()

    assert cost_cold == 0.0, f"cold (n_b=0) drain must cost 0, got {cost_cold}"
    assert n_b > 0, f"hot node should have windowed hits, got {n_b}"
    assert cost_hot > cost_cold, (
        f"hot drain {cost_hot} must exceed cold {cost_cold}"
    )
    expected_hot = n_b * (curves.c_kv_ms(2) + curves.c_m_ms(2)) * 1000.0
    assert abs(cost_hot - expected_hot) < 1e-6, (
        f"hot cost {cost_hot} != n_b·(c_kv+c_m) {expected_hot}"
    )
    print(
        f"  PASS  7  LPB drain cost hot={cost_hot:.1f}us (n_b={n_b}) > "
        f"cold={cost_cold:.1f}us"
    )


def test_8_evict_without_calibration_must_not_crash():
    """Reproducing test (#343): a hybrid model served WITHOUT a calibrated cost
    model (base: --radix-eviction-policy lru, no SGLANG_CSIGMA_*) still uses
    MambaRadixCache, and its eviction path must NOT require the cost model.

    The per-victim LPB-loss telemetry (`evict_mamba` / `evict_full` accumulating
    `_cumulative_evicted_*_lpb_loss`) is consumed ONLY by the Budgeter; with no
    calibration it is dead and must be skipped, not raise. Pre-fix, both methods
    open with an unconditional `get_cost_curves()` which RAISES 'HiMA requires a
    calibrated cost model' -> the base scheduler SIGQUITs on the first eviction
    (observed: base completed 501/800, silently invalidating the case1 A/B by
    dropping the 299 hardest requests as errors)."""
    import sglang.srt.budgeter.cost_model as cm
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    try:
        get_global_server_args()
    except Exception:
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="Qwen/Qwen3.5-9B")
        )

    saved_singleton = cm._singleton
    saved_env = {
        k: os.environ.pop(k)
        for k in list(os.environ)
        if k.startswith("SGLANG_CSIGMA")
    }
    cm._singleton = None  # force the uncalibrated state base serving runs in
    try:
        assert not cm.has_cost_curves(), (
            "fixture precondition: no calibration should be resolvable "
            "(env cleared, singleton reset)"
        )
        cache, pool, alloc = _make_real_cache(policy="lru", size=8, kv_size=256)
        for i in range(6):
            _insert_real(cache, pool, alloc, [i * 10 + j for j in range(4)])
        # Pre-fix: get_cost_curves() at the top of each RAISES RuntimeError.
        n_m = cache.evict_mamba(1)
        n_k = cache.evict_full(1)
        assert n_m >= 0 and n_k >= 0
        print(
            f"  PASS  8  evict without calibration OK "
            f"(mamba_evicted={n_m} full_evicted={n_k})"
        )
    finally:
        cm._singleton = saved_singleton
        os.environ.update(saved_env)


def main() -> int:
    tests = [
        test_1_real_init_sets_ground_truth_bytes,
        test_2_zero_byte_api_must_raise_not_swallow,
        test_3_evict_mamba_order_lpb_vs_lru,
        test_4_predict_equals_real_evict_full_with_cascade,
        test_5_dynamic_cap_bb_divides_by_max_size_not_size,
        test_6_predict_mamba_equals_real_evict_mamba_with_cascade,
        test_7_lpb_drain_cost_hot_gt_cold,
        test_8_evict_without_calibration_must_not_crash,
    ]
    print(f"\n#2/#262 MambaRadixCache REAL-pool tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n#2/#262: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
