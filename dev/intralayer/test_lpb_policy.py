"""#181 — LPB eviction policy: KV-side + mamba-side sort-key correctness.

Pins the design.md §"Shared cost model" / paper §sec:design-l1
eq:lpb-lru formula `ℓ(b) = n_b · c_i(s_b) / B_b`:

- `n_b` = `TreeNode.hits_in_window()` (sliding-window hit count).
- `c_i(s_b)` = `CostCurves.c_kv_ms(s_b)` / `c_m_ms(s_b)` depending
  on which buffer the node holds. `s_b = len(node.key)`.
- `B_b` = `value.numel()` for KV, `mamba_value.numel() ×
  bytes_per_mamba_slot` for mamba; SUM for hybrid nodes.

Sub-tests:
- A: KV-side `TreeNode.lpb_priority` honours formula (n_b factor,
  c_kv_ms(s_b) factor, B_b denominator).
- B: KV-side `LPBStrategy.get_priority(node)` returns
  `(lpb_priority, last_access_time)` for stable tiebreak.
- C: KV-side `record_hit()` wires through `_match_prefix_helper`
  iff `eviction_policy="lpb"` (LRU mode stays zero-overhead).
- D: KV-side `RadixCache` accepts `eviction_policy="lpb"` via
  `CacheInitParams`; raises on typos.
- E: Mamba-side `TreeNode.eviction_priority` includes the new
  c_i factor (regression for #181 mamba half).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch


def test_1_kv_lpb_priority_formula():
    """A: KV `lpb_priority` = n_b · c_kv(s_b) / B_b. Verify each
    factor by varying one at a time."""
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.mem_cache.radix_cache import RadixKey

    def _make_node(n_tokens: int, n_pages: int, n_hits: int) -> TreeNode:
        n = TreeNode()
        n.key = RadixKey(token_ids=list(range(n_tokens)), extra_key=None)
        n.value = torch.arange(n_pages, dtype=torch.int64)
        for _ in range(n_hits):
            n.record_hit()
        return n

    # Sanity: n_hits=0 → 0 (degenerate-but-safe).
    n0 = _make_node(n_tokens=8, n_pages=4, n_hits=0)
    assert n0.lpb_priority() == 0.0

    # Hold s_b and B_b constant; vary n_hits → priority scales linearly.
    n1 = _make_node(n_tokens=8, n_pages=4, n_hits=1)
    n3 = _make_node(n_tokens=8, n_pages=4, n_hits=3)
    assert n1.lpb_priority() > 0
    assert abs(n3.lpb_priority() / n1.lpb_priority() - 3.0) < 1e-9, (
        f"n_b factor broken: 3×hits → priority ratio = "
        f"{n3.lpb_priority() / n1.lpb_priority():.4f}, expected 3.0"
    )

    # Hold n_hits and B_b constant; vary s_b → priority scales with c_kv(s_b).
    # c_kv is α·s² + β·s + γ; ratio of c_kv at two s values is
    # NOT linear in s (quadratic dominant for large s).
    from sglang.srt.budgeter.cost_model import get_cost_curves
    curves = get_cost_curves()
    n_short = _make_node(n_tokens=64, n_pages=4, n_hits=1)
    n_long = _make_node(n_tokens=512, n_pages=4, n_hits=1)
    expected_ratio = curves.c_kv_ms(512) / curves.c_kv_ms(64)
    actual_ratio = n_long.lpb_priority() / n_short.lpb_priority()
    assert abs(actual_ratio - expected_ratio) < 1e-6, (
        f"c_kv(s_b) factor broken: 64→512 tokens ratio actual="
        f"{actual_ratio:.4f}, expected={expected_ratio:.4f}"
    )

    # Hold n_hits and s_b constant; vary B_b (n_pages) → priority
    # scales as 1/B_b.
    n_thin = _make_node(n_tokens=8, n_pages=4, n_hits=2)
    n_fat = _make_node(n_tokens=8, n_pages=16, n_hits=2)
    # n_fat has 4× the bytes → 1/4 the priority.
    assert abs(n_thin.lpb_priority() / n_fat.lpb_priority() - 4.0) < 1e-9, (
        f"B_b factor broken: 4× bytes → priority ratio = "
        f"{n_thin.lpb_priority() / n_fat.lpb_priority():.4f}, expected 4.0"
    )

    print("  PASS  1  KV lpb_priority honours n_b × c_kv(s_b) / B_b "
          "(each factor verified in isolation)")


def test_2_lpb_strategy_returns_tuple_for_tiebreak():
    """B: `LPBStrategy.get_priority(node)` returns
    `(lpb_priority, last_access_time)` so never-hit nodes (lpb=0)
    fall back to LRU order amongst themselves rather than colliding
    on equal-priority key."""
    from sglang.srt.mem_cache.evict_policy import LPBStrategy
    from sglang.srt.mem_cache.radix_cache import TreeNode

    n_a = TreeNode()  # last_access_time = monotonic NOW
    n_b = TreeNode()  # later
    assert n_b.last_access_time > n_a.last_access_time
    s = LPBStrategy()
    pa, pb = s.get_priority(n_a), s.get_priority(n_b)
    assert isinstance(pa, tuple) and len(pa) == 2
    assert pa < pb, (
        f"tiebreak on last_access_time broken; "
        f"older node {pa} should be < newer {pb}"
    )
    print("  PASS  2  LPBStrategy returns (loss, last_access) — "
          "older zero-loss node evicted first by tiebreak")


def test_3_record_hit_wired_only_under_lpb():
    """C: `_match_prefix_helper` calls `child.record_hit()` ONLY
    when policy=lpb. Under any other policy the deque stays empty
    (zero-overhead non-LPB path)."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
    from sglang.srt.mem_cache.radix_cache import RadixKey

    # Build a tiny radix cache + insert one prefix.
    def _cache(policy: str) -> RadixCache:
        p = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            enable_kv_cache_events=False,
            eviction_policy=policy,
        )
        return RadixCache(p)

    def _do_match(c: RadixCache, key_ids: list) -> TreeNode:
        # Manually populate one child node to test the match-walk.
        child = TreeNode()
        child.key = RadixKey(token_ids=key_ids, extra_key=None)
        child.value = torch.arange(len(key_ids), dtype=torch.int64)
        child.parent = c.root_node
        c.root_node.children[child.key.child_key(c.page_size)] = child
        _, last = c._match_prefix_helper(
            c.root_node, RadixKey(token_ids=key_ids, extra_key=None)
        )
        return last

    for policy in ("lru", "lfu", "fifo"):
        c = _cache(policy)
        last = _do_match(c, [1, 2, 3, 4])
        # In non-LPB modes, the deque stays empty.
        assert len(last._hit_times) == 0, (
            f"{policy}: record_hit should NOT fire in non-LPB mode; "
            f"got {len(last._hit_times)} hits"
        )

    c = _cache("lpb")
    last = _do_match(c, [1, 2, 3, 4])
    assert len(last._hit_times) == 1, (
        f"lpb: record_hit should fire once on the matched node; "
        f"got {len(last._hit_times)} hits"
    )
    print("  PASS  3  record_hit fires iff policy=lpb (zero-overhead "
          "non-LPB path verified across lru/lfu/fifo)")


def test_4_radix_cache_accepts_lpb_policy():
    """D: `RadixCache(eviction_policy="lpb")` constructs with
    `LPBStrategy`; typo policy raises ValueError listing the
    supported set including `lpb`."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.evict_policy import LPBStrategy
    from sglang.srt.mem_cache.radix_cache import RadixCache

    p_ok = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        enable_kv_cache_events=False,
        eviction_policy="lpb",
    )
    c = RadixCache(p_ok)
    assert isinstance(c.eviction_strategy, LPBStrategy), (
        f"eviction_policy='lpb' did not select LPBStrategy; "
        f"got {type(c.eviction_strategy).__name__}"
    )

    p_typo = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        enable_kv_cache_events=False,
        eviction_policy="lpb_wrong",
    )
    try:
        RadixCache(p_typo)
    except ValueError as e:
        assert "lpb" in str(e), (
            f"ValueError must list 'lpb' in supported policies; got: {e}"
        )
    else:
        raise AssertionError(
            "RadixCache must raise on unknown policy (got silent build "
            "with 'lpb_wrong')"
        )
    print("  PASS  4  RadixCache accepts policy='lpb' → LPBStrategy; "
          "typo raises with 'lpb' in error message")


def test_5_mamba_lpb_includes_c_i_factor():
    """E: mamba-side `eviction_priority` includes `c_i(s_b)`
    factor (#181 fix). Two nodes with same n_hits + same bytes
    but different s_b lengths must produce different priorities
    proportional to `c_kv_ms(s) + c_m_ms(s)`."""
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode as MTreeNode
    from sglang.srt.mem_cache.radix_cache import RadixKey

    def _make(n_tokens: int, n_pages: int, n_mamba: int, n_hits: int):
        n = MTreeNode()
        n.key = RadixKey(token_ids=list(range(n_tokens)), extra_key=None)
        n.value = torch.arange(n_pages, dtype=torch.int64)
        n.mamba_value = torch.arange(n_mamba, dtype=torch.int64)
        for _ in range(n_hits):
            n.record_hit()
        return n

    # Two nodes: same hits, same byte counts, different s_b.
    n_short = _make(n_tokens=64, n_pages=4, n_mamba=1, n_hits=1)
    n_long = _make(n_tokens=512, n_pages=4, n_mamba=1, n_hits=1)
    # Pre-#181 mamba: ratio would be 1.0 (n_hits / bytes only).
    # Post-#181: ratio = (c_kv(512) + c_m(512)) / (c_kv(64) + c_m(64)).
    from sglang.srt.budgeter.cost_model import get_cost_curves
    curves = get_cost_curves()
    expected = (
        (curves.c_kv_ms(512) + curves.c_m_ms(512))
        / (curves.c_kv_ms(64) + curves.c_m_ms(64))
    )
    actual = n_long.eviction_priority() / n_short.eviction_priority()
    assert expected != 1.0, "test setup degenerate"
    assert abs(actual - expected) < 1e-6, (
        f"mamba LPB sort key did NOT pick up c_i(s_b) factor; "
        f"actual ratio={actual:.4f}, expected={expected:.4f}. "
        f"Pre-#181 the ratio would be 1.0 (n/B alone)."
    )
    print(f"  PASS  5  mamba LPB sort key now includes c_i(s_b): "
          f"long/short = {actual:.3f} (expected {expected:.3f})")


def test_6_record_hit_walks_multi_level_path():
    """C-extended (audit Lens 1 BLOCKER): `record_hit` must fire on
    EVERY intermediate child along the prefix-walk, and NOT on the
    root. Pre-audit test_3 only verified the single leaf got a hit;
    a buggy impl that recorded on `node` (parent) instead of `child`
    would also leave the leaf with one hit and pass test_3.
    """
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import (
        RadixCache, RadixKey, TreeNode,
    )

    c = RadixCache(CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        enable_kv_cache_events=False,
        eviction_policy="lpb",
    ))

    # Build a 3-level chain: root → mid → leaf, with keys [1,2,3,4]
    # split as [1,2] / [3,4].
    mid = TreeNode()
    mid.key = RadixKey(token_ids=[1, 2], extra_key=None)
    mid.value = torch.arange(2, dtype=torch.int64)
    mid.parent = c.root_node
    c.root_node.children[mid.key.child_key(c.page_size)] = mid

    leaf = TreeNode()
    leaf.key = RadixKey(token_ids=[3, 4], extra_key=None)
    leaf.value = torch.arange(2, dtype=torch.int64)
    leaf.parent = mid
    mid.children[leaf.key.child_key(c.page_size)] = leaf

    # Walk the full path.
    c._match_prefix_helper(
        c.root_node, RadixKey(token_ids=[1, 2, 3, 4], extra_key=None),
    )

    assert len(c.root_node._hit_times) == 0, (
        f"root should NEVER record_hit (it's not a 'visited child'); "
        f"got {len(c.root_node._hit_times)} hits on root"
    )
    assert len(mid._hit_times) == 1, (
        f"mid-level child must record_hit on the walk; got "
        f"{len(mid._hit_times)}"
    )
    assert len(leaf._hit_times) == 1, (
        f"leaf must record_hit on the walk; got {len(leaf._hit_times)}"
    )
    print("  PASS  6  record_hit walks every intermediate child "
          "(root=0, mid=1, leaf=1) on a 3-level prefix match")


def test_7_cache_does_not_stale_on_window_expiry():
    """Audit-driven (Lens 4 + bug uncovered during audit): a prior
    `_cached_lpb_priority` memoization went stale when `hits_in_window`
    pruned a deque entry — the cache was only invalidated on
    `record_hit`, never on passive time advancement. This test pins
    the post-fix contract: `lpb_priority` reflects the current
    sliding-window `hits_in_window`, even after old entries expire
    with no record_hit in between.
    """
    import time as _time
    from sglang.srt.mem_cache.radix_cache import (
        RadixKey, TreeNode,
    )

    # Shrink window so the test runs in < 1s.
    orig_window = TreeNode.lpb_window_s
    TreeNode.lpb_window_s = 0.05
    try:
        n = TreeNode()
        n.key = RadixKey(token_ids=list(range(8)), extra_key=None)
        n.value = torch.arange(4, dtype=torch.int64)
        n.record_hit()
        p_with_hit = n.lpb_priority()
        assert p_with_hit > 0
        # Wait for the entry to fall out of the window.
        _time.sleep(0.1)
        p_after_expiry = n.lpb_priority()
        assert p_after_expiry == 0.0, (
            f"BUG (cache-staleness): after window expiry the cached "
            f"priority should be 0 (hit pruned), but got "
            f"{p_after_expiry}. Pre-fix the memoized value persisted."
        )
    finally:
        TreeNode.lpb_window_s = orig_window
    print("  PASS  7  lpb_priority reflects hits_in_window after "
          "window expiry (no stale cache from prior record_hit)")


def test_8_kv_lpb_pre_181_sentinel_ratio():
    """Audit-driven (Lens 5 — KV-side mirror of test_5). Two KV-only
    nodes with the same hits and bytes but different s_b (token
    counts) MUST produce different `lpb_priority` proportional to
    `c_kv_ms(s_b)`. Pre-#181 KV had no LPB at all; the explicit
    sentinel guards against any future refactor that drops the
    `c_kv_ms` factor.
    """
    from sglang.srt.budgeter.cost_model import get_cost_curves
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

    def _make(n_tokens: int, n_pages: int, n_hits: int) -> TreeNode:
        n = TreeNode()
        n.key = RadixKey(token_ids=list(range(n_tokens)), extra_key=None)
        n.value = torch.arange(n_pages, dtype=torch.int64)
        for _ in range(n_hits):
            n.record_hit()
        return n

    n_short = _make(n_tokens=64, n_pages=4, n_hits=1)
    n_long = _make(n_tokens=512, n_pages=4, n_hits=1)
    curves = get_cost_curves()
    expected = curves.c_kv_ms(512) / curves.c_kv_ms(64)
    actual = n_long.lpb_priority() / n_short.lpb_priority()
    assert expected != 1.0, "test setup degenerate (c_kv constant?)"
    assert abs(actual - expected) < 1e-6, (
        f"KV LPB sort key dropped the c_kv_ms factor; long/short "
        f"ratio actual={actual:.4f}, expected={expected:.4f}. "
        f"Without c_i factor the ratio would be 1.0."
    )
    print(f"  PASS  8  KV LPB sort key honours c_kv(s_b): "
          f"long/short = {actual:.3f} (degenerate-ratio 1.0 fails)")


# ============================================================
# Audit-driven reproducing tests (round 2). Pre-fix: FAIL.
# ============================================================

def test_9_mamba_radix_cache_honors_eviction_policy_boot_flag():
    """BLOCKER 1 reproducer: `MambaRadixCache.evict_mamba` /
    `evict_full` previously gated LPB on `SGLANG_LPB_LRU` env var,
    NOT on `params.eviction_policy`. Server launched with
    `--eviction-policy lpb` silently fell back to LRU on hybrid
    models. Fix: read `params.eviction_policy` like plain
    `RadixCache` does.

    Pre-fix: `MambaRadixCache.eviction_policy` attribute does not
    exist; `_should_use_lpb()` does not exist. Test fails on attr
    access. Post-fix: attribute is set + helper returns True iff
    policy=='lpb', regardless of env state.
    """
    import os
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

    # Bypass full __init__ — only assert plumbing fields are
    # populated. Real construction needs a HybridReqToTokenPool +
    # MambaPool that this test deliberately doesn't bring up.
    saved_env = os.environ.pop("SGLANG_LPB_LRU", None)
    try:
        # Simulate post-fix state: policy field carries the boot flag.
        c_lpb = MambaRadixCache.__new__(MambaRadixCache)
        c_lpb.eviction_policy = "lpb"
        assert c_lpb._should_use_lpb() is True, (
            "MambaRadixCache._should_use_lpb() must read "
            "params.eviction_policy (post-fix), not env var. "
            "Pre-fix this method doesn't exist; AttributeError fails "
            "the test (which is the reproducer signal)."
        )

        c_lru = MambaRadixCache.__new__(MambaRadixCache)
        c_lru.eviction_policy = "lru"
        assert c_lru._should_use_lpb() is False, (
            "policy='lru' must yield use_lpb=False even if env var "
            "is unset."
        )

        # Env-var presence must NOT override an explicit 'lru' policy.
        os.environ["SGLANG_LPB_LRU"] = "1"
        assert c_lru._should_use_lpb() is False, (
            "Env var should NOT override params.eviction_policy. "
            "The boot flag is the source of truth post-#181."
        )
    finally:
        os.environ.pop("SGLANG_LPB_LRU", None)
        if saved_env is not None:
            os.environ["SGLANG_LPB_LRU"] = saved_env
    print("  PASS  9  MambaRadixCache._should_use_lpb honors "
          "params.eviction_policy (no env-var fallback)")


def test_10a_hi_radix_cache_record_hit_walks_prefix():
    """BLOCKER 2 reproducer (KV-only HiRadixCache):
    `HiRadixCache._match_prefix_helper` overrides
    `RadixCache._match_prefix_helper` and does NOT call
    `record_hit`. Result: every prefix match on a HiRadix-enabled
    server produces `n_b ≡ 0` for every node → LPB silently
    degenerates to LRU under `--enable-hierarchical-cache`.

    Test bypasses HiRadixCache.__init__ (heavy host-memory pool
    setup) and drives `_match_prefix_helper` directly on a
    minimum-shape tree.
    """
    import time
    from sglang.srt.mem_cache.evict_policy import LPBStrategy
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

    cache = HiRadixCache.__new__(HiRadixCache)
    cache.page_size = 1
    cache.disable = False
    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey(token_ids=[], extra_key=None)
    cache.eviction_strategy = LPBStrategy()  # post-fix gate

    leaf = TreeNode()
    leaf.key = RadixKey(token_ids=[1, 2, 3, 4], extra_key=None)
    leaf.value = torch.arange(4, dtype=torch.int64)
    leaf.parent = cache.root_node
    cache.root_node.children[leaf.key.child_key(cache.page_size)] = leaf

    # Pre-condition: leaf has no recorded hits.
    assert leaf.hits_in_window() == 0

    cache._match_prefix_helper(
        cache.root_node, RadixKey(token_ids=[1, 2, 3, 4], extra_key=None)
    )

    # Pre-fix: `record_hit` not called → still 0. Post-fix: 1.
    assert leaf.hits_in_window() == 1, (
        f"HiRadixCache._match_prefix_helper must call record_hit on "
        f"matched nodes when policy=LPB. Got hits_in_window={leaf.hits_in_window()}; "
        f"expected 1. Pre-fix value 0 is the BLOCKER 2 signal."
    )
    print("  PASS  10a HiRadixCache._match_prefix_helper records "
          "hit under LPB")


def test_10b_hi_mamba_radix_cache_record_hit_walks_prefix():
    """BLOCKER 2 reproducer (hybrid HiMambaRadixCache): same as
    test_10a but for the hybrid+hierarchical cache shape. Uses the
    mamba TreeNode (which always records — no LPBStrategy gate;
    matches `MambaRadixCache._match_prefix_helper` always-record
    pattern)."""
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode as MambaTreeNode
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache = HiMambaRadixCache.__new__(HiMambaRadixCache)
    cache.page_size = 1
    cache.disable = False
    cache.root_node = MambaTreeNode()
    cache.root_node.key = RadixKey(token_ids=[], extra_key=None)
    cache.root_node.value = []
    cache.root_node.mamba_value = None

    leaf = MambaTreeNode()
    leaf.key = RadixKey(token_ids=[1, 2, 3], extra_key=None)
    leaf.value = torch.arange(3, dtype=torch.int64)
    # mamba_value populated so the matched-but-no-mamba path doesn't
    # zero out best_value_len.
    leaf.mamba_value = torch.tensor([0], dtype=torch.int64)
    leaf.parent = cache.root_node
    cache.root_node.children[leaf.key.child_key(cache.page_size)] = leaf

    assert leaf.hits_in_window() == 0

    cache._match_prefix_helper(RadixKey(token_ids=[1, 2, 3], extra_key=None))

    assert leaf.hits_in_window() == 1, (
        f"HiMambaRadixCache._match_prefix_helper must call "
        f"record_hit on matched nodes (matching MambaRadixCache "
        f"pattern of always-record). Got hits_in_window="
        f"{leaf.hits_in_window()}; expected 1. Pre-fix value 0 is "
        f"the BLOCKER 2 signal."
    )
    print("  PASS  10b HiMambaRadixCache._match_prefix_helper "
          "records hit on visited node")


def test_11_radix_cache_evict_picks_lowest_lpb_priority():
    """Integration: drive `RadixCache.evict()` under LPB with
    multiple leaves at distinct (n_b, s_b, B_b) and assert the
    chosen victim is the LOWEST-lpb-priority leaf. None of the
    pre-existing unit tests exercise `evict()`'s heap selection —
    they only test `lpb_priority()` in isolation. This test would
    catch any regression in `RadixCache.evict()`'s integration with
    `LPBStrategy.get_priority`.
    """
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import (
        RadixCache,
        RadixKey,
        TreeNode,
    )

    p = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        enable_kv_cache_events=False,
        eviction_policy="lpb",
    )
    c = RadixCache(p)

    # Stub allocator: evict() calls .free(values).
    class _NoopAlloc:
        def free(self, _v):
            pass
    c.token_to_kv_pool_allocator = _NoopAlloc()

    def _add(token_ids, hits):
        n = TreeNode()
        n.key = RadixKey(token_ids=token_ids, extra_key=None)
        n.value = torch.arange(len(token_ids), dtype=torch.int64)
        n.parent = c.root_node
        c.root_node.children[n.key.child_key(c.page_size)] = n
        c.evictable_leaves.add(n)
        for _ in range(hits):
            n.record_hit()
        return n

    # Three leaves. lpb = n_hits * c_kv(s_b) / size_bytes.
    # All s_b match size_bytes (per-token page) → ratio reduces
    # to n_hits * c_kv_ms(s_b) / s_b. Lowest lpb wins as victim.
    cold = _add(token_ids=[1, 2, 3, 4], hits=0)          # lpb = 0
    warm = _add(token_ids=[10, 11, 12, 13], hits=3)
    hot  = _add(token_ids=[20, 21, 22, 23], hits=20)
    cold.last_access_time = 1.0
    warm.last_access_time = 2.0
    hot.last_access_time = 3.0

    # Demand 4 tokens — exactly one leaf's worth. Should evict
    # `cold` (lpb=0 is the lowest under LPB).
    c.evict(EvictParams(num_tokens=4))
    assert cold not in c.evictable_leaves, (
        "evict() should have picked the cold (lpb=0) leaf first; "
        "it's still in the heap. Pre-fix this might pass for the "
        "wrong reason if record_hit doesn't fire (all leaves "
        "tied at 0 → LRU tiebreak picks oldest, which is `cold` "
        "by last_access_time anyway). Test_8 covers the formula; "
        "this one covers the *integration*."
    )
    assert hot in c.evictable_leaves and warm in c.evictable_leaves, (
        "hot + warm should be spared (higher lpb)"
    )
    print(f"  PASS  11 RadixCache.evict under LPB picks lowest-"
          f"lpb leaf (cold lpb=0, warm + hot spared)")


def test_12_split_node_carries_hit_times():
    """`_split_node` previously copied `hit_count` to `new_node`
    but NOT `_hit_times` deque. Newly-split prefix nodes started
    with empty hit history → biased to evict first under LPB.
    Post-fix: deque is moved from `child` to `new_node` (the
    shared prefix path inherits hit history; the smaller tail
    starts fresh, which it must, since it's now distinct content).
    """
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import (
        RadixCache,
        RadixKey,
        TreeNode,
    )

    p = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        enable_kv_cache_events=False,
        eviction_policy="lpb",
    )
    c = RadixCache(p)

    # Long node covering tokens [1,2,3,4,5,6]; record several hits.
    n = TreeNode()
    n.key = RadixKey(token_ids=[1, 2, 3, 4, 5, 6], extra_key=None)
    n.value = torch.arange(6, dtype=torch.int64)
    n.parent = c.root_node
    c.root_node.children[n.key.child_key(c.page_size)] = n
    for _ in range(5):
        n.record_hit()
    assert n.hits_in_window() == 5

    new_node = c._split_node(n.key, n, split_len=3)

    # Post-fix: prefix part (`new_node`) carries the 5 hits;
    # tail part (`n`) starts empty.
    assert new_node.hits_in_window() == 5, (
        f"_split_node must move _hit_times deque from child to "
        f"new_node (the shared-prefix segment). Got "
        f"new_node.hits_in_window={new_node.hits_in_window()}; "
        f"expected 5. Pre-fix value 0 is the carryover-bug signal."
    )
    assert n.hits_in_window() == 0, (
        f"after split, the tail (child node, holding only the "
        f"suffix) should start fresh; got n.hits_in_window="
        f"{n.hits_in_window()}; expected 0."
    )
    print(f"  PASS  12 _split_node moves _hit_times to new_node "
          f"(5 hits preserved; tail resets)")


def test_13_evict_paths_consult_should_use_lpb_not_env():
    """Gate-wiring guard: `MambaRadixCache.evict_full` and
    `evict_mamba` must derive `use_lpb` from `_should_use_lpb()`
    (which reads `params.eviction_policy`), NOT from
    `os.environ["SGLANG_LPB_LRU"]`. Spy on `_should_use_lpb` and
    confirm both eviction entry points invoke it.

    This is the regression guard that would have caught the original
    BLOCKER 1 (env-var gate) had it existed — the prior code never
    called any policy helper, it read os.environ inline.
    """
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

    c = MambaRadixCache.__new__(MambaRadixCache)
    c.disable = False
    c.eviction_policy = "lpb"
    # Cumulative tally evict_mamba/evict_full increment at their tail
    # (real __init__ seeds 0); the empty-LRU run still touches it.
    c._cumulative_evicted_mamba_slots = 0
    c._cumulative_evicted_kv_tokens = 0

    calls = {"n": 0}
    real = MambaRadixCache._should_use_lpb

    def _spy(self):
        calls["n"] += 1
        return real(self)

    c._should_use_lpb = _spy.__get__(c, MambaRadixCache)

    # Stub the LRU list so the eviction loop is a no-op (returns None
    # → zero victims). We only need to reach the `use_lpb =
    # self._should_use_lpb()` line.
    class _EmptyLRU:
        cache: dict = {}  # _lpb_build_*_heap iterates .cache.values()
        def get_lru_no_lock(self):
            return None
        def get_leaf_lru_no_lock(self):
            return None
    c.mamba_lru_list = _EmptyLRU()
    c.full_lru_list = _EmptyLRU()

    n_mamba = c.evict_mamba(1)
    n_full = c.evict_full(1)

    assert n_mamba == 0 and n_full == 0, "empty LRU → zero evictions"
    assert calls["n"] == 2, (
        f"evict_mamba + evict_full must each call _should_use_lpb "
        f"exactly once; got {calls['n']}. If 0, the eviction path "
        f"still reads os.environ inline (BLOCKER 1 regressed)."
    )
    print("  PASS  13 evict_mamba + evict_full both consult "
          "_should_use_lpb (no inline env read)")


def test_14_swa_rejects_lpb_loudly():
    """SWARadixCache has no LPB plumbing (separate TreeNode, no
    `record_hit` / LPBStrategy) — #261. It must FAIL LOUD when asked
    for `--radix-eviction-policy lpb` rather than silently running
    LRU (the exact silent-degradation class that #181 closed on the
    other variants). The guard runs at the very top of `__init__`,
    before the SWA-pool isinstance assert, so it's reachable with
    dummy params.
    """
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        eviction_policy="lpb",
        sliding_window_size=128,
    )
    try:
        SWARadixCache(params)
    except NotImplementedError as e:
        assert "lpb" in str(e).lower() and "swa" in str(e).lower(), (
            f"raise must name both 'lpb' and 'SWA': {e}"
        )
        assert "#261" in str(e), "raise should point to the tracking issue #261"
    else:
        raise AssertionError(
            "SWARadixCache(eviction_policy='lpb') must raise "
            "NotImplementedError, not silently run LRU"
        )
    # Sanity: lru policy does NOT trip the guard (it proceeds to the
    # SWA-pool isinstance assert, which fails on our None allocator —
    # proving the guard let lru through to the real construction).
    params_lru = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        eviction_policy="lru",
        sliding_window_size=128,
    )
    try:
        SWARadixCache(params_lru)
    except NotImplementedError:
        raise AssertionError("lru policy must NOT trip the lpb guard")
    except (AssertionError, AttributeError):
        pass  # expected: downstream SWA-pool assert/attr on None allocator
    print("  PASS  14 SWARadixCache rejects lpb loudly (lru passes "
          "the guard)")


def test_15_lpb_and_lru_evict_different_victims():
    """The load-bearing LPB claim test_11 does NOT make: on a tree where
    the lowest-ℓ(b) leaf and the LRU-oldest leaf are DIFFERENT nodes,
    LPB must evict the low-ℓ(b) leaf while LRU evicts the oldest — they
    pick OPPOSITE victims. test_11's `cold` leaf is both lpb=0 AND oldest,
    so it passes even if LPB silently fell back to LRU. Here the two
    orderings are crossed, so a silent fallback fails loudly.

    Per-token pages → B_b == s_b, so ℓ(b) = n_b·c_kv(s_b)/s_b:
      node_low : n_b=1,  last_access=99 (NEWEST) → min ℓ(b), LRU would spare
      node_high: n_b=20, last_access=1  (OLDEST) → max ℓ(b), LRU would evict
    LPB evicts node_low; LRU evicts node_high.
    """
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

    class _NoopAlloc:
        def free(self, _v):
            pass

    def _build(policy):
        p = CacheInitParams(
            disable=False, req_to_token_pool=None,
            token_to_kv_pool_allocator=None, page_size=1,
            enable_kv_cache_events=False, eviction_policy=policy,
        )
        c = RadixCache(p)
        c.token_to_kv_pool_allocator = _NoopAlloc()

        def _add(token_ids, hits, last_access):
            n = TreeNode()
            n.key = RadixKey(token_ids=token_ids, extra_key=None)
            n.value = torch.arange(len(token_ids), dtype=torch.int64)
            n.parent = c.root_node
            c.root_node.children[n.key.child_key(c.page_size)] = n
            c.evictable_leaves.add(n)
            for _ in range(hits):
                n.record_hit()
            n.last_access_time = last_access
            return n

        low = _add([1, 2, 3, 4], hits=1, last_access=99.0)
        high = _add([10, 11, 12, 13], hits=20, last_access=1.0)
        return c, low, high

    # LPB: evicts the min-ℓ(b) leaf = node_low, SPARES node_high.
    c, low, high = _build("lpb")
    c.evict(EvictParams(num_tokens=4))
    assert low not in c.evictable_leaves, (
        "LPB must evict the low-ℓ(b) leaf (node_low)"
    )
    assert high in c.evictable_leaves, (
        "LPB must SPARE the high-ℓ(b) leaf even though it is LRU-oldest — "
        "if this fails, LPB silently fell back to LRU"
    )

    # LRU: same tree → evicts the oldest = node_high (OPPOSITE victim).
    c2, low2, high2 = _build("lru")
    c2.evict(EvictParams(num_tokens=4))
    assert high2 not in c2.evictable_leaves, (
        "LRU must evict the oldest leaf (node_high)"
    )
    assert low2 in c2.evictable_leaves, "LRU must spare the newest leaf"

    print("  PASS  15 LPB and LRU evict OPPOSITE victims (LPB→low-ℓ(b), "
          "LRU→oldest); test_11's tie no longer hides a silent fallback")


def main() -> int:
    tests = [
        test_1_kv_lpb_priority_formula,
        test_2_lpb_strategy_returns_tuple_for_tiebreak,
        test_3_record_hit_wired_only_under_lpb,
        test_4_radix_cache_accepts_lpb_policy,
        test_5_mamba_lpb_includes_c_i_factor,
        test_6_record_hit_walks_multi_level_path,
        test_7_cache_does_not_stale_on_window_expiry,
        test_8_kv_lpb_pre_181_sentinel_ratio,
        test_9_mamba_radix_cache_honors_eviction_policy_boot_flag,
        test_10a_hi_radix_cache_record_hit_walks_prefix,
        test_10b_hi_mamba_radix_cache_record_hit_walks_prefix,
        test_11_radix_cache_evict_picks_lowest_lpb_priority,
        test_12_split_node_carries_hit_times,
        test_13_evict_paths_consult_should_use_lpb_not_env,
        test_14_swa_rejects_lpb_loudly,
        test_15_lpb_and_lru_evict_different_victims,
    ]
    print(f"\n#181 LPB eviction policy tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n#181: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
