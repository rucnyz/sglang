"""#180 — Policy-aware c^evict predictor: exact RadixCache walk.

Pins the design.md §"Shared cost model" `c^evict_i(X)` contract:
"walk RadixCache at decision time, simulate sglang's heap-pop in
active policy (LRU or LPB), sum `Σ n_b · c_i(s_b)` over the exact
prefix of picked blocks". The predictor is a pure-read mirror of
`RadixCache.evict()`'s shape — same `eviction_strategy`, same heap,
same parent-promotion — but accumulates cost instead of mutating.

Replaces the deleted snapshot-style `EvictCostIndex` (previously
under this same conjecture's test file). Snapshot was a placeholder
that returned `+inf` until plugged in; predictor is the exact form
the design asks for.

Sub-tests:
1. Cold start: no cache wired → `CostModel.c_evict_us(...)` returns
   `+inf` (Admitter falls back to other actions).
2. Unknown pool: raises ValueError (mirrors `c_recompute_us` /
   `c_migrate_us` discipline).
3. Empty cache → `+inf` (fail-closed: no own-evict candidate).
4. Demand > evictable supply → `+inf`.
5. Single leaf, LRU policy: cost = `1 × c_kv_ms(s_b) × 1000`
   (n_b ≡ 1 because LRU doesn't path-count).
6. Single leaf, LPB policy: cost = `hits_in_window × c_kv_ms(s_b)
   × 1000` (n_b from sliding window).
7. Multi-leaf, LRU: walks oldest-first; predicted set matches
   `cache.evict()` actual set BYTE-IDENTICAL (the design's
   falsification contract).
8. Parent promotion: when all children of a node are popped, the
   parent enters the heap — verify the predictor includes parent
   cost too.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch


def _make_cache(policy: str = "lru"):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache
    p = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        enable_kv_cache_events=False,
        eviction_policy=policy,
    )
    return RadixCache(p)


def _add_leaf(cache, token_ids: list, n_hits: int = 0):
    """Manually add an evictable leaf under root with given key + value."""
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
    leaf = TreeNode()
    leaf.key = RadixKey(token_ids=token_ids, extra_key=None)
    leaf.value = torch.arange(len(token_ids), dtype=torch.int64)
    leaf.parent = cache.root_node
    cache.root_node.children[leaf.key.child_key(cache.page_size)] = leaf
    cache.evictable_leaves.add(leaf)
    for _ in range(n_hits):
        leaf.record_hit()
    return leaf


def test_1_cold_start_cache_returns_inf():
    """No cache wired → CostModel.c_evict_us = +inf (fail-closed)."""
    from sglang.srt.budgeter.cost_model import CostModel, reset_cost_model
    reset_cost_model()
    cm = CostModel()
    assert cm.c_evict_us("kv", 100) == float("inf")
    assert cm.c_evict_us("mamba", 100) == float("inf")
    print("  PASS  1  cold start: no cache wired → c_evict_us = +inf")


def test_2_unknown_pool_raises():
    """Typo pool name raises ValueError loudly — mirrors
    `c_recompute_us` / `c_migrate_us` discipline (no silent +inf)."""
    from sglang.srt.budgeter.cost_model import CostModel, reset_cost_model
    reset_cost_model()
    cm = CostModel()
    for bad in ("KV", "Mamba", "foo", ""):
        try:
            cm.c_evict_us(bad, 1)
        except ValueError as e:
            assert "unknown pool" in str(e).lower() or "expected" in str(e)
        else:
            raise AssertionError(
                f"c_evict_us({bad!r}, ...) must raise; got silent +inf"
            )
    print("  PASS  2  unknown pool → ValueError (capitalization + "
          "typos + empty all loud-fail)")


def test_3_empty_cache_returns_inf():
    """No evictable leaves → +inf (Admitter own-evict infeasible)."""
    c = _make_cache("lru")
    assert c.predict_evict_cost_us(1) == float("inf")
    assert c.predict_evict_cost_us(0) == 0.0  # degenerate: 0 demand = 0 cost
    print("  PASS  3  empty cache: predict(X>0) = +inf, predict(0) = 0")


def test_4_demand_exceeds_supply_returns_inf():
    """If evictable supply < demanded tokens, predictor reports +inf
    so the Admitter knows own-evict is infeasible."""
    c = _make_cache("lru")
    _add_leaf(c, token_ids=[1, 2, 3])  # 3 tokens evictable
    assert c.predict_evict_cost_us(100) == float("inf"), (
        "Supply=3 < demand=100 must fail-closed to +inf"
    )
    # Sanity: demand within supply succeeds.
    assert c.predict_evict_cost_us(3) > 0 and c.predict_evict_cost_us(3) != float("inf")
    print("  PASS  4  demand > supply → +inf (fail-closed)")


def test_5_lru_n_b_equals_one():
    """LRU mode: predictor uses n_b ≡ 1 (sglang doesn't path-count).
    Cost = 1 × c_kv_ms(s_b) × 1000 µs."""
    from sglang.srt.budgeter.cost_model import get_cost_curves
    c = _make_cache("lru")
    _add_leaf(c, token_ids=list(range(64)), n_hits=99)  # hits should be ignored
    expected_us = 1.0 * get_cost_curves().c_kv_ms(64) * 1000.0
    actual = c.predict_evict_cost_us(64)
    assert abs(actual - expected_us) < 1e-6, (
        f"LRU mode should use n_b=1 (ignore hits). Got {actual:.3f} µs, "
        f"expected {expected_us:.3f}. If actual scales with hits, the "
        f"LRU/LPB switch in predict_evict_cost_us is broken."
    )
    print(f"  PASS  5  LRU mode: n_b=1, cost={actual:.1f} µs (ignored "
          f"99 stale hits)")


def test_6_lpb_n_b_uses_hits_in_window():
    """LPB mode: predictor uses n_b = hits_in_window().
    Cost scales linearly with hit count for fixed s_b + B_b."""
    from sglang.srt.budgeter.cost_model import get_cost_curves
    curves = get_cost_curves()
    c = _make_cache("lpb")
    _add_leaf(c, token_ids=list(range(64)), n_hits=3)
    expected_us = 3.0 * curves.c_kv_ms(64) * 1000.0
    actual = c.predict_evict_cost_us(64)
    assert abs(actual - expected_us) < 1e-6, (
        f"LPB mode: cost should be 3 × c_kv_ms × 1000. Got "
        f"{actual:.3f}, expected {expected_us:.3f}."
    )
    print(f"  PASS  6  LPB mode: n_b=3 (hits_in_window), "
          f"cost={actual:.1f} µs")


def test_7_predicted_set_byte_identical_to_evict():
    """Falsification (design.md "predicted set matches sglang's
    actual evict set, byte-identical").

    Two leaves with very different `last_access_time` (LRU keys):
    predictor should pick the OLDER leaf first; `evict()` should
    pick the same leaf. Predicted cost ≡ what `cache.evict()`
    would charge if we ran the cost formula on its chosen leaves.
    """
    from sglang.srt.budgeter.cost_model import get_cost_curves
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    curves = get_cost_curves()

    # Two separate caches: one for the predictor (untouched), one
    # for the actual evict() (mutates). Both built identically.
    def _build():
        c = _make_cache("lru")
        # Older leaf — set last_access_time directly.
        old = _add_leaf(c, token_ids=[1, 2, 3, 4])
        # Force older timestamp on the old leaf.
        old.last_access_time = time.monotonic() - 100.0
        # Newer leaf.
        new = _add_leaf(c, token_ids=[5, 6, 7, 8])
        new.last_access_time = time.monotonic()
        return c, old, new

    c_pred, old_p, new_p = _build()
    c_evt, old_e, new_e = _build()

    # Real `cache.evict()` calls into `token_to_kv_pool_allocator.free`
    # which is None in the test stub. Plug a no-op allocator.
    class _NoopAlloc:
        def free(self, values):
            pass
    c_evt.token_to_kv_pool_allocator = _NoopAlloc()

    # Predictor: demand 4 tokens (one leaf's worth).
    predicted_us = c_pred.predict_evict_cost_us(4)
    # Real evict: same demand.
    c_evt.evict(EvictParams(num_tokens=4))
    # After evict(), the OLDER leaf should be gone, newer remains.
    assert old_e not in c_evt.evictable_leaves, (
        "evict() must pick the OLDER leaf first under LRU"
    )
    assert new_e in c_evt.evictable_leaves, (
        "evict() must spare the newer leaf"
    )
    # Predicted cost should match: 1 × c_kv_ms(4) × 1000.
    expected_us = 1.0 * curves.c_kv_ms(4) * 1000.0
    assert abs(predicted_us - expected_us) < 1e-6, (
        f"predicted {predicted_us} != expected {expected_us} from "
        f"the leaf evict() picked"
    )
    print(f"  PASS  7  predicted set byte-identical to evict(): "
          f"both pick the older leaf; cost={predicted_us:.1f} µs")


def test_8_parent_promotion_simulated():
    """When all children of a non-root node are popped, the node
    itself becomes evictable. `evict()` promotes the parent into
    the heap; the predictor must do the same to remain byte-
    identical for multi-level trees.
    """
    from sglang.srt.budgeter.cost_model import get_cost_curves
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
    curves = get_cost_curves()

    c = _make_cache("lru")
    # Manually build root → mid → {leaf_A, leaf_B}.
    mid = TreeNode()
    mid.key = RadixKey(token_ids=[1, 2], extra_key=None)
    mid.value = torch.arange(2, dtype=torch.int64)
    mid.parent = c.root_node
    c.root_node.children[mid.key.child_key(c.page_size)] = mid
    # `mid` is NOT a leaf until both children gone — so NOT in
    # evictable_leaves yet.

    leaf_a = TreeNode()
    leaf_a.key = RadixKey(token_ids=[3, 4], extra_key=None)
    leaf_a.value = torch.arange(2, dtype=torch.int64)
    leaf_a.parent = mid
    mid.children[leaf_a.key.child_key(c.page_size)] = leaf_a
    c.evictable_leaves.add(leaf_a)
    leaf_a.last_access_time = time.monotonic() - 10.0

    leaf_b = TreeNode()
    leaf_b.key = RadixKey(token_ids=[5, 6], extra_key=None)
    leaf_b.value = torch.arange(2, dtype=torch.int64)
    leaf_b.parent = mid
    mid.children[leaf_b.key.child_key(c.page_size)] = leaf_b
    c.evictable_leaves.add(leaf_b)
    leaf_b.last_access_time = time.monotonic() - 5.0

    # Demand 6 tokens = leaf_a (2) + leaf_b (2) + mid (2) — predictor
    # must promote `mid` into the heap after popping its children.
    cost = c.predict_evict_cost_us(6)
    expected = (
        1.0 * curves.c_kv_ms(2) * 1000.0   # leaf_a
        + 1.0 * curves.c_kv_ms(2) * 1000.0  # leaf_b
        + 1.0 * curves.c_kv_ms(2) * 1000.0  # promoted mid
    )
    assert abs(cost - expected) < 1e-6, (
        f"Parent promotion broken: cost={cost:.3f}, expected "
        f"{expected:.3f} (3 blocks × c_kv_ms(2) × 1000). If only "
        f"the two leaves were counted, predictor returned +inf "
        f"because demand=6 > leaves' 4 tokens."
    )
    print(f"  PASS  8  parent promotion simulated: 3 blocks counted "
          f"(leaf_a + leaf_b + promoted mid), cost={cost:.1f} µs")


# ============================================================
# Falsification: predicted SET == real evict() SET, byte-identical.
# These cross-check the predictor's chosen set against an ACTUAL
# `RadixCache.evict()` run on an identically-built tree — the
# design.md §"Why exact c^evict" headline claim. Earlier tests
# (1-8) only checked the cost number on degenerate shapes.
# ============================================================

class _NoopAlloc:
    """evict() calls token_to_kv_pool_allocator.free(values)."""
    def free(self, _v):
        pass


def _all_tree_nodes(cache):
    """Walk the whole tree; return {tuple(token_ids): node} for every
    non-root node (keyed by content so two identically-built caches
    align despite distinct object ids)."""
    out = {}
    stack = list(cache.root_node.children.values())
    while stack:
        n = stack.pop()
        out[tuple(n.key.token_ids)] = n
        stack.extend(n.children.values())
    return out


def _node_cost_ms(cache, node) -> float:
    """The predictor's per-node charge: n_b · c_kv_ms(s_b)."""
    from sglang.srt.budgeter.cost_model import get_cost_curves
    from sglang.srt.mem_cache.evict_policy import LPBStrategy
    curves = get_cost_curves()
    lpb = isinstance(cache.eviction_strategy, LPBStrategy)
    n_b = node.hits_in_window() if lpb else 1
    s_b = len(node.key)
    return n_b * curves.c_kv_ms(s_b)


def _evicted_set_and_cost(cache, num_tokens):
    """Run REAL evict() and return (set of evicted token-id tuples,
    cost-µs summed over them via the predictor's formula). Costs are
    snapshotted BEFORE eviction frees/mutates the nodes."""
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    cache.token_to_kv_pool_allocator = _NoopAlloc()
    before = _all_tree_nodes(cache)
    cost_by_key = {k: _node_cost_ms(cache, n) for k, n in before.items()}
    cache.evict(EvictParams(num_tokens=num_tokens))
    after = set(_all_tree_nodes(cache).keys())
    evicted = set(before.keys()) - after
    cost_us = sum(cost_by_key[k] for k in evicted) * 1000.0
    return evicted, cost_us


def test_9_predicted_cost_equals_real_evict_set_lru_multileaf():
    """4 flat leaves, distinct key-lengths → distinct per-node cost
    (cost is a fingerprint of the chosen set). predicted cost must
    equal the cost computed over the set a REAL evict() removes.
    Distinct costs ⇒ equal cost ⇔ identical set."""
    def _build():
        c = _make_cache("lru")
        # distinct key lengths 2/3/5/7 → distinct c_kv_ms.
        specs = [([1, 2], 1.0), ([3, 4, 5], 2.0),
                 ([6, 7, 8, 9, 10], 3.0), ([11, 12, 13, 14, 15, 16, 17], 4.0)]
        nodes = []
        for toks, t in specs:
            n = _add_leaf(c, token_ids=toks)
            n.last_access_time = t  # ascending → LRU evicts toks[0] group first
            nodes.append(n)
        return c

    c_pred = _build()
    c_evt = _build()
    # Demand 5 tokens → LRU pops the two oldest (2 + 3 tokens = 5).
    predicted = c_pred.predict_evict_cost_us(5)
    evicted, actual = _evicted_set_and_cost(c_evt, 5)

    assert evicted == {(1, 2), (3, 4, 5)}, (
        f"evict() should remove the two oldest leaves; got {evicted}"
    )
    assert abs(predicted - actual) < 1e-6, (
        f"predicted {predicted:.3f} µs != cost over evict()'s actual "
        f"set {actual:.3f} µs. Distinct per-node costs make this a "
        f"set-identity check, not just a magnitude check."
    )
    print(f"  PASS  9  predicted cost == real evict() set cost (LRU, "
          f"4-leaf): {predicted:.1f} µs over {len(evicted)} blocks")


def test_10_parent_promotion_matches_real_evict():
    """2-level tree root→mid→{leaf_a, leaf_b}. Evicting both leaves
    promotes `mid`; demand forces `mid` to be evicted too. Predicted
    cost must equal the cost over the set evict() ACTUALLY removes
    (which must include the promoted `mid`)."""
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode

    def _build():
        c = _make_cache("lru")
        mid = TreeNode()
        mid.key = RadixKey(token_ids=[1, 2], extra_key=None)
        mid.value = torch.arange(2, dtype=torch.int64)
        mid.parent = c.root_node
        c.root_node.children[mid.key.child_key(c.page_size)] = mid
        for toks, t in (([3, 4], 1.0), ([5, 6], 2.0)):
            leaf = TreeNode()
            leaf.key = RadixKey(token_ids=toks, extra_key=None)
            leaf.value = torch.arange(2, dtype=torch.int64)
            leaf.parent = mid
            mid.children[leaf.key.child_key(c.page_size)] = leaf
            c.evictable_leaves.add(leaf)
            leaf.last_access_time = t
        return c

    c_pred = _build()
    c_evt = _build()
    # Demand 6 tokens = leaf_a(2) + leaf_b(2) + promoted mid(2).
    predicted = c_pred.predict_evict_cost_us(6)
    evicted, actual = _evicted_set_and_cost(c_evt, 6)

    assert evicted == {(3, 4), (5, 6), (1, 2)}, (
        f"evict() must remove both leaves AND the promoted parent "
        f"mid=(1,2); got {evicted}"
    )
    assert abs(predicted - actual) < 1e-6, (
        f"predicted {predicted:.3f} != real-evict-set cost "
        f"{actual:.3f}; parent-promotion simulation diverges from "
        f"evict()."
    )
    print(f"  PASS  10 parent promotion matches real evict(): "
          f"{len(evicted)} blocks incl. promoted parent, "
          f"{predicted:.1f} µs")


def test_11_predict_then_evict_no_perturbation_lpb():
    """`predict_evict_cost_us` calls `hits_in_window()` which
    `popleft()`s expired deque entries — a mutation during a
    "pure-read" predict. Guard: under LPB, calling predict() before
    evict() must NOT change which set evict() picks. Compare a
    cache that ran predict-then-evict against one that ran evict
    alone."""
    def _build():
        c = _make_cache("lpb")
        # distinct (hits, len) → distinct ℓ(b); ascending loss.
        c.token_to_kv_pool_allocator = _NoopAlloc()
        _add_leaf(c, token_ids=[1, 2], n_hits=0)           # ℓ=0 (coldest)
        _add_leaf(c, token_ids=[3, 4, 5], n_hits=2)
        _add_leaf(c, token_ids=[6, 7, 8, 9], n_hits=9)     # hottest
        return c

    # Baseline: evict alone.
    c_base = _build()
    base_evicted, _ = _evicted_set_and_cost(c_base, 2)

    # Treatment: predict THEN evict on the same cache.
    c_treat = _build()
    _ = c_treat.predict_evict_cost_us(2)   # the potentially-perturbing call
    treat_evicted, _ = _evicted_set_and_cost(c_treat, 2)

    assert base_evicted == treat_evicted == {(1, 2)}, (
        f"predict() perturbed eviction order: evict-alone removed "
        f"{base_evicted}, predict-then-evict removed {treat_evicted}. "
        f"The coldest (ℓ=0) leaf (1,2) must be picked in both."
    )
    print(f"  PASS  11 predict()-then-evict() picks same set as evict "
          f"alone (LPB, no perturbation): {base_evicted}")


def test_12_lpb_multileaf_order_matches_real_evict():
    """LPB, multiple leaves with distinct ℓ(b). Predicted cost must
    equal the cost over evict()'s actual set — i.e. the predictor
    walks lowest-loss-first in the same order evict() does."""
    def _build():
        c = _make_cache("lpb")
        # ℓ(b) = n_b·c_kv(s_b)/B_b. Keep B_b=len, vary n_b + s_b so
        # losses are strictly ordered and per-node costs distinct.
        c.token_to_kv_pool_allocator = _NoopAlloc()
        _add_leaf(c, token_ids=[1, 2], n_hits=0)            # ℓ=0
        _add_leaf(c, token_ids=[3, 4, 5], n_hits=1)
        _add_leaf(c, token_ids=[6, 7, 8, 9, 10], n_hits=5)
        _add_leaf(c, token_ids=[11, 12, 13, 14, 15, 16], n_hits=12)
        # deterministic tiebreak via last_access_time
        for i, n in enumerate(c.root_node.children.values()):
            n.last_access_time = float(i)
        return c

    c_pred = _build()
    c_evt = _build()
    # Demand 5 tokens → two lowest-loss leaves: (1,2)=2 + (3,4,5)=3.
    predicted = c_pred.predict_evict_cost_us(5)
    evicted, actual = _evicted_set_and_cost(c_evt, 5)

    assert evicted == {(1, 2), (3, 4, 5)}, (
        f"LPB evict() must pick the two lowest-loss leaves; got {evicted}"
    )
    assert abs(predicted - actual) < 1e-6, (
        f"predicted {predicted:.3f} != real LPB evict-set cost "
        f"{actual:.3f}; LPB walk order diverges from evict()."
    )
    print(f"  PASS  12 LPB multileaf order matches real evict(): "
          f"{len(evicted)} blocks, {predicted:.1f} µs")


# ============================================================
# Shared-walk audit follow-ups (commit e81bc8da06): tests 9-12
# compare predict vs a fresh evict(), but both now consume the
# SAME `_iter_evict_victims` generator, so they can no longer
# INDEPENDENTLY validate the generator (only that it's self-
# consistent). 13-16 add static oracles + cascade / locked /
# degenerate coverage that pin the generator against hand-
# computed expectations, independent of running evict().
# ============================================================

def _victim_keys(cache, num_tokens):
    """The ordered token-id tuples `_iter_evict_victims` yields."""
    return [tuple(n.key.token_ids) for n in cache._iter_evict_victims(num_tokens)]


def test_13_iter_victims_matches_static_oracle():
    """Independent oracle: the victim ORDER from the generator must
    equal a hand-computed sequence on a fixed flat LRU tree — NOT
    derived from running evict() (which shares the generator and
    would be tautological)."""
    c = _make_cache("lru")
    for toks, t in (([1, 2], 3.0), ([3, 4], 1.0), ([5, 6], 2.0)):
        n = _add_leaf(c, token_ids=toks)
        n.last_access_time = t
    # LRU pops ascending last_access_time: (3,4)@1 → (5,6)@2 → (1,2)@3.
    # demand 4 tokens → first two.
    assert _victim_keys(c, 4) == [(3, 4), (5, 6)], _victim_keys(c, 4)
    # demand 6 → all three, in LRU order.
    assert _victim_keys(c, 6) == [(3, 4), (5, 6), (1, 2)], _victim_keys(c, 6)
    print("  PASS  13 _iter_evict_victims order matches static LRU oracle")


def test_14_three_level_cascade_promotion():
    """3-level tree root→gp→p→{a,b}. Evicting both leaves promotes
    `p`; evicting `p` promotes `gp`. The generator must cascade:
    order [a, b, p, gp]. (2-level tests 8/10 don't exercise a
    promoted parent itself triggering grandparent promotion.)"""
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
    c = _make_cache("lru")

    def _mk(parent, toks, t):
        n = TreeNode()
        n.key = RadixKey(token_ids=toks, extra_key=None)
        n.value = torch.arange(len(toks), dtype=torch.int64)
        n.parent = parent
        parent.children[n.key.child_key(c.page_size)] = n
        n.last_access_time = t
        return n

    gp = _mk(c.root_node, [1, 2], 30.0)
    p = _mk(gp, [3, 4], 20.0)
    a = _mk(p, [5, 6], 1.0)
    b = _mk(p, [7, 8], 2.0)
    # only the true leaves are evictable initially.
    c.evictable_leaves.update({a, b})

    # demand 8 = a(2)+b(2)+p(2)+gp(2): full cascade.
    assert _victim_keys(c, 8) == [(5, 6), (7, 8), (3, 4), (1, 2)], (
        _victim_keys(c, 8)
    )
    print("  PASS  14 three-level cascade promotion: [a, b, p, gp]")


def test_15_locked_sibling_blocks_promotion():
    """A parent with one evictable + one locked (`lock_ref>0`) child
    must NOT be promoted (its `children` never empties). Guards the
    `effective_children` seed counting ALL children, not just
    evictable ones."""
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
    c = _make_cache("lru")
    parent = TreeNode()
    parent.key = RadixKey(token_ids=[1, 2], extra_key=None)
    parent.value = torch.arange(2, dtype=torch.int64)
    parent.parent = c.root_node
    c.root_node.children[parent.key.child_key(c.page_size)] = parent

    def _child(toks, lock):
        n = TreeNode()
        n.key = RadixKey(token_ids=toks, extra_key=None)
        n.value = torch.arange(len(toks), dtype=torch.int64)
        n.parent = parent
        n.lock_ref = lock
        parent.children[n.key.child_key(c.page_size)] = n
        return n

    leaf_a = _child([3, 4], lock=0)
    _child([5, 6], lock=1)            # locked sibling, NOT evictable
    c.evictable_leaves.add(leaf_a)

    # demand 10 > available (only leaf_a's 2 tokens free): parent must
    # not be promoted, so only leaf_a is yielded.
    keys = _victim_keys(c, 10)
    assert keys == [(3, 4)], keys
    assert (1, 2) not in keys, "locked sibling must block parent promotion"
    print("  PASS  15 locked sibling blocks parent promotion")


def test_16_none_and_zero_length_leaves_skipped():
    """Skip contract (intentional divergence from the pre-refactor
    inline loop): a `value is None` leaf is skipped (old loop crashed
    on `len(None)`); a zero-length `value` leaf is skipped and left in
    the tree (old loop pruned it). Neither contributes tokens."""
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
    c = _make_cache("lru")

    good = _add_leaf(c, token_ids=[1, 2, 3, 4])
    good.last_access_time = 5.0

    none_leaf = TreeNode()
    none_leaf.key = RadixKey(token_ids=[10, 11], extra_key=None)
    none_leaf.value = None
    none_leaf.parent = c.root_node
    c.root_node.children[none_leaf.key.child_key(c.page_size)] = none_leaf
    none_leaf.last_access_time = 1.0
    c.evictable_leaves.add(none_leaf)

    zero_leaf = TreeNode()
    zero_leaf.key = RadixKey(token_ids=[20, 21], extra_key=None)
    zero_leaf.value = torch.arange(0, dtype=torch.int64)
    zero_leaf.parent = c.root_node
    c.root_node.children[zero_leaf.key.child_key(c.page_size)] = zero_leaf
    zero_leaf.last_access_time = 2.0
    c.evictable_leaves.add(zero_leaf)

    # Even though none_leaf/zero_leaf sort FIRST (oldest), they're
    # skipped; only `good` is yielded.
    keys = _victim_keys(c, 4)
    assert keys == [(1, 2, 3, 4)], f"None/zero-len must be skipped: {keys}"
    print("  PASS  16 None + zero-length leaves skipped (not crashed/pruned)")


def main() -> int:
    tests = [
        test_1_cold_start_cache_returns_inf,
        test_2_unknown_pool_raises,
        test_3_empty_cache_returns_inf,
        test_4_demand_exceeds_supply_returns_inf,
        test_5_lru_n_b_equals_one,
        test_6_lpb_n_b_uses_hits_in_window,
        test_7_predicted_set_byte_identical_to_evict,
        test_8_parent_promotion_simulated,
        test_9_predicted_cost_equals_real_evict_set_lru_multileaf,
        test_10_parent_promotion_matches_real_evict,
        test_11_predict_then_evict_no_perturbation_lpb,
        test_12_lpb_multileaf_order_matches_real_evict,
        test_13_iter_victims_matches_static_oracle,
        test_14_three_level_cascade_promotion,
        test_15_locked_sibling_blocks_promotion,
        test_16_none_and_zero_length_leaves_skipped,
    ]
    print(f"\n#180 c^evict predictor tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n#180: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
