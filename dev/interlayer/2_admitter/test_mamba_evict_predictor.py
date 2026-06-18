"""#259 — MambaRadixCache c^evict predictor (full / KV side).

Validates that `MambaRadixCache.predict_evict_cost_us(pool="kv")`
prices the EXACT set `evict_full` evicts — they share the pure-read
`_iter_evict_full_victims` generator (single source of truth, the
ideal-architecture refactor), so the priced set is byte-identical to
the evicted set by construction. These tests pin that the generator
itself is correct via independent oracles + a real `evict_full` run.

Cost basis: a hybrid full-tree leaf carries both a KV value and a
mamba snapshot; evicting it loses both, so recompute is
`c_kv(s_b) + c_m(s_b)` — the same `c_i` decomposition
`eviction_priority()` uses for the LPB sort key (design.md line 578
+ §"Why exact c^evict").

NOTE on the LRU-promotion semantics: the shared generator selects
victims by GLOBAL-LRU (heapify by `last_access_time`) + parent-
promotion — the same global ordering the LPB path's heap already
uses, and the clean/consistent semantics. The pre-refactor
`full_lru_list` leaf-chain walk was an APPROXIMATION of this (it
followed `get_prev_leaf` and only re-fetched the global leaf-LRU on a
narrow promotion-detect condition), so in rare promotion edges the
two could pick a different victim order. `_insert_helper` stamps
`last_access_time` root→leaf (parent BEFORE child), so a promoted
parent generally has a LOWER time than its just-evicted children and
global-LRU surfaces it earlier than the chain walk would. This is an
intentional move to consistent global-LRU; own_evict cost stays
fail-closed-safe regardless. test_5 pins the promotion order;
test_10 pins the global-LRU-vs-old-chain divergence explicitly.

Sub-tests:
1. static LRU oracle for `_plan_full_eviction` victim order.
2. predict cost == cost over the set a REAL `evict_full` removes.
3. fail-closed: demand > evictable supply → +inf.
4. LPB order: lowest `eviction_priority()` first.
5. parent promotion: 2-level tree, full eviction promotes the parent.
6. predict-then-evict_full picks the same set (no perturbation).
7. pool='mamba' predict == real evict_mamba set cost (#275/#270 drain).
8. unknown pool raises ValueError.
9. tombstone parent: not a victim, swept internally, priced byte-exact.
10. global-LRU vs old chain-walk divergence (intentional), pinned.
11. wiring gate: exact-type includes base, excludes subclasses.
12. multi-tombstone cascade (chain ≥2): both swept, byte-exact.
13. LPB promoted-parent (with hits): predict no-perturbation.
14. swept tombstone at the demand boundary: exact token count.
15. partial tombstone (child remains): NOT swept, survives.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch


class _NoopKVAlloc:
    def free(self, _v):
        pass


class _NoopMambaPool:
    def free(self, _v):
        pass


class _StubReqToToken:
    def __init__(self):
        self.mamba_pool = _NoopMambaPool()


def _make_hybrid_cache(policy: str = "lru"):
    """A MambaRadixCache with __init__ bypassed — just the fields the
    eviction paths read, plus stub allocator/pool so a real
    `evict_full` can run without GPU."""
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, LRUList
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode

    c = MambaRadixCache.__new__(MambaRadixCache)
    c.page_size = 1
    c.disable = False
    c.eviction_policy = policy
    c.full_lru_list = LRUList(mamba=False)
    c.mamba_lru_list = LRUList(mamba=True)
    c.token_to_kv_pool_allocator = _NoopKVAlloc()
    c.req_to_token_pool = _StubReqToToken()
    # counters touched by _evict_leaf_node via record_recovery_len_*
    c.full_evictable_size_ = 0
    c.mamba_evictable_size_ = 0
    # EWMA accumulators record_recovery_len_kv/rec read+write (real
    # __init__ seeds them 0.0); a real evict_full/evict_mamba touches them.
    c._slow_recovery_len_kv_ewma = 0.0
    c._slow_recovery_len_rec_ewma = 0.0
    # Cumulative cache-eviction tallies evict_full/evict_mamba increment
    # at their tail (the Budgeter's grow-side signal); real __init__ → 0.
    c._cumulative_evicted_kv_tokens = 0
    c._cumulative_evicted_mamba_slots = 0
    root = TreeNode()
    root.key = RadixKey([], None)
    root.value = []
    root.full_lock_ref = 1
    root.mamba_lock_ref = 1
    c.root_node = root
    return c


def _add_node(cache, parent, token_ids, t, n_hits=0):
    """Add a hybrid node (KV value + mamba snapshot) under `parent`,
    register it in both LRU lists with access time `t`."""
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode
    from sglang.srt.mem_cache.radix_cache import RadixKey
    n = TreeNode()
    n.key = RadixKey(token_ids, None)
    n.value = torch.arange(len(token_ids), dtype=torch.int64)
    n.mamba_value = torch.tensor([0], dtype=torch.int64)
    n.parent = parent
    parent.children[n.key.child_key(cache.page_size)] = n
    n.last_access_time = t
    cache.full_lru_list.insert_mru(n)
    cache.mamba_lru_list.insert_mru(n)
    for _ in range(n_hits):
        n.record_hit()
    return n


def _victim_keys(cache, num_tokens):
    victims, _swept = cache._plan_full_eviction(num_tokens)
    return [tuple(n.key.token_ids) for n in victims]


def _swept_keys(cache, num_tokens):
    _victims, swept = cache._plan_full_eviction(num_tokens)
    return [tuple(n.key.token_ids) for n in swept]


def _all_full_nodes(cache):
    out = {}
    stack = list(cache.root_node.children.values())
    while stack:
        n = stack.pop()
        out[tuple(n.key.token_ids)] = n
        stack.extend(n.children.values())
    return out


def test_1_static_lru_oracle():
    """Generator order == hand-computed global-LRU on a flat tree."""
    c = _make_hybrid_cache("lru")
    _add_node(c, c.root_node, [1, 2], t=3.0)
    _add_node(c, c.root_node, [3, 4], t=1.0)
    _add_node(c, c.root_node, [5, 6], t=2.0)
    assert _victim_keys(c, 4) == [(3, 4), (5, 6)], _victim_keys(c, 4)
    assert _victim_keys(c, 6) == [(3, 4), (5, 6), (1, 2)], _victim_keys(c, 6)
    print("  PASS  1  static LRU oracle: lowest last_access first")


def test_2_predict_equals_real_evict_full_set():
    """predict cost == cost summed over the set a REAL evict_full
    removes (distinct per-node costs ⇒ set fingerprint)."""
    from sglang.srt.budgeter.cost_model import get_cost_curves
    curves = get_cost_curves()

    def _build():
        c = _make_hybrid_cache("lru")
        # distinct key lengths → distinct c_kv+c_m per node.
        _add_node(c, c.root_node, [1, 2], t=3.0)
        _add_node(c, c.root_node, [3, 4, 5], t=1.0)
        _add_node(c, c.root_node, [6, 7, 8, 9, 10], t=2.0)
        return c

    c_pred = _build()
    c_evt = _build()
    predicted = c_pred.predict_evict_cost_us(5, pool="kv")

    before = set(_all_full_nodes(c_evt).keys())
    cost_by_key = {}
    for k, n in _all_full_nodes(c_evt).items():
        s_b = len(n.key)
        cost_by_key[k] = curves.c_kv_ms(s_b) + curves.c_m_ms(s_b)  # n_b=1 (lru)
    c_evt.evict_full(5)
    after = set(_all_full_nodes(c_evt).keys())
    evicted = before - after
    actual = sum(cost_by_key[k] for k in evicted) * 1000.0

    # LRU demand 5 → (3,4,5)@1 (3 tok) + (6,7,8,9,10)@2 (5 tok) = 8 ≥ 5.
    assert evicted == {(3, 4, 5), (6, 7, 8, 9, 10)}, evicted
    assert abs(predicted - actual) < 1e-6, (
        f"predicted {predicted:.3f} != real evict_full set cost "
        f"{actual:.3f}"
    )
    print(f"  PASS  2  predict == real evict_full set cost: "
          f"{predicted:.1f} µs over {len(evicted)} leaves")


def test_3_fail_closed_demand_exceeds_supply():
    c = _make_hybrid_cache("lru")
    _add_node(c, c.root_node, [1, 2], t=1.0)   # 2 tokens evictable
    assert c.predict_evict_cost_us(100, pool="kv") == float("inf")
    val = c.predict_evict_cost_us(2, pool="kv")
    assert val > 0 and val != float("inf")
    print("  PASS  3  fail-closed: demand > supply → +inf")


def test_4_lpb_order_lowest_priority_first():
    """LPB: generator pops lowest eviction_priority() first."""
    c = _make_hybrid_cache("lpb")
    # ℓ(b) = n_b·(c_kv+c_m)/B_b. Vary hits so losses are ordered.
    _add_node(c, c.root_node, [1, 2], t=1.0, n_hits=0)        # ℓ=0 (coldest)
    _add_node(c, c.root_node, [3, 4], t=2.0, n_hits=5)
    _add_node(c, c.root_node, [5, 6], t=3.0, n_hits=20)       # hottest
    # demand 2 → only the coldest (1,2).
    assert _victim_keys(c, 2) == [(1, 2)], _victim_keys(c, 2)
    print("  PASS  4  LPB order: coldest (ℓ=0) leaf first")


def test_5_parent_promotion_global_lru():
    """2-level tree root→mid→{a,b}; mid is a hybrid internal node.
    Evicting both leaves promotes mid; demand forces mid too. Global-
    LRU order = [a, b, mid] (a,b oldest; mid promoted after)."""
    c = _make_hybrid_cache("lru")
    mid = _add_node(c, c.root_node, [1, 2], t=10.0)
    _add_node(c, mid, [3, 4], t=1.0)
    _add_node(c, mid, [5, 6], t=2.0)
    # mid has children → not an initial leaf; surfaces via promotion.
    assert _victim_keys(c, 6) == [(3, 4), (5, 6), (1, 2)], _victim_keys(c, 6)

    # And a REAL evict_full frees all three.
    c2 = _make_hybrid_cache("lru")
    mid2 = _add_node(c2, c2.root_node, [1, 2], t=10.0)
    _add_node(c2, mid2, [3, 4], t=1.0)
    _add_node(c2, mid2, [5, 6], t=2.0)
    before = set(_all_full_nodes(c2).keys())
    c2.evict_full(6)
    evicted = before - set(_all_full_nodes(c2).keys())
    assert evicted == {(3, 4), (5, 6), (1, 2)}, evicted
    print("  PASS  5  parent promotion (global-LRU): [a, b, mid]")


def test_6_predict_then_evict_no_perturbation():
    """predict_evict_cost_us must not perturb which set evict_full
    picks (LPB hits_in_window popleft guard)."""
    def _build():
        c = _make_hybrid_cache("lpb")
        _add_node(c, c.root_node, [1, 2], t=1.0, n_hits=0)
        _add_node(c, c.root_node, [3, 4], t=2.0, n_hits=3)
        _add_node(c, c.root_node, [5, 6], t=3.0, n_hits=9)
        return c

    c_base = _build()
    before = set(_all_full_nodes(c_base).keys())
    c_base.evict_full(2)
    base_evicted = before - set(_all_full_nodes(c_base).keys())

    c_treat = _build()
    _ = c_treat.predict_evict_cost_us(2, pool="kv")
    before_t = set(_all_full_nodes(c_treat).keys())
    c_treat.evict_full(2)
    treat_evicted = before_t - set(_all_full_nodes(c_treat).keys())

    assert base_evicted == treat_evicted == {(1, 2)}, (
        f"predict perturbed eviction: base={base_evicted} "
        f"treat={treat_evicted}"
    )
    print(f"  PASS  6  predict-then-evict_full no perturbation: "
          f"{base_evicted}")


def test_7_pool_mamba_predict_equals_real_evict_mamba():
    """pool='mamba' (#275/#270 reuse-aware drain cost) prices the EXACT
    set `evict_mamba` frees: an internal tombstone (c_m), a leaf
    (c_kv + c_m), and the tombstone-leaf cascade (c_kv). CPU mirror of
    test_mamba_real_pool::test_6. Build root -> mid=[1,2] (oldest) ->
    leaf=[3,4]; draining 2 mamba slots LRU-first tombstones mid (mamba),
    then leaf-evicts the child (KV+mamba) whose cascade sweeps mid (KV)."""
    from sglang.srt.budgeter.cost_model import get_cost_curves

    curves = get_cost_curves()
    c = _make_hybrid_cache("lru")
    mid = _add_node(c, c.root_node, [1, 2], t=1.0)       # oldest, internal
    _add_node(c, mid, [3, 4], t=2.0)                     # child leaf, newest

    before = dict(_all_full_nodes(c))
    use_lpb = c._should_use_lpb()
    snap = {
        k: (len(n.key), n.mamba_value is not None,
            n.hits_in_window() if use_lpb else 1)
        for k, n in before.items()
    }

    predicted = c.predict_evict_cost_us(2, pool="mamba")
    freed = c.evict_mamba(2)
    after = _all_full_nodes(c)

    actual_ms = 0.0
    for k, (s_b, had_mamba, n_b) in snap.items():
        c_kv = curves.c_kv_ms(s_b)
        c_m = curves.c_m_ms(s_b)
        if k not in after:
            actual_ms += n_b * (c_kv + (c_m if had_mamba else 0.0))
        elif had_mamba and after[k].mamba_value is None:
            # Internal mamba eviction (snapshot dropped, KV kept): recovering
            # the snapshot needs a full prefix re-prefill (interleaved layers),
            # so it costs the whole-prefix total c_kv + c_m — not c_m alone
            # (#298; matches the single folded curve and eviction_priority).
            actual_ms += n_b * (c_kv + c_m)
    actual = actual_ms * 1000.0

    assert freed == 2, f"evict_mamba freed {freed} != 2"
    assert abs(predicted - actual) < 1e-6, (
        f"predict('mamba') {predicted:.4f} != real evict_mamba cost {actual:.4f}"
    )
    print(
        f"  PASS  7  predict('mamba') == real evict_mamba cost incl cascade: "
        f"{predicted:.1f}us, freed {freed} slots"
    )


def test_8_unknown_pool_raises_value_error():
    c = _make_hybrid_cache("lru")
    for bad in ("KV", "foo", ""):
        try:
            c.predict_evict_cost_us(1, pool=bad)
        except (ValueError, NotImplementedError):
            pass
        else:
            raise AssertionError(f"pool={bad!r} must raise")
    print("  PASS  8  unknown pool raises")


def _tombstone(node):
    """Turn a hybrid internal node into a tombstone (mamba evicted,
    KV value retained) — what evict_mamba's `_tombstone_internal_node`
    produces: mamba_value=None + removed from the mamba LRU list."""
    node.mamba_value = None


def _make_tombstone(cache, node):
    """Production-faithful tombstoning: drop from the mamba LRU list
    FIRST (while mamba_value present), THEN null mamba_value — the
    order evict_mamba uses around `_tombstone_internal_node`."""
    cache.mamba_lru_list.remove_node(node)
    node.mamba_value = None


def test_9_tombstone_parent_not_yielded_no_crash():
    """Audit C.2 reproducer (#259 (3/3)): a tombstone internal node
    (mamba_value=None, KV value retained, still in full_lru_list, left
    by a prior evict_mamba) that becomes childless must NOT be yielded
    as an `_evict_leaf_node` victim — `_evict_leaf_node` asserts
    `mamba_value is not None`. The OLD evict_full swept tombstones
    internally; the shared-walk generator must cascade past them.

    Pre-fix: evict_full(6) promotes the tombstone `mid` → assert
    'leaf node mamba value is not None' → crash. Post-fix: the
    generator skips the tombstone (cascades), real evict_full sweeps
    it internally, no crash."""
    from sglang.srt.mem_cache.mamba_radix_cache import LRUList

    def _build():
        c = _make_hybrid_cache("lru")
        # root → mid (TOMBSTONE) → {a, b}
        mid = _add_node(c, c.root_node, [1, 2], t=10.0)
        # Mirror evict_mamba's internal-node tombstoning: drop from the
        # mamba LRU list FIRST (while mamba_value is still present),
        # THEN null mamba_value.
        c.mamba_lru_list.remove_node(mid)
        _tombstone(mid)
        _add_node(c, mid, [3, 4], t=1.0)
        _add_node(c, mid, [5, 6], t=2.0)
        return c, mid

    # Plan: tombstone is NOT a victim but IS in swept_tombstones.
    c, mid = _build()
    assert set(_victim_keys(c, 6)) == {(3, 4), (5, 6)}, _victim_keys(c, 6)
    assert (1, 2) in _swept_keys(c, 6), (
        f"tombstone mid=(1,2) must be in swept_tombstones; got "
        f"{_swept_keys(c, 6)}"
    )

    # Real evict_full must NOT crash and frees the leaves (+ sweeps the
    # tombstone internally). delta includes the swept tombstone's KV.
    c2, mid2 = _build()
    before = set(_all_full_nodes(c2).keys())
    n = c2.evict_full(6)
    after = set(_all_full_nodes(c2).keys())
    assert (3, 4) not in after and (5, 6) not in after, "leaves not freed"
    assert (1, 2) not in after, "tombstone parent not swept"
    assert n == 6, f"expected 6 tokens freed (2+2 leaves + 2 tombstone), got {n}"

    # Predictor must now COUNT the swept tombstone's c_kv (byte-exact,
    # cost = 2 leaves·(c_kv+c_m)(2) + swept tombstone·c_kv(2).
    from sglang.srt.budgeter.cost_model import get_cost_curves
    cur = get_cost_curves()
    c3, _ = _build()
    predicted = c3.predict_evict_cost_us(6, pool="kv")
    expected = (
        2 * (cur.c_kv_ms(2) + cur.c_m_ms(2))   # leaves (3,4),(5,6)
        + 1 * cur.c_kv_ms(2)                    # swept tombstone (1,2)
    ) * 1000.0
    assert abs(predicted - expected) < 1e-6, (
        f"predictor must include swept tombstone c_kv: got {predicted:.3f}, "
        f"expected {expected:.3f}"
    )
    print(f"  PASS  9  tombstone not a victim but swept + priced "
          f"(no crash); freed {n} tok, cost {predicted:.1f} µs")


def test_10_global_lru_vs_old_chain_divergence():
    """Audit C.3 (#263): the shared-walk generator selects victims by
    GLOBAL-LRU (heapify by last_access_time + promotion). This DIFFERS
    from the pre-refactor `full_lru_list` leaf-chain walk in promotion
    edges — pinned here, transparently.

    Tree:  root → p(t=0.5) → leaf_x(t=1.0)
           root → leaf_y(t=2.0)
    `p` is a real hybrid internal node with the LOWEST access time but
    is not evictable until its child `leaf_x` is gone.

    Demand = 4 tokens (two 2-token nodes):
      * NEW global-LRU: evict leaf_x@1 → p promoted (t=0.5, now the
        globally-oldest) → evict p. Set = {leaf_x, p}.
      * OLD chain walk WOULD have taken leaf_x then leaf_y@2 (the next
        leaf toward MRU), skipping the just-promoted lower-time p →
        set {leaf_x, leaf_y}.
    The NEW behavior is the consistent/correct LRU (evict the truly
    oldest first); this test documents the intentional change."""
    from sglang.srt.mem_cache.mamba_radix_cache import TreeNode
    from sglang.srt.mem_cache.radix_cache import RadixKey

    def _build():
        c = _make_hybrid_cache("lru")
        p = _add_node(c, c.root_node, [1, 2], t=0.5)        # oldest, internal
        _add_node(c, p, [3, 4], t=1.0)                       # leaf_x
        _add_node(c, c.root_node, [5, 6], t=2.0)             # leaf_y
        return c

    c = _build()
    # Plan victims for demand 4.
    assert _victim_keys(c, 4) == [(3, 4), (1, 2)], (
        f"NEW global-LRU must evict leaf_x then promoted p; got "
        f"{_victim_keys(c, 4)}"
    )
    # Real evict_full frees the same set; leaf_y@2 is spared.
    c2 = _build()
    before = set(_all_full_nodes(c2).keys())
    c2.evict_full(4)
    evicted = before - set(_all_full_nodes(c2).keys())
    assert evicted == {(3, 4), (1, 2)}, (
        f"global-LRU set should be {{leaf_x, p}}; the newer leaf_y must "
        f"survive. Got {evicted}"
    )
    assert (5, 6) in set(_all_full_nodes(c2).keys()), "leaf_y must survive"
    print("  PASS  10 global-LRU evicts truly-oldest (promoted p) before "
          "newer leaf_y — intentional divergence from old chain walk")


def test_11_wiring_gate_excludes_subclasses():
    """#263: the scheduler wires `set_evict_cache("kv", tree_cache)`
    only for the BASE RadixCache / MambaRadixCache whose evict() uses
    the shared walk — via an EXACT type check, so Hi*/LMC subclasses
    (which override evict()) are excluded. Pin that exact-type
    semantics: isinstance would wrongly include subclasses."""
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache

    _WIRED = (RadixCache, MambaRadixCache)

    # A subclass (stand-in for HiRadixCache/LMCRadixCache, which
    # override evict()) must NOT match the exact-type gate.
    class _SubCache(RadixCache):
        pass

    plain = RadixCache.__new__(RadixCache)
    mamba = MambaRadixCache.__new__(MambaRadixCache)
    sub = _SubCache.__new__(_SubCache)

    assert type(plain) in _WIRED, "plain RadixCache must wire"
    assert type(mamba) in _WIRED, "MambaRadixCache must wire"
    assert type(sub) not in _WIRED, (
        "subclass overriding evict() must be EXCLUDED (exact-type gate); "
        "isinstance would wrongly wire it and mispredict its eviction"
    )
    assert isinstance(sub, RadixCache), "sanity: sub IS a RadixCache subclass"
    print("  PASS  11 wiring gate: exact-type includes base, excludes "
          "evict()-overriding subclasses")


def test_12_multi_tombstone_cascade():
    """#263 audit follow-up: a chain root→t1(tombstone)→t2(tombstone)
    →leaf. Evicting `leaf` must cascade-sweep BOTH t2 then t1 (depth
    >1). Pins the planner's cascade `while` vs
    `_iteratively_delete_tombstone_leaf`'s multi-level walk."""
    from sglang.srt.budgeter.cost_model import get_cost_curves
    cur = get_cost_curves()

    def _build():
        c = _make_hybrid_cache("lru")
        t1 = _add_node(c, c.root_node, [1, 2], t=5.0)
        t2 = _add_node(c, t1, [3, 4], t=4.0)
        _add_node(c, t2, [5, 6], t=1.0)   # leaf
        _make_tombstone(c, t1)
        _make_tombstone(c, t2)
        return c

    c = _build()
    victims, swept = c._plan_full_eviction(6)
    assert [tuple(n.key.token_ids) for n in victims] == [(5, 6)], victims
    # cascade order: t2 (nearer leaf) then t1.
    assert [tuple(n.key.token_ids) for n in swept] == [(3, 4), (1, 2)], swept

    c2 = _build()
    before = set(_all_full_nodes(c2).keys())
    n = c2.evict_full(6)
    assert before - set(_all_full_nodes(c2).keys()) == {(5, 6), (3, 4), (1, 2)}
    assert n == 6, f"leaf(2)+t2(2)+t1(2)=6 expected, got {n}"

    c3 = _build()
    predicted = c3.predict_evict_cost_us(6, pool="kv")
    expected = (
        (cur.c_kv_ms(2) + cur.c_m_ms(2))   # leaf (hybrid)
        + cur.c_kv_ms(2) + cur.c_kv_ms(2)   # t2, t1 (KV-only)
    ) * 1000.0
    assert abs(predicted - expected) < 1e-6, (predicted, expected)
    print("  PASS  12 multi-tombstone cascade (t2→t1): swept both, "
          "evict_full + predictor byte-exact")


def test_13_lpb_promoted_parent_no_perturbation():
    """#263: test_6 only covers a flat tree. Here a REAL parent with
    hits gets PROMOTED under LPB — predict (which calls
    eviction_priority→hits_in_window→popleft on the promoted parent)
    must not perturb which set a subsequent evict_full picks."""
    def _build():
        c = _make_hybrid_cache("lpb")
        p = _add_node(c, c.root_node, [1, 2], t=10.0, n_hits=4)  # real, hits
        _add_node(c, p, [3, 4], t=1.0, n_hits=1)
        _add_node(c, p, [5, 6], t=2.0, n_hits=1)
        return c

    c_base = _build()
    before = set(_all_full_nodes(c_base).keys())
    c_base.evict_full(6)
    base = before - set(_all_full_nodes(c_base).keys())

    c_treat = _build()
    _ = c_treat.predict_evict_cost_us(6, pool="kv")
    _ = c_treat.predict_evict_cost_us(6, pool="kv")  # twice
    before_t = set(_all_full_nodes(c_treat).keys())
    c_treat.evict_full(6)
    treat = before_t - set(_all_full_nodes(c_treat).keys())

    assert base == treat == {(3, 4), (5, 6), (1, 2)}, (base, treat)
    print("  PASS  13 LPB promoted-parent (with hits): predict×2 then "
          "evict_full picks same set (no perturbation)")


def test_14_swept_tombstone_at_demand_boundary():
    """#263: demand landing EXACTLY on leaf+tombstone token sum. The
    planner counts swept tokens toward the stop, so the boundary must
    free exactly that set — no over/under-eviction."""
    def _build():
        c = _make_hybrid_cache("lru")
        mid = _add_node(c, c.root_node, [1, 2], t=10.0)   # → tombstone, 2 tok
        _add_node(c, mid, [3, 4], t=1.0)                   # leaf, 2 tok
        _make_tombstone(c, mid)
        _add_node(c, c.root_node, [7, 8], t=9.0)           # spare leaf, 2 tok
        return c

    # demand 4 = leaf(2) + swept tombstone(2): the spare leaf must survive.
    c = _build()
    victims, swept = c._plan_full_eviction(4)
    assert [tuple(n.key.token_ids) for n in victims] == [(3, 4)], victims
    assert [tuple(n.key.token_ids) for n in swept] == [(1, 2)], swept

    c2 = _build()
    before = set(_all_full_nodes(c2).keys())
    n = c2.evict_full(4)
    assert n == 4, f"exactly 4 tokens (leaf+tombstone), got {n}"
    survivors = set(_all_full_nodes(c2).keys())
    assert (7, 8) in survivors, "spare leaf must survive the boundary"
    print("  PASS  14 swept tombstone at demand boundary: exactly "
          "leaf+tombstone freed, spare leaf survives")


def test_15_partial_tombstone_not_swept():
    """#263: the dangerous inverse — a tombstone that retains a child
    at the stop must NOT be in swept_tombstones and must survive a
    real evict_full (guards a 'counted-but-not-freed' planner bug)."""
    def _build():
        c = _make_hybrid_cache("lru")
        mid = _add_node(c, c.root_node, [1, 2], t=10.0)
        _add_node(c, mid, [3, 4], t=1.0)   # leaf_a (oldest)
        _add_node(c, mid, [5, 6], t=5.0)   # leaf_b (newer, survives)
        _make_tombstone(c, mid)
        return c

    # demand 2 = only leaf_a; mid keeps leaf_b → NOT childless → not swept.
    c = _build()
    victims, swept = c._plan_full_eviction(2)
    assert [tuple(n.key.token_ids) for n in victims] == [(3, 4)], victims
    assert swept == [], f"tombstone with a remaining child must NOT sweep: {swept}"

    c2 = _build()
    before = set(_all_full_nodes(c2).keys())
    c2.evict_full(2)
    survivors = set(_all_full_nodes(c2).keys())
    assert (1, 2) in survivors and (5, 6) in survivors, (
        f"tombstone mid + leaf_b must survive partial eviction: {survivors}"
    )
    print("  PASS  15 partial tombstone (child remains): not swept, "
          "survives evict_full")


def main() -> int:
    tests = [
        test_1_static_lru_oracle,
        test_2_predict_equals_real_evict_full_set,
        test_3_fail_closed_demand_exceeds_supply,
        test_4_lpb_order_lowest_priority_first,
        test_5_parent_promotion_global_lru,
        test_6_predict_then_evict_no_perturbation,
        test_7_pool_mamba_predict_equals_real_evict_mamba,
        test_8_unknown_pool_raises_value_error,
        test_9_tombstone_parent_not_yielded_no_crash,
        test_10_global_lru_vs_old_chain_divergence,
        test_11_wiring_gate_excludes_subclasses,
        test_12_multi_tombstone_cascade,
        test_13_lpb_promoted_parent_no_perturbation,
        test_14_swept_tombstone_at_demand_boundary,
        test_15_partial_tombstone_not_swept,
    ]
    print(f"\n#259 MambaRadixCache c^evict (full/KV) tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n#259: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
