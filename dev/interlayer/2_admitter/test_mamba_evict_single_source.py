"""Characterization: the base MambaRadixCache mamba-side eviction must
have a SINGLE victim-ordering source consumed by BOTH `evict_mamba`
(the actuator) and `_plan_mamba_eviction` / `predict_evict_cost_us`
(the cost predictor), mirroring the KV side where `evict_full` and
`predict_evict_cost_us(pool="kv")` share `_plan_full_eviction`.

These tests pin the OBSERVABLE eviction contract:
  (1) the ORDER and SET of nodes `evict_mamba` frees, and
  (2) the priced set / cost from `_plan_mamba_eviction` +
      `predict_evict_cost_us(pool="mamba")`,
under BOTH `lru` and `lpb` policies, on flat + nested + tombstone
trees. They must stay byte-identical across the single-source refactor:
if the extracted `_iter_mamba_victims` generator changes which nodes
are evicted or in what order, one of these assertions fails.

Scope: BASE `MambaRadixCache` only. `HiMambaRadixCache.evict_mamba`
overrides with an LRU-only host-offload lifecycle (different policy,
different freeing path) and is intentionally left on its own path, so
it is not covered here.

Fixtures reuse `test_mamba_evict_predictor._make_hybrid_cache` /
`_add_node` (real `MambaRadixCache`, `__init__` bypassed, stub
allocator/pool, no GPU).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_mamba_evict_predictor import _add_node, _make_hybrid_cache  # noqa: E402

from sglang.srt.budgeter.cost_model import get_cost_curves  # noqa: E402


def _all_full_nodes(cache):
    out = {}
    stack = list(cache.root_node.children.values())
    while stack:
        n = stack.pop()
        out[tuple(n.key.token_ids)] = n
        stack.extend(n.children.values())
    return out


def _mamba_present(cache):
    """Keys of nodes that still hold a device mamba snapshot."""
    return {
        k for k, n in _all_full_nodes(cache).items() if n.mamba_value is not None
    }


def _evict_mamba_order(build, mamba_num):
    """Run a REAL `evict_mamba(mamba_num)` on a fresh tree from `build`
    and return the keys whose mamba snapshot got freed, in the ORDER
    they were freed (snapshot before/after diff is order-agnostic, so
    we reconstruct order from the plan's victim sequence; the SET is
    the ground truth and the order is pinned via the plan separately)."""
    c = build()
    before = _mamba_present(c)
    freed = c.evict_mamba(mamba_num)
    after_present = _mamba_present(c)
    dropped = before - after_present
    return freed, dropped


def _plan_keys(c, mamba_num):
    leaf_v, internal_v, swept = c._plan_mamba_eviction(mamba_num)
    return (
        [tuple(n.key.token_ids) for n in leaf_v],
        [tuple(n.key.token_ids) for n in internal_v],
        [tuple(n.key.token_ids) for n in swept],
    )


def test_1_lru_flat_order_and_set():
    """LRU flat tree: the mamba LRU walk orders by mamba_lru_list
    membership (insertion order, tail = first inserted = oldest), NOT
    by `last_access_time` (which only drives the LPB heap / the KV
    `_plan_full_eviction`). The plan's leaf-victim ORDER matches the
    walk and the freed SET matches `evict_mamba`."""
    def _build():
        c = _make_hybrid_cache("lru")
        _add_node(c, c.root_node, [1, 2], t=3.0)   # inserted first → LRU tail
        _add_node(c, c.root_node, [3, 4], t=1.0)
        _add_node(c, c.root_node, [5, 6], t=2.0)
        return c

    # LRU mamba walk: oldest-inserted first → (1,2), (3,4) for demand 2.
    c = _build()
    leaf, internal, swept = _plan_keys(c, 2)
    assert leaf == [(1, 2), (3, 4)], leaf
    assert internal == [] and swept == [], (internal, swept)

    freed, dropped = _evict_mamba_order(_build, 2)
    assert freed == 2, freed
    assert dropped == {(1, 2), (3, 4)}, dropped
    print("  PASS  1  LRU flat: plan leaf-order == evict_mamba set "
          "(insertion-order, oldest first)")


def test_2_lpb_phase1_cold_tail_then_heap():
    """LPB two-phase: Phase-1 contiguous cold (hit_count==0) tail run
    first, then Phase-2 heap over hit-bearing nodes by priority. Pin
    both the plan order and the real evict_mamba set."""
    def _build():
        c = _make_hybrid_cache("lpb")
        # tail (oldest) cold run: (3,4)@t1 h0, (5,6)@t2 h0 ; then hot (1,2)@t3 h9.
        _add_node(c, c.root_node, [1, 2], t=3.0, n_hits=9)   # hottest, newest
        _add_node(c, c.root_node, [3, 4], t=1.0, n_hits=0)   # cold tail (oldest)
        _add_node(c, c.root_node, [5, 6], t=2.0, n_hits=0)   # cold tail
        return c

    # demand 2 → the two cold-tail nodes (Phase 1), hot (1,2) spared.
    c = _build()
    leaf, internal, swept = _plan_keys(c, 2)
    assert leaf == [(3, 4), (5, 6)], leaf
    freed, dropped = _evict_mamba_order(_build, 2)
    assert freed == 2 and dropped == {(3, 4), (5, 6)}, (freed, dropped)

    # demand 3 → both cold + the hot one via Phase-2 heap.
    c3 = _build()
    leaf3, _i, _s = _plan_keys(c3, 3)
    assert leaf3 == [(3, 4), (5, 6), (1, 2)], leaf3
    freed3, dropped3 = _evict_mamba_order(_build, 3)
    assert dropped3 == {(3, 4), (5, 6), (1, 2)}, dropped3
    print("  PASS  2  LPB: Phase-1 cold tail then Phase-2 heap, plan == evict set")


def test_3_lpb_hot_tail_skips_to_heap():
    """LPB where the LRU tail node ITSELF has hits: Phase-1 is skipped
    and selection goes straight to the heap (lowest priority first)."""
    def _build():
        c = _make_hybrid_cache("lpb")
        # All have hits → no cold tail. Heap orders by eviction_priority.
        _add_node(c, c.root_node, [1, 2], t=1.0, n_hits=20)   # tail, hottest
        _add_node(c, c.root_node, [3, 4], t=2.0, n_hits=2)    # coldest-by-priority
        _add_node(c, c.root_node, [5, 6], t=3.0, n_hits=5)
        return c

    c = _build()
    leaf, _i, _s = _plan_keys(c, 1)
    # lowest priority = fewest hits per byte = (3,4) (2 hits).
    assert leaf == [(3, 4)], leaf
    freed, dropped = _evict_mamba_order(_build, 1)
    assert freed == 1 and dropped == {(3, 4)}, (freed, dropped)
    print("  PASS  3  LPB hot tail: skip Phase-1, heap picks lowest-priority")


def test_4_internal_tombstone_then_leaf_cascade():
    """Nested root→mid(oldest)→leaf. LRU drain of 2 slots: mid is an
    INTERNAL victim (snapshot freed, KV tombstone, stays in tree), then
    the leaf frees KV+mamba and its cascade sweeps mid's KV. Pin the
    plan classification + the real evict_mamba freed-set/cost."""
    curves = get_cost_curves()

    def _build():
        c = _make_hybrid_cache("lru")
        mid = _add_node(c, c.root_node, [1, 2], t=1.0)   # oldest → internal
        _add_node(c, mid, [3, 4], t=2.0)                 # child leaf, newer
        return c

    c = _build()
    leaf, internal, swept = _plan_keys(c, 2)
    assert internal == [(1, 2)], internal
    assert leaf == [(3, 4)], leaf
    assert swept == [(1, 2)], swept   # mid's KV swept by the leaf cascade

    # Real evict_mamba: both mamba snapshots gone, both KV gone (cascade).
    c2 = _build()
    before = _mamba_present(c2)
    freed = c2.evict_mamba(2)
    after = _all_full_nodes(c2)
    assert freed == 2, freed
    assert before - _mamba_present(c2) == {(1, 2), (3, 4)}, before
    assert after == {}, f"leaf+cascade should empty the tree, got {after.keys()}"

    # Predictor prices the exact set (internal priced by whole-prefix total).
    c3 = _build()
    predicted = c3.predict_evict_cost_us(2, pool="mamba")
    expected = (
        (curves.c_kv_ms(2) + curves.c_m_ms(2))   # internal mid: whole-prefix total
        + (curves.c_kv_ms(2) + curves.c_m_ms(2))  # leaf (3,4): c_kv + c_m
    ) * 1000.0
    assert abs(predicted - expected) < 1e-6, (predicted, expected)
    print("  PASS  4  nested: internal tombstone + leaf cascade, plan==evict==price")


def test_5_predict_then_evict_no_perturbation():
    """Calling `predict_evict_cost_us(pool='mamba')` (which consumes the
    same victim source) must NOT perturb which set a subsequent real
    `evict_mamba` frees — the source is pure-read."""
    def _build():
        c = _make_hybrid_cache("lpb")
        _add_node(c, c.root_node, [1, 2], t=3.0, n_hits=9)
        _add_node(c, c.root_node, [3, 4], t=1.0, n_hits=0)
        _add_node(c, c.root_node, [5, 6], t=2.0, n_hits=0)
        return c

    c_base = _build()
    base_freed = c_base.evict_mamba(2)
    base_drop = _mamba_present(_build()) - _mamba_present(c_base)
    # Reconstruct base dropped set from a clean run.
    cb = _build()
    before_b = _mamba_present(cb)
    cb.evict_mamba(2)
    base_drop = before_b - _mamba_present(cb)

    c_treat = _build()
    _ = c_treat.predict_evict_cost_us(2, pool="mamba")
    _ = c_treat.predict_evict_cost_us(2, pool="mamba")   # twice
    before_t = _mamba_present(c_treat)
    c_treat.evict_mamba(2)
    treat_drop = before_t - _mamba_present(c_treat)

    assert base_freed == 2, base_freed
    assert base_drop == treat_drop == {(3, 4), (5, 6)}, (base_drop, treat_drop)
    print("  PASS  5  predict('mamba')×2 then evict_mamba: same set (pure-read)")


def test_6_predict_equals_real_evict_mamba_cost_lpb():
    """End-to-end: under LPB, `predict_evict_cost_us(pool='mamba')` ==
    the cost summed over the EXACT set a real `evict_mamba` frees, with
    n_b = hits_in_window weighting. Pins price↔evict equivalence."""
    curves = get_cost_curves()

    def _build():
        c = _make_hybrid_cache("lpb")
        # distinct lengths → distinct per-node cost; mixed hits.
        _add_node(c, c.root_node, [1, 2], t=1.0, n_hits=0)        # cold tail
        _add_node(c, c.root_node, [3, 4, 5], t=2.0, n_hits=0)     # cold tail
        _add_node(c, c.root_node, [6, 7, 8, 9], t=3.0, n_hits=4)  # hot
        return c

    c = _build()
    use_lpb = c._should_use_lpb()
    snap = {
        k: (len(n.key), n.hits_in_window() if use_lpb else 1)
        for k, n in _all_full_nodes(c).items()
    }
    predicted = c.predict_evict_cost_us(3, pool="mamba")

    c2 = _build()
    before = _mamba_present(c2)
    freed = c2.evict_mamba(3)
    dropped = before - _mamba_present(c2)

    actual_ms = 0.0
    for k in dropped:
        s_b, n_b = snap[k]
        actual_ms += n_b * (curves.c_kv_ms(s_b) + curves.c_m_ms(s_b))
    actual = actual_ms * 1000.0

    assert freed == 3, freed
    assert abs(predicted - actual) < 1e-6, (predicted, actual, dropped)
    print(f"  PASS  6  LPB predict('mamba')={predicted:.1f}us == real evict cost "
          f"over {len(dropped)} nodes")


def main() -> int:
    tests = [
        test_1_lru_flat_order_and_set,
        test_2_lpb_phase1_cold_tail_then_heap,
        test_3_lpb_hot_tail_skips_to_heap,
        test_4_internal_tombstone_then_leaf_cascade,
        test_5_predict_then_evict_no_perturbation,
        test_6_predict_equals_real_evict_mamba_cost_lpb,
    ]
    print(f"\nmamba single-source characterization (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nsingle-source: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
