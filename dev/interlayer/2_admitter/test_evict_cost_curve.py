"""c^evict prefix-curve cache (evict_cost_curve.py): the Admitter's
per-arrival pricing must come off the O(evictable-nodes) walk without
changing the priced values.

Invariants:
  1. Fresh curve == exact walk at every queried target (plain RadixCache
     has no tombstone cascade, so fidelity is exact).
  2. Fail-closed: targets beyond the evictable supply price +inf in both
     modes.
  3. Eviction invalidates the curve; the rebuilt curve matches the exact
     walk on the mutated tree.
  4. Age expiry rebuilds; within max_age the same curve object is reused
     (one walk amortized over many queries).
  5. Env=0 disables the cache (exact-walk behavior, no curve objects).
"""
import os
import sys

# get_cost_curves() fails-fast without a calibrated cost model; use the
# Qwen3.5-9B/H200 calibration (run_arm.sh default branch) for pricing.
os.environ.setdefault("SGLANG_CSIGMA_KV_ALPHA", "1.0214961938707212e-07")
os.environ.setdefault("SGLANG_CSIGMA_KV_BETA", "0.024570739655696554")
os.environ.setdefault("SGLANG_CSIGMA_KV_GAMMA", "5.97224986310455")
os.environ.setdefault("SGLANG_CSIGMA_M_ALPHA", "0.0")
os.environ.setdefault("SGLANG_CSIGMA_M_BETA", "0.0")
os.environ.setdefault("SGLANG_CSIGMA_L_STAR", "0.0")

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.evict_cost_curve import EvictCostCurve


class _StubAllocator:
    """evict() calls allocator.free(value); RadixCache.__init__ reads
    .device. Nothing else is needed."""

    device = "cpu"

    def free(self, value):
        pass


def _build_cache(n_entries: int = 40) -> RadixCache:
    p = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=_StubAllocator(),
        page_size=1,
        enable_kv_cache_events=False,
        eviction_policy="lpb",
    )
    c = RadixCache(p)
    from sglang.srt.mem_cache.radix_cache import InsertParams

    base = 0
    for i in range(n_entries):
        length = 8 + (i * 7) % 96
        ids = list(range(base, base + length))
        base += length
        c.insert(
            InsertParams(
                key=RadixKey(token_ids=ids, extra_key=None),
                value=torch.arange(length, dtype=torch.int64),
            )
        )
    return c


def _exact(cache, x):
    with envs.SGLANG_XPOOL_EVICT_CURVE_MAX_AGE_S.override(0.0):
        return cache.predict_evict_cost_us(x, pool="kv")


def _cached(cache, x):
    with envs.SGLANG_XPOOL_EVICT_CURVE_MAX_AGE_S.override(60.0):
        return cache.predict_evict_cost_us(x, pool="kv")


def test_1_fresh_curve_matches_exact():
    c = _build_cache()
    supply = c.evictable_size()
    assert supply > 0
    targets = [1, 8, supply // 4, supply // 2, supply - 1, supply]
    for x in targets:
        e, g = _exact(c, x), _cached(c, x)
        assert e == g, f"x={x}: exact {e} != cached {g}"
    print(f"test_1 OK  (curve == exact at {len(targets)} targets, supply={supply})")


def test_2_fail_closed_beyond_supply():
    c = _build_cache()
    supply = c.evictable_size()
    assert _exact(c, supply + 1) == float("inf")
    assert _cached(c, supply + 1) == float("inf")
    print("test_2 OK  (+inf beyond supply in both modes)")


def test_3_evict_invalidates_and_rebuild_matches():
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    c = _build_cache()
    _cached(c, 8)  # populate curve
    assert c._evict_curve_cache._curves, "curve should be cached"
    c.evict(EvictParams(num_tokens=c.evictable_size() // 3))
    assert not c._evict_curve_cache._curves, "evict() must invalidate"
    supply = c.evictable_size()
    for x in [1, supply // 2, supply]:
        e, g = _exact(c, x), _cached(c, x)
        assert e == g, f"post-evict x={x}: exact {e} != cached {g}"
    print("test_3 OK  (evict invalidates; rebuilt curve matches on mutated tree)")


def test_4_age_expiry_and_reuse():
    c = _build_cache()
    _cached(c, 8)
    curve1 = c._evict_curve_cache._curves["kv"]
    _cached(c, 16)
    assert c._evict_curve_cache._curves["kv"] is curve1, "within max_age: reuse"
    curve1.built_at -= 3600.0  # force expiry
    _cached(c, 16)
    assert c._evict_curve_cache._curves["kv"] is not curve1, "expired: rebuild"
    print("test_4 OK  (reuse within max_age, rebuild after expiry)")


def test_5_env_zero_disables():
    c = _build_cache()
    _exact(c, 8)
    assert not c._evict_curve_cache._curves, "max_age=0 must not build curves"
    print("test_5 OK  (max_age=0 leaves the cache unused)")


def test_6_curve_lookup_semantics():
    # Direct curve semantics: cumulative boundaries and fail-closed.
    curve = EvictCostCurve([(10, 5.0), (10, 7.0), (5, 100.0)])
    assert curve.lookup(0) == 0.0
    assert curve.lookup(1) == 5.0
    assert curve.lookup(10) == 5.0
    assert curve.lookup(11) == 12.0
    assert curve.lookup(25) == 112.0
    assert curve.lookup(26) == float("inf")
    print("test_6 OK  (cumulative boundary + fail-closed lookup)")


if __name__ == "__main__":
    for name in sorted(k for k in dir() if k.startswith("test_")):
        globals()[name]()
    print("\nALL TESTS PASSED")
