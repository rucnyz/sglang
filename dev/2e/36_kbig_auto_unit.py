"""Unit test for SGLANG_K_BIG_AUTO_THRESHOLD adaptive disable.

Verifies:
  - When SGLANG_K_BIG_AUTO_THRESHOLD=0 (default), K_BIG always active.
  - When SGLANG_K_BIG_AUTO_THRESHOLD=0.5 and mamba_usage<0.5, K_BIG is
    auto-disabled (insert keeps mamba_value).
  - When SGLANG_K_BIG_AUTO_THRESHOLD=0.5 and mamba_usage>=0.5, K_BIG
    activates as usual.
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("SGLANG_K_BIG", "8")
os.environ.setdefault("SGLANG_HPB_LRU", "1")

import torch

from sglang.srt.mem_cache.mamba_radix_cache import (
    MambaRadixCache, RadixKey, InsertParams, TreeNode,
)


class StubMambaPool:
    def __init__(self, total: int = 100, used: int = 0):
        self.size = total
        self._used = used
    def available_size(self): return self.size - self._used
    def free(self, x): pass
    def fork_from(self, mv): return mv


class StubReqToToken:
    def __init__(self, mamba_pool):
        self.mamba_pool = mamba_pool
        self.req_to_token = None


def make_cache(mamba_pool):
    cache = MambaRadixCache.__new__(MambaRadixCache)
    cache.disable = False
    cache.page_size = 1
    cache.is_eagle = False
    cache.full_evictable_size_ = 0
    cache.mamba_evictable_size_ = 0
    cache.full_protected_size_ = 0
    cache.mamba_protected_size_ = 0
    cache.enable_mamba_extra_buffer = False
    from sglang.srt.mem_cache.mamba_radix_cache import LRUList
    cache.full_lru_list = LRUList(mamba=False)
    cache.mamba_lru_list = LRUList(mamba=True)
    cache.req_to_token_pool = StubReqToToken(mamba_pool)
    class StubAllocator:
        def free(self, *a, **k): pass
    cache.token_to_kv_pool_allocator = StubAllocator()
    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey([], extra_key=None)
    cache.root_node.value = torch.tensor([], dtype=torch.int64)
    return cache


def insert_at_depth(cache, depth: int):
    key = RadixKey(list(range(1, depth + 1)), extra_key=None)
    value = torch.arange(1, depth + 1, dtype=torch.int64)
    mamba_slot = torch.tensor([42], dtype=torch.int64)
    return cache.insert(InsertParams(
        key=key, value=value, mamba_value=mamba_slot, prev_prefix_len=0,
    ))


def test_auto_disable_low_usage():
    print("== Test 1: K_BIG_AUTO_THRESHOLD=0.5, mamba_usage=0.1 → K_BIG auto-disabled ==")
    os.environ["SGLANG_K_BIG_AUTO_THRESHOLD"] = "0.5"
    os.environ["SGLANG_K_BIG"] = "8"
    pool = StubMambaPool(total=100, used=10)  # 10% usage, below 50%
    cache = make_cache(pool)
    # depth=13 would normally suppress; with auto-disable it should NOT suppress.
    result = insert_at_depth(cache, 13)
    print(f"  depth=13 result: prefix_len={result.prefix_len}, mamba_exist={result.mamba_exist}")
    # Walk tree to verify the depth-13 node has mamba_value (snapshot taken).
    node = cache.root_node.children[1]
    assert node.mamba_value is not None, \
        f"with auto-disable, depth-13 should retain snapshot (mamba_usage=0.1 < 0.5)"
    print(f"  → PASS (K_BIG auto-disabled when mamba_usage<threshold)")
    print()


def test_auto_keeps_kbig_high_usage():
    print("== Test 2: K_BIG_AUTO_THRESHOLD=0.5, mamba_usage=0.7 → K_BIG active ==")
    os.environ["SGLANG_K_BIG_AUTO_THRESHOLD"] = "0.5"
    os.environ["SGLANG_K_BIG"] = "8"
    pool = StubMambaPool(total=100, used=70)  # 70% usage, above 50%
    cache = make_cache(pool)
    # depth=13 with K_BIG=8: suppress. Trailing KV freed, no leaf created.
    result = insert_at_depth(cache, 13)
    print(f"  depth=13 result: prefix_len={result.prefix_len}, mamba_exist={result.mamba_exist}")
    assert 1 not in cache.root_node.children, \
        f"with K_BIG active at depth 13, no tombstone leaf should be created"
    assert result.mamba_exist == True, \
        f"K_BIG suppression should set mamba_exist=True"
    print(f"  → PASS (K_BIG fires normally at high mamba_usage)")
    print()


def test_default_zero_threshold():
    print("== Test 3: K_BIG_AUTO_THRESHOLD=0 (default) → always-on K_BIG, no probing ==")
    os.environ["SGLANG_K_BIG_AUTO_THRESHOLD"] = "0"
    os.environ["SGLANG_K_BIG"] = "8"
    pool = StubMambaPool(total=100, used=5)  # 5% usage
    cache = make_cache(pool)
    result = insert_at_depth(cache, 13)
    print(f"  depth=13 result: prefix_len={result.prefix_len}, mamba_exist={result.mamba_exist}")
    assert 1 not in cache.root_node.children, \
        f"with auto-threshold=0, K_BIG always fires regardless of usage"
    print(f"  → PASS (default behavior preserved)")
    print()


def main():
    test_auto_disable_low_usage()
    test_auto_keeps_kbig_high_usage()
    test_default_zero_threshold()
    print("== ALL PASS: K_BIG_AUTO_THRESHOLD adaptive control ready ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
