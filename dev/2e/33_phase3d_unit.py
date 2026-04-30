"""
Phase 3.d unit test — heterogeneous granularity in MambaRadixCache.

Verifies that with SGLANG_K_BIG=8, mamba_value snapshots are inserted
ONLY at depths that are multiples of 8 (the K_big), and small-page
inserts at non-aligned depths get KV but no mamba_value (tombstone).

Direct unit test on the radix tree: bypass the engine entirely, drive
insert() with synthetic keys + mamba slots.

Run:
  PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
    .venv/bin/python -u dev/2e/33_phase3d_unit.py
"""
from __future__ import annotations
import os
import sys

# Set K_big to 8 BEFORE importing.
os.environ.setdefault("SGLANG_K_BIG", "8")

import torch

from sglang.srt.mem_cache.mamba_radix_cache import (
    MambaRadixCache,
    RadixKey,
    InsertParams,
    MatchPrefixParams,
    TreeNode,
)


class StubMambaPool:
    def __init__(self): self.size = 64
    def free(self, x): pass


class StubReqToToken:
    def __init__(self):
        self.mamba_pool = StubMambaPool()
        self.req_to_token = None


def make_cache():
    """Build a MambaRadixCache without engine plumbing — just enough to
    exercise insert/match. Bypass __init__ heavy parts."""
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
    cache.req_to_token_pool = StubReqToToken()

    # Allocator stub for free() calls during eviction (not exercised here).
    class StubAllocator:
        def free(self, *a, **k): pass
    cache.token_to_kv_pool_allocator = StubAllocator()

    cache.root_node = TreeNode()
    cache.root_node.key = RadixKey([], extra_key=None)
    cache.root_node.value = torch.tensor([], dtype=torch.int64)
    cache.root_node.lock_ref = 0
    return cache


def test_kbig_alignment():
    print("== Test 1: K_big=8 alignment — only aligned-depth inserts get mamba_value ==")
    print(f"  SGLANG_K_BIG = {os.environ.get('SGLANG_K_BIG')}")

    # Insert at depth 8 (aligned). Should get mamba_value.
    cache = make_cache()
    key = RadixKey(list(range(1, 9)), extra_key=None)  # 8 tokens
    value = torch.arange(1, 9, dtype=torch.int64)
    mamba_slot = torch.tensor([42], dtype=torch.int64)
    result = cache.insert(InsertParams(
        key=key, value=value, mamba_value=mamba_slot, prev_prefix_len=0,
    ))
    print(f"  insert at depth 8 (aligned): prefix_len={result.prefix_len}, mamba_exist={result.mamba_exist}")

    # Walk the tree to find the new node.
    node = cache.root_node.children[1]  # child key = first token = 1
    assert node.mamba_value is not None, \
        f"depth-8 (aligned) insert should have mamba_value, got None"
    print(f"  → depth-8 node has mamba_value (snapshot taken). GOOD.")

    # Now insert a different key at depth 5 (NOT aligned).
    cache2 = make_cache()
    key5 = RadixKey(list(range(1, 6)), extra_key=None)  # 5 tokens
    value5 = torch.arange(1, 6, dtype=torch.int64)
    mamba_slot2 = torch.tensor([43], dtype=torch.int64)
    result2 = cache2.insert(InsertParams(
        key=key5, value=value5, mamba_value=mamba_slot2, prev_prefix_len=0,
    ))
    print(f"  insert at depth 5 (NOT aligned): prefix_len={result2.prefix_len}, mamba_exist={result2.mamba_exist}")

    # Should be a tombstone — node has KV value but no mamba_value.
    node5 = cache2.root_node.children[1]
    assert node5.value is not None and len(node5.value) == 5, \
        f"depth-5 node should have KV value of length 5, got {node5.value}"
    assert node5.mamba_value is None, \
        f"depth-5 (non-aligned) insert should NOT have mamba_value, got {node5.mamba_value}"
    assert result2.mamba_exist == True, \
        f"non-aligned insert must signal mamba_exist=True so caller frees the fork (got {result2.mamba_exist})"
    print(f"  → depth-5 node is a tombstone (KV yes, mamba no). GOOD.")
    print(f"  → mamba_exist=True signals caller to free its fork. GOOD.")

    print("PASS Test 1\n")


def test_kbig_default_off():
    print("== Test 2: SGLANG_K_BIG default (0/unset) preserves existing behavior ==")
    # Save and clear.
    saved = os.environ.pop("SGLANG_K_BIG", None)
    try:
        cache = make_cache()
        key = RadixKey(list(range(1, 6)), extra_key=None)  # 5 tokens, NOT aligned
        value = torch.arange(1, 6, dtype=torch.int64)
        mamba_slot = torch.tensor([99], dtype=torch.int64)
        result = cache.insert(InsertParams(
            key=key, value=value, mamba_value=mamba_slot, prev_prefix_len=0,
        ))
        node = cache.root_node.children[1]
        # Without K_big, the old behavior: every insert gets mamba_value.
        assert node.mamba_value is not None, \
            f"with K_big=0 (default), all inserts should have mamba_value"
        assert result.mamba_exist == False, \
            f"with K_big=0, mamba_exist should be False (radix took the slot)"
        print(f"  → K_big=0 path: depth-5 still has mamba_value (legacy). GOOD.")
    finally:
        if saved is not None:
            os.environ["SGLANG_K_BIG"] = saved
        else:
            os.environ["SGLANG_K_BIG"] = "8"
    print("PASS Test 2\n")


def test_kbig_match_falls_back():
    print("== Test 3: match walks past tombstone to nearest big-page ancestor ==")
    cache = make_cache()
    # Insert at depth 8 (big-page snapshot).
    key8 = RadixKey(list(range(1, 9)), extra_key=None)
    value8 = torch.arange(1, 9, dtype=torch.int64)
    cache.insert(InsertParams(key=key8, value=value8, mamba_value=torch.tensor([1]), prev_prefix_len=0))

    # Now extend with a tombstone leaf at depth 13 (not aligned, K_big=8).
    key13 = RadixKey(list(range(1, 14)), extra_key=None)
    value13 = torch.arange(1, 14, dtype=torch.int64)
    cache.insert(InsertParams(key=key13, value=value13, mamba_value=torch.tensor([2]), prev_prefix_len=0))

    # Use the lower-level _match_prefix_helper to avoid global_server_args
    # dependency. It returns (value_list, last_node, best_value_len). We
    # care about best_value_len: with K_big=8, only the depth-8 ancestor
    # has mamba_value, so best_value_len should be 8 not 13.
    value, last_node, best_value_len = cache._match_prefix_helper(
        RadixKey(list(range(1, 14)), extra_key=None)
    )
    print(f"  match key_len=13 → best_value_len = {best_value_len}, last_node has mamba_value? {last_node.mamba_value is not None}")
    # best_value_len is the count of value-list entries up to the deepest
    # ancestor with mamba_value, NOT the token depth. With our two inserts
    # (depth-8 with snapshot, depth-13 tombstone), value=[d8_tensor, d13_tensor]
    # and best_last_node = depth-8 node. best_value_len = 1 (only d8 entry
    # was accumulated before we passed the snapshot-bearing node).
    assert last_node.mamba_value is not None, \
        f"match should land at the big-page ancestor (depth 8), got tombstone"
    assert best_value_len == 1, \
        f"value-list length up to big-page ancestor should be 1, got {best_value_len}"
    # The big-page node's KV tensor has 8 tokens — verify by summing
    # the matched value tensors.
    matched_kv_len = sum(t.numel() for t in value[:best_value_len])
    assert matched_kv_len == 8, \
        f"matched KV token count should be 8, got {matched_kv_len}"
    print(f"  → match falls back to depth-8 big-page ancestor (KV length 8). GOOD.")
    print("PASS Test 3\n")


def main() -> int:
    test_kbig_alignment()
    test_kbig_default_off()
    test_kbig_match_falls_back()
    print("== ALL PASS: Phase 3.d heterogeneous granularity ready ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
