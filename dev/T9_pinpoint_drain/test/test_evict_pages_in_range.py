"""T9: RadixCache.evict_pages_in_range unit tests.

Builds a synthetic radix tree with known node→pages assignment, calls the
new pinpoint API, and verifies:
  - only nodes overlapping the cap range get evicted
  - locked nodes (lock_ref > 0) are skipped
  - partial-overlap nodes are evicted whole (collateral over-eviction is
    allowed; doc'd in §3.2.3)
  - returned count matches actual freed page total
  - allocator.free is invoked with the right page tensors
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _build_cache(node_specs):
    """Build a RadixCache stand-in with the minimum surface
    `evict_pages_in_range` exercises. `node_specs` is a list of (page_ids,
    lock_ref) tuples; each becomes one leaf node off the root.
    """
    from sglang.srt.mem_cache.radix_cache import RadixCache

    cache = RadixCache.__new__(RadixCache)
    cache.disable = False
    cache.evictable_size_ = 0
    cache.protected_size_ = 0

    root = SimpleNamespace(
        children={},
        parent=None,
        key=SimpleNamespace(token_ids=[]),
        lock_ref=0,
        evicted=False,
        value=None,
    )
    cache.root_node = root

    leaves = []
    for i, (pages, lock_ref) in enumerate(node_specs):
        v = torch.tensor(pages, dtype=torch.int64)
        node = SimpleNamespace(
            children={},
            parent=root,
            key=SimpleNamespace(
                token_ids=list(range(len(pages))),
                child_key=lambda *_args, **_kw: i,
            ),
            lock_ref=lock_ref,
            evicted=False,
            value=v,
        )
        # Attach: simulate what insert would do.
        root.children[i] = node
        cache.evictable_size_ += len(pages)
        leaves.append(node)

    # The real RadixCache uses an OrderedSet; SimpleNamespace isn't hashable
    # so we model it as a list with the operations the method needs.
    cache.evictable_leaves = [n for n in leaves if n.lock_ref == 0]

    # Allocator stub: track freed pages.
    freed = []

    def fake_free(value):
        freed.append(value.tolist())

    cache.token_to_kv_pool_allocator = SimpleNamespace(free=fake_free)

    # Stub out methods evict_pages_in_range calls.
    def fake_delete_leaf(node):
        # Mirror the real _delete_leaf: pop from parent, drop from
        # evictable_leaves, update size accounting. We don't need to
        # update_leaf_status on parent for this stub.
        for k, v in list(node.parent.children.items()):
            if v is node:
                node.parent.children.pop(k)
                break
        cache.evictable_size_ -= len(node.value)
        if node in cache.evictable_leaves:
            cache.evictable_leaves.remove(node)

    cache._delete_leaf = fake_delete_leaf
    cache._record_remove_event = lambda _node: None
    cache.update_eviction_metrics = lambda *_a, **_kw: None

    return cache, freed


def test_evicts_only_overlapping_leaves():
    cache, freed = _build_cache([
        ([10, 11, 12, 13], 0),  # outside cap range
        ([57, 58, 59, 60], 0),  # inside cap range
        ([100, 101], 0),        # outside cap range
    ])

    n = cache.evict_pages_in_range(50, 70)
    assert n == 4, f"expected 4 pages freed, got {n}"
    assert freed == [[57, 58, 59, 60]]
    # The other two nodes still in the tree.
    assert len(cache.root_node.children) == 2
    print(f"[T9] in-range eviction: freed={n} pages, kept 2/3 nodes")


def test_skips_locked_nodes():
    cache, freed = _build_cache([
        ([57, 58, 59, 60], 0),  # in range, evictable
        ([61, 62, 63, 64], 1),  # in range BUT locked
    ])

    n = cache.evict_pages_in_range(50, 70)
    assert n == 4, f"expected 4 pages (locked one skipped), got {n}"
    assert freed == [[57, 58, 59, 60]]
    # Locked node still attached.
    assert len(cache.root_node.children) == 1
    print(f"[T9] locked-skip: freed={n}, locked node retained")


def test_partial_overlap_evicts_whole_node():
    """Node value [55, 56, 57, 58]; cap range [57, 70). The node spans
    the boundary. Per design (over-eviction allowed under T2 placement
    bias), the whole node evicts and pages 55-56 are released too."""
    cache, freed = _build_cache([
        ([55, 56, 57, 58], 0),
    ])

    n = cache.evict_pages_in_range(57, 70)
    assert n == 4, f"expected 4 pages (whole node), got {n}"
    assert freed == [[55, 56, 57, 58]]
    print(f"[T9] partial-overlap: whole node evicted, n={n}")


def test_no_match_returns_zero():
    cache, freed = _build_cache([
        ([1, 2, 3], 0),
        ([10, 20, 30], 0),
    ])

    n = cache.evict_pages_in_range(100, 200)
    assert n == 0
    assert freed == []
    assert len(cache.root_node.children) == 2
    print("[T9] no-match: returned 0, no eviction")


def test_empty_range_returns_zero():
    cache, freed = _build_cache([
        ([57, 58], 0),
    ])

    # high <= low: no-op
    n = cache.evict_pages_in_range(70, 70)
    assert n == 0
    n = cache.evict_pages_in_range(70, 50)
    assert n == 0
    print("[T9] empty-range: returned 0")


def test_disable_returns_zero():
    cache, freed = _build_cache([
        ([57, 58], 0),
    ])
    cache.disable = True

    n = cache.evict_pages_in_range(50, 70)
    assert n == 0
    assert freed == []
    print("[T9] disabled cache: returned 0")


def main():
    test_evicts_only_overlapping_leaves()
    test_skips_locked_nodes()
    test_partial_overlap_evicts_whole_node()
    test_no_match_returns_zero()
    test_empty_range_returns_zero()
    test_disable_returns_zero()
    print("\nT9 evict_pages_in_range test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
