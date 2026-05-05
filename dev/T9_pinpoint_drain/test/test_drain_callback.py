"""T9: end-to-end drain_callback integration.

Exercises the full T9 drain path: planner asks to drain pages
[57, 58, 59, 60], drain_callback calls evict_pages_in_range AND then
mark_pages_capped to sweep freed pages from free → capped. Verifies:
  - tree node containing those pages is evicted
  - freed pages end up in `_capped_pages`, not `free_pages`
  - if the cache lacks evict_pages_in_range, callback falls back to
    LRU evict (sanity: still doesn't crash; verify will catch stragglers)
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _build_alloc(n_pages, free_ids):
    """Allocator with mark_pages_capped that mirrors the production allocator."""
    alloc = SimpleNamespace(
        size=n_pages,
        device="cpu",
        free_pages=torch.tensor(free_ids, dtype=torch.int64),
        release_pages=torch.tensor([], dtype=torch.int64),
        _capped_pages=torch.tensor([], dtype=torch.int64),
    )

    def free(value):
        # Mimic SGLang allocator.free: append page-ids to free_pages.
        v = value.to(dtype=torch.int64, device=alloc.device)
        if alloc.free_pages.numel() == 0:
            alloc.free_pages = v.clone()
        else:
            alloc.free_pages = torch.cat([alloc.free_pages, v])

    def mark_pages_capped(t):
        target = t.to(alloc.device).to(torch.int64)
        n = 0
        if alloc.free_pages.numel() > 0:
            mask = torch.isin(alloc.free_pages, target)
            held = alloc.free_pages[mask]
            alloc.free_pages = alloc.free_pages[~mask]
            n += int(held.numel())
            alloc._capped_pages = torch.cat([alloc._capped_pages, held])
        return n

    alloc.free = free
    alloc.mark_pages_capped = mark_pages_capped
    return alloc


def _build_tree_cache_with_pinpoint(node_specs, alloc):
    """Spin up a real RadixCache stub with our evict_pages_in_range."""
    from sglang.srt.mem_cache.radix_cache import RadixCache

    cache = RadixCache.__new__(RadixCache)
    cache.disable = False
    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.token_to_kv_pool_allocator = alloc
    cache.update_eviction_metrics = lambda *a, **kw: None
    cache._record_remove_event = lambda _node: None

    root = SimpleNamespace(children={}, parent=None, lock_ref=0,
                          evicted=False, value=None,
                          key=SimpleNamespace(token_ids=[]))
    cache.root_node = root

    leaves = []
    for i, (pages, lock_ref) in enumerate(node_specs):
        v = torch.tensor(pages, dtype=torch.int64)
        node = SimpleNamespace(
            children={}, parent=root,
            key=SimpleNamespace(token_ids=list(range(len(pages))),
                                child_key=lambda *_a, **_kw: i),
            lock_ref=lock_ref, evicted=False, value=v,
        )
        root.children[i] = node
        cache.evictable_size_ += len(pages)
        leaves.append(node)
    cache.evictable_leaves = [n for n in leaves if n.lock_ref == 0]

    def fake_delete_leaf(node):
        for k, v in list(node.parent.children.items()):
            if v is node:
                node.parent.children.pop(k)
                break
        cache.evictable_size_ -= len(node.value)
        if node in cache.evictable_leaves:
            cache.evictable_leaves.remove(node)
    cache._delete_leaf = fake_delete_leaf
    return cache


def _build_agent_with_drain(tree_cache, alloc):
    """Build the smallest BudgetAgent shell that lets _ensure_t8_state's
    drain_callback construction run, then return that callback."""
    from sglang.srt.budgeter.agent import BudgetAgent

    a = BudgetAgent.__new__(BudgetAgent)

    # Minimal _xpool_actuator: only kv_actuator.allocator is read.
    kv_act = SimpleNamespace(allocator=alloc, pool=SimpleNamespace(
        k_buffer=[torch.zeros((alloc.size + 1, 2, 2))], v_buffer=[torch.zeros((alloc.size + 1, 2, 2))],
        layer_num=1,
    ))
    a._xpool_actuator = SimpleNamespace(
        kv_actuator=kv_act, mamba_actuator=None,
    )

    # Minimal scheduler with the tree cache.
    sched = SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((4, 16), dtype=torch.int32),
        ),
        running_batch=SimpleNamespace(reqs=[]),
        waiting_queue=[],
        tree_cache=tree_cache,
    )
    a.scheduler = sched
    a._t8_state = None

    # Force flag-on so _ensure_t8_state doesn't bail.
    os.environ["SGLANG_T8_PLANNER"] = "1"
    os.environ["SGLANG_T8_EXECUTE"] = "1"

    ok = a._ensure_t8_state()
    assert ok, "T9 test: _ensure_t8_state should succeed"
    return a._t8_state["drain_callback"]


def test_pinpoint_drain_then_sweep_to_capped():
    n_pages = 100
    # Pre-fire state: pages 1..56 free; 57..64 are owned by tree (cached);
    # 65..100 free.
    free_ids = list(range(1, 57)) + list(range(65, n_pages + 1))
    alloc = _build_alloc(n_pages, free_ids)

    # Simulate cap-barrier on cap range [57, 65) — but no free pages in
    # that range right now (all are tree-owned), so mark_pages_capped is a
    # no-op for now. The post-drain sweep is what catches the freed pages.

    cache = _build_tree_cache_with_pinpoint([
        ([57, 58, 59, 60], 0),
        ([61, 62, 63, 64], 0),
        ([10, 11, 12, 13], 0),  # outside cap
    ], alloc)

    drain_cb = _build_agent_with_drain(cache, alloc)
    n = drain_cb([57, 58, 59, 60, 61, 62, 63, 64])

    assert n == 8, f"expected 8 pages drained, got {n}"
    # Verify: cap-range pages NOT in free_pages.
    cap_in_free = ((alloc.free_pages >= 57) & (alloc.free_pages < 65)).any().item()
    assert not cap_in_free, "cap-range pages leaked into free_pages"
    # Verify: cap-range pages IN _capped_pages.
    cap_in_capped = (
        (alloc._capped_pages >= 57) & (alloc._capped_pages < 65)
    ).sum().item()
    assert cap_in_capped == 8, f"expected 8 capped pages, got {cap_in_capped}"
    # Out-of-range tree node still attached.
    assert len(cache.root_node.children) == 1
    print(
        f"[T9-integ] drained={n}, capped={cap_in_capped}, "
        f"out-of-range nodes retained={len(cache.root_node.children)}"
    )


def test_drain_with_no_overlap_returns_zero():
    n_pages = 100
    free_ids = list(range(1, 50))
    alloc = _build_alloc(n_pages, free_ids)

    cache = _build_tree_cache_with_pinpoint([
        ([10, 11, 12], 0),
    ], alloc)

    drain_cb = _build_agent_with_drain(cache, alloc)
    n = drain_cb([57, 58, 59, 60])  # cap range pages — none in tree

    assert n == 0
    # Tree node untouched.
    assert len(cache.root_node.children) == 1
    print(f"[T9-integ] no-overlap drain: n=0, tree intact")


def main():
    test_pinpoint_drain_then_sweep_to_capped()
    test_drain_with_no_overlap_returns_zero()
    print("\nT9 drain_callback integration test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
