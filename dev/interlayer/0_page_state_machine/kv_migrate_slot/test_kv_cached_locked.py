"""#271 HIGH-1 (audit) — _cached_kv_slots must exclude LOCKED radix nodes too.

The Migration source set is "LIVE-UNCACHED" = allocated − free − capped −
CACHED. `_cached_kv_slots` supplies the CACHED set. The bug: it was built
from `_iter_drain_victims`, i.e. the cost-order EVICTION-victim walk, which
seeds only from `evictable_leaves` and skips locked parents
(`RadixCache._iter_evict_victims`: `if parent.lock_ref != 0: continue`). So
KV slots backing a LOCKED shared-prefix node (held by >1 running req) are
absent from the cached set, get classified LIVE-uncached, and become a
migration SOURCE. Migrating one rewrites only the FIRST owner's
`req_to_token` (`rewrite_kv_token_indices` returns on first hit) → every
co-sharer reads the freed src = silent KV corruption + orphaned radix node.

Fix: `_cached_kv_slots` walks the WHOLE tree from the root (locked AND
evictable nodes), not the eviction-victim subset. These CPU tests build a
faithful tree_cache: a root with one LOCKED node and one evictable node, and
a `_plan_full_eviction` that — like the real walk — yields only the
evictable one. Pre-fix: the locked slots leak in as a migration source
(RED). Post-fix: they're excluded (GREEN).
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "1"  # un-gate the walk (#271 step 5)


class _Node:
    """Minimal radix TreeNode stand-in: KV slots in `value`, a `children`
    dict, and a `lock_ref` so the eviction-victim walk can mimic skipping
    locked nodes."""

    def __init__(self, value=None, lock_ref=0):
        self.value = (
            torch.tensor(value, dtype=torch.int64) if value is not None else None
        )
        self.children = {}
        self.lock_ref = lock_ref


class _FakeTree:
    """A tree_cache exposing BOTH the full-tree structure (root_node →
    children, what the FIXED `_cached_kv_slots` walks) AND the hybrid
    eviction-victim API (`_plan_full_eviction` / `full_evictable_size`, what
    the BUGGY one consumed) — so the test exercises the real production path
    on both sides of the fix."""

    def __init__(self, root, evictable_nodes):
        self.root_node = root
        self._evictable = list(evictable_nodes)

    def full_evictable_size(self):
        return sum(int(n.value.numel()) for n in self._evictable)

    def _plan_full_eviction(self, num_tokens):
        # Mirrors the real walk: yields ONLY evictable (unlocked) victims;
        # locked nodes are never returned. swept tombstones: none.
        return list(self._evictable), []


def _provider(tree, allocator=None, tps=4, n_pages=10):
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider
    kv_act = types.SimpleNamespace(_tokens_per_page=lambda: tps, n_pages=n_pages)
    sched = types.SimpleNamespace(
        token_to_kv_pool_allocator=allocator, tree_cache=tree,
    )
    return SchedulerOwnerProvider(
        scheduler=sched, kv_actuator=kv_act, mamba_actuator=None,
    )


def _alloc(size, free_ids):
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    class _StubKV:
        page_size = 1
        def can_move_kv_cache(self):
            return True
        def get_kv_size_bytes(self):
            return 0
        def move_kv_cache(self, tgt, src):
            pass

    a = TokenToKVPoolAllocator(
        size=size, dtype=torch.float16, device="cpu",
        kvcache=_StubKV(), need_sort=False,
    )
    a.free_pages = torch.tensor(free_ids, dtype=torch.int64)
    return a


def test_1_cached_set_includes_locked_nodes():
    """The CACHED set must union LOCKED and evictable node slots. A locked
    shared-prefix node holds [20,21,22,23]; an evictable node holds [30,31].
    Pre-fix (eviction-walk): only {30,31}. Post-fix (full-tree): both."""
    locked = _Node(value=[20, 21, 22, 23], lock_ref=2)
    evictable = _Node(value=[30, 31], lock_ref=0)
    root = _Node()
    root.children = {0: locked, 1: evictable}
    prov = _provider(_FakeTree(root, evictable_nodes=[evictable]))
    cached = prov._cached_kv_slots()
    assert {20, 21, 22, 23} <= cached, (
        f"LOCKED shared-prefix slots must be in the cached set; got {sorted(cached)}"
    )
    assert {30, 31} <= cached, "evictable cached slots must remain present"
    print("  PASS  1  _cached_kv_slots unions LOCKED + evictable node slots")


def test_2_locked_prefix_page_is_not_a_migration_source():
    """End-to-end through `_kv_live_pages_in_cost_order`: a page composed
    entirely of LOCKED shared-prefix slots must NOT be selected as a
    migration source. Layout (tps=4, 10 pages, size=40):
      p1[4-7] live single-owner → genuine SOURCE; p2/p3/p4/p6 partial → 8
      donors; p5[20-23] LOCKED prefix (cached, not free); p7-9 whole-free.
    Pre-fix p5 leaks in as a 2nd source (8 donors suffice for both) and its
    locked slots get migrated. Post-fix only p1 is a source."""
    tps, n_pages, size = 4, 10, 40
    # Donors on partial pages 2,3,4,6 (2 free each) + whole-free pages 7,8,9.
    free_ids = [8, 9, 12, 13, 16, 17, 24, 25] + list(range(28, 40))
    a = _alloc(size, free_ids)
    locked = _Node(value=[20, 21, 22, 23], lock_ref=2)
    root = _Node()
    root.children = {0: locked}
    prov = _provider(_FakeTree(root, evictable_nodes=[]), allocator=a,
                     tps=tps, n_pages=n_pages)
    out = prov._kv_live_pages_in_cost_order()
    src_pages = {pid for pid, _ in out}
    assert 5 not in src_pages, (
        f"LOCKED-prefix page 5 must NOT be a migration source; got {out}"
    )
    assert out == [(1, ((4, 8), (5, 9), (6, 12), (7, 13)))], (
        f"only the live single-owner page 1 may be a source; got {out}"
    )
    # And none of the locked slots may appear as a migrated src anywhere.
    migrated_srcs = {s for _, moves in out for (s, _d) in moves}
    assert migrated_srcs.isdisjoint({20, 21, 22, 23}), (
        f"locked slots must never be migrated; got srcs {sorted(migrated_srcs)}"
    )
    print("  PASS  2  locked-prefix page excluded from migration sources")


def main() -> int:
    tests = [
        test_1_cached_set_includes_locked_nodes,
        test_2_locked_prefix_page_is_not_a_migration_source,
    ]
    print(f"\n#271 HIGH-1 locked-node cached-exclusion tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\n#271 HIGH-1: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
