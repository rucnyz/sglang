"""#264 — mamba_to_kv fire verify path (latent crash found on the
2026-06-01 Qwen3.5-9B boot).

`XPoolActuator.execute_async`'s post-cap-barrier verify checked
"did any capped target page leak back into the SOURCE allocator's
free list?" by reading `src_alloc.free_pages` / `_capped_pages` —
KV paged-allocator TENSOR state. For a `mamba_to_kv` fire the source
is the mamba pool, whose `_MambaCapAllocator` is SLOT-based
(`free_slots` / `_capped_slots`) and has no `free_pages` → the fire
raised `AttributeError: free_pages`. So `mamba_to_kv` — a real
production fire direction (XPoolPlanner can decide it) and the
self-reversing c^xfer probe's reverse leg — was latently broken.

Fix: a polymorphic `count_reachable_capped(cap_t) -> int` on BOTH
allocators (KV: free_pages∖_capped_pages; mamba: free_slots∖
_capped_slots), so the actuator's verify is allocator-representation-
agnostic. This test pins the semantics on both + their equivalence.

Sub-tests:
1. KV allocator: a leaked capped target (in free_pages, not in
   _capped_pages) is counted; a properly-capped one is not.
2. _MambaCapAllocator: slot-based twin — same semantics on
   free_slots / _capped_slots.
3. equivalence: identical (free, capped, cap_t) configuration gives
   the same count on both representations.
4. the actuator verify now calls count_reachable_capped (no
   free_pages access) — guards the regression.
"""
from __future__ import annotations

import sys
import threading

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch


def _kv_with_free_capped(free, capped):
    """A KV allocator whose LOGICAL state is: `free` ids free, `capped` ids
    drained (held out). In the CappedFreeList model a drained page STAYS in the
    free list and is named in `marks` (alloc skips it) — so `free_ids` holds all
    `free` ids and `marks` holds the `capped` subset. The allocatable set is
    `free ∖ capped`; a "leak" (a cap target still allocatable) is an id that is
    in `free` but not in `marks`."""
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.capped_free_list import _NO_TAIL
    a = TokenToKVPoolAllocator(
        size=64, dtype=torch.bfloat16, device="cpu", kvcache=None,
        need_sort=True,
    )
    a._fl.free_ids = torch.tensor(sorted(free), dtype=torch.int64)
    a._fl.marks = torch.tensor(list(capped), dtype=torch.int64)
    a._fl.pending = torch.empty(0, dtype=torch.int64)
    a._fl.tail_lo = _NO_TAIL
    return a


def test_1_kv_allocator_count_reachable_capped():
    # Logical state: 1,2,3,4 free; 3,4 capped (so free_ids = {1,2}, marks={3,4}).
    a = _kv_with_free_capped(free=[1, 2, 3, 4], capped=[3, 4])
    # cap_t = the fire's target slots {3,4}. Both are properly capped (not in
    # the free list) → 0 reachable (the healthy post-cap-barrier state).
    assert a.count_reachable_capped(torch.tensor([3, 4])) == 0
    # If target 2 was supposed to be capped but leaked (still in the free
    # list, not capped) → 1 reachable.
    assert a.count_reachable_capped(torch.tensor([2, 3, 4])) == 1
    # A target not in the free list at all → 0.
    assert a.count_reachable_capped(torch.tensor([99])) == 0
    print("  PASS  1  KV count_reachable_capped: capped→0, leaked→1, absent→0")


def _make_mamba_cap_alloc(free, capped):
    from sglang.srt.arena.mamba_actuator import _MambaCapAllocator

    class _StubPool:
        def __init__(self):
            self.free_slots = torch.tensor(free, dtype=torch.int64)
            self._capped_slots = torch.tensor(capped, dtype=torch.int64)
            self._alloc_lock = threading.Lock()
    alloc = _MambaCapAllocator.__new__(_MambaCapAllocator)
    alloc.pool = _StubPool()
    alloc.device = torch.device("cpu")
    return alloc


def test_2_mamba_allocator_count_reachable_capped():
    # Mamba mark removes capped slots from free_slots, so the healthy
    # state has the capped targets NOT in free_slots → 0 reachable.
    a = _make_mamba_cap_alloc(free=[1, 2], capped=[3, 4])
    assert a.count_reachable_capped(torch.tensor([3, 4])) == 0
    # Leak: a target slot reappeared in free_slots and isn't capped → 1.
    a2 = _make_mamba_cap_alloc(free=[1, 2, 5], capped=[3, 4])
    assert a2.count_reachable_capped(torch.tensor([5])) == 1
    # Target absent from free → 0.
    assert a2.count_reachable_capped(torch.tensor([99])) == 0
    print("  PASS  2  mamba count_reachable_capped: capped→0, leaked→1, absent→0")


def test_3_equivalence_across_representations():
    free = [1, 2, 3, 4, 5]
    capped = [4, 5]
    cap_t = torch.tensor([2, 4, 5])   # 2 leaked (free, not capped); 4,5 ok
    # KV (free_ids∖marks rep) and mamba (Convention-A free_slots rep) hold the
    # SAME logical config; count_reachable_capped must agree across the two.
    kv = _kv_with_free_capped(free=free, capped=capped)
    mamba = _make_mamba_cap_alloc(free=free, capped=capped)
    n_kv = kv.count_reachable_capped(cap_t)
    n_mamba = mamba.count_reachable_capped(cap_t)
    assert n_kv == n_mamba == 1, (n_kv, n_mamba)
    print(f"  PASS  3  representations agree: KV={n_kv} mamba={n_mamba} (both 1)")


def test_4_actuator_verify_uses_polymorphic_method():
    """Regression guard: execute_async's verify must route through
    `count_reachable_capped`, NOT read `src_alloc.free_pages`
    directly (the latent mamba_to_kv crash)."""
    import inspect
    from sglang.srt.arena import xpool_actuator
    src = inspect.getsource(xpool_actuator.XPoolActuator._execute_async_locked)
    assert "count_reachable_capped" in src, (
        "execute_async verify must call count_reachable_capped"
    )
    assert ".free_pages" not in src, (
        "execute_async must NOT read src_alloc.free_pages directly — "
        "that crashes on a mamba source (mamba_to_kv fire)"
    )
    print("  PASS  4  actuator verify uses count_reachable_capped "
          "(no direct free_pages access)")


def main() -> int:
    tests = [
        test_1_kv_allocator_count_reachable_capped,
        test_2_mamba_allocator_count_reachable_capped,
        test_3_equivalence_across_representations,
        test_4_actuator_verify_uses_polymorphic_method,
    ]
    print(f"\n#264 mamba_to_kv verify tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n#264: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
