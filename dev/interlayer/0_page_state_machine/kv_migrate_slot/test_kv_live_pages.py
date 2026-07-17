"""#271 step 4 — SchedulerOwnerProvider._live_pages_in_cost_order KV branch.

Stage-3 Migration consolidates scattered LIVE KV slots so a cross-fire k2m
can free (transfer) a whole chunk. KV "pages" are arena chunks of `tps`
token-slots (tps ≈ tokens_per_chunk, large), so KV is highly fragmentable —
unlike mamba (tps=1, atomic-inert). The walk must:
  - SOURCE = a fully-LIVE-UNCACHED page (all tps slots live, none cached/free),
  - DONORS = free slots on PARTIAL (kept) pages,
  - skip whole-free pages (Stage-1 payload), capped pages (mid-fire), page 0,
  - EXCLUDE cached slots (audit H2: migrating a cached/shared slot would
    orphan the radix node + other reqs' rows — Migration is LIVE-uncached),
emitting (freed_page, ((src,dst),...)) with each source slot paired to a
distinct donor. CPU: real TokenToKVPoolAllocator + a fake actuator/cache.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

# The walk is fail-closed-gated behind SGLANG_XPOOL_KV_MIGRATE (#271 step 5,
# default OFF until #291). The walk-logic tests enable it; test_4 pins the gate.
os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "1"


def _provider(allocator, tps, n_pages, cached_slots):
    from sglang.srt.budgeter.scheduler_owner_provider import SchedulerOwnerProvider
    kv_act = types.SimpleNamespace(
        _tokens_per_page=lambda: tps, n_pages=n_pages,
    )
    sched = types.SimpleNamespace(
        token_to_kv_pool_allocator=allocator, tree_cache=None,
    )
    prov = SchedulerOwnerProvider(
        scheduler=sched, kv_actuator=kv_act, mamba_actuator=None,
    )
    # Inject the cached-KV-slot set (production reuses _iter_drain_victims;
    # here we isolate the page-classification logic from the radix walk).
    prov._cached_kv_slots = lambda: set(cached_slots)
    return prov


def _alloc(size):
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    class _StubKV:
        page_size = 1
        def can_move_kv_cache(self):
            return True
        def get_kv_size_bytes(self):
            return 0
        def move_kv_cache(self, tgt, src):
            pass

    return TokenToKVPoolAllocator(
        size=size, dtype=torch.float16, device="cpu",
        kvcache=_StubKV(), need_sort=False,
    )


def test_1_consolidates_fully_live_page_into_scattered_donors():
    """tps=4, 6 pages (slots 0..23):
      p0 sentinel; p1 [4-7] fully-live → SOURCE; p2 [8,9 free | 10,11 live]
      and p3 [12,13 free | 14,15 live] → 4 donors; p4 [16-19] whole-free
      (skip); p5 [20-23] live-but-CACHED (excluded → not a source).
    Expect one move set emptying p1 into the 4 partial-page donors."""
    a = _alloc(size=24)
    a.free_pages = torch.tensor([8, 9, 12, 13, 16, 17, 18, 19], dtype=torch.int64)
    prov = _provider(a, tps=4, n_pages=6, cached_slots={20, 21, 22, 23})
    out = prov._live_pages_in_cost_order("kv")
    assert out == [(1, ((4, 8), (5, 9), (6, 12), (7, 13)))], (
        f"expected p1 consolidated into donors [8,9,12,13]; got {out}"
    )
    print("  PASS  1  KV migration: fully-live page → scattered partial-page "
          "donors (cached + whole-free excluded)")


def test_2_no_donors_when_only_whole_free():
    """A fully-live source but NO partial pages (donors only from whole-free
    pages, which are Stage-1 payload, not donors) → no migration."""
    a = _alloc(size=12)  # tps=4, 3 pages
    # p1 [4-7] fully-live (source); p2 [8-11] whole-free (not a donor).
    a.free_pages = torch.tensor([8, 9, 10, 11], dtype=torch.int64)
    prov = _provider(a, tps=4, n_pages=3, cached_slots=set())
    out = prov._live_pages_in_cost_order("kv")
    assert out == [], f"whole-free pages are not donors; expected []; got {out}"
    print("  PASS  2  no partial-page donors → no migration (whole-free is "
          "Stage-1 payload, not a donor)")


def test_3_atomic_tps1_yields_nothing():
    """tps=1 (atomic): every page is 1-live or 1-free, no partial pages →
    no scattered donors → [] (mirrors mamba atomic-inert, #269)."""
    a = _alloc(size=8)
    a.free_pages = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
    prov = _provider(a, tps=1, n_pages=8, cached_slots=set())
    out = prov._live_pages_in_cost_order("kv")
    assert out == [], f"atomic tps=1 must yield no migration; got {out}"
    print("  PASS  3  atomic (tps=1) KV → [] (no consolidation possible)")


def test_4_fail_closed_disabled_by_default():
    """#271 step-5 gate: SGLANG_XPOOL_KV_MIGRATE defaults OFF — even a
    migratable layout (test_1's) yields NO moves, so a cross_migrate
    candidate degrades to free-only/drain until #291 validates the path and
    enable_kv_cache_copy is wired. Flipping the flag re-enables it."""
    a = _alloc(size=24)
    a.free_pages = torch.tensor([8, 9, 12, 13, 16, 17, 18, 19], dtype=torch.int64)
    prov = _provider(a, tps=4, n_pages=6, cached_slots={20, 21, 22, 23})
    prev = os.environ.get("SGLANG_XPOOL_KV_MIGRATE")
    try:
        os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "0"
        assert prov._live_pages_in_cost_order("kv") == [], (
            "default-OFF gate must yield no KV migrations"
        )
        os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "1"
        assert prov._live_pages_in_cost_order("kv"), (
            "flag ON re-enables the same migratable layout"
        )
    finally:
        if prev is None:
            os.environ["SGLANG_XPOOL_KV_MIGRATE"] = "1"  # restore module default
        else:
            os.environ["SGLANG_XPOOL_KV_MIGRATE"] = prev
    print("  PASS  4  fail-closed gate (SGLANG_XPOOL_KV_MIGRATE default OFF)")


def main() -> int:
    tests = [
        test_1_consolidates_fully_live_page_into_scattered_donors,
        test_2_no_donors_when_only_whole_free,
        test_3_atomic_tps1_yields_nothing,
        test_4_fail_closed_disabled_by_default,
    ]
    print(f"\n#271 step 4 KV live-pages walk tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\n#271 step 4: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
