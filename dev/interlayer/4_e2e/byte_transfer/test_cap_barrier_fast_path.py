"""cap_barrier clamp-first fast path — correctness + perf (case3 regression fix).

Root cause being locked in (Nemotron-3-120B TP4, case3 conc=128, -9% tps):
each k2m fire's cap_barrier ran on the SCHEDULER thread and expanded + marked
the planner's FULL offered page set (n_pages=80 -> 655,360 token-slot ids via
a Python list -> CUDA tensor, ~200-240 ms), then clamped the grant to the dst
headroom (~5 pages) and unmarked the ~94% surplus. At the Admitter's fire
cadence (~400 fires/rank/rep) this stole ~97 s/rep of scheduler time — the
entire regression (fire records: cap_barrier_us p50=240ms across all runs).

Fix under test (python/sglang/srt/arena/):
  1. kv_actuator / mamba_actuator: `expand_pages_to_token_slots_tensor` —
     vectorized, one tensor op, same contract (page-0 rejected).
  2. xpool_actuator.cap_barrier: free-only plans clamp BEFORE expand/mark and
     mark only the kept pages. Legacy mark-all-then-restore is kept for
     drain/migration plans (Stage-0 entangled; 0 such plans observed in prod).

Tests:
  1/2. tensor expand == list expand (kv, mamba; non-contiguous pages)
  3.   page-0 rejected loudly; empty input -> empty tensor
  4.   fast path end-state == legacy mark-all-then-restore semantics on a
       REAL CappedFreeList-backed allocator (free set, capped set, kept ids)
  5.   FireToken math (per_src/unmapped_total/cap_slots_count) matches the
       legacy formula, including dst-headroom clamp and LCM floor
  6.   perf: fast path cap_barrier < 20 ms at the prod shape (80 pages x
       8192 tps) where legacy cost ~200 ms (scheduler-thread budget)
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/data/yuzhou/projects/sglang/python")

import torch

from sglang.srt.mem_cache.capped_free_list import CappedFreeList


# ---------- stubs (pattern from test_chunk_slot_unit.py) ----------

class _StubArena:
    def __init__(self, tps, max_chunks=100_000):
        self.tokens_per_chunk = tps
        self.max_chunks_per_pool = max_chunks
        self.max_tokens = tps * max_chunks


class _StubKVPool:
    def __init__(self, size, tps):
        self.size = size
        self.page_size = 1
        self._kv_arena = _StubArena(tps)

    def set_capacity_tokens(self, n_tokens):
        pass


class _StubMambaAllocator:
    device = "cpu"

    def set_capacity(self, n):
        pass


class _StubMambaPool:
    def __init__(self, size, tps):
        self.size = size
        self.live_size = size
        self._mamba_temporal_arena = _StubArena(tps)
        self._capped_slots = torch.empty(0, dtype=torch.int64, device="cpu")
        self.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device="cpu")
        self._allocator = _StubMambaAllocator()

    def set_capacity_slots(self, n_slots):
        pass


class _CappedListAllocator:
    """Real CappedFreeList wrapped with the allocator surface cap_barrier
    touches: mark/unmark_pages_capped + device. `size` covers every slot the
    test can mark."""

    def __init__(self, n_slots, device="cpu"):
        self.device = device
        self._fl = CappedFreeList(n_slots, device, need_sort=True, boot_cap=n_slots)

    def mark_pages_capped(self, t):
        return self._fl.mark(t)

    def unmark_pages_capped(self, t):
        return self._fl.unmark(t)

    # observable state for equivalence assertions
    def capped_ids_sorted(self):
        return torch.sort(self._fl.capped_ids())[0]

    def free_ids_sorted(self):
        ids = torch.cat([self._fl.free_ids, self._fl.pending])
        return torch.sort(ids)[0]


class _FakeMTA:
    """Duck-typed MultiTensorArena for _all_subpool_names/lcm_pages."""

    def __init__(self, n_layers, n_kinds):
        self.n_layers = n_layers
        self.n_kinds = n_kinds

    def _pool_name(self, i):
        return f"p{i}"


class _FakeActuator:
    """Duck-typed per-pool actuator: real expand fns bound to a stub pool,
    real CappedFreeList allocator, controllable grow headroom."""

    def __init__(self, real_actuator, allocator, headroom_pages):
        self._real = real_actuator
        self.allocator = allocator
        self._headroom = headroom_pages

    def expand_pages_to_token_slots(self, page_ids):
        return self._real.expand_pages_to_token_slots(page_ids)

    def expand_pages_to_token_slots_tensor(self, page_ids, device):
        return self._real.expand_pages_to_token_slots_tensor(page_ids, device)

    def grow_headroom_pages(self):
        return self._headroom


def _build_kv_real(tps):
    from sglang.srt.arena.kv_actuator import KVArenaActuator

    class _A:  # allocator stub only for KVArenaActuator.__init__
        size = 10_000_000
        device = "cpu"

    return KVArenaActuator(pool=_StubKVPool(size=10_000_000, tps=tps), allocator=_A())


def _build_mamba_real(tps):
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator

    return MambaArenaActuator(pool=_StubMambaPool(size=4096, tps=tps))


def _build_xpool(n_kv_sub=(8, 2), n_mamba_sub=(40, 2), kv_act=None, mamba_act=None):
    """XPoolActuator via __new__ (skip the arena-identity __init__ checks);
    set exactly the fields cap_barrier reads."""
    from sglang.srt.arena.xpool_actuator import XPoolActuator

    x = XPoolActuator.__new__(XPoolActuator)
    x.kv = _FakeMTA(*n_kv_sub)
    x.mamba = _FakeMTA(*n_mamba_sub)
    x.kv_actuator = kv_act
    x.mamba_actuator = mamba_act
    x.n_kv_subpools = n_kv_sub[0] * n_kv_sub[1]
    x.n_mamba_subpools = n_mamba_sub[0] * n_mamba_sub[1]
    x.stage0_handler = None
    x.kv_serving_floor_tokens = 0
    return x


def _make_plan(direction, pages, map_dst, seq=1):
    from sglang.srt.arena.fire_plan import FirePlan

    return FirePlan(
        direction=direction,
        pages_to_unmap=list(pages),
        pages_to_map_dst=map_dst,
        plan_seq=seq,
        drains=(),
        migrations=(),
    )


# ---------- tests ----------

def test_1_tensor_expand_equals_list_expand_kv():
    act = _build_kv_real(tps=8192)
    pages = [3, 7, 100, 101, 999]  # non-contiguous
    ref = torch.tensor(
        act.expand_pages_to_token_slots(pages), dtype=torch.int64
    )
    out = act.expand_pages_to_token_slots_tensor(pages, "cpu")
    assert torch.equal(ref, out), "kv tensor expand != list expand"
    print("test_1 OK  (kv tensor expand == list expand)")


def test_2_tensor_expand_equals_list_expand_mamba():
    act = _build_mamba_real(tps=1)
    pages = [1, 2, 50, 51]
    ref = torch.tensor(
        act.expand_pages_to_token_slots(pages), dtype=torch.int64
    )
    out = act.expand_pages_to_token_slots_tensor(pages, "cpu")
    assert torch.equal(ref, out), "mamba tensor expand != list expand"
    # tps>1 mamba variant too
    act8 = _build_mamba_real(tps=8)
    ref8 = torch.tensor(
        act8.expand_pages_to_token_slots(pages), dtype=torch.int64
    )
    out8 = act8.expand_pages_to_token_slots_tensor(pages, "cpu")
    assert torch.equal(ref8, out8)
    print("test_2 OK  (mamba tensor expand == list expand, tps=1 and tps=8)")


def test_3_page0_rejected_and_empty_ok():
    act = _build_kv_real(tps=64)
    try:
        act.expand_pages_to_token_slots_tensor([4, 0, 9], "cpu")
        raise AssertionError("page 0 not rejected")
    except ValueError:
        pass
    empty = act.expand_pages_to_token_slots_tensor([], "cpu")
    assert empty.numel() == 0 and empty.dtype == torch.int64
    print("test_3 OK  (page-0 rejected loudly; empty -> empty tensor)")


def test_4_fast_path_endstate_equals_legacy_semantics():
    """The core claim: mark(kept) ≡ mark(all) -> unmark(surplus) on the real
    CappedFreeList, and the returned cap_t is exactly the kept slots."""
    tps = 64
    n_slots = 200_000
    offered = list(range(10, 90))  # 80 pages, like prod
    headroom = 5                   # dst clamp, like prod

    kv_real = _build_kv_real(tps=tps)
    mamba_real = _build_mamba_real(tps=1)

    # --- fast path (the new code) ---
    alloc_fast = _CappedListAllocator(n_slots)
    kv_act = _FakeActuator(kv_real, alloc_fast, headroom_pages=10**9)
    mamba_act = _FakeActuator(mamba_real, _CappedListAllocator(4096), headroom)
    x = _build_xpool(kv_act=kv_act, mamba_act=mamba_act)
    plan = _make_plan("kv_to_mamba", offered, map_dst=len(offered))
    token = x.cap_barrier(plan)
    assert not token.aborted

    # --- legacy reference semantics on an identical allocator ---
    alloc_ref = _CappedListAllocator(n_slots)
    all_slots = torch.tensor(
        kv_real.expand_pages_to_token_slots(offered), dtype=torch.int64
    )
    alloc_ref.mark_pages_capped(all_slots)
    per_src = token.per_src
    surplus = offered[per_src:]
    if surplus:
        surplus_slots = torch.tensor(
            kv_real.expand_pages_to_token_slots(surplus), dtype=torch.int64
        )
        alloc_ref.unmark_pages_capped(surplus_slots)
    kept_ref = torch.tensor(
        kv_real.expand_pages_to_token_slots(offered[:per_src]),
        dtype=torch.int64,
    )

    assert torch.equal(torch.sort(token.cap_t)[0], torch.sort(kept_ref)[0]), \
        "fast-path cap_t != legacy kept slots"
    assert torch.equal(
        alloc_fast.capped_ids_sorted(), alloc_ref.capped_ids_sorted()
    ), "fast-path capped set != legacy capped set"
    assert torch.equal(
        alloc_fast.free_ids_sorted(), alloc_ref.free_ids_sorted()
    ), "fast-path free set != legacy free set"
    print(f"test_4 OK  (end-state equivalence; per_src={per_src}, "
          f"kept_slots={token.cap_t.numel()})")


def test_5_token_math_matches_legacy_formula():
    """per_src / unmapped_total: replicate the legacy formula literally and
    compare, across headroom / LCM / offered-size corners."""
    tps = 16
    kv_real = _build_kv_real(tps=tps)
    mamba_real = _build_mamba_real(tps=1)

    cases = [
        # (n_kv_sub, n_mamba_sub, n_offered, map_dst, headroom)
        ((8, 2), (40, 2), 80, 80, 5),      # prod-like: heavy clamp
        ((8, 2), (40, 2), 80, 80, 10**9),  # no clamp
        ((8, 2), (40, 2), 3, 3, 10**9),    # below one LCM unit -> 0
        ((4, 1), (6, 1), 24, 24, 7),       # lcm(4,6)=12 rounding
        ((8, 2), (40, 2), 80, 40, 10**9),  # dst asks less than offered
    ]
    for n_kv, n_mamba, n_off, map_dst, headroom in cases:
        offered = list(range(1, 1 + n_off))
        alloc = _CappedListAllocator(200_000)
        kv_act = _FakeActuator(kv_real, alloc, 10**9)
        mamba_act = _FakeActuator(mamba_real, _CappedListAllocator(4096), headroom)
        x = _build_xpool(n_kv_sub=n_kv, n_mamba_sub=n_mamba,
                         kv_act=kv_act, mamba_act=mamba_act)
        plan = _make_plan("kv_to_mamba", offered, map_dst=map_dst)
        token = x.cap_barrier(plan)

        # legacy formula, verbatim from the pre-fix code
        import math
        n_src = n_kv[0] * n_kv[1]
        n_dst = n_mamba[0] * n_mamba[1]
        lcm_n = math.lcm(n_src, n_dst)
        target_src_total = n_src * len(offered)
        target_dst_total = min(n_dst * map_dst, n_dst * headroom)
        total = (min(target_src_total, target_dst_total) // lcm_n) * lcm_n
        exp_per_src = total // n_src if n_src else 0
        exp_unmapped = exp_per_src * n_src

        assert token.per_src == exp_per_src, \
            f"per_src {token.per_src} != {exp_per_src} for {n_kv},{n_mamba},{headroom}"
        assert token.unmapped_total == exp_unmapped
        assert token.cap_slots_count == exp_per_src * tps
    print(f"test_5 OK  (token math == legacy formula, {len(cases)} corners)")


def test_6_fast_path_perf_budget():
    """Prod shape: 80 pages x 8192 tps. Relative budget (load-immune):
    the fast path must beat the legacy expand+tensor cost by >=5x measured
    in the same process (absolute numbers swing 40x with machine load)."""
    tps = 8192
    kv_real = _build_kv_real(tps=tps)
    mamba_real = _build_mamba_real(tps=1)
    alloc = _CappedListAllocator(10_000_000)
    kv_act = _FakeActuator(kv_real, alloc, 10**9)
    mamba_act = _FakeActuator(mamba_real, _CappedListAllocator(4096), 5)
    x = _build_xpool(kv_act=kv_act, mamba_act=mamba_act)

    pages = list(range(100, 180))

    # Legacy cost proxy: full-offered-set python expand + list->tensor
    # (the dominant terms of the pre-fix cap_barrier).
    legacy = []
    for _ in range(3):
        t0 = time.perf_counter()
        slots = kv_real.expand_pages_to_token_slots(pages)
        torch.tensor(slots, dtype=torch.int64)
        legacy.append((time.perf_counter() - t0) * 1000)
    t_legacy = min(legacy)

    times = []
    for seq in range(5):
        plan = _make_plan("kv_to_mamba", pages, map_dst=80, seq=seq + 1)
        t0 = time.perf_counter()
        token = x.cap_barrier(plan)
        times.append((time.perf_counter() - t0) * 1000)
        # undo the mark so each iteration starts clean
        if token.cap_t.numel():
            alloc.unmark_pages_capped(token.cap_t)
    best = min(times)
    assert best < t_legacy / 5, \
        f"fast path {best:.1f}ms not >=5x faster than legacy {t_legacy:.1f}ms"
    print(f"test_6 OK  (fast path {best:.2f}ms vs legacy expand+tensor "
          f"{t_legacy:.1f}ms, {t_legacy/best:.0f}x)")


def test_7_kv_serving_floor_clamps_k2m():
    """k2m fires must never shrink KV available below the serving floor
    (one prefill chunk + decode headroom) — the missing invariant behind
    the 9B dynamic OOM crash ('Available full tokens: 6408, evictable 0')."""
    tps = 64
    kv_real = _build_kv_real(tps=tps)
    mamba_real = _build_mamba_real(tps=1)

    class _FloorAlloc(_CappedListAllocator):
        def __init__(self, n_slots, avail):
            super().__init__(n_slots)
            self._avail = avail

        def available_size(self):
            return self._avail

    # KV has 1000 tokens available; floor 500 -> at most (1000-500)//64 = 7
    # shrinkable pages regardless of what the planner offers.
    alloc = _FloorAlloc(200_000, avail=1000)
    kv_act = _FakeActuator(kv_real, alloc, 10**9)
    kv_act._tokens_per_page = lambda: tps
    mamba_act = _FakeActuator(mamba_real, _CappedListAllocator(4096), 10**9)
    x = _build_xpool(n_kv_sub=(1, 1), n_mamba_sub=(1, 1),
                     kv_act=kv_act, mamba_act=mamba_act)
    x.kv_serving_floor_tokens = 500

    plan = _make_plan("kv_to_mamba", list(range(10, 90)), map_dst=80)
    token = x.cap_barrier(plan)
    assert token.per_src == 7, f"floor clamp failed: per_src={token.per_src}"

    # floor disabled (0) -> no clamp
    alloc2 = _FloorAlloc(200_000, avail=1000)
    kv_act2 = _FakeActuator(kv_real, alloc2, 10**9)
    kv_act2._tokens_per_page = lambda: tps
    x2 = _build_xpool(n_kv_sub=(1, 1), n_mamba_sub=(1, 1),
                      kv_act=kv_act2, mamba_act=mamba_act)
    x2.kv_serving_floor_tokens = 0
    token2 = x2.cap_barrier(_make_plan("kv_to_mamba", list(range(10, 90)),
                                       map_dst=80, seq=2))
    assert token2.per_src == 80, f"floor=0 should not clamp: {token2.per_src}"

    # avail already below floor -> zero-page fire (fail closed, no underflow)
    alloc3 = _FloorAlloc(200_000, avail=400)
    kv_act3 = _FakeActuator(kv_real, alloc3, 10**9)
    kv_act3._tokens_per_page = lambda: tps
    x3 = _build_xpool(n_kv_sub=(1, 1), n_mamba_sub=(1, 1),
                      kv_act=kv_act3, mamba_act=mamba_act)
    x3.kv_serving_floor_tokens = 500
    token3 = x3.cap_barrier(_make_plan("kv_to_mamba", list(range(10, 90)),
                                       map_dst=80, seq=3))
    assert token3.per_src == 0, f"below-floor must grant 0: {token3.per_src}"
    print("test_7 OK  (k2m serving floor: clamps to headroom, 0-disables, "
          "fails closed below floor)")


def test_8_floor_does_not_touch_m2k():
    """The KV serving floor is a k2m (KV-source) guard only; m2k fires
    (mamba source) are governed by the admission-cap floor elsewhere."""
    tps = 64
    kv_real = _build_kv_real(tps=tps)
    mamba_real = _build_mamba_real(tps=1)
    mamba_alloc = _CappedListAllocator(200_000)
    mamba_act = _FakeActuator(mamba_real, mamba_alloc, 10**9)
    kv_act = _FakeActuator(kv_real, _CappedListAllocator(200_000), 10**9)
    x = _build_xpool(n_kv_sub=(1, 1), n_mamba_sub=(1, 1),
                     kv_act=kv_act, mamba_act=mamba_act)
    x.kv_serving_floor_tokens = 10**9  # absurd floor; must not affect m2k

    plan = _make_plan("mamba_to_kv", list(range(10, 50)), map_dst=40)
    token = x.cap_barrier(plan)
    assert token.per_src == 40, f"m2k must ignore the kv floor: {token.per_src}"
    print("test_8 OK  (m2k unaffected by the kv serving floor)")


if __name__ == "__main__":
    test_1_tensor_expand_equals_list_expand_kv()
    test_2_tensor_expand_equals_list_expand_mamba()
    test_3_page0_rejected_and_empty_ok()
    test_4_fast_path_endstate_equals_legacy_semantics()
    test_5_token_math_matches_legacy_formula()
    test_6_fast_path_perf_budget()
    test_7_kv_serving_floor_clamps_k2m()
    test_8_floor_does_not_touch_m2k()
    print("\nALL TESTS PASSED")
