"""A1 (#282) — KV allocator must be GROWABLE past boot (port of the #279
MambaPool dynamic-cap pattern to the KV side).

ROOT CAUSE this pins: m2k (grow KV by shrinking mamba) regresses on real
cc traces because the KV allocator CANNOT grow past its boot size —
`set_capacity_pages` can only restore `_capped_pages`, which are bounded
by `[1, self.size]`, and at boot `_cap == size` so `_capped_pages` is
empty. The KV *arena* reserves VA headroom (43M tokens) but the allocator
accounting caps at boot. So m2k shrinks mamba (real loss) while KV can't
grow (zero gain) → pure loss. This is the #279 bug (MambaPool
`max_size == size`) on the KV side; the fix mirrors MambaPool: construct
with `max_size > size`, hold the deferred `[size+1, max_size]` range in
`_capped_pages`, and let `set_capacity_pages` grow into it — while
`available_size`/live capacity stays correct (only capped ids within the
live cap subtract).

Test-first (bug-workflow): these are RED until the allocator gains the
`max_size` dynamic-cap support, then GREEN.

CPU-only (stub kvcache); no GPU needed.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch


class _StubKV:
    """Minimal KVCache stand-in — the allocator stores it but the
    capacity/cap paths under test never call into it."""
    def __init__(self):
        self.page_size = 1
        self.move_calls = []  # records (tgt_loc, src_loc) for migrate_slot

    def get_kv_size_bytes(self):
        return 0

    def can_move_kv_cache(self) -> bool:
        # Models a movable MHA pool (the authoritative guard migrate_slot
        # checks — see MHATokenToKVPool.can_move_kv_cache).
        return True

    def move_kv_cache(self, tgt_loc, src_loc):
        # Record the byte-move the allocator delegates; the real per-layer
        # copy is exercised on GPU by the #271 spike (test_kv_migrate_replay).
        self.move_calls.append((tgt_loc.tolist(), src_loc.tolist()))


def _make_alloc(size, max_size=None):
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    kwargs = dict(
        size=size, dtype=torch.float16, device="cpu",
        kvcache=_StubKV(), need_sort=False,
    )
    if max_size is not None:
        kwargs["max_size"] = max_size
    return TokenToKVPoolAllocator(**kwargs)


def test_1_back_compat_no_max_size():
    """max_size omitted → behaves exactly as before (size==boot, no
    headroom, available_size==size)."""
    a = _make_alloc(size=8)
    assert a.size == 8, f"size={a.size}"
    assert a._capped_pages.numel() == 0, "no headroom when max_size omitted"
    # available_size = live capacity = 8 (minus padded slot 0 convention)
    assert a.available_size() == 8, f"available_size={a.available_size()}"
    print("  PASS  1  back-compat: no max_size → size=8, no capped headroom")


def test_2_dynamic_headroom_boots_capped():
    """max_size > size → allocator boots at LIVE cap `size`, with the
    deferred [size+1, max_size] range held in _capped_pages; available_size
    reflects the live cap (size), NOT max_size."""
    a = _make_alloc(size=8, max_size=32)
    assert a.max_size == 32, f"max_size={a.max_size}"
    # live capacity at boot = 8 (the deferred [9..32] are capped, unmapped)
    assert a.available_size() == 8, (
        f"boot live capacity must be 8 (not max_size); got "
        f"{a.available_size()}"
    )
    capped = set(a._capped_pages.tolist())
    assert capped == set(range(9, 33)), (
        f"_capped_pages must hold deferred [9..32]; got {sorted(capped)}"
    )
    print("  PASS  2  max_size=32 → boot live=8, _capped=[9..32]")


def test_3_grow_past_boot():
    """The core fix: set_capacity_pages can grow the live cap ABOVE boot
    size, into the headroom — the page-ids become allocatable."""
    a = _make_alloc(size=8, max_size=32)
    assert a.available_size() == 8
    a.set_capacity_pages(20)             # grow live cap 8 → 20
    assert a.available_size() == 20, (
        f"after grow to 20, live capacity must be 20; got "
        f"{a.available_size()}"
    )
    # Convention A: free_pages spans the full id space; "allocatable" is
    # free MINUS _capped. After grow to 20, [9..20] are un-capped (now
    # allocatable); (20..32] stay in _capped.
    capped = set(a._capped_pages.tolist())
    assert capped == set(range(21, 33)), (
        f"after grow to 20, _capped must be (20..32]; got {sorted(capped)}"
    )
    assert not ({9, 15, 20} & capped), "grown ids must no longer be capped"
    print("  PASS  3  set_capacity_pages grows live cap 8 → 20 past boot")


def test_4_grow_shrink_grow_consistent():
    """Grow to max, shrink, grow again — live capacity + capped stay
    consistent (no double-count, no leak)."""
    a = _make_alloc(size=4, max_size=16)
    a.set_capacity_pages(16)
    assert a.available_size() == 16, f"grow-to-max live={a.available_size()}"
    a.set_capacity_pages(6)
    assert a.available_size() == 6, f"shrink live={a.available_size()}"
    a.set_capacity_pages(16)
    assert a.available_size() == 16, f"re-grow live={a.available_size()}"
    # all ids restored, none lost
    assert set(a.free_pages.tolist()) == set(range(1, 17)), (
        f"re-grow must restore [1..16]; got {sorted(a.free_pages.tolist())}"
    )
    print("  PASS  4  grow → shrink → grow restores all ids, no leak")


def test_5_actuator_max_tokens_is_arena_max():
    """KVArenaActuator must cap at the ARENA's max (growable), not the
    boot pool.size. (Stub the pool surface the actuator reads.)"""
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    import types

    class _Arena:
        tokens_per_chunk = 1
        max_tokens = 1000
    pool = types.SimpleNamespace(
        _kv_arena=_Arena(), size=100, page_size=1,
    )
    alloc = _make_alloc(size=100, max_size=1000)
    act = KVArenaActuator(pool, alloc)
    assert act.max_tokens >= 1000, (
        f"KVArenaActuator.max_tokens must be the arena max (>=1000, "
        f"growable), not boot pool.size (100); got {act.max_tokens}"
    )
    print(f"  PASS  5  KVArenaActuator.max_tokens={act.max_tokens} (arena max, growable)")


def test_6_set_capacity_shrink_available_consistent():
    """#283 pre-existing-bug repro (no max_size): set_capacity_pages must
    keep available_size consistent. The OLD set_capacity_pages removed
    capped ids from free_pages while available_size subtracts capped →
    double-count → available=0 after shrink-to-4 on size-8. After the
    Convention-A reconciliation, available == live cap."""
    a = _make_alloc(size=8)
    assert a.available_size() == 8
    a.set_capacity_pages(4)            # shrink live cap to 4
    assert a.available_size() == 4, (
        f"#283: after set_capacity_pages(4), available must be 4 "
        f"(was 0 pre-fix: free/capped double-counted); got "
        f"{a.available_size()}"
    )
    a.set_capacity_pages(8)            # grow back
    assert a.available_size() == 8, f"re-grow available={a.available_size()}"
    print("  PASS  6  #283: set_capacity_pages shrink/grow keeps available consistent")


def test_7_alloc_never_returns_capped():
    """alloc() must never hand out an id that is capped (deferred headroom
    OR mark/set_capacity-capped). The whole point of Convention A: capped
    ids may remain in free_pages but alloc filters them."""
    a = _make_alloc(size=8, max_size=32)
    # live cap 8 → only ids [1..8] allocatable; [9..32] are deferred-capped
    got = a.alloc(8)
    assert got is not None, "alloc(8) within live cap must succeed"
    ids = set(int(x) for x in got.tolist())
    assert ids.issubset(set(range(1, 9))), (
        f"alloc returned capped/headroom ids: {sorted(ids)}"
    )
    # pool now exhausted at live cap → next alloc fails (no leak into headroom)
    assert a.alloc(1) is None, "must NOT allocate into deferred headroom"
    print("  PASS  7  alloc never returns capped/headroom ids")


def test_8_grow_then_alloc_uses_new_pages():
    """After growing the live cap, the newly-uncapped ids become
    allocatable (the actual point of A1: m2k grows KV → more usable KV)."""
    a = _make_alloc(size=8, max_size=32)
    a.alloc(8)                          # exhaust boot live cap
    assert a.alloc(1) is None
    a.set_capacity_pages(16)            # grow live cap 8 → 16
    got = a.alloc(8)                    # 8 more should now be allocatable
    assert got is not None, "after grow, the new headroom must be allocatable"
    ids = set(int(x) for x in got.tolist())
    assert ids.issubset(set(range(9, 17))), (
        f"grown alloc must use ids [9..16]; got {sorted(ids)}"
    )
    print("  PASS  8  grow → newly-uncapped ids become allocatable")


def test_9_set_capacity_and_mark_compose():
    """set_capacity_pages (range cap) and mark_pages_capped (specific ids)
    must compose under one convention — capped count is the union, no
    double-count, available stays correct."""
    a = _make_alloc(size=16)
    a.set_capacity_pages(12)            # cap (12,16] → 4 capped
    assert a.available_size() == 12
    a.mark_pages_capped(torch.tensor([3, 5], dtype=torch.int64))  # +2 within live
    assert a.available_size() == 10, (
        f"set_capacity(12) + mark(2 live ids) → available 10; got "
        f"{a.available_size()}"
    )
    a.unmark_pages_capped(torch.tensor([3, 5], dtype=torch.int64))
    assert a.available_size() == 12, f"unmark → 12; got {a.available_size()}"
    print("  PASS  9  set_capacity_pages + mark_pages_capped compose (no double-count)")


def test_10_capped_invariant_holds():
    """_capped_pages.numel() must never exceed self.size (the #162
    fail-fast), including with dynamic headroom."""
    a = _make_alloc(size=8, max_size=32)
    a._assert_capped_invariant()        # boot: capped=[9..32]=24 <= size=32
    a.set_capacity_pages(32)            # full grow
    a._assert_capped_invariant()
    a.set_capacity_pages(8)             # back to boot
    a._assert_capped_invariant()
    print("  PASS  10  _capped invariant holds across grow/shrink")


def test_11_need_sort_path():
    """need_sort=True alloc path (free_group / merge_and_sort) must also
    respect capping (#161 coverage)."""
    a = _make_alloc(size=8, max_size=32)
    # need_sort default False in _make_alloc; build one with need_sort
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    a = TokenToKVPoolAllocator(
        size=8, dtype=torch.float16, device="cpu",
        kvcache=_StubKV(), need_sort=True, max_size=32,
    )
    got = a.alloc(8)
    assert got is not None and set(int(x) for x in got.tolist()).issubset(set(range(1, 9)))
    assert a.alloc(1) is None, "need_sort path must not alloc into headroom"
    print("  PASS  11  need_sort=True path respects cap (#161)")



def test_13_set_capacity_out_of_range_clamped():
    """HIGH (audit-2): set_capacity_pages must clamp out-of-range n to
    [1, size] — never corrupt _cap or trip the invariant on a LATER op.
    Pre-fix: n>size silently set _cap past size → next shrink crashed the
    invariant; n<0 built arange over slot 0 / negatives."""
    a = _make_alloc(size=8, max_size=32)
    a.set_capacity_pages(100)            # over-grow → clamp to 32 (ceiling)
    assert a._cap == 32, f"over-grow must clamp _cap to size; got {a._cap}"
    a._assert_capped_invariant()
    a.set_capacity_pages(4)              # follow-up shrink must stay consistent
    assert a.available_size() == 4 and a._cap == 4
    a._assert_capped_invariant()
    a.set_capacity_pages(0)              # under-floor → clamp to 1
    assert a._cap == 1, f"n=0 must clamp to 1; got {a._cap}"
    a._assert_capped_invariant()
    a.set_capacity_pages(-5)             # negative → clamp to 1 (no slot-0 cap)
    assert a._cap == 1 and 0 not in set(a._capped_pages.tolist())
    a._assert_capped_invariant()
    print("  PASS  13  set_capacity_pages clamps out-of-range n (no corruption)")


def test_14_max_size_less_than_size_raises():
    """LOW (audit-2): max_size < size is a construction error → raise."""
    raised = False
    try:
        _make_alloc(size=16, max_size=8)
    except ValueError:
        raised = True
    assert raised, "max_size < size must raise ValueError"
    print("  PASS  14  max_size < size raises")


def test_15_grow_leaves_marks_intact():
    """HIGH (audit-2): a grow must un-cap ONLY its range (cap, n] and leave
    mark_pages_capped ids OUTSIDE that range capped (the diff comment's
    explicit contract — previously unasserted)."""
    a = _make_alloc(size=16)
    a.mark_pages_capped(torch.tensor([3, 5], dtype=torch.int64))  # mark within live
    a.set_capacity_pages(4)              # shrink → capped {3,5} ∪ {5..16} = {3,5..16}
    assert a.available_size() == 3, f"live after shrink={a.available_size()}"
    a.set_capacity_pages(16)             # grow back: un-cap (4,16], keep mark {3}
    capped = set(a._capped_pages.tolist())
    assert capped == {3}, (
        f"grow must leave mark {{3}} capped, un-cap only the range; got {sorted(capped)}"
    )
    assert a.available_size() == 15, f"live=size-1mark=15; got {a.available_size()}"
    print("  PASS  15  grow un-caps range, leaves mark_pages_capped ids intact")


def test_16_clear_after_grow_preserves_cap():
    """HIGH (audit-2): clear() (flush_cache mid-run) must rebuild to the
    CURRENT live cap, not reset to boot — _cap preserved, the free list rebuilt
    to the live cap [1,cap], _capped rebuilt to (cap, size]."""
    a = _make_alloc(size=8, max_size=32)
    a.set_capacity_pages(20)             # grow live cap to 20
    a.clear()                            # flush
    assert a._cap == 20, f"clear must preserve _cap=20; got {a._cap}"
    assert a.available_size() == 20, f"live after clear={a.available_size()}"
    # The free list holds exactly the allocatable live cap [1,20]; the capped
    # tail [21,32] is NOT in it (that residency was the per-token isin tax).
    assert set(a.free_pages.tolist()) == set(range(1, 21)), "free = live cap [1,20]"
    assert set(a._capped_pages.tolist()) == set(range(21, 33)), "capped=(20,32]"
    print("  PASS  16  clear() after grow preserves live cap (not reset to boot)")



def test_18_r1_usage_under_load_after_grow():
    """full_token_usage = used / live_size AFTER a grow with live allocations.
    live=300, used=150 -> 0.5, not 150/1000 (ceiling) nor 150/100 (boot)."""
    a = _make_alloc(size=100, max_size=1000)
    a.set_capacity_pages(300)
    a.alloc(150)
    available = a.available_size()
    evictable = 0
    num_used = a.live_size - (available + evictable)
    usage = num_used / a.live_size if a.live_size > 0 else 0.0
    assert num_used == 150, f"used must be 150; got {num_used}"
    assert abs(usage - 0.5) < 1e-9, (
        f"usage must be 150/live(300)=0.5, not /ceiling(1000) nor /boot(100); "
        f"got {usage}"
    )
    print("  PASS  18  R1 under load: full_token_usage = used/live after grow")


def test_19_many_grow_shrink_cycles_no_leak():
    """LOW (audit-2): repeated grow/shrink must not leak or duplicate ids
    (guards slow torch.cat accumulation in _capped_pages / free_pages)."""
    a = _make_alloc(size=8, max_size=64)
    for _ in range(12):
        a.set_capacity_pages(64)
        a.set_capacity_pages(8)
    a.set_capacity_pages(64)
    free = a.free_pages.tolist()
    assert len(free) == len(set(free)), "duplicate ids after many cycles"
    assert set(free) == set(range(1, 65)), f"id set drifted: {sorted(set(free))[:5]}..."
    assert a._capped_pages.numel() == 0, "fully grown → no capped"
    assert a.available_size() == 64
    a._assert_capped_invariant()
    print("  PASS  19  12× grow/shrink cycles: no leak, no duplicate ids")


def test_20_concurrent_alloc_vs_set_capacity():
    """LOW (audit-2, best-effort): the _alloc_lock must keep alloc()/free()
    consistent with concurrent set_capacity_pages (the cross-pool actuator
    runs on a worker thread). Non-deterministic, but exercises the lock and
    asserts no crash + invariant + no duplicate live ids."""
    import threading
    a = _make_alloc(size=256, max_size=1024)
    stop = [False]
    errs = []

    def capper():
        try:
            n = 256
            while not stop[0]:
                n = 1024 if n == 256 else 256
                a.set_capacity_pages(n)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))

    t = threading.Thread(target=capper)
    t.start()
    try:
        held = []
        for _ in range(2000):
            g = a.alloc(1)
            if g is not None:
                held.append(g)
            if len(held) > 50:
                a.free(torch.cat(held[:25])); held = held[25:]
    except Exception as e:  # noqa: BLE001
        errs.append(repr(e))
    finally:
        stop[0] = True; t.join(timeout=5)
    assert not errs, f"concurrency errors: {errs[:3]}"
    a._assert_capped_invariant()
    print("  PASS  20  concurrent alloc/free vs set_capacity_pages: no crash, invariant holds")


def test_21_paged_allocator_is_boot_only_scope():
    """LOW (audit-2): A1 dynamic-cap is page_size==1-scoped. Document the
    boundary: PagedTokenToKVPoolAllocator.__init__ does NOT take max_size
    (boot-only); only TokenToKVPoolAllocator does. (The base
    available_size capped-subtraction asymmetry for paged is pre-existing
    and out of A1 scope — the mixin only wires max_size for page_size==1.)"""
    import inspect
    from sglang.srt.mem_cache.allocator import (
        TokenToKVPoolAllocator, PagedTokenToKVPoolAllocator,
    )
    assert "max_size" in inspect.signature(TokenToKVPoolAllocator.__init__).parameters, (
        "page_size==1 allocator must accept max_size (A1)"
    )
    assert "max_size" not in inspect.signature(
        PagedTokenToKVPoolAllocator.__init__
    ).parameters, (
        "Paged allocator is out of A1 scope (boot-only); must NOT silently "
        "accept max_size"
    )
    print("  PASS  21  A1 scope boundary: max_size is page_size==1-only (paged boot-only)")


def test_22_leak_diagnostic_excludes_headroom():
    """The leak diagnostic's set arithmetic must exclude the dynamic-cap
    headroom (capped pages) from the expected set, so only genuine leaks
    are flagged. Mirrors SchedulerInvariantChecker._check_mamba_pool logic
    (lines 146-151 in invariant_checker.py) without instantiating the full
    checker dataclass."""
    a = _make_alloc(size=8, max_size=32)
    a._fl.free_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    a._fl.pending = torch.empty(0, dtype=torch.int64)
    cached = {8}
    free = set(a.free_pages.tolist()) | set(a.release_pages.tolist())
    expected = set(range(1, a.size + 1))
    capped = a._capped_pages
    if capped is not None and capped.numel() > 0:
        expected -= set(capped.tolist())
    leaked = expected - free - cached
    assert leaked == {7}, f"only genuine leak id 7 should remain; got {leaked}"
    for hid in (9, 20, 32):
        assert hid not in leaked, f"headroom id {hid} must not be flagged leaked"
    print("  PASS  22  leak diagnostic excludes dynamic-cap headroom (flags real leak only)")


def test_23_kv_arena_universal_and_hybrid_forwards():
    """Boot-sanity regression: the m2k grow-KV wiring reads
    `token_to_kv_pool._kv_arena` directly. That crashed on a hybrid (Mamba)
    model because HybridLinearKVPool did not expose `_kv_arena` — the arena
    lives on its inner full_kv_pool. The design-faithful fix: declare
    `_kv_arena` on the KVCache base (universal, default None → no getattr
    fallback at the call site) and forward it from the hybrid wrapper to its
    inner pool. Pin both so a future refactor can't silently reintroduce the
    AttributeError or the getattr crutch."""
    from sglang.srt.mem_cache.memory_pool import KVCache, HybridLinearKVPool

    # Universal: every KVCache exposes _kv_arena, default None (back-compat:
    # non-arena builds → max_size=None → boot-only allocator).
    assert KVCache._kv_arena is None, (
        "KVCache base must declare _kv_arena=None so callers do a direct "
        "None check (not getattr) regardless of subclass"
    )
    # Hybrid wrapper forwards to its inner full-attention pool.
    assert isinstance(HybridLinearKVPool.__dict__.get("_kv_arena"), property), (
        "HybridLinearKVPool must forward _kv_arena to full_kv_pool via a "
        "property (consistent with its other full_kv_pool forwards)"
    )

    class _InnerArena:
        _kv_arena = "ARENA"

    class _InnerNone:
        _kv_arena = None

    class _FakeHybrid(HybridLinearKVPool):
        def __init__(self, inner):
            self.full_kv_pool = inner

    assert _FakeHybrid(_InnerArena())._kv_arena == "ARENA", (
        "hybrid must surface the inner pool's live arena"
    )
    assert _FakeHybrid(_InnerNone())._kv_arena is None, (
        "hybrid must surface None when the inner pool is not arena-backed"
    )
    print("  PASS  23  _kv_arena universal on KVCache + hybrid forwards to inner pool")


def test_24_clear_preserves_unmark_grow():
    """CRITICAL (audit 2026-06-07): the production cross-fire GROW path is
    `unmark_pages_capped` (NOT `set_capacity_pages`). If unmark doesn't
    advance the live cap `_cap`, `clear()` (flush_cache) rebuilds
    `_capped_pages` from the stale boot `_cap` and SILENTLY reverts the grown
    KV back to boot — and orphans the arena handles mamba donated. This is
    the KV twin of the #224 MambaPool.clear() orphan bug; MambaPool.unmark_slots
    advances `self.size = max(self.size, restore.max())` and is safe, so KV
    must mirror it.

    Grow via the real unmark path, flush, assert the grow survives."""
    a = _make_alloc(size=8, max_size=32)
    assert a.available_size() == 8
    # Cross-fire grow: un-cap the 4 lowest headroom ids (what the actuator's
    # arena.grow → expand → unmark_token_slots does).
    a.unmark_pages_capped(torch.arange(9, 13, dtype=torch.int64))
    assert a.available_size() == 12, "grow via unmark must raise available"
    a.clear()
    assert a.available_size() == 12, (
        f"clear() must PRESERVE the cross-fire grow (arena chunks stay "
        f"mapped); got available={a.available_size()} (reverted to boot → "
        f"grow lost + donated handles orphaned)"
    )
    # And the grown ids must be the ones still allocatable, not re-capped.
    live = set(a.alloc(12).tolist())
    assert 9 in live and 12 in live, f"grown ids must be allocatable: {sorted(live)}"
    print("  PASS  24  clear() preserves unmark-grown KV cap (no revert/orphan)")


def test_25_clear_preserves_mark_shrink():
    """Symmetric to test_24: KV-as-source shrink (k2m) uses
    `mark_pages_capped` to cap the tail pages whose chunks are unmapped. If
    mark doesn't lower the live cap, `clear()` REINSTATES those ids into the
    live set → available > backed → the next alloc hands out a slot whose
    chunk is unmapped (CUDA illegal access). Shrink, flush, assert the shrink
    survives."""
    a = _make_alloc(size=16, max_size=16)  # no headroom: pure live pool
    assert a.available_size() == 16
    # Shrink: cap the top-4 tail ids (fire_planner picks descending page-ids).
    a.mark_pages_capped(torch.arange(13, 17, dtype=torch.int64))
    assert a.available_size() == 12, "shrink via mark must lower available"
    a.clear()
    assert a.available_size() == 12, (
        f"clear() must PRESERVE the cross-fire shrink (those chunks are "
        f"unmapped); got available={a.available_size()} (reinstated unmapped "
        f"pages → alloc would return an unbacked slot)"
    )
    print("  PASS  25  clear() preserves mark-shrunk KV cap (no reinstate)")


def test_26_unmark_out_of_range_fails_fast():
    """HIGH (audit 2026-06-07): the allocator page-id ceiling is `max_size`
    (= boot × SGLANG_XPOOL_KV_MAX_FACTOR), but the KV arena's VA `max_tokens`
    is far larger. Under sustained m2k fires the arena can map chunks whose
    token-slot ids exceed `max_size`. `unmark_pages_capped` filters via
    `torch.isin` against `_capped_pages` (max id = size), so an id > size is
    SILENTLY dropped — the arena cuMemMap'd the chunk (consuming a donated
    handle + HBM) but the allocator never exposes it: a pure leak, invisible.
    Per #162/#205 fail-fast, an id beyond the ceiling must CRASH LOUDLY at
    the mutation, not silently drop."""
    a = _make_alloc(size=8, max_size=12)
    a.unmark_pages_capped(torch.arange(9, 13, dtype=torch.int64))  # grow to ceiling
    assert a.available_size() == 12
    raised = False
    try:
        # Arena grew past max_size → slot ids 13,14,15 the allocator cannot
        # represent. Must fail-fast, not silently no-op.
        a.unmark_pages_capped(torch.tensor([13, 14, 15], dtype=torch.int64))
    except (AssertionError, ValueError, RuntimeError):
        raised = True
    assert raised, (
        "unmark_pages_capped must fail-fast on ids > size (page-id ceiling) "
        "— silently dropping them orphans the arena chunk (handle+HBM leak)"
    )
    print("  PASS  26  unmark_pages_capped fails fast on out-of-ceiling ids")


def test_27_concurrent_crossfire_mutators_vs_alloc():
    """#287 (audit 2026-06-07 A1 L2, defense-in-depth): exercise the real
    cross-fire thread split. Production (XPoolActuator) NEVER caps an allocator
    from a thread other than the one allocating from it: the fire WORKER thread
    only GROWS the dst (`unmark_pages_capped` / cap-bump — additive), while the
    SHRINK (`cap_barrier` → `mark_pages_capped`) runs on the SCHEDULER thread and
    caps only planner-selected FREE pages (`plan.pages_to_unmap`). This test
    mirrors that split: the worker thread only unmarks (grows) the headroom band
    [257..280]; the scheduler thread (main) does the shrink, capping only
    currently-FREE band pages selected via the production `free_page_mask()`.
    A hand-rolled worker that capped the band unconditionally would model the
    un-production behaviour of capping a page an in-flight req holds (and was
    timing-flaky); reusing the production thread roles + `free_page_mask`
    selection makes that structurally impossible (#313). Asserts the guarantees
    that MUST hold regardless of interleaving: no crash, the `_capped_pages`
    invariant, no duplicate live id, `alloc()` never returns a capped (unmapped)
    page, and — once quiescent — a consistent non-negative live count.

    Scope note: `set_capacity_pages` (admission cap) is deliberately NOT in
    this mix. Composing it with the cross-fire mark/unmark deltas is the #175
    unified-per-ID-cap territory (the admission `_cap` and the per-ID
    mark/unmark set are decoupled); and `available_size()` is intentionally
    lock-free on the hot path, so a torn read during a concurrent admission
    resize is a benign transient, not a corruption — neither belongs in a
    cross-fire thread-safety assertion."""
    import threading
    import time
    a = _make_alloc(size=256, max_size=1024)
    stop = [False]
    errs = []
    band = torch.arange(257, 281, dtype=torch.int64)

    def worker():
        # Worker thread's real job: GROW (expose headroom). `unmark_pages_capped`
        # is additive — it only REMOVES ids from `_capped_pages`, so it can never
        # cap a page an in-flight req holds. This is the actual cross-thread
        # surface (XPoolActuator.execute_async: the worker does the dst unmark).
        # A short sleep models the real fire cadence (a fire is ~1/s, NOT a tight
        # spin loop) and keeps this unfair-lock busy-loop from starving the
        # alloc thread — the cross-fire correctness under test does not depend
        # on hammering the lock at maximum rate.
        try:
            while not stop[0]:
                a.unmark_pages_capped(band)
                time.sleep(0.0005)
        except Exception as e:  # noqa: BLE001
            errs.append(("worker", repr(e)))

    def shrink_free_band():
        with a._alloc_lock:
            free_set = set(a._fl.free_ids.tolist())
        free_band = torch.tensor(
            [int(x) for x in band.tolist() if x in free_set],
            dtype=torch.int64,
        )
        if free_band.numel() > 0:
            a.mark_pages_capped(free_band)

    t = threading.Thread(target=worker)
    t.start()
    bad_alloc = []
    try:
        held = []
        for i in range(3000):
            g = a.alloc(1)
            if g is not None:
                # Cheap capped check against the IMPLICIT representation (an id
                # is capped iff it is in the tail `>= tail_lo` or in the small
                # `marks` set) — never materialize the full tail in this hot
                # loop. In the new model alloc cannot return a capped id by
                # construction (the free list excludes them); this still
                # catches any race that violated that.
                gid = int(g[0].item())
                with a._alloc_lock:
                    capped = gid >= a._fl.tail_lo or (
                        a._fl.marks.numel() > 0 and gid in a._fl.marks.tolist()
                    )
                if capped:
                    bad_alloc.append(gid)
                held.append(g)
            if i % 7 == 0:
                shrink_free_band()           # scheduler-thread shrink (free only)
            if len(held) > 50:
                a.free(torch.cat(held[:25])); held = held[25:]
    except Exception as e:  # noqa: BLE001
        errs.append(("main", repr(e)))
    finally:
        stop[0] = True
        t.join(timeout=5)

    assert not errs, f"concurrency errors: {errs[:3]}"
    assert not bad_alloc, (
        f"alloc returned currently-capped (unmapped) ids: {bad_alloc[:5]} — "
        f"a capped page leaked into the live free path under the cross-fire "
        f"mark/unmark race"
    )
    a._assert_capped_invariant()
    # Quiescent (worker stopped) → free_pages / _capped_pages are consistent.
    assert a.available_size() >= 0, "available_size underflowed (quiescent)"
    print("  PASS  27  concurrent unmark/mark vs alloc/free: no crash, no "
          "capped-id handed, invariant holds")


def test_28_migrate_slot_swaps_dst_live_src_free():
    """#271 step 1: the KV migrate_slot primitive relocates a LIVE slot
    src→dst so a cross-pool fire can free src's page. It moves the slot's
    k/v bytes (delegated to the kvcache's move_kv_cache) and SWAPS allocator
    state: dst leaves the free set (now live with src's relocated data), src
    RE-ENTERS it (now free — data moved away; cap_barrier caps src's whole
    page next). The swap is available_size-neutral (one live↔one free) until
    the page-level cap — unlike capping src directly, which would break
    Convention A's `capped ⊆ free_pages`."""
    a = _make_alloc(size=16)
    # Make src=5 LIVE (allocated → removed from free); dst=12 stays free.
    a._fl.free_ids = a._fl.free_ids[a._fl.free_ids != 5]
    src, dst = 5, 12
    avail_before = a.available_size()
    assert dst in a.free_pages.tolist() and src not in a.free_pages.tolist()

    ok = a.migrate_slot(src, dst)
    assert ok, "migrate_slot should succeed: dst free, src!=dst"
    free = a.free_pages.tolist()
    assert dst not in free, "dst must leave free (now live with relocated data)"
    assert src in free, "src must re-enter free (data relocated away)"
    # byte-move delegated to the kvcache with (tgt=[dst], src=[src]).
    assert a._kvcache.move_calls == [([dst], [src])], (
        f"migrate_slot must move bytes via move_kv_cache([dst],[src]); "
        f"got {a._kvcache.move_calls}"
    )
    # available_size neutral until cap_barrier caps src's page.
    assert a.available_size() == avail_before, (
        f"the live↔free swap must be available-neutral; "
        f"{a.available_size()} != {avail_before}"
    )
    a._assert_capped_invariant()

    # Guards: src==dst and dst-not-free both refuse.
    assert a.migrate_slot(7, 7) is False, "src==dst must refuse"
    assert a.migrate_slot(5, 5) is False, "src==dst must refuse"
    # dst=5 is now free again (we migrated into it being free? no, 5 is src,
    # now free) — pick a genuinely-live dst: 6 is live (removed? no). Use a
    # live id: src 5 is free now; make 8 live then try migrating into it.
    a._fl.free_ids = a._fl.free_ids[a._fl.free_ids != 8]  # 8 now live
    assert a.migrate_slot(3, 8) is False, "dst not free must refuse"
    print("  PASS  28  migrate_slot swaps dst→live / src→free, byte-move "
          "delegated, available-neutral")


def test_29_migrate_slot_fail_fast_guards():
    """#271 step 1 audit fixes: migrate_slot fail-fasts (#162/#205) on
    structural misuse rather than silently swapping state or crashing deep:
      - page_size != 1 (paged allocator: src/dst are page-ids, not slots) → raise
      - kvcache without move_kv_cache (e.g. MLA) → raise (no silent state swap
        without moving bytes)
      - slot-0 sentinel (#226) as src or dst → refuse (return False)."""
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    # page_size != 1 → raise
    a = _make_alloc(size=16)
    a._fl.free_ids = a._fl.free_ids[a._fl.free_ids != 5]  # make 5 live
    a.page_size = 2
    raised = False
    try:
        a.migrate_slot(5, 12)
    except RuntimeError:
        raised = True
    assert raised, "migrate_slot on a page_size!=1 allocator must raise"

    # kvcache without move_kv_cache → raise (MLA-like)
    class _NoMoveKV:
        page_size = 1
        def get_kv_size_bytes(self):
            return 0
        def can_move_kv_cache(self) -> bool:
            # MLA-like / copy-disabled: migrate_slot's authoritative guard must
            # raise rather than swap slot state without moving the bytes.
            return False

    a2 = TokenToKVPoolAllocator(
        size=16, dtype=torch.float16, device="cpu",
        kvcache=_NoMoveKV(), need_sort=False,
    )
    a2._fl.free_ids = a2._fl.free_ids[a2._fl.free_ids != 5]
    raised = False
    try:
        a2.migrate_slot(5, 12)
    except RuntimeError:
        raised = True
    assert raised, (
        "migrate_slot must raise when the kvcache has no move_kv_cache "
        "(MLA) — never swap slot state without moving the bytes"
    )
    # state must be untouched after the fail-fast raise (no half-swap).
    assert 12 in a2.free_pages.tolist() and 5 not in a2.free_pages.tolist(), (
        "fail-fast must leave free_pages untouched (raise before mutation)"
    )

    # slot-0 sentinel → refuse (return False, not raise)
    a3 = _make_alloc(size=16)
    a3._fl.free_ids = a3._fl.free_ids[a3._fl.free_ids != 5]
    assert a3.migrate_slot(0, 5) is False, "src==0 sentinel must refuse"
    assert a3.migrate_slot(5, 0) is False, "dst==0 sentinel must refuse"
    print("  PASS  29  migrate_slot fail-fasts on paged / no-move kvcache / "
          "slot-0 sentinel")


def main() -> int:
    tests = [
        test_1_back_compat_no_max_size,
        test_2_dynamic_headroom_boots_capped,
        test_3_grow_past_boot,
        test_4_grow_shrink_grow_consistent,
        test_5_actuator_max_tokens_is_arena_max,
        test_6_set_capacity_shrink_available_consistent,
        test_7_alloc_never_returns_capped,
        test_8_grow_then_alloc_uses_new_pages,
        test_9_set_capacity_and_mark_compose,
        test_10_capped_invariant_holds,
        test_11_need_sort_path,
        test_12_scheduler_usage_uses_live_not_ceiling,
        test_13_set_capacity_out_of_range_clamped,
        test_14_max_size_less_than_size_raises,
        test_15_grow_leaves_marks_intact,
        test_16_clear_after_grow_preserves_cap,
        test_17_actuator_grow_roundtrip_alloc,
        test_18_r1_usage_under_load_after_grow,
        test_19_many_grow_shrink_cycles_no_leak,
        test_20_concurrent_alloc_vs_set_capacity,
        test_21_paged_allocator_is_boot_only_scope,
        test_22_leak_diagnostic_excludes_headroom,
        test_23_kv_arena_universal_and_hybrid_forwards,
        test_24_clear_preserves_unmark_grow,
        test_25_clear_preserves_mark_shrink,
        test_26_unmark_out_of_range_fails_fast,
        test_27_concurrent_crossfire_mutators_vs_alloc,
        test_28_migrate_slot_swaps_dst_live_src_free,
        test_29_migrate_slot_fail_fast_guards,
    ]
    print(f"\nA1 (#282) KV-growable tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:120]}")
    print(f"\n#282 Phase-0: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
