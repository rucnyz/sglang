"""Tests for the unified MambaSlotAllocator (CappedFreeList-based).

Guards the architectural invariant: MambaSlotAllocator is the SINGLE source
of truth for mamba slot allocation + dynamic cap, mirroring KV's
TokenToKVPoolAllocator. MambaPool is storage-only (tensors, no free-list).

Every test here has a KV-side twin in test_kv_growable.py; the mamba
allocator must behave identically.
"""

import threading
import time

import torch
import pytest


def _make_alloc(size: int, max_size: int = None, device: str = "cpu"):
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
    return MambaSlotAllocator(size=size, device=device, max_size=max_size)


# ---- boot invariants ----

def test_back_compat_no_max_size():
    """No max_size = boot-only, no headroom, available == size."""
    a = _make_alloc(size=64)
    assert a.available_size() == 64
    assert a.live_size == 64
    assert a.size == 64


def test_dynamic_headroom_boots_capped():
    """max_size > size: headroom [size+1, max_size] is capped at boot."""
    a = _make_alloc(size=64, max_size=256)
    assert a.size == 256
    assert a.live_size == 64
    assert a.available_size() == 64
    g = a.alloc(64)
    assert g is not None
    assert a.alloc(1) is None, "must not alloc into capped headroom"


# ---- grow / shrink ----

def test_grow_past_boot():
    """set_capacity grows live cap; newly-uncapped slots become allocatable."""
    a = _make_alloc(size=8, max_size=32)
    a.set_capacity(16)
    assert a.live_size == 16
    assert a.available_size() == 16
    g = a.alloc(12)
    assert g is not None and len(g) == 12


def test_shrink_below_boot():
    """set_capacity shrinks; newly-capped slots leave the allocatable set."""
    a = _make_alloc(size=16, max_size=32)
    a.set_capacity(8)
    assert a.live_size == 8
    assert a.available_size() == 8
    assert a.alloc(9) is None


def test_grow_shrink_grow_consistent():
    """Round-trip: grow-max, shrink, re-grow. No id leak or double-count."""
    a = _make_alloc(size=8, max_size=64)
    a.set_capacity(64)
    assert a.live_size == 64 and a.available_size() == 64
    a.set_capacity(8)
    assert a.live_size == 8 and a.available_size() == 8
    a.set_capacity(32)
    assert a.live_size == 32 and a.available_size() == 32


# ---- mark / unmark (cross-pool cap) ----

def test_mark_unmark_compose():
    """mark caps specific free slots; unmark restores them."""
    a = _make_alloc(size=16, max_size=32)
    before = a.available_size()
    ids = torch.tensor([5, 6, 7], dtype=torch.int64)
    a.mark(ids)
    assert a.available_size() == before - 3
    assert a.alloc(before) is None, "marked slots not allocatable"
    a.unmark(ids)
    assert a.available_size() == before


def test_alloc_never_returns_capped():
    """Core safety: alloc within live cap succeeds, into headroom fails."""
    a = _make_alloc(size=8, max_size=32)
    g = a.alloc(8)
    assert g is not None
    # all live slots taken; headroom exists but must not be allocatable
    assert a.alloc(1) is None


# ---- clear preserves cap ----

def test_clear_preserves_cap():
    """clear() rebuilds free-list within the CURRENT live cap, not boot."""
    a = _make_alloc(size=8, max_size=32)
    a.set_capacity(16)
    a.alloc(10)
    a.clear()
    assert a.live_size == 16
    assert a.available_size() == 16, "clear must restore to live cap, not boot or max"


# ---- invariant assert ----

def test_invariant_holds_after_mutations():
    """_assert_invariant passes after a sequence of grow/shrink/alloc/free/mark."""
    a = _make_alloc(size=8, max_size=64)
    a._assert_invariant()
    a.set_capacity(32)
    a._assert_invariant()
    g = a.alloc(10)
    a._assert_invariant()
    a.mark(torch.tensor([20, 21], dtype=torch.int64))
    a._assert_invariant()
    if g is not None:
        a.free(g)
    a._assert_invariant()
    a.unmark(torch.tensor([20, 21], dtype=torch.int64))
    a._assert_invariant()
    a.set_capacity(8)
    a._assert_invariant()


# ---- #1 CRITICAL: dual-state agreement (pool forwards to allocator) ----

def test_pool_and_allocator_agree():
    """MambaPool.available_size/live_size forward to allocator (single source of truth)."""
    import types
    a = _make_alloc(size=64, max_size=128)
    pool = types.SimpleNamespace(_allocator=a)
    from sglang.srt.mem_cache.memory_pool import MambaPool
    assert MambaPool.available_size(pool) == a.available_size()
    assert MambaPool.live_size.fget(pool) == a.live_size
    a.set_capacity(96)
    assert MambaPool.available_size(pool) == a.available_size()
    assert MambaPool.live_size.fget(pool) == a.live_size


# ---- #2 clear preserves unmark-grown cap (KV test_24 twin) ----

def test_clear_preserves_unmark_grow():
    """Grow via unmark (the cross-fire path), then clear: grow must survive."""
    a = _make_alloc(size=8, max_size=32)
    band = torch.arange(9, 17, dtype=torch.int64)
    a.unmark(band)
    assert a.live_size == 16
    a.alloc(10)
    a.clear()
    assert a.live_size == 16, "clear must preserve unmark-grown cap"
    assert a.available_size() == 16


# ---- #3 clear preserves mark-shrunk cap (KV test_25 twin) ----

def test_clear_preserves_mark_shrink():
    """Shrink via mark (cross-fire cap), then clear: shrink must survive."""
    a = _make_alloc(size=16)
    ids = torch.tensor([5, 6, 7], dtype=torch.int64)
    a.mark(ids)
    avail_before = a.available_size()
    a.clear()
    assert a.available_size() == avail_before, "clear must preserve mark-shrunk state"


# ---- #4 grow leaves marks intact (KV test_15 twin) ----

def test_grow_leaves_marks_intact():
    """set_capacity grow un-caps range BUT leaves mark-capped ids still capped."""
    a = _make_alloc(size=16, max_size=64)
    ids = torch.tensor([5, 6], dtype=torch.int64)
    a.mark(ids)
    before = a.available_size()
    a.set_capacity(32)
    assert a.available_size() == before + 16, "grow adds 16 but marks stay"
    g = a.alloc(before + 16)
    assert g is not None
    assert a.alloc(1) is None, "marks still block those 2 slots"


# ---- #5 alloc_group excludes capped slots ----

def test_alloc_group_excludes_capped():
    """alloc_group_begin pre-allocates from non-capped slots only."""
    a = _make_alloc(size=8, max_size=32)
    a.alloc_group_begin(8)
    slots = []
    for _ in range(8):
        s = a.alloc(1)
        if s is not None:
            slots.append(int(s.item()))
    a.alloc_group_end()
    assert len(slots) == 8
    for s in slots:
        assert 1 <= s <= 8, f"slot {s} outside live cap [1,8]"


# ---- #6 set_capacity out-of-range clamped (KV test_13 twin) ----

def test_set_capacity_clamps():
    """set_capacity(0) clamps to 1, set_capacity(>max) clamps to max."""
    a = _make_alloc(size=8, max_size=32)
    a.set_capacity(0)
    assert a.live_size == 1
    a.set_capacity(999)
    assert a.live_size == 32


# ---- #7 max_size < size raises (KV test_14 twin) ----

def test_max_size_less_than_size_raises():
    """max_size < size must raise ValueError."""
    with pytest.raises(ValueError):
        _make_alloc(size=16, max_size=8)


# ---- #8 many grow/shrink cycles no leak (KV test_19 twin) ----

def test_many_grow_shrink_cycles():
    """Repeated grow/shrink must not leak or duplicate ids."""
    a = _make_alloc(size=8, max_size=64)
    for _ in range(12):
        a.set_capacity(64)
        a.set_capacity(8)
    a._assert_invariant()
    assert a.live_size == 8
    assert a.available_size() == 8


# ---- alloc_group batch optimization (upstream API) ----

def test_alloc_group_begin_end():
    """alloc_group_begin pre-allocates; alloc_group_end returns unused."""
    a = _make_alloc(size=16)
    a.alloc_group_begin(4)
    slots = [a.alloc(1) for _ in range(3)]
    assert all(s is not None for s in slots)
    a.alloc_group_end()  # returns 1 unused slot
    assert a.available_size() == 16 - 3


# ---- concurrent safety (mirrors test_27 on KV side) ----

def test_concurrent_mark_unmark_vs_alloc():
    """Worker thread marks/unmarks while main allocs/frees. No crash, no capped-id handed out."""
    a = _make_alloc(size=64, max_size=256)
    stop = [False]
    errs = []
    band = torch.arange(65, 90, dtype=torch.int64)

    def worker():
        try:
            while not stop[0]:
                a.unmark(band)
                time.sleep(0.0005)
        except Exception as e:
            errs.append(repr(e))

    t = threading.Thread(target=worker)
    t.start()
    bad = []
    try:
        held = []
        for _ in range(500):
            g = a.alloc(1)
            if g is not None:
                held.append(g)
            if len(held) > 20:
                a.free(torch.cat(held[:10]))
                held = held[10:]
    except Exception as e:
        errs.append(repr(e))
    finally:
        stop[0] = True
        t.join(timeout=5)
    assert not errs, f"errors: {errs}"
    a._assert_invariant()


# ---- actuator dst-grow must route through the allocator (single source of truth) ----

def _storage_pool(alloc):
    """A real MambaPool wired to `alloc` as its single-source-of-truth
    allocator, carrying only the fields the CPU unmark path touches
    (storage-only: no conv/temporal tensors)."""
    from sglang.srt.mem_cache.memory_pool import MambaPool
    pool = object.__new__(MambaPool)
    pool._allocator = alloc
    pool.size = alloc.live_size
    pool.device = torch.device("cpu")
    pool._alloc_lock = threading.Lock()
    # Legacy shadow fields, so the OLD MambaPool.unmark_slots path is still
    # constructible for the regression comparison; the fix must NOT rely on them.
    pool._capped_slots = torch.arange(
        alloc.live_size + 1, alloc.size + 1, dtype=torch.int64
    )
    pool.free_slots = torch.arange(1, alloc.live_size + 1, dtype=torch.int64)
    return pool


def test_actuator_unmark_token_slots_grows_live_size():
    """The k2m dst-grow restore MUST move the allocator's live_size (the single
    source of truth the admission cap reads), not just legacy MambaPool shadow
    state. Reproduces the swarm "grew events: 0": before the fix
    MambaArenaActuator.unmark_token_slots routed to MambaPool.unmark_slots
    (shadow only), so mamba_allocator.live_size stayed pinned at boot and
    BudgetAgent._maybe_update_admission_cap never raised max_running."""
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator, _MambaCapAllocator
    alloc = _make_alloc(size=8, max_size=32)  # live_size=8, capped tail [9..32]
    assert alloc.live_size == 8
    pool = _storage_pool(alloc)
    # Real actuator method; skip __init__ (arena/CUDA) since unmark_token_slots
    # only touches self.pool + self.allocator.
    act = object.__new__(MambaArenaActuator)
    act.pool = pool
    act.allocator = _MambaCapAllocator(pool)
    act.unmark_token_slots([9, 10, 11, 12])  # restore 4 capped tail slots
    assert pool.live_size == 12, (
        f"actuator grow must reach the allocator: live_size={pool.live_size}, want 12"
    )
    assert pool._allocator.available_size() == 12
    assert pool.size == 12  # engine-visible bound reconciled from the allocator


def test_actuator_grow_headroom_pages_bounds_cross_fire():
    """A k2m cross-fire must not grant dst mamba slots past the allocator's
    physical ceiling (max_size): the arena's chunk-id space is deliberately far
    larger than the CappedFreeList id space, so an unbounded grant hands out
    chunk ids that expand past max_size and unmark_token_slots fail-fasts
    (the swarm 'unmark: id N exceeds ceiling' crash). grow_headroom_pages
    reports the page budget XPoolActuator clamps the grant to."""
    import types
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator, _MambaCapAllocator

    alloc = _make_alloc(size=8, max_size=12)  # live_size=8, id-space ceiling 12
    assert alloc.live_size == 8 and alloc.max_size == 12
    pool = _storage_pool(alloc)
    pool._mamba_temporal_arena = types.SimpleNamespace(tokens_per_chunk=1)  # tps=1
    act = object.__new__(MambaArenaActuator)
    act.pool = pool
    act.allocator = _MambaCapAllocator(pool)

    # headroom = (max_size 12 - live_size 8) // tps 1 = 4 pages: the exact grant
    # ceiling the cross-fire clamp uses.
    assert act.grow_headroom_pages() == 4

    # A grant WITHIN headroom (ids 9..12) restores fine, growing live to 12.
    act.unmark_token_slots([9, 10, 11, 12])
    assert pool.live_size == 12
    assert act.grow_headroom_pages() == 0  # at the ceiling: no further grant

    # A grant PAST the ceiling (id 13 > max_size 12) is exactly what the clamp
    # must prevent upstream; unclamped, the allocator fail-fasts (the crash).
    with pytest.raises(AssertionError):
        act.unmark_token_slots([13])


# ---- admission gate must reflect free-backable capacity across BOTH sub-pools ----

def test_hybrid_mamba_admittable_reqs():
    """A hybrid request needs a mamba active slot in addition to a req slot. The
    admission gate (Scheduler.get_num_allocatable_reqs) bounds the batch by
    mamba_admittable_reqs = (free mamba + evictable cached snapshots) /
    slots-per-req, mirroring the KV available+evictable gate. This must (a) fall
    to 0 when mamba is exhausted AND nothing is evictable, so the request DEFERS
    instead of crashing alloc_req_slots (the swarm crash), and (b) NOT throttle a
    lightly-loaded pool by ignoring evictable cached snapshots (the base
    regression the free-only gate caused)."""
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    pool = object.__new__(HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer_lazy = False  # -> PREFIX_CACHE factor = 3

    # (a) crash guard: mamba exhausted AND nothing evictable -> gate 0 (defer).
    mamba = _make_alloc(size=6)
    pool.mamba_allocator = mamba
    assert mamba.alloc(6) is not None and mamba.available_size() == 0
    assert pool.mamba_admittable_reqs(mamba_evictable=0) == 0

    # (b) no-throttle: evictable cached snapshots count toward backable capacity
    # (alloc_req_slots evicts them before allocating), so free 0 + evictable 90
    # backs (0 + 90) // 3 = 30 requests, NOT 0.
    assert pool.mamba_admittable_reqs(mamba_evictable=90) == 30

    # lightly loaded: free 6 + evictable 0 -> 6 // 3 = 2.
    pool.mamba_allocator = _make_alloc(size=6)
    assert pool.mamba_admittable_reqs(mamba_evictable=0) == 2


# ---- per-request mamba budget must exclude a request's own COW source -------

def _mamba_tree_with_source(source_slots, other_evictable_slots=0):
    """A minimal real MambaRadixCache (CPU) carrying one evictable shared-prefix
    snapshot of `source_slots` mamba slots, plus optionally an unrelated
    `other_evictable_slots` snapshot. Reuses the production inc_lock_ref /
    mamba_evictable_size bookkeeping (the lock-drops-evictable mechanism the
    KV side relies on), so the test exercises the real code path, not an
    imitation."""
    from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache, TreeNode

    tree = object.__new__(MambaRadixCache)
    tree.disable = False
    tree.root_node = TreeNode()
    tree.mamba_evictable_size_ = 0
    tree.mamba_protected_size_ = 0
    tree.full_evictable_size_ = 0
    tree.full_protected_size_ = 0

    def _add_snapshot(mamba_len):
        n = TreeNode()
        n.parent = tree.root_node
        n.value = torch.arange(1, dtype=torch.int64)  # 1 KV token (walk needs .value)
        n.mamba_value = torch.arange(mamba_len, dtype=torch.int64)
        n.mamba_lock_ref = 0
        n.full_lock_ref = 0
        tree.mamba_evictable_size_ += mamba_len
        tree.full_evictable_size_ += len(n.value)
        return n

    source = _add_snapshot(source_slots)
    if other_evictable_slots:
        _add_snapshot(other_evictable_slots)
    return tree, source


def _make_adder(mamba_alloc, tree):
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = object.__new__(HybridReqToTokenPool)
    pool.enable_mamba_extra_buffer_lazy = False  # PREFIX_CACHE factor = 3
    pool.mamba_allocator = mamba_alloc
    # The adder reaches the pool via tree_cache.req_to_token_pool (the same
    # path Scheduler uses), so wire it on the tree, not the adder.
    tree.req_to_token_pool = pool

    adder = object.__new__(PrefillAdder)
    adder.tree_cache = tree
    adder.is_hybrid_ssm_cache = True
    adder.mamba_slot_offset = 0
    return adder


def test_prefill_adder_defers_own_cow_source():
    """The swarm crash: a hybrid request COWs from the ONE shared cached mamba
    snapshot at pool saturation. The batch-level gate (mamba_admittable_reqs)
    counts that snapshot as reclaimable BEFORE the match, so it over-admits;
    but the request pins the snapshot as its COW source (inc_lock_ref, the same
    lock the KV side uses), so evict can never reclaim it -> alloc_req_slots
    would crash instead of deferring.

    The per-request mamba budget (PrefillAdder.rem_mamba_slots, read INSIDE the
    _lock_node block after the source is locked, mirroring the KV
    rem_total_tokens check) must exclude the just-locked source and defer.
    """
    import types

    # Saturated pool: 6 slots, 5 already live -> 1 free at gate time. The only
    # evictable mamba is the shared source snapshot (3 slots).
    mamba = _make_alloc(size=6)
    assert mamba.alloc(5) is not None and mamba.available_size() == 1
    tree, source = _mamba_tree_with_source(source_slots=3)
    adder = _make_adder(mamba, tree)

    # (1) batch gate BEFORE the match over-admits: (free 1 + evictable 3)//3 = 1.
    assert adder.tree_cache.req_to_token_pool.mamba_admittable_reqs(
        tree.mamba_evictable_size(), tree.supports_mamba()
    ) == 1

    # (2) match time: the COW dst active slot is allocated (req.mamba_pool_idx
    # set), consuming the last free slot; then the source is locked (the
    # _lock_node inc_lock_ref in add_one_req).
    dst = mamba.alloc(1)
    assert dst is not None and mamba.available_size() == 0
    req = types.SimpleNamespace(mamba_pool_idx=dst[0])
    tree.inc_lock_ref(source)
    assert tree.mamba_evictable_size() == 0  # source excluded once locked

    # (3) per-request budget now sees no reclaimable mamba: 0 free + 0 evictable.
    # The request still needs its ping-pong buffers (factor 3 minus the 1 COW-dst
    # already held = 2), so it MUST defer.
    assert adder.rem_mamba_slots == 0
    assert adder._mamba_slots_needed(req) == 2
    assert adder._mamba_slots_needed(req) > adder.rem_mamba_slots, "must defer"


def test_prefill_adder_no_false_defer_with_other_evictable():
    """No-throttle guard: when a DIFFERENT session's snapshot is evictable
    (not the request's COW source), the budget must count it, so the COW
    request is admitted -- eviction of the unrelated snapshot backs its
    ping-pong buffers. Mirrors the KV base no-regression requirement."""
    import types

    mamba = _make_alloc(size=9)
    assert mamba.alloc(5) is not None and mamba.available_size() == 4
    # source snapshot 3 slots + an unrelated evictable snapshot 3 slots.
    tree, source = _mamba_tree_with_source(source_slots=3, other_evictable_slots=3)
    adder = _make_adder(mamba, tree)

    dst = mamba.alloc(1)  # COW dst
    req = types.SimpleNamespace(mamba_pool_idx=dst[0])
    tree.inc_lock_ref(source)  # source locked; other snapshot stays evictable
    assert tree.mamba_evictable_size() == 3  # the unrelated snapshot only

    # rem = free 3 + evictable 3 - offset 0 = 6 >= needed 2 -> admit (no defer).
    assert adder.rem_mamba_slots == 6
    assert adder._mamba_slots_needed(req) == 2
    assert adder._mamba_slots_needed(req) <= adder.rem_mamba_slots, "must admit"


def test_prefill_adder_mamba_offset_accumulates():
    """The budget is cumulative across a batch (symmetric with
    rem_total_token_offset): each admitted request's fresh mamba demand is
    deducted, so a later request in the same pass sees the earlier consumption
    and defers before over-committing."""
    import types

    mamba = _make_alloc(size=12)
    assert mamba.alloc(6) is not None and mamba.available_size() == 6
    tree, _ = _mamba_tree_with_source(source_slots=0)  # no cached snapshots
    adder = _make_adder(mamba, tree)

    fresh = types.SimpleNamespace(mamba_pool_idx=None)  # no COW dst held
    assert adder._mamba_slots_needed(fresh) == 3  # full active + ping-pong
    assert adder.rem_mamba_slots == 6  # 6 free + 0 evictable

    # Admit two fresh requests: offset accumulates 3 + 3 = 6.
    adder.mamba_slot_offset += adder._mamba_slots_needed(fresh)
    adder.mamba_slot_offset += adder._mamba_slots_needed(fresh)
    assert adder.rem_mamba_slots == 0
    # A third fresh request needs 3 but the pass has already committed all 6.
    assert adder._mamba_slots_needed(fresh) > adder.rem_mamba_slots, "must defer"
