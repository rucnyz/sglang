"""Reproducing test for #329 — every mamba-alloc site must DEGRADE, never
assert, when the pool is over-drained.

The working-set mamba floor (#297) lets the Budgeter drain mamba to its LIVE
working set. In the m2k regime (KV full by design, mamba borrowed FOR KV) a
mamba-alloc can then find nothing: the pool is full of active + locked-cache
slots, `evict_mamba` frees nothing, and a k2m grow has no idle KV to lend back.
Three sibling sites asserted "Can not alloc mamba cache" / "Not enough space
for mamba cache" on that genuine scarcity, turning back-pressure into a crash
(the 262k agentreplay SIGQUIT). Each must instead degrade gracefully:

  1. COW (`MambaRadixCache._match_post_processor`, slot via
     `_cow_mamba_slot_or_none`): the request cannot copy the cached mamba state,
     so it must NOT claim the matched mamba prefix. Degrade to a mamba cache
     MISS (`_no_mamba_match_result`: empty device_indices + root last_node) so
     it re-prefills from scratch. The matched KV prefix is coupled to that
     cached state (both end at `best_value_len`) and is dropped with it.
  2. Caching fork (`MambaRadixCache._fork_mamba_with_recovery` ->
     `cache_unfinished_req`): returns None; the caller skips caching this
     snapshot (the request keeps its live state and continues).
  3. Active slot (`HybridReqToTokenPool.alloc`): rolls back the fresh req_pool
     slots + fresh mamba slots and returns None, so the scheduler back-pressures
     (the same None the req-slot-exhausted branch already returns).

Test-first: each test asserts the over-drained pool DEGRADES (no assert). Pre-
fix these raised AssertionError("Can not alloc mamba cache" / "Not enough space
for mamba cache") — the RED witness this file was written against.

Baseline safety: with the Budgeter OFF (no grow hooks wired, default split),
mamba is never drained below boot, so `mamba_pool.alloc` always succeeds on the
first try and none of these degrade branches is ever entered. The success path
is byte-identical to stock sglang; only the terminal failure changed from
assert to degrade.
"""
import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch  # noqa: E402

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams  # noqa: E402
from sglang.srt.mem_cache.mamba_radix_cache import (  # noqa: E402
    LRUList,
    MambaRadixCache,
    TreeNode,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402


# --------------------------------------------------------------------------- #
# Stub mamba pool: a fixed number of free slots; evict / grow add slots back.
# Mirrors dev/interlayer/3_budgeter/mamba_fork_grow/test_fork_grow_recovery.py.
# --------------------------------------------------------------------------- #
class _StubMambaPool:
    def __init__(self, free=0):
        self._free = free
        self.alloc_calls = 0
        self.copy_calls = 0

    def alloc(self, n):
        self.alloc_calls += 1
        if self._free < n:
            return None
        self._free -= n
        return torch.arange(n, dtype=torch.int64)

    def copy_from(self, src, dst):
        self.copy_calls += 1


def _cow_cache(*, pool_free, evict_frees, grow_frees):
    """A MambaRadixCache with __init__ bypassed, wired only for the COW path.
    `evict_frees`/`grow_frees` = slots the stubbed evict / grow hook free."""
    c = MambaRadixCache.__new__(MambaRadixCache)
    c.disable = False
    c.device = "cpu"
    pool = _StubMambaPool(free=pool_free)
    c.req_to_token_pool = types.SimpleNamespace(mamba_pool=pool)

    def _evict(_params):
        pool._free += evict_frees

    c.evict = _evict
    # inc/dec_lock_ref touch mamba_*_size_ accounting + node refs; stub to noop
    # so the COW slot logic is exercised in isolation.
    c.inc_lock_ref = lambda node: None
    c.dec_lock_ref = lambda node, params=None: None

    hook_calls = {"n": 0}
    if grow_frees is None:
        c._mamba_grow_hook = None
    else:
        def _hook(n):
            hook_calls["n"] += 1
            pool._free += grow_frees
            return grow_frees > 0
        c._mamba_grow_hook = _hook

    # Minimal root for _no_mamba_match_result.
    root = TreeNode()
    root.key = RadixKey([], None)
    root.value = []
    root.full_lock_ref = 1
    root.mamba_lock_ref = 1
    c.root_node = root
    return c, pool, hook_calls


def _matched_node(cache):
    """A matched leaf carrying a mamba snapshot (the COW source)."""
    n = TreeNode()
    n.key = RadixKey([1, 2, 3], None)
    n.value = torch.arange(3, dtype=torch.int64)
    n.mamba_value = torch.tensor([7], dtype=torch.int64)
    n.parent = cache.root_node
    return n


# --------------------------------------------------------------------------- #
# Site 1: COW slot helper + post-processor degrade.
# --------------------------------------------------------------------------- #
def test_cow_slot_free_no_recovery():
    """A free slot -> first alloc succeeds, no evict / grow."""
    c, pool, hook = _cow_cache(pool_free=1, evict_frees=0, grow_frees=0)
    slot = c._cow_mamba_slot_or_none(_matched_node(c))
    assert slot is not None
    assert pool.alloc_calls == 1 and hook["n"] == 0


def test_cow_slot_evict_recovers():
    """Pool empty, evict frees a cold-cache slot -> grow not fired."""
    c, pool, hook = _cow_cache(pool_free=0, evict_frees=1, grow_frees=1)
    slot = c._cow_mamba_slot_or_none(_matched_node(c))
    assert slot is not None
    assert hook["n"] == 0, "grow is last resort; evict already freed a slot"


def test_cow_slot_grow_recovers():
    """Pool empty, evict frees nothing, grow hook frees a slot -> succeeds."""
    c, pool, hook = _cow_cache(pool_free=0, evict_frees=0, grow_frees=1)
    slot = c._cow_mamba_slot_or_none(_matched_node(c))
    assert slot is not None
    assert hook["n"] == 1


def test_cow_slot_none_when_over_drained():
    """The #329 regime: pool empty, evict frees nothing, grow can't borrow KV
    -> helper returns None (NOT assert)."""
    c, pool, hook = _cow_cache(pool_free=0, evict_frees=0, grow_frees=0)
    slot = c._cow_mamba_slot_or_none(_matched_node(c))
    assert slot is None, "over-drained COW must return None, not assert"
    assert hook["n"] == 1, "the grow hook must be attempted once"


def test_cow_slot_none_no_hook():
    """No grow hook wired (Budgeter off): pool empty, evict frees nothing ->
    None (no assert). The degrade is unconditional; it does not depend on a
    hook being present."""
    c, pool, hook = _cow_cache(pool_free=0, evict_frees=0, grow_frees=None)
    slot = c._cow_mamba_slot_or_none(_matched_node(c))
    assert slot is None


def test_cow_post_processor_degrades_to_mamba_miss():
    """The crash site itself: `_match_post_processor` with cow_mamba=True, a
    matched node carrying a mamba snapshot, req without a mamba slot, and an
    over-drained pool. TODAY (pre-fix) this asserts "Can not alloc mamba cache".
    AFTER the fix it returns the no-mamba-match result: empty device_indices +
    root last_node (the request re-prefills its mamba from scratch), and
    req.mamba_pool_idx stays None (it gets a fresh active slot later)."""
    c, pool, hook = _cow_cache(pool_free=0, evict_frees=0, grow_frees=0)
    c.full_lru_list = LRUList(mamba=False)
    c.mamba_lru_list = LRUList(mamba=True)
    last_node = _matched_node(c)
    c.full_lru_list.insert_mru(last_node)
    c.mamba_lru_list.insert_mru(last_node)
    req = types.SimpleNamespace(mamba_pool_idx=None)
    params = MatchPrefixParams(
        key=RadixKey([1, 2, 3], None), req=req, cow_mamba=True
    )
    # value/best_value_len as _match_prefix_helper would hand them in: the
    # matched KV up to the mamba-bearing node.
    value = [last_node.value]
    result = c._match_post_processor(params, value, last_node, best_value_len=3)
    assert result.device_indices.numel() == 0, (
        "COW miss must drop the matched KV prefix (coupled to the cached "
        "mamba state); got a non-empty prefix")
    assert result.last_device_node is c.root_node, (
        "COW miss must point at root so the request re-prefills from scratch")
    assert req.mamba_pool_idx is None, (
        "COW miss must leave the req without a mamba slot; it gets a fresh "
        "active slot at alloc time")
    assert pool.copy_calls == 0, "no mamba state may be copied on a COW miss"


def test_cow_post_processor_success_path_unchanged():
    """Baseline: a free slot -> COW copies the cached state, req gets the slot,
    and the full matched prefix + last_node are returned (stock behavior)."""
    c, pool, hook = _cow_cache(pool_free=1, evict_frees=0, grow_frees=0)
    c.full_lru_list = LRUList(mamba=False)
    c.mamba_lru_list = LRUList(mamba=True)
    last_node = _matched_node(c)
    c.full_lru_list.insert_mru(last_node)
    c.mamba_lru_list.insert_mru(last_node)
    req = types.SimpleNamespace(mamba_pool_idx=None)
    params = MatchPrefixParams(
        key=RadixKey([1, 2, 3], None), req=req, cow_mamba=True
    )
    value = [last_node.value]
    result = c._match_post_processor(params, value, last_node, best_value_len=3)
    assert result.last_device_node is last_node
    assert result.device_indices.numel() == 3
    assert req.mamba_pool_idx is not None, "COW success sets the req mamba slot"
    assert pool.copy_calls == 1


# --------------------------------------------------------------------------- #
# Site 2: caching fork. (_fork_mamba_with_recovery already has detailed tests
# in mamba_fork_grow/test_fork_grow_recovery.py; here we pin the #329 change:
# final failure returns None instead of asserting.)
# --------------------------------------------------------------------------- #
class _ForkStubPool:
    def __init__(self, free=0):
        self._free = free

    def fork_from(self, mamba_value):
        if self._free <= 0:
            return None
        self._free -= 1
        return ("slot",)


def _fork_cache(*, pool_free, evict_frees, grow_frees):
    c = MambaRadixCache.__new__(MambaRadixCache)
    pool = _ForkStubPool(free=pool_free)
    c.req_to_token_pool = types.SimpleNamespace(mamba_pool=pool)
    c.evict = lambda params: setattr(pool, "_free", pool._free + evict_frees)
    if grow_frees is None:
        c._mamba_grow_hook = None
    else:
        c._mamba_grow_hook = lambda n: (
            setattr(pool, "_free", pool._free + grow_frees) or grow_frees > 0
        )
    return c, pool


def test_fork_returns_none_when_over_drained():
    """Pool empty, evict + grow free nothing -> fork returns None (NOT assert).
    Pre-fix this asserted "Can not alloc mamba cache"."""
    c, pool = _fork_cache(pool_free=0, evict_frees=0, grow_frees=0)
    assert c._fork_mamba_with_recovery(("src",)) is None


def test_fork_returns_none_when_over_drained_no_hook():
    """No grow hook wired (Budgeter off): pool empty, evict frees nothing ->
    fork returns None (no assert). Stock sglang asserted here, but the degrade
    is harmless: with the Budgeter off the pool is never over-drained so this
    branch is unreachable in practice (see baseline-safety in the docstring)."""
    c, pool = _fork_cache(pool_free=0, evict_frees=0, grow_frees=None)
    assert c._fork_mamba_with_recovery(("src",)) is None


def test_fork_success_path_unchanged():
    """A free slot -> fork succeeds immediately (stock behavior)."""
    c, pool = _fork_cache(pool_free=1, evict_frees=0, grow_frees=0)
    assert c._fork_mamba_with_recovery(("src",)) is not None


def test_cache_unfinished_req_skips_on_fork_none():
    """The fork-None contract reaches `cache_unfinished_req`: when the fork
    can't allocate, the request must SKIP caching (keep its live state) instead
    of inserting a None snapshot. We verify the skip path is taken: prefix_indices
    set to the live KV, no insert attempted."""
    c, pool = _fork_cache(pool_free=0, evict_frees=0, grow_frees=0)
    c.disable = False
    c.enable_mamba_extra_buffer = False
    c.page_size = 1
    fill_ids = [1, 2, 3, 4]
    req_to_token = torch.zeros((1, len(fill_ids)), dtype=torch.int64)
    req_to_token[0] = torch.arange(len(fill_ids), dtype=torch.int64)
    c.req_to_token_pool.req_to_token = req_to_token
    c.req_to_token_pool.get_mamba_indices = lambda idx: torch.tensor([0])

    inserted = {"n": 0}
    c.insert = lambda params: inserted.__setitem__("n", inserted["n"] + 1)

    req = types.SimpleNamespace(
        req_pool_idx=0,
        fill_ids=fill_ids,
        extra_key=None,
        cache_protected_len=0,
        mamba_last_track_seqlen=None,
        prefix_indices=None,
    )
    c.cache_unfinished_req(req)
    assert inserted["n"] == 0, "fork-None must skip the radix insert"
    assert req.prefix_indices is not None and len(req.prefix_indices) == len(fill_ids), (
        "skip path must set prefix_indices to the full live KV")


# --------------------------------------------------------------------------- #
# Site 3: HybridReqToTokenPool.alloc active-slot.
# --------------------------------------------------------------------------- #
class _ActiveStubMambaPool:
    def __init__(self, free):
        self._free = free
        self.freed = []

    def alloc(self, n):
        if self._free < n:
            return None
        self._free -= n
        out = torch.arange(n, dtype=torch.int64)
        return out

    def free(self, idx):
        self.freed.append(int(idx.numel()))
        self._free += int(idx.numel())


def _active_pool(*, req_slots, mamba_free, grow_frees):
    """A HybridReqToTokenPool with __init__ bypassed, wired only for alloc."""
    p = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    p.size = req_slots
    p.free_slots = list(range(req_slots))
    p.enable_mamba_extra_buffer = False
    p.mamba_pool = _ActiveStubMambaPool(free=mamba_free)
    # Real pool maps req_pool_idx -> mamba_idx via a tensor (indexed by the
    # list of selected req slots on the success path).
    p.req_index_to_mamba_index_mapping = torch.zeros(req_slots, dtype=torch.int32)
    if grow_frees is None:
        p._mamba_active_grow_hook = None
    else:
        p._mamba_active_grow_hook = lambda n: (
            setattr(p.mamba_pool, "_free", p.mamba_pool._free + grow_frees)
            or grow_frees > 0
        )
    return p


def _active_req():
    return types.SimpleNamespace(
        req_pool_idx=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        is_chunked=0,
        kv_committed_len=0,
        is_dllm=lambda: False,
    )


def test_active_alloc_returns_none_and_rolls_back_when_over_drained():
    """The #329 active-slot regime: req_pool slots exist but the mamba pool is
    empty and no idle KV to grow from. alloc must roll back the fresh req_pool
    slot and return None (NOT assert "Not enough space for mamba cache")."""
    # Map mapping write needs a tensor-indexable container; use a dict subclass
    # that ignores the success-path mapping write (not reached on the degrade).
    p = _active_pool(req_slots=4, mamba_free=0, grow_frees=0)
    req = _active_req()
    free_before = list(p.free_slots)
    out = p.alloc([req])
    assert out is None, "over-drained active alloc must return None, not assert"
    assert req.req_pool_idx is None, "fresh req_pool slot must be rolled back"
    assert req.mamba_pool_idx is None
    assert sorted(p.free_slots) == sorted(free_before), (
        "the fresh req_pool slot must be returned to free_slots on rollback")


def test_active_alloc_returns_none_no_hook():
    """No grow hook (Budgeter off): empty mamba pool -> None + rollback, no
    assert. (Unreachable when Budgeter off, since mamba is never drained below
    boot; pinned for completeness.)"""
    p = _active_pool(req_slots=4, mamba_free=0, grow_frees=None)
    req = _active_req()
    assert p.alloc([req]) is None
    assert req.req_pool_idx is None


def test_active_alloc_grow_recovers():
    """Empty mamba pool but the grow hook frees a slot -> alloc succeeds (the
    on-demand k2m path), req gets both slots."""
    p = _active_pool(req_slots=4, mamba_free=0, grow_frees=1)
    req = _active_req()
    out = p.alloc([req])
    assert out is not None
    assert req.req_pool_idx is not None and req.mamba_pool_idx is not None


def test_active_alloc_success_path_unchanged():
    """Baseline: mamba slot available -> alloc succeeds first try, no rollback,
    no grow hook fired (the stock success path)."""
    p = _active_pool(req_slots=4, mamba_free=4, grow_frees=0)
    req = _active_req()
    out = p.alloc([req])
    assert out is not None
    assert req.req_pool_idx is not None and req.mamba_pool_idx is not None
    assert p.mamba_pool.freed == [], "success path frees nothing"


def test_active_alloc_partial_batch_rollback():
    """A 2-req batch where the first gets a mamba slot and the second can't:
    both fresh req_pool slots AND the first req's mamba slot must be rolled
    back, leaving the pool exactly as before."""
    p = _active_pool(req_slots=4, mamba_free=1, grow_frees=0)
    r1, r2 = _active_req(), _active_req()
    free_before = list(p.free_slots)
    mamba_free_before = p.mamba_pool._free
    out = p.alloc([r1, r2])
    assert out is None
    assert r1.req_pool_idx is None and r2.req_pool_idx is None
    assert r1.mamba_pool_idx is None and r2.mamba_pool_idx is None
    assert sorted(p.free_slots) == sorted(free_before)
    assert p.mamba_pool._free == mamba_free_before, (
        "the first req's mamba slot must be freed back on rollback")


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError:
                failures += 1
                print("FAIL", name)
                traceback.print_exc()
            except Exception:
                failures += 1
                print("ERROR", name)
                traceback.print_exc()
    sys.exit(1 if failures else 0)
