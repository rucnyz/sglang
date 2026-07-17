"""Reproducing test for the k2m KV working-set floor.

A k2m fire (grow mamba by donating idle KV pages) had NO KV floor, so on the
agent-swarm workload (recurrent-bound, c_M-folded) repeated fires shrank the KV
pool until a prefill chunk no longer fit:

  RuntimeError: Out of memory ... Try to allocate 8192 tokens.
  Available full tokens: 3809 (full_available_size=3809 + full_evictable_size=0)

The floor must leave one INDIVISIBLE prefill chunk of GENUINELY-FREE KV: a
chunk cannot be back-filled on demand once the mamba donor is saturated (the
reactive `_grow_kv_from_mamba` hook returns False). Crucially it must NOT
credit evictable KV cache (as the mamba floor credits its donatable snapshots):
a k2m fire donates FREE pages only, and the swarm's shared-prefix cache is
COW-locked by the batch that needs the chunk, so a `free + evictable >= chunk`
floor lets the evictable evaporate and OOM anyway (the KV twin of the #339 COW
over-count). `kv_used` already includes the cache, so the floor is

    kv_floor = kv_used + chunked_prefill_size

which guarantees post-drain `available >= chunked_prefill_size`. Reuses the
pool-agnostic `_mamba_drain_floor` clamp.
"""
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _agent_with_chunk_floor(chunk):
    from sglang.srt.budgeter.agent import BudgetAgent
    a = object.__new__(BudgetAgent)
    a._kv_prefill_chunk_floor = chunk
    return a


def test_kv_working_set_floor_is_kv_used_plus_chunk():
    """The floor reserves ALL of kv_used (running + cache) plus one prefill
    chunk of genuinely-free capacity — it does NOT credit evictable cache,
    which can evaporate (COW-lock / admission consumption) before the prefill."""
    a = _agent_with_chunk_floor(8192)
    assert a._kv_working_set_floor_slots(kv_used=15000) == 15000 + 8192
    assert a._kv_working_set_floor_slots(kv_used=0) == 8192


def test_kv_drain_clamped_to_leave_a_free_prefill_chunk():
    """`_mamba_drain_floor` (reused for KV) bounds the drain so post-drain
    live_size stays >= kv_used + chunk, i.e. >= one prefill chunk of FREE KV
    remains regardless of evictable cache dynamics."""
    from sglang.srt.budgeter.agent import _mamba_drain_floor
    a = _agent_with_chunk_floor(8192)
    floor = a._kv_working_set_floor_slots(kv_used=20000)  # 28192

    # roomy pool (live 40000): a small request passes untouched.
    assert _mamba_drain_floor(40000, floor, 1, 100) == 100
    # near the floor (live 29000): donatable = 29000-28192 = 808, clamped.
    assert _mamba_drain_floor(29000, floor, 1, 5000) == 808
    # at/below the floor (live 28000 < 28192): refuse — a free chunk must stay.
    assert _mamba_drain_floor(28000, floor, 1, 5000) == 0


def test_post_drain_kv_always_holds_a_free_prefill_chunk():
    """The invariant the crash violates: after the MAXIMUM allowed k2m drain,
    KV available (GENUINELY FREE, not counting evaporate-able evictable) >= one
    prefill chunk, so alloc_token_slots never OOMs even if all cache is
    evicted/locked."""
    from sglang.srt.budgeter.agent import _mamba_drain_floor
    chunk = 8192
    a = _agent_with_chunk_floor(chunk)
    saw_clamped = False
    for kv_live in (30000, 50000, 200000):
        for kv_used in (5000, 20000):
            if kv_used > kv_live:
                continue
            kv_free = kv_live - kv_used
            floor = a._kv_working_set_floor_slots(kv_used)
            drained = _mamba_drain_floor(kv_live, floor, 1, 10 ** 9)  # unbounded ask
            post_available = kv_free - drained
            assert 0 <= post_available <= kv_free
            # genuinely-free capacity alone (no evictable credit) >= a chunk,
            # OR the pool started with < a chunk free (drain is then 0).
            assert post_available >= min(chunk, kv_free), (
                f"live={kv_live} used={kv_used} drained={drained} -> free "
                f"{post_available} < min(chunk={chunk}, pre_free={kv_free})"
            )
            if 0 < drained < kv_free:
                saw_clamped = True
    assert saw_clamped, "sweep must exercise a floor-clamped (partial) drain"
