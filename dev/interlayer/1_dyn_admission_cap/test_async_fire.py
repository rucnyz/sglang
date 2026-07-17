"""Async fire: cap_barrier is mark-only, cuMem* on worker, deferred apply.

Tests that the scheduler-thread cap_barrier is fast (no cuMem*), the worker
does the physical transfer, and apply_pending_fires exposes new capacity.
Uses real ChunkArena + SharedHandlePool on CUDA.
"""
import os
import sys
import threading
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
os.environ.setdefault("SGLANG_ARENA_CHUNK_BYTES", str(2 * 1024 * 1024))

import pytest
import torch

from sglang.srt.arena.chunk_arena import ChunkArena, SharedHandlePool
from sglang.srt.arena.fire_plan import FirePlan

HAS_CUDA = torch.cuda.is_available()
CHUNK = 2 * 1024 * 1024


def _skip_no_cuda():
    if not HAS_CUDA:
        pytest.skip("CUDA required")


class _FakePool:
    def __init__(self, size):
        self.size = size


class _MTAShim:
    """Thin wrapper around ChunkArena to satisfy XPoolActuator's interface."""
    def __init__(self, arena, pool_name_prefix, n_layers=1, n_kinds=1, subpool_offset=0):
        self._arena = arena
        self.n_layers = n_layers
        self.n_kinds = n_kinds
        self._prefix = pool_name_prefix
        self._subpool_offset = subpool_offset

    def _pool_name(self, i):
        return f"{self._prefix}{self._subpool_offset + i}"

    def _c_index(self, i):
        return self._subpool_offset + i


class _FakeAllocator:
    def __init__(self):
        self.device = torch.device("cuda:0")
    def mark_pages_capped(self, t):
        return len(t)
    def unmark_pages_capped(self, t):
        pass
    def count_referenced(self, t):
        return 0
    def count_reachable_capped(self, t):
        return 0


class _FakeActuator:
    def __init__(self, pool, arena, tpc=1):
        self.pool = pool
        self._arena_ref = arena
        self.allocator = _FakeAllocator()
        self._tpc = tpc

    def _tokens_per_page(self):
        return self._tpc

    def expand_pages_to_token_slots(self, pages):
        return list(pages)

    def grow_headroom_pages(self):
        # #340 clamp: real actuators bound the grant to (max_size-live_size);
        # the fake pool has unbounded headroom so the clamp never binds here.
        return 1 << 30

    def unmark_token_slots(self, slots):
        pass


def _make_xpool_actuator(n_src=10, n_dst=0):
    """Build a real XPoolActuator with two ChunkArenas sharing a pool."""
    _skip_no_cuda()
    total = n_src + n_dst
    pool = SharedHandlePool(device_id=0, chunk_size=CHUNK, n_handles=total)
    src_arena = ChunkArena(device_id=0, chunk_size=CHUNK, n_handles=0,
                           pool_capacities=[("src0", total)],
                           external_handle_pool=pool)
    for _ in range(n_src):
        src_arena.grow("src0", 1)
    dst_arena = ChunkArena(device_id=0, chunk_size=CHUNK, n_handles=0,
                           pool_capacities=[("dst0", total)],
                           external_handle_pool=pool)
    for _ in range(n_dst):
        dst_arena.grow("dst0", 1)
    src_mta = _MTAShim(src_arena, "src")
    dst_mta = _MTAShim(dst_arena, "dst")
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    src_pool = _FakePool(n_src)
    dst_pool = _FakePool(n_dst)
    src_act = _FakeActuator(src_pool, src_arena)
    dst_act = _FakeActuator(dst_pool, dst_arena)
    actuator = XPoolActuator(
        kv_arena=dst_mta, mamba_arena=src_mta,
        shared_pool=pool,
        kv_actuator=dst_act, mamba_actuator=src_act,
    )
    return actuator, pool, src_arena, dst_arena, src_pool, dst_pool


def _make_plan(pages, direction="mamba_to_kv"):
    return FirePlan(
        direction=direction,
        pages_to_unmap=list(pages),
        pages_to_map_dst=len(pages),
        plan_seq=1,
    )


# ---- Test 1: cap_barrier is mark-only (fast) ----

def test_cap_barrier_does_not_shrink_grow():
    """cap_barrier should NOT do cuMemUnmap/Map. Source chunks stay mapped
    after cap_barrier, destination stays empty."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(10, 0)
    assert src.pool_mapped_chunks("src0") == 10
    assert dst.pool_mapped_chunks("dst0") == 0

    plan = _make_plan(list(range(3)))
    token = act.cap_barrier(plan)
    assert not token.aborted

    assert src.pool_mapped_chunks("src0") == 10, "cap_barrier must NOT shrink src"
    assert dst.pool_mapped_chunks("dst0") == 0, "cap_barrier must NOT grow dst"
    assert token.granted_in_barrier is None, "no in-barrier grow"
    src.cleanup()
    dst.cleanup()


# ---- Test 2: execute_async does shrink + grow + pending ----

def test_execute_async_transfers_and_queues_pending():
    """execute_async should cuMemUnmap src + cuMemMap dst, then push a
    _PendingApply entry."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(10, 0)
    free_before = pool.free_count()

    plan = _make_plan(list(range(3)))
    token = act.cap_barrier(plan)
    result = act.execute_async(token)

    assert not result.aborted
    assert result.unmapped_pages > 0
    assert result.granted_pages > 0
    assert pool.free_count() == free_before, "handles recycled, not leaked"

    with act._pending_lock:
        assert len(act._pending_applies) == 1, "one pending entry"
        p = act._pending_applies[0]
        assert p.shrunk_pages > 0
        assert p.grown_pages > 0
        assert len(p.dst_token_slots) > 0
    src.cleanup()
    dst.cleanup()


# ---- Test 2b: the SYNCHRONOUS execute() must apply inline (rep2-OOM repro) ----

def test_execute_applies_pending_inline():
    """The synchronous `execute()` entrypoint (used by the on-demand KV/mamba
    grows `_grow_kv_from_mamba` / `_grow_mamba_from_kv` and the Admitter) must
    make the grown dst capacity IMMEDIATELY visible, NOT deferred to the next
    tick's `apply_pending_fires`.

    Reproduces the rep2 OOM on a repeated KV-bound burst: `alloc_token_slots`
    gets None, calls the on-demand `_grow_kv_from_mamba` -> `execute()`, which
    physically cuMemMaps the KV chunks and returns granted>0; but pre-fix the
    metadata unmark was left on `_pending_applies` (applied only next tick), so
    the caller's immediate retry `allocator.alloc()` still saw the OLD (small)
    capacity and raised 'Out of memory' even though the grow succeeded. The
    async worker path (cap_barrier + execute_async, applied on the tick) is
    unchanged; only the synchronous `execute()` must self-complete."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(10, 0)
    sp.size = 100
    dp.size = 0
    plan = _make_plan(list(range(3)))
    result = act.execute(plan)  # synchronous entrypoint
    assert not result.aborted and result.granted_pages > 0
    # Pre-fix: execute() leaves the transfer's metadata deferred.
    assert dp.size > 0, "execute() must expose grown dst capacity INLINE"
    assert sp.size < 100, "execute() must shrink src pool.size inline"
    assert sp.size + dp.size == 100, "total conserved"
    with act._pending_lock:
        assert len(act._pending_applies) == 0, (
            "execute() must not leave the transfer deferred to the next tick"
        )
    src.cleanup()
    dst.cleanup()


# ---- Test 3: apply_pending updates pool sizes ----

def test_apply_pending_updates_pool_sizes():
    """apply_pending_fires should update src/dst pool.size and clear the
    pending queue."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(10, 0)
    sp.size = 100
    dp.size = 0

    plan = _make_plan(list(range(3)))
    token = act.cap_barrier(plan)
    act.execute_async(token)

    n_applied = act.apply_pending_fires()
    assert n_applied == 1
    assert sp.size < 100, "src pool.size should decrease"
    assert dp.size > 0, "dst pool.size should increase"
    assert sp.size + dp.size == 100, "total conserved"

    with act._pending_lock:
        assert len(act._pending_applies) == 0, "queue cleared"
    src.cleanup()
    dst.cleanup()


# ---- Test 4: handle conservation across full cycle ----

def test_handle_conservation_full_cycle():
    """Handles are conserved: pool.free_count unchanged after mark →
    worker → apply cycle."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(10, 0)
    free_before = pool.free_count()
    total_mapped_before = src.pool_mapped_chunks("src0") + dst.pool_mapped_chunks("dst0")

    plan = _make_plan(list(range(5)))
    token = act.cap_barrier(plan)
    act.execute_async(token)
    act.apply_pending_fires()

    total_mapped_after = src.pool_mapped_chunks("src0") + dst.pool_mapped_chunks("dst0")
    assert pool.free_count() == free_before, "no handle leak"
    assert total_mapped_after == total_mapped_before, "total mapped chunks conserved"
    src.cleanup()
    dst.cleanup()


# ---- Test 5: cap_barrier latency < 2ms ----

def test_cap_barrier_latency():
    """cap_barrier (mark-only) should complete in < 2ms."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(10, 0)
    plan = _make_plan(list(range(3)))

    t0 = time.perf_counter()
    token = act.cap_barrier(plan)
    dt_ms = (time.perf_counter() - t0) * 1000

    assert dt_ms < 2.0, f"cap_barrier took {dt_ms:.1f}ms, target < 2ms"
    assert not token.aborted
    src.cleanup()
    dst.cleanup()


# ---- Test 6: multiple fires queue and apply correctly ----

def test_multiple_pending_fires():
    """Multiple fires should accumulate in the pending queue and all apply."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(20, 0)
    sp.size = 200
    dp.size = 0

    for i in range(3):
        plan = FirePlan(
            direction="mamba_to_kv",
            pages_to_unmap=[i * 2, i * 2 + 1],
            pages_to_map_dst=2,
            plan_seq=i + 1,
        )
        token = act.cap_barrier(plan)
        act.execute_async(token)

    with act._pending_lock:
        assert len(act._pending_applies) == 3

    n = act.apply_pending_fires()
    assert n == 3
    assert dp.size > 0
    assert sp.size + dp.size == 200
    src.cleanup()
    dst.cleanup()


# ---- Test 7: scheduler can alloc during async fire ----

def test_concurrent_alloc_during_fire():
    """Simulates scheduler allocating from dst while worker fires.
    The dst capacity only appears after apply_pending, not during the fire."""
    act, pool, src, dst, sp, dp = _make_xpool_actuator(10, 0)

    plan = _make_plan(list(range(3)))
    token = act.cap_barrier(plan)

    assert dst.pool_mapped_chunks("dst0") == 0, "dst empty before worker runs"

    barrier = threading.Barrier(2, timeout=5)
    worker_done = threading.Event()

    def worker():
        barrier.wait()
        act.execute_async(token)
        worker_done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    barrier.wait()
    time.sleep(0.01)

    with act._pending_lock:
        pending_during = len(act._pending_applies)

    worker_done.wait(timeout=10)
    assert worker_done.is_set()

    with act._pending_lock:
        pending_after = len(act._pending_applies)
    assert pending_after >= 1, "worker pushed pending"

    act.apply_pending_fires()
    assert dp.size > 0, "capacity available after apply"
    t.join(timeout=5)
    src.cleanup()
    dst.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
