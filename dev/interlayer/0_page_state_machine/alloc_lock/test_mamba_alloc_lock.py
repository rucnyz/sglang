"""#222 — MambaPool needs `_alloc_lock`.

Parallel to KV-side `BaseTokenToKVPoolAllocator._alloc_lock`
(`allocator.py:81`). MambaPool exposes the same cross-thread race
surface:

  worker thread   : ``MambaArenaActuator.unmark_token_slots`` /
                    ``set_capacity_slots`` SHRINK   ─ reads + rebinds
                    ``free_slots``, ``_capped_slots``, ``self.size``
  scheduler thread: ``alloc`` / ``free`` / ``migrate_slot``         ─ reads + rebinds
                    the same fields

Without a lock the dominant failure is set_capacity_slots SHRINK
racing with alloc:

  1. alloc reads ``self.free_slots`` (mask shape N)
  2. worker filters ``free_slots`` to ids ≤ n_slots, rebinds to
     shape M < N
  3. worker sets ``self.size = n_slots``
  4. alloc indexes the OLD tensor (length N) → returns slot IDs
     > n_slots that are about to be cuMemUnmap'd
  5. scheduler hands those IDs to a forward pass → unmapped VA → crash

The KV-side test in ``race.py`` proved the symmetric bug there; this
file pins the contract for MambaPool.

Pure-Python, CPU tensors, no GPU/sglang boot. Uses the production
``MambaPool`` class with ``__new__`` to skip the heavy state init —
we want to exercise only the allocator bookkeeping that owns the
race, not the conv/temporal tensor backing.
"""
from __future__ import annotations

import inspect
import sys
import threading

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch
from sglang.srt.arena.mamba_actuator import _MambaCapAllocator
from sglang.srt.mem_cache.memory_pool import MambaPool


def _build_pool(size: int = 256, max_size: int | None = None) -> MambaPool:
    """Minimal MambaPool sufficient for the bookkeeping paths.

    Bypasses ``__init__`` (which allocates GPU-flavored conv + temporal
    tensors) and sets just the fields ``alloc`` / ``free`` /
    ``set_capacity_slots`` / ``unmark_slots`` read.
    """
    if max_size is None:
        max_size = size
    pool = MambaPool.__new__(MambaPool)
    pool.size = size
    pool.max_size = max_size
    pool.device = "cpu"
    pool.num_mamba_layers = 1
    pool._alloc_lock = threading.Lock()
    pool.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device="cpu")
    if size < max_size:
        pool._capped_slots = torch.arange(
            size + 1, max_size + 1, dtype=torch.int64, device="cpu"
        )
    else:
        pool._capped_slots = torch.empty(0, dtype=torch.int64, device="cpu")
    # Real conv/temporal at small width: shape (layers, max_size+1, hidden)
    conv = [torch.zeros((1, max_size + 1, 4), dtype=torch.float32, device="cpu")]
    temporal = torch.zeros((1, max_size + 1, 4), dtype=torch.float32, device="cpu")
    pool.mamba_cache = MambaPool.State(conv=conv, temporal=temporal)
    return pool


# ---------- test_1: _alloc_lock attribute exists ----------


def test_1_alloc_lock_attribute_exists():
    """`MambaPool.__init__` must initialize `self._alloc_lock` to a
    `threading.Lock`-shaped object. Direct attribute access (no
    `getattr(..., None)` defensive lookup) — see memory rule
    feedback_no_getattr_none_state.
    """
    pool = _build_pool(size=8)
    # _build_pool bypasses __init__; the production fix sets _alloc_lock
    # in __init__ AND we mirror that in _build_pool so all other tests
    # don't trip on missing attribute. The contract test below verifies
    # the production __init__ source actually sets it.
    init_src = inspect.getsource(MambaPool.__init__)
    assert "self._alloc_lock" in init_src, (
        "BUG (#222): MambaPool.__init__ must set self._alloc_lock = "
        "threading.Lock() (parallel to allocator.py:81). Worker-thread "
        "set_capacity_slots / unmark_slots race scheduler-thread alloc "
        "/ free without it."
    )
    assert "threading.Lock" in init_src or "Lock()" in init_src, (
        "BUG (#222): self._alloc_lock must be a threading.Lock, not a "
        "no-op stub."
    )
    print(f"  PASS  1  MambaPool.__init__ sets self._alloc_lock")


# ---------- test_2: mutation methods acquire the lock ----------


def test_2_mutation_methods_acquire_lock():
    """Every method that mutates {free_slots, _capped_slots, self.size}
    must wrap its body with `with self._alloc_lock:`. Symmetric to the
    KV side where allocator.py wraps alloc / free / mark_pages_capped /
    unmark_pages_capped / set_capacity_pages.
    """
    must_lock = [
        "alloc",
        "free",
        "migrate_slot",
        "clear",
        "set_capacity_slots",
        "unmark_slots",
    ]
    missing = []
    for name in must_lock:
        method = getattr(MambaPool, name)
        src = inspect.getsource(method)
        if "self._alloc_lock" not in src:
            missing.append(name)
    assert not missing, (
        f"BUG (#222): the following MambaPool mutation methods do NOT "
        f"acquire self._alloc_lock: {missing}. Each must wrap its "
        f"body with `with self._alloc_lock:` (see allocator.py:185, "
        f":219, :283, :317, :404, :465 for the symmetric KV-side "
        f"contract)."
    )
    print(f"  PASS  2  all {len(must_lock)} MambaPool mutators acquire _alloc_lock")


# ---------- test_2b: _MambaCapAllocator mark/unmark acquire pool._alloc_lock ----------


def test_2b_capallocator_methods_acquire_pool_lock():
    """`_MambaCapAllocator.mark_pages_capped` and `unmark_pages_capped`
    are called by the cross-pool actuator on the **worker thread** to
    perform the m2k cap_barrier (`xpool_actuator.cap_barrier` →
    `dst_pool.allocator.mark_pages_capped(...)`). They mutate
    `pool.free_slots` and `pool._capped_slots` directly.

    Without `with pool._alloc_lock:` they race scheduler-thread
    `alloc` / `free` — the same race the six `MambaPool` mutators
    were just locked against. The lock surface is incomplete unless
    these two also acquire it.

    Symmetric to KV side, where the equivalent `mark_pages_capped`
    lives on the allocator itself (which already holds the lock) —
    mamba's separate `_MambaCapAllocator` class must reach into
    `pool._alloc_lock`.
    """
    missing = []
    for name in ("mark_pages_capped", "unmark_pages_capped"):
        method = getattr(_MambaCapAllocator, name)
        src = inspect.getsource(method)
        if "pool._alloc_lock" not in src:
            missing.append(name)
    assert not missing, (
        f"BUG (#222 audit BLOCKER): _MambaCapAllocator.{missing} do NOT "
        f"acquire pool._alloc_lock. Worker-thread cap_barrier mutates "
        f"pool.free_slots + pool._capped_slots while scheduler is in "
        f"alloc/free → same race the six MambaPool mutators were just "
        f"locked against."
    )
    print(f"  PASS  2b _MambaCapAllocator.mark/unmark_pages_capped "
          f"acquire pool._alloc_lock")


# ---------- test_3: concurrent set_capacity_slots SHRINK vs alloc ----------


def test_3_concurrent_setcap_shrink_vs_alloc_no_orphan():
    """Worker SHRINKs the cap while scheduler is alloc'ing. After both
    threads finish, no slot ID that was allocated should now sit above
    the final cap (`self.size`) — that would mean scheduler holds a
    slot whose chunk is about to be cuMemUnmap'd.
    """
    SIZE = 256
    ITERATIONS = 2000
    pool = _build_pool(size=SIZE, max_size=SIZE)

    allocated: set[int] = set()
    set_lock = threading.Lock()
    errors: list[str] = []
    worker_exc: list[BaseException] = []
    stop = threading.Event()

    def scheduler_thread():
        rng = torch.Generator().manual_seed(0)
        cnt = 0
        while not stop.is_set() and cnt < ITERATIONS:
            n = int(torch.randint(1, 4, (1,), generator=rng).item())
            res = pool.alloc(n)
            if res is not None:
                ids = set(int(x) for x in res.tolist())
                with set_lock:
                    overlap = ids & allocated
                    if overlap:
                        errors.append(
                            f"DOUBLE-ALLOC: slots {sorted(overlap)} returned "
                            f"while still in allocated"
                        )
                        stop.set()
                        return
                    allocated.update(ids)
            # Occasional free to keep capacity churning
            if allocated and cnt % 3 == 0:
                with set_lock:
                    if allocated:
                        victim = next(iter(allocated))
                        allocated.discard(victim)
                pool.free(torch.tensor([victim], dtype=torch.int64))
            cnt += 1

    def worker_thread():
        try:
            cnt = 0
            cap = SIZE
            while not stop.is_set() and cnt < ITERATIONS:
                cap = SIZE // 2 if cap == SIZE else SIZE
                pool.set_capacity_slots(cap)
                cnt += 1
        except BaseException as e:
            worker_exc.append(e)
            stop.set()

    t_sched = threading.Thread(target=scheduler_thread, name="scheduler")
    t_worker = threading.Thread(target=worker_thread, name="worker")
    t_sched.start(); t_worker.start()
    t_sched.join(timeout=30); t_worker.join(timeout=30)
    stop.set()

    if worker_exc:
        raise AssertionError(
            f"Race detected as worker crash: {type(worker_exc[0]).__name__}: "
            f"{worker_exc[0]}. set_capacity_slots reads self.free_slots "
            f"non-atomically with alloc's slice+rebind."
        )
    if errors:
        for e in errors[:5]:
            print(f"    {e}")
        raise AssertionError(
            f"Race detected: {len(errors)} double-allocations observed in "
            f"{ITERATIONS} iterations. set_capacity_slots and alloc are "
            f"both mutating self.free_slots without a lock."
        )
    # Post-hoc invariant: no allocated slot ID should now sit above the
    # final cap. (Cleaner asserted via pool internals.)
    above_cap = {sid for sid in allocated if sid > pool.size}
    if above_cap:
        raise AssertionError(
            f"ORPHAN: scheduler holds {len(above_cap)} slot IDs above final "
            f"cap={pool.size} (e.g. {sorted(above_cap)[:5]}). Without "
            f"_alloc_lock the SHRINK path can complete while scheduler "
            f"has already returned a tail slot — its chunk is about to "
            f"be cuMemUnmap'd."
        )
    fs_set = set(int(x) for x in pool.free_slots.tolist())
    overlap = allocated & fs_set
    if overlap:
        raise AssertionError(
            f"Inconsistency: {len(overlap)} slots in BOTH allocated and "
            f"free_slots, e.g. {sorted(overlap)[:5]}"
        )
    print(f"    {ITERATIONS} interleavings, no double-alloc / orphan")


# ---------- test_4: concurrent unmark_slots vs alloc ----------


def test_4_concurrent_unmark_vs_alloc_no_double_return():
    """Worker thread restores capped slots while scheduler is allocating.
    `unmark_slots` does two non-atomic writes:
        self._capped_slots = capped[~in_restore]
        self.free_slots    = torch.cat([self.free_slots, restore])

    Without a lock, an alloc reading free_slots between those two
    writes can miss the restored ids (mildly suboptimal). But a worse
    failure: alloc snapshots ``self.free_slots`` BEFORE the cat, then
    rebinds ``self.free_slots = self.free_slots[need_size:]`` AFTER the
    cat — the worker's cat is silently undone, restored slots drop on
    the floor, capped accounting goes out of sync.
    """
    SIZE = 128
    MAX_SIZE = 256
    ITERATIONS = 2000
    pool = _build_pool(size=SIZE, max_size=MAX_SIZE)

    errors: list[str] = []
    worker_exc: list[BaseException] = []
    stop = threading.Event()

    def scheduler_thread():
        rng = torch.Generator().manual_seed(1)
        cnt = 0
        while not stop.is_set() and cnt < ITERATIONS:
            n = int(torch.randint(1, 4, (1,), generator=rng).item())
            res = pool.alloc(n)
            if res is not None:
                pool.free(res)
            cnt += 1

    def worker_thread():
        try:
            cnt = 0
            while not stop.is_set() and cnt < ITERATIONS:
                # Round-trip: capture some live ids → unmark restores them.
                # set_capacity_slots SHRINK gives _capped_slots fresh content.
                pool.set_capacity_slots(SIZE // 2)
                pool.set_capacity_slots(SIZE)
                # Now exercise unmark with a few IDs the SHRINK just capped.
                if pool._capped_slots.numel() > 0:
                    ids = pool._capped_slots[:2]
                    pool.unmark_slots(ids)
                cnt += 1
        except BaseException as e:
            worker_exc.append(e)
            stop.set()

    t_sched = threading.Thread(target=scheduler_thread, name="scheduler")
    t_worker = threading.Thread(target=worker_thread, name="worker")
    t_sched.start(); t_worker.start()
    t_sched.join(timeout=30); t_worker.join(timeout=30)
    stop.set()

    if worker_exc:
        raise AssertionError(
            f"Race detected as worker crash: {type(worker_exc[0]).__name__}: "
            f"{worker_exc[0]}."
        )
    # Invariant: free_slots ∩ _capped_slots == ∅
    fs = set(int(x) for x in pool.free_slots.tolist())
    cs = set(int(x) for x in pool._capped_slots.tolist())
    overlap = fs & cs
    if overlap:
        raise AssertionError(
            f"Inconsistency: {len(overlap)} slots in BOTH free_slots and "
            f"_capped_slots, e.g. {sorted(overlap)[:5]}. unmark_slots "
            f"non-atomic write was interrupted by alloc."
        )
    # Invariant: |free_slots| + |_capped_slots| ≤ max_size
    total = pool.free_slots.numel() + pool._capped_slots.numel()
    if total > pool.max_size:
        raise AssertionError(
            f"Inconsistency: |free|+|capped|={total} > max_size={pool.max_size}"
        )
    print(f"    {ITERATIONS} interleavings, free/_capped sets disjoint")


# ---------- runner ----------


def main():
    tests = [
        ("1 MambaPool.__init__ sets self._alloc_lock",
         test_1_alloc_lock_attribute_exists),
        ("2 mutation methods acquire _alloc_lock",
         test_2_mutation_methods_acquire_lock),
        ("2b _MambaCapAllocator mark/unmark acquire pool._alloc_lock",
         test_2b_capallocator_methods_acquire_pool_lock),
        ("3 concurrent set_capacity_slots SHRINK vs alloc",
         test_3_concurrent_setcap_shrink_vs_alloc_no_orphan),
        ("4 concurrent unmark_slots vs alloc",
         test_4_concurrent_unmark_vs_alloc_no_double_return),
    ]
    passed = 0
    for name, fn in tests:
        print(f"\n  TEST {name}")
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            import traceback
            traceback.print_exc()
    print(f"\n#222 MambaPool _alloc_lock: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
