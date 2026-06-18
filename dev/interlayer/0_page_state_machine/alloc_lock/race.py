"""alloc_lock race — verify the async-fire claims under concurrent allocator mutation.

Subagent review of commit 56e8237098 flagged two race/correctness
concerns. This file REPRODUCES them as failing tests *before* applying
fixes (so we know the claims are real, not theoretical).

  test_1: concurrent set_capacity_pages (worker) vs alloc/free (scheduler)
          on the same TokenToKVPoolAllocator — same-page-twice race
  test_2: worker exception path leaks capped pages (no unmark on except)

Pure-Python, CPU tensors, no GPU/sglang boot. Uses the production
TokenToKVPoolAllocator class directly.
"""
from __future__ import annotations

import queue
import sys
import threading
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator


def _build_allocator(size=4096):
    """Pure-CPU instance; kvcache=None is OK because alloc/free/cap don't
    touch the kv buffer (only allocator-side bookkeeping)."""
    alloc = TokenToKVPoolAllocator(
        size=size, dtype=torch.bfloat16, device="cpu",
        kvcache=None, need_sort=False,
    )
    return alloc


# ---------- test_1: concurrent set_capacity_pages vs alloc race ----------

def test_1_concurrent_setcap_vs_alloc_no_double_alloc():
    """Race claim (subagent review issue #1):
    worker thread:  set_capacity_pages(cap+k) -> torch.cat onto self.free_pages
    scheduler thread: alloc(n) -> slice + rebind self.free_pages
    No lock, no atomic. The cat can clobber a slice that just happened →
    same page returned to two alloc() calls.

    Detection: allocated_set tracks page IDs currently allocated.
    Every alloc result must contain pages NOT already in the set;
    every free must remove pages that ARE in the set."""
    SIZE = 4096
    ITERATIONS = 5000
    alloc = _build_allocator(size=SIZE)

    # Start with cap at SIZE/2 so set_capacity has room to oscillate.
    alloc.set_capacity_pages(SIZE // 2)

    allocated_set: set[int] = set()
    set_lock = threading.Lock()  # ONLY to make our test bookkeeping safe,
                                  # NOT to protect the allocator. The allocator
                                  # is the SUT and we test its raw thread-safety.
    errors: list[str] = []
    worker_exc: list[BaseException] = []
    stop = threading.Event()

    def scheduler_thread():
        rng = torch.Generator().manual_seed(0)
        cnt = 0
        while not stop.is_set() and cnt < ITERATIONS:
            n = int(torch.randint(1, 8, (1,), generator=rng).item())
            res = alloc.alloc(n)
            if res is not None:
                ids = set(int(x) for x in res.tolist())
                with set_lock:
                    overlap = ids & allocated_set
                    if overlap:
                        errors.append(
                            f"DOUBLE-ALLOC: pages {sorted(overlap)} returned "
                            f"while still in allocated_set"
                        )
                        stop.set()
                        return
                    allocated_set.update(ids)
            # Occasionally free a random page back
            if allocated_set and cnt % 3 == 0:
                with set_lock:
                    if allocated_set:
                        victim = next(iter(allocated_set))
                        allocated_set.discard(victim)
                # Free outside the bookkeeping lock so it can race the worker
                alloc.free(torch.tensor([victim], dtype=torch.int64))
            cnt += 1

    def worker_thread():
        try:
            cnt = 0
            cap = SIZE // 2
            while not stop.is_set() and cnt < ITERATIONS:
                # Oscillate cap between SIZE/2 and SIZE — exercises both
                # shrink and grow branches of set_capacity_pages.
                cap = SIZE if cap == SIZE // 2 else SIZE // 2
                alloc.set_capacity_pages(cap)
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
        # Race can manifest as crash too: between
        # `mask = self.free_pages > n_pages` and `self.free_pages[mask]`,
        # scheduler rebinds self.free_pages to a different shape →
        # IndexError. In production this crashes the worker thread,
        # capped pages leak (test_2), and budgeter goes silent.
        raise AssertionError(
            f"Race detected as worker crash: {type(worker_exc[0]).__name__}: "
            f"{worker_exc[0]}. set_capacity_pages reads self.free_pages "
            f"non-atomically with alloc's slice+rebind."
        )
    if errors:
        for e in errors[:5]:
            print(f"    {e}")
        raise AssertionError(
            f"Race detected: {len(errors)} double-allocations observed in "
            f"{ITERATIONS} iterations. set_capacity_pages and alloc are "
            f"both mutating self.free_pages without a lock."
        )
    # If no double-alloc, also check that all pages in allocated_set are
    # NOT in free_pages (no page handed out yet still in free list)
    fp_set = set(int(x) for x in alloc.free_pages.tolist())
    overlap = allocated_set & fp_set
    if overlap:
        raise AssertionError(
            f"Inconsistency: {len(overlap)} pages in BOTH allocated_set "
            f"and free_pages, e.g. {sorted(overlap)[:5]}"
        )
    print(f"    {ITERATIONS} interleavings, no double-alloc detected")


# ---------- test_2: worker exception leaks capped pages ----------

def test_2_worker_exception_does_not_leak_capped_pages():
    """Subagent review issue #2: when XPoolActuator.execute_async
    raises (e.g. transient ctypes/cuMem failure), pages already removed
    from free_pages by cap_barrier must be restored — otherwise they
    leak permanently from src pool capacity.

    Drive the REAL BudgetAgent worker loop:
      1. Build an agent + replace its actuator with a stub that raises
      2. Call cap_barrier (or simulate its effect)
      3. Push a token to the agent's queue
      4. Wait for worker to process
      5. Assert: cap_target pages no longer in _capped_pages
                 AND back in free_pages
    """
    from sglang.srt.budgeter.agent import BudgetAgent

    SIZE = 4096
    alloc = _build_allocator(size=SIZE)
    initial_live = alloc.live_size
    initial_free_count = int(alloc.free_pages.numel())

    # Build a minimal agent (no scheduler needed — we bypass tick())
    class _DummyScheduler:
        pass
    agent = BudgetAgent.__new__(BudgetAgent)  # bypass __init__
    agent.log_enabled = False
    agent._log_fp = None

    # Stub actuator whose execute_async raises
    class _StubActuator:
        class _Shared:
            def free_count(self): return 0
        shared = _Shared()
        def execute_async(self, token):
            raise RuntimeError("simulated cuMem failure")
    agent._actuator = _StubActuator()

    # Build the token + cap_barrier effect
    cap_target = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17],
                              dtype=torch.int64)
    moved = alloc.mark_pages_capped(cap_target)
    assert moved == 8

    # Stub src_act so worker can find allocator for rollback
    class _StubSrcAct:
        allocator = alloc
    class _StubToken:
        src_act = _StubSrcAct()
        cap_t = cap_target

    # Drive worker: queue the token, start thread, wait, sentinel-stop
    agent._fire_queue = queue.Queue(maxsize=4)
    agent._fire_queue.put_nowait(_StubToken())
    agent._fire_queue.put_nowait(None)  # shutdown sentinel
    worker = threading.Thread(target=agent._fire_worker_loop, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "worker did not finish in 5s"

    final_live = alloc.live_size
    final_free_count = int(alloc.free_pages.numel())
    capped_count = int(alloc._capped_pages.numel())
    print(f"    after worker exception: live={final_live} "
          f"free={final_free_count} capped={capped_count}")

    if final_live != initial_live or capped_count != 0:
        raise AssertionError(
            f"LEAK: worker exception path didn't roll back cap_barrier. "
            f"live {initial_live}→{final_live}, capped {capped_count}, "
            f"free {initial_free_count}→{final_free_count}. "
            f"agent.py worker except must call unmark_pages_capped(token.cap_t)."
        )
    assert final_free_count == initial_free_count, (
        f"live recovered but free_count {initial_free_count}→{final_free_count}"
    )
    print(f"    PASS: capped pages restored, allocator back to initial state")


# ---------- runner ----------

def main():
    tests = [
        ("1 concurrent set_capacity_pages vs alloc — no double-alloc",
         test_1_concurrent_setcap_vs_alloc_no_double_alloc),
        ("2 worker exception path does not leak capped pages",
         test_2_worker_exception_does_not_leak_capped_pages),
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
    print(f"\nalloc_lock race: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
