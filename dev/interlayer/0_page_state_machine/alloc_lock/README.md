# alloc_lock — worker↔scheduler concurrency correctness

Tests for the cross-thread mutation paths in the allocator. Both
KV-side (`BaseTokenToKVPoolAllocator`) and mamba-side (`MambaPool`)
allocators expose mutation entry points that the cross-pool actuator
worker thread and the scheduler thread both call concurrently — the
`_alloc_lock` is the synchronization contract.

Design ref: [`../../design.md`](../../design.md) §"Threading model"
(scheduler vs worker thread) + §"Transfer protocol" Stage 0/1 (where
cap_barrier on the scheduler thread interleaves with worker fire
execution).

## KV-side tests (`race.py`)

### test_1 — concurrent `set_capacity_pages` vs `alloc/free` race

Reproduces a real race in `python/sglang/srt/mem_cache/allocator.py`
between:
- Worker thread: `dst_act.cap_allocator_only` →
  `set_capacity_pages` reads `self.free_pages`, computes mask, then
  re-reads to index.
- Scheduler thread: `alloc()` does `self.free_pages[:n]` then
  `self.free_pages = self.free_pages[n:]`.

Between worker's mask-compute and mask-apply, scheduler rebinds
`self.free_pages` to a different shape → `IndexError: shape [4087]
vs [4083]`. Real Python race, not CUDA-async. Reproduced reliably
in ~30s of concurrent load.

**Fix**: per-allocator `threading.Lock` (`_alloc_lock`) wrapping
mark/unmark_pages_capped + set_capacity_pages + alloc + free +
merge_and_sort_free.

### test_2 — worker exception leaks capped pages

Reproduces leak path: `BudgetAgent._fire_worker_loop`'s
`except Exception: continue` swallowed cuMem/ctypes failures, never
restoring `cap_barrier`-removed pages → permanent capacity leak per
failure. Test drives the real agent worker with a stub actuator that
raises; asserts `cap_target` pages return to `free_pages`.

**Fix**: in worker except branch, call
`token.src_act.allocator.unmark_pages_capped(token.cap_t)` (with
nested try for rollback failure path).

## Mamba-side tests (`test_mamba_alloc_lock.py`)

Symmetric race on the mamba side: worker
`MambaArenaActuator.unmark_token_slots` / `set_capacity_slots` SHRINK
mutates `free_slots` + `_capped_slots` + `self.size` while scheduler
`alloc` / `free` / `migrate_slot` reads + rebinds the same fields.
Same failure mode as the KV side — without a lock the worker SHRINK
can complete in the middle of scheduler `alloc.free_slots[:n]`, the
scheduler ends up holding a slot ID that's about to be unmapped, and
the next forward pass hits unmapped VA.

**Fix**: `self._alloc_lock = threading.Lock()` in `MambaPool.__init__`,
wraps `alloc` / `free` / `migrate_slot` / `clear` /
`set_capacity_slots` / `unmark_slots`. The `_MambaCapAllocator`
mark/unmark methods (called on the worker thread for m2k cap_barrier
rollback) also acquire the same lock. Mirrors
`BaseTokenToKVPoolAllocator` 1:1.

Five tests: `_alloc_lock` attribute exists, all six MambaPool mutators
acquire it, `_MambaCapAllocator` mark/unmark acquire `pool._alloc_lock`
(audit BLOCKER caught), and two concurrency stress tests
(set_capacity_slots SHRINK vs alloc; unmark_slots vs alloc).

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
.venv/bin/python dev/interlayer/0_page_state_machine/alloc_lock/race.py
.venv/bin/python dev/interlayer/0_page_state_machine/alloc_lock/test_mamba_alloc_lock.py
```

Pure-Python, no GPU; takes ~10s combined. Tests use the production
`TokenToKVPoolAllocator` + `BudgetAgent` + `MambaPool` classes (no
mocking of the SUT for the KV race + leak; mamba uses `__new__` to
bypass conv/temporal tensor allocation but exercises the real
allocator bookkeeping).

## Result

- `race.py`: 2/2 PASS post-fix. Pre-fix both FAIL (IndexError + LEAK).
- `test_mamba_alloc_lock.py`: 5/5 PASS post-fix. Pre-fix 0/5 (same
  IndexError symptom + missing-attribute on the contract tests).

## Why byte_transfer / idle_no_regression didn't catch these

Both happy-path: only 5-22 fires total, scheduler.alloc rate low
(RPS=4) → race window rarely trips. No exception injection → leak
path never exercised. This folder's tests use accelerated loops +
direct exception injection to make the race / leak observable in
seconds.

Commit: `9e6e349f50`

---

## Follow-up — TODO 1 closed (root cause: pytorch issue 165419)

After active-fix v2 (`ed6da1eda9`) dropped phantom fires to 0 on idle_no_regression
idle, ~+3.5% TTFT-inter cost remained. Bisected in two layers:

1. **3-phase bisect** (`bisect_3phase.sh`, N=3 paired):
   `inter` vs `arena_only` Δ = +0.24% — **budgeter contributes ~0**.
   All the cost is in arena tensor backing.

2. **Arena sub-bisect** (`bisect_arena_path.sh`, N=3 paired):
   `arena_fromblob` vs `arena_mempool` Δ = -2.94% SE 1.42 (|Δ|/SE
   = 2.07, sig). 3/3 paired runs faster on from_blob. Matches the
   magnitude of the residual cost.

Root cause: `torch.cuda.MemPool` silently disables expandable_segments
process-wide (pytorch issue 165419), costing ~3% TTFT under live
attention + CUDA graph capture. The from_blob path bypasses MemPool
entirely (uses `at::from_blob` over cuMemMap-backed VA).

**Fix applied**: `multi_tensor_arena.py` now uses from_blob as the
only path; the MemPool branch was deleted (-44 lines). Single code
path. Commits: `cd3902bcc6` (verdict + bisect), `241463552d` (cleanup).

Validated after cleanup: vmm_boot_smoke 16/16 PASS, cuda_graph_safety 3/3 PASS, idle_no_regression sanity PASS
(Δ = -1.16% / +2.70%, well within noise band).

**Lesson learned**: 4 prior micro-tests (lock, mempool_demo,
tick_cost, arena_tensor_perf) all individually exonerated their
piece, but none ran real attention + CUDA graph capture concurrently
— so all missed the real cost. The `mempool_penalty_demo.py`
N=10 result (no penalty on isolated alloc kernels) is preserved as
historical evidence that the penalty only emerges under real serving.
