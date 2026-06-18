# Test Depth Audit: dyn_admission_cap Phases 1-7 + xpool_actuator

**Audit Date:** 2026-05-27  
**Focus:** Mock vs. Real execution; logic coverage; edge cases; integration risk.

---

## Executive Summary

**Overall Verdict:** **MIXED — SUFFICIENT for happy path, GAPS in edge-case coverage**

The test suite validates **core data-ptr-stable mechanics** (Phase 1-4, 5b) and **happy-path admission-cap orchestration** (Phase 5). However:

1. **Test 5 (Budgeter) is now using REAL pools** — a major improvement that exercises genuine grow/shrink semantics. ✓
2. **Phase 5b (CUDA graph replay)** provides critical P0 validation that VA stability works end-to-end. ✓
3. **Edge cases are under-tested:** boundary conditions (shrink-to-0, grow-to-exact-max), held-slot race patterns, and concurrent mamba pool changes.
4. **xpool balanced-atomic test is pure math** — no integration test validates that correct math still breaks when calling the wrong chunk_arena methods or passing wrong indices.

---

## Per-Test-File Summary

### test_phase1.py (ReqTokenVAArena, 6 tests)
**Verdict: DEEP**

- **Coverage:** All 6 acceptance gates exercised with real CUDA VA + mapping.
- **Mock vs Real:** REAL arena (ChunkArena, cuMemMap, cuMemUnmap). No mocks.
- **Strengths:**
  - Data-ptr stability across multiple grow/shrink cycles (test_4) is critical P0.
  - Unmapped row access correctly skipped (no false positives from stale reads).
  - Cleanup + secondary arena construction validates no global state corruption.
- **Gaps:** No test of partial failure (e.g., arena.grow() returns fewer chunks than requested). Partial grow is caught by assertion, but not validated in a test.

---

### test_phase2.py (ReqToTokenPool grow/shrink, 6 tests)
**Verdict: DEEP**

- **Coverage:** Back-compat (torch.zeros), dynamic-cap (VA-arena backed), grow, shrink, free/alloc.
- **Mock vs Real:** REAL ReqToTokenPool, REAL arena. Minimal _StubReq shims only mock the request object API (req_pool_idx, is_chunked, kv_committed_len).
- **Strengths:**
  - Free/alloc round-trip on grown slots (test_4) validates slot recycling.
  - Shrink rejection on held slots (test_5) enforces the precondition check.
  - Back-compat mode (test_1) confirms zero-init fallback for existing code.
- **Gaps:**
  - No test of shrink to 0 (edge case: `shrink(0)` when size=2 or size=4).
  - No test of grow when arena.grow() partially succeeds (would trigger assertion; not tested).
  - No concurrent alloc/free/grow interleaving (simulated stress).

---

### test_phase3.py (HybridReqToTokenPool, 3 tests)
**Verdict: SUFFICIENT**

- **Coverage:** Back-compat, dynamic-cap mode, mapping tensor writable in grown range.
- **Mock vs Real:** REAL HybridReqToTokenPool + Mamba2CacheParams. Minimal mamba state initialization (cache_params shape only; no actual conv/temporal allocations beyond what the pool does).
- **Strengths:**
  - Mapping tensor pre-allocated at max_size, not re-allocated on grow (test_2).
  - Grown indices writable with persistence (test_3).
- **Gaps:**
  - Only 3 tests. No test of mapping data corruption if shape mismatch happens.
  - No test of HybridReqToTokenPool.shrink() (only via reusing test_2's grow).
  - Skip fallback for older Mamba2CacheParams API versions hides potential schema drift.

---

### test_phase4.py (FutureMap, 4 tests)
**Verdict: SUFFICIENT**

- **Coverage:** Back-compat, grow updates future_limit, token_ids_buf stable, grow rejects overshoot.
- **Mock vs Real:** REAL FutureMap with SpeculativeAlgorithm.NONE. No mocks.
- **Strengths:**
  - Circular-buffer wrap-around logic implicitly tested via size calculations.
  - Overshoot rejection (test_4) guards against alloc bugs.
- **Gaps:**
  - No test of grow when new_max_running_requests < max_running_requests (no-op behavior).
  - No test of alloc_future_indices wrapping at future_limit after grow (indirect through other tests, but not explicit).
  - No test with speculative decoding (ALGORITHM != NONE), so lazy_init_buf untested.

---

### test_phase5.py (BudgetAgent._maybe_update_admission_cap, 6 tests)
**Verdict: DEEP** ⭐ **RECENTLY REFACTORED TO REAL POOLS**

- **Coverage:** Real MambaPool, HybridReqToTokenPool, FutureMap. No stubs. Exercises _maybe_update_admission_cap against actual grow/shrink.
- **Mock vs Real:** 
  - ✓ REAL: MambaPool.set_capacity_slots() → mamba_pool.live_size changes → Budgeter detects and cascades.
  - ✓ REAL: HybridReqToTokenPool.grow() called in the pool.grow() branch.
  - ✓ REAL: FutureMap.grow() called via sched.future_map.grow().
- **Strengths:**
  - test_2 (mamba grew, pool followed) validates the cascade end-to-end.
  - test_5 (held slot blocks shrink, retry on release) exercises retry logic with real RuntimeError.
  - test_6 (KV-only model) guards the mamba_pool attribute check.
- **Gaps:**
  - No test of **held slot blocks grow** — only shrink is tested (test_5). Grow should succeed even with held slots, but no explicit test.
  - No test of **concurrent mamba changes** (e.g., mamba grows then shrinks before Budgeter ticks; does ratio stay stable?).
  - No test of `future_map.grow()` raising ValueError (ceiling guards it, but not tested directly).
  - No test of **partial shrink failure due to held slots beyond [new_size, size)** (test_5 only blocks the one slot; what if multiple held?).

---

### test_phase5b_cuda_graph.py (CUDA graph + grow, 4 tests)
**Verdict: DEEP** ⭐ **CRITICAL P0**

- **Coverage:** CUDA graph capture before/after grow, data-ptr stability, boundary grows.
- **Mock vs Real:** REAL ReqToTokenPool, REAL arena, REAL CUDA graph capture.
- **Strengths:**
  - test_1 (pre/post-grow replay) is the WHOLE POINT of VA stability. Failure here = D8 crash.
  - Bonus assertion (row 1 rewrite post-capture) proves graphs read LIVE data, not stale.
  - test_2 (exact max_size), test_3 (sequential grows), test_4 (shrink to 1) cover boundary conditions.
- **Gaps:**
  - Only tests ReqToTokenPool; MambaPool has its own CUDA-graph-capture risk (test_phase7 test_5 covers it).
  - No test of graph capture spanning unmapped rows (would fault on replay — correctly rejected, but not tested as an anti-test).
  - No test of grow DURING graph capture (should fail gracefully; untested).

---

### test_phase7.py (MambaPool dynamic resize, 6 tests)
**Verdict: DEEP**

- **Coverage:** Back-compat, dynamic pre-allocation, set_capacity_slots grow/shrink, CUDA graph capture on conv_state.
- **Mock vs Real:** REAL MambaPool, REAL Mamba2CacheParams, REAL conv/temporal allocations.
- **Strengths:**
  - test_3 (set_capacity_slots grow past init, write to newly exposed slot) validates writable grow.
  - test_4 (grow → shrink → grow) cycles the _capped_slots / free_slots split correctly.
  - test_5 (CUDA graph on conv_state, grow, continue) proves VA stability for Mamba conv state.
  - test_6 (overshoot clamps to max_size) guards against alloc bugs.
- **Gaps:**
  - No test of **shrink when held slot is in the shrunk range** (test_phase5.py test_5 covers ReqToTokenPool; no MambaPool shrink safety test).
  - No test of **temporal_state arena mode** (SGLANG_MAMBA_ARENA=1) — tests use stacked layout only. Shared arena temporal_state has different growth semantics (MultiTensorArena).
  - No test of **set_capacity_slots(0)** (edge case).
  - No test of **rapid set_capacity_slots changes** (e.g., 8 → 12 → 6 → 14) to stress the _capped_slots reordering.

---

### test_xpool_balanced_atomic.py (math test, 6 tests)
**Verdict: SURFACE-LEVEL** ⚠️ **MATH ONLY, NO INTEGRATION**

- **Coverage:** LCM-balanced transfer math: `total = floor(target_total / lcm) * lcm`, per_src = total / n_src, per_dst = total / n_dst.
- **Mock vs Real:** Pure Python math. No arenas, no chunk_arena.shrink_explicit(), no chunk_arena.grow().
- **Invariants Tested:**
  - ✓ Cross-pool atomic: per_src × n_src == per_dst × n_dst == total
  - ✓ Bounded: per_src ≤ len_unmap, per_dst ≤ pages_dst
  - ✓ Smoke: 6×7×4 = 168 combos verify invariants hold.
- **Critical GAP: NO INTEGRATION TEST**
  - The math is correct, but **no test validates the math is used correctly when calling chunk_arena**:
    - Does xpool_actuator compute pages_to_unmap from the math result correctly?
    - Does it call chunk_arena.shrink_explicit on the right pages?
    - Does it call chunk_arena.grow on the right destination?
    - Does passing wrong indices to shrink_explicit silently succeed but shrink wrong data?
  - **Example failure mode:** Math says per_src=3 (correct), but code calls `chunk_arena.shrink_explicit(src_pool, pages=2)` (bug). Test would pass; D8 would OOM mamba because fewer pages freed than math promised.

---

## Uncovered Production Code Paths

### ReqTokenVAArena (req_token_arena.py)
- ✗ `set_mapped_bytes()` partial grow failure (assertion path; not tested)
- ✓ Nominal grow/shrink/cleanup (Phase 1 covers)

### ReqToTokenPool (memory_pool.py)
- ✗ `shrink()` to 0 (edge case)
- ✗ `grow()` followed immediately by `shrink()` to size < grow target (interleaving test)
- ✓ Back-compat mode (Phase 2 test_1)
- ✓ Dynamic grow/shrink (Phase 2 + 5b)

### HybridReqToTokenPool (memory_pool.py)
- ✗ `shrink()` method (only grow tested)
- ✗ Mamba slot mapping consistency post-grow (implied but not explicit)

### MambaPool (memory_pool.py)
- ✗ `set_capacity_slots(0)` (edge case)
- ✗ Rapid set_capacity_slots changes (stress)
- ✗ Temporal state arena mode (SGLANG_MAMBA_ARENA=1 with MultiTensorArena)
- ✓ Back-compat mode (Phase 7 test_1)
- ✓ Dynamic pre-alloc + grow/shrink (Phase 7 tests 2-6)

### FutureMap (overlap_utils.py)
- ✗ Grow with new_max < old max (no-op behavior — implicit, not explicit)
- ✗ Speculative decoding (ALGORITHM != NONE) — lazy_init_buf never called in tests
- ✓ Non-speculative (ALGORITHM.NONE) grow/alloc (Phase 4)

### BudgetAgent._maybe_update_admission_cap (agent.py)
- ✗ **Held slot blocks grow** (grow should succeed; no test)
- ✗ **Future_map.grow() raises ValueError** (user ceiling or max capped it; no direct test)
- ✗ **Concurrent mamba changes** (mamba grows twice before Budgeter ticks; ratio stable?)
- ✗ **Multiple held slots in shrink range** (test_5 only 1 slot; sparse held set untested)
- ✓ First tick + ratio snapshot (Phase 5 test_1)
- ✓ Mamba grow cascade (Phase 5 test_2)
- ✓ User ceiling (Phase 5 test_3)
- ✓ Mamba shrink (Phase 5 test_4)
- ✓ Shrink blocked + retry (Phase 5 test_5)
- ✓ KV-only no-op (Phase 5 test_6)

### XPoolActuator.balanced_atomic math + integration (xpool_actuator.py)
- ✗ **CRITICAL: No integration test** linking math → chunk_arena calls
  - Does expand_pages_to_token_slots() produce the right indices?
  - Does shrink_explicit(pages) shrink the right memory?
  - Does grow(pages) map the right handles?
  - Failure modes:
    - Math correct, indices wrong → wrong pages unmapped/mapped
    - Math correct, order wrong → deallocation before migration fails
    - Math correct, count wrong → fewer/more bytes moved than promised

---

## Recommended Additional Tests

### P0 (Blocking — High Integration Risk)

#### **Test: xpool_balanced_atomic_integration** (test_xpool_balanced_atomic.py, new)
```python
def test_xpool_balanced_atomic_integration():
    """P0: Math → real chunk_arena calls. Validate math result is used correctly
    by the actuator when calling shrink_explicit + grow.
    
    Setup:
      - Create two MultiTensorArena pools on shared SharedHandlePool
      - Populate src pool with N pages
      - Run balanced_atomic logic: compute per_src, per_dst
      - Call actuator.shrink_explicit(src_pool, pages_to_shrink)
      - Call actuator.grow(dst_pool, pages_to_grow) with freed handles
    
    Assertions:
      - src free-count decreased by per_src
      - dst free-count increased by per_dst
      - Moved pages are readable in dst (not corrupted)
      - Old src pages unreachable post-shrink (no lingering refs)
    
    Why: Math test passes but actuator calls chunk_arena wrong → D8 OOMs
    """
```

#### **Test: Budgeter_held_slot_does_not_block_grow** (test_phase5.py, new)
```python
def test_budgeter_held_slot_does_not_block_grow():
    """P0: grow() must succeed even with held slots. Only shrink() is blocked.
    
    Setup:
      - Hold slot 25 (remove from free_slots)
      - Mamba grows, triggering Budgeter to grow pool past 25
    
    Assertions:
      - pool.size increases (grow succeeds despite held slot)
      - New free_slots include slots > 25 (up to new cap)
    
    Why: If grow is also blocked by held slots, admission can deadlock
    """
```

#### **Test: test_phase5b_shrink_to_zero** (test_phase5b_cuda_graph.py, new)
```python
def test_shrink_to_zero():
    """P0: Boundary case. Shrink to 1 works (test_4); shrink to 0 should also
    succeed or be explicitly rejected with clear error.
    
    Setup:
      - Pool size=2, max_size=8, all free
      - shrink(0) → should succeed or raise ValueError with "min_size"
    
    Assertions:
      - If succeeds: size==0, free_slots==[]
      - If raises: message is clear (not an assertion)
    
    Why: Boundary logic error could cause weird state (size=0 but tensor nonzero)
    """
```

---

### P1 (Soon — Coverage Gaps)

#### **Test: concurrent_mamba_changes_stable_ratio** (test_phase5.py, new)
```python
def test_concurrent_mamba_changes_stable_ratio():
    """P1: Multiple mamba changes before Budgeter ticks; ratio stays stable.
    
    Setup:
      - Init: mamba=99, pool=33, ratio=3
      - Mamba grows 99 → 198
      - Before Budgeter ticks, mamba shrinks 198 → 120
      - Budgeter ticks once
    
    Assertions:
      - Pool grows to min(ceiling, 120//3) = 40 (not 66 then back down)
      - _last_mamba_size == 120 (not intermediate 198)
    
    Why: Budgeter should use final state, not intermediate snapshots
    """
```

#### **Test: held_multiple_slots_in_shrink_range** (test_phase5.py, new)
```python
def test_held_multiple_slots_in_shrink_range():
    """P1: Shrink blocked by ANY held slot in range, not just one.
    
    Setup:
      - Pool size=8, hold slots [5, 6, 7]
      - Attempt shrink(4) (would drop [4..8])
    
    Assertions:
      - Raises RuntimeError mentioning 5,6,7 (or at least one of them)
      - Pool.size unchanged at 8
    
    Why: Test_5 only checks slot 25 in range [20..32]; sparse coverage
    """
```

#### **Test: FutureMap_speculative_lazy_init** (test_phase4.py, new)
```python
def test_future_map_speculative_lazy_init():
    """P1: Speculative decoding initializes buffers on first alloc_future_indices.
    
    Setup:
      - FutureMap with SpeculativeAlgorithm != NONE
      - Allocate draft_input via model_worker_batch
      - Call store_to_map_for_new_batch() → triggers _lazy_init_buf
    
    Assertions:
      - topk_p_buf, topk_index_buf, etc. all allocated post-lazy-init
      - Data written + read back correctly
    
    Why: Current tests skip speculative; latent bug in lazy init untested
    """
```

#### **Test: MambaPool_set_capacity_rapid_cycles** (test_phase7.py, new)
```python
def test_mamba_pool_set_capacity_rapid_cycles():
    """P1: Rapid set_capacity_slots changes; _capped_slots reordered correctly.
    
    Setup:
      - Pool size=4, max_size=32
      - set_capacity_slots(8) → free=[1..8], capped=[9..32]
      - set_capacity_slots(16) → free=[1..16], capped=[17..32]
      - set_capacity_slots(6) → free=[1..6], capped=[7..32]
      - set_capacity_slots(24) → free=[1..24], capped=[25..32]
    
    Assertions:
      - free_slots + _capped_slots always partition [1..max_size]
      - No duplicates, no gaps
      - Write to any id in free_slots succeeds
    
    Why: Set_capacity_slots_logic with moving _capped_slots indices may have
         off-by-one; rapid changes stress reordering
    """
```

---

### P2 (Nice-to-have — Lower Risk)

#### **Test: ReqTokenVAArena_partial_grow_assertion** (test_phase1.py, new)
```python
def test_req_token_va_arena_partial_grow_assertion():
    """P2: If ChunkArena.grow() returns fewer chunks than requested,
    ReqTokenVAArena.set_mapped_bytes() raises RuntimeError (not silent no-op).
    
    Setup:
      - Mock ChunkArena.grow() to return 1 chunk when 2 requested
    
    Assertions:
      - Raises RuntimeError mentioning "partial grow" or "handle pool exhausted"
    
    Why: Silent partial grow can cause unmapped-row access; assertion is good
    """
```

#### **Test: HybridReqToTokenPool_shrink** (test_phase3.py, new)
```python
def test_hybrid_req_to_token_pool_shrink():
    """P2: HybridReqToTokenPool.shrink() mirrors ReqToTokenPool behavior.
    
    Setup:
      - Pool size=4, max_size=8, all free
      - Shrink to 2
    
    Assertions:
      - size == 2
      - mamba_index_mapping shape unchanged (8,) but only [0:2] valid
      - Re-grow to 4 re-populates mapping
    
    Why: Phase 3 only tests grow; shrink untested for Hybrid variant
    """
```

#### **Test: FutureMap_grow_noop** (test_phase4.py, new)
```python
def test_future_map_grow_noop():
    """P2: grow(smaller_size) is no-op; data_ptr unchanged.
    
    Setup:
      - max_running=100, max_max=128
      - grow(100) → no-op
      - grow(50) → no-op
    
    Assertions:
      - max_running unchanged at 100
      - token_ids_buf.data_ptr() unchanged
    
    Why: Boundary logic (grow should be idempotent on equal or smaller)
    """
```

---

## Specific Feedback on xpool_balanced_atomic Test

**Current Status:** Pure-math validation. Correct invariants verified.

**Gap:** **No integration test linking math → chunk_arena calls.**

The test currently validates:
- ✓ LCM alignment
- ✓ Invariant per_src × n_src == per_dst × n_dst == total
- ✓ Bounding (per_src ≤ unmap, per_dst ≤ growth)

But it **does NOT test**:
- ✗ Does xpool_actuator use the computed per_src / per_dst correctly?
- ✗ Does it pass the right pages to chunk_arena.shrink_explicit()?
- ✗ Does it pass the right handles to chunk_arena.grow()?
- ✗ Is the order correct (shrink before map, not after)?
- ✗ Do freed handles actually end up in the right pool?

**Recommended Single Integration Test:**
```
test_xpool_balanced_atomic_integration (see P0 above)
```

This test would:
1. Create two real MultiTensorArena pools on a shared SharedHandlePool
2. Populate src pool with known pages
3. Call actuator method that computes balanced_atomic + executes shrink_explicit + grow
4. Verify: free-count deltas match per_src/per_dst, moved data is readable, old src pages gone

**Risk if skipped:** Math is "right" but actuator calls chunk_arena wrong → silent data loss or deallocation-of-wrong-pages → D8 crashes during cross-pool transfer.

---

## Documentation Gaps

### Test Naming & Clarity
| File | Clarity | Gap |
|------|---------|-----|
| test_phase1.py | Excellent | None (clear docstrings, explicit test names) |
| test_phase2.py | Excellent | None |
| test_phase3.py | Good | Mamba API assumption hidden in skip fallback; not explained |
| test_phase4.py | Good | No explanation of circular buffer wrap logic |
| test_phase5.py | Excellent | ⭐ "real classes" comment now accurate; doc is up-to-date |
| test_phase5b_cuda_graph.py | Excellent | P0 test well-documented |
| test_phase7.py | Excellent | Clear per-layout explanation |
| test_xpool_balanced_atomic.py | Good | Missing note: "math-only, no chunk_arena calls" |

### Contributor Onboarding
A new contributor would struggle with:
1. **Why each phase matters** (docstrings are good, but big-picture flow missing)
2. **Where to add integration tests** (test_xpool_balanced_atomic.py should note "pure math; see audit for integration gap")
3. **CUDA graph safety story** (why test_phase5b is P0; audit explains, but should be in test file comment)

---

## Summary: Test Depth Verdict

| Phase | Verdict | Key Finding |
|-------|---------|-------------|
| Phase 1 (ReqTokenVAArena) | **DEEP** | All acceptance gates, no mocks, VA stability critical |
| Phase 2 (ReqToTokenPool) | **DEEP** | Back-compat + dynamic, real arena, grow/shrink/free cycles |
| Phase 3 (HybridReqToTokenPool) | **SUFFICIENT** | Hybrid variant growth, but shrink untested (P1) |
| Phase 4 (FutureMap) | **SUFFICIENT** | Non-speculative tested; speculative deferred (P1) |
| Phase 5 (Budgeter) | **DEEP** ⭐ | **NOW REAL POOLS** — cascading grow/shrink validated end-to-end |
| Phase 5b (CUDA graph) | **DEEP** ⭐ | P0 validation; replay post-grow proves VA stability |
| Phase 7 (MambaPool) | **DEEP** | Back-compat + dynamic, conv_state CUDA graph, grow/shrink cycles |
| xpool balanced-atomic | **SURFACE-LEVEL** ⚠️ | Math correct; no integration test (P0 gap) |

**Aggregate Risk:** **Low-to-Medium**
- **Low:** Core VA-stability + happy-path admission cap (tests cover real code, not mocks)
- **Medium:** Edge cases (held slots, rapid cycles, boundary values) + integration (math → chunk_arena calls)

**Blockers for Production:**
- Add xpool_balanced_atomic_integration test (P0)
- Verify held-slot doesn't block grow (P0)
- Shrink-to-zero boundary (P0)

---

**Audit Prepared By:** Claude Code (Anthropic)
