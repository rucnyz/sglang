# Test Coverage Audit: dyn_admission_cap Phase 1–5

**Audit date:** 2026-05-26  
**Scope:** 25 unit tests across 5 phases; design + implementation complete  
**Goal:** Identify admission-time edge cases and failure modes not yet covered

---

## 1. Race Conditions

### Current Test Coverage

- **Phase 2** (ReqToTokenPool): Tests are purely sequential; alloc/free isolated
- **Phase 5** (Budgeter): Mock-only; no concurrent scheduler thread simulation
- **No tests** for simultaneous operations: alloc happening while grow is in progress

### Gaps

**Gap 1.1: Grow During Admission Alloc**
- Scenario: Worker thread calls `pool.alloc([req1, req2])` (list-slicing free_slots). Simultaneously, scheduler tick calls `pool.grow(new_size)` (extending free_slots). Python list operations are atomic under GIL, but interleaving of extend() and slice() may cause logic errors if free_slots is read mid-operation.
- **Test case:** `test_grow_during_alloc_interleave` (P1)
  - Setup: Pool with 100 slots, 50 free, threads ready
  - Thread A: alloc([r1, r2, ...]) — reads free_slots, removes 10 items
  - Thread B: grow(120) — extends free_slots with 20 new ids
  - Assertion: Final free_slots has exactly 60 items (50 - 10 + 20)
  - Captures: race between list[:] slicing (line 298) and extend() (line 239)

**Gap 1.2: Grow During Attention Forward (cuMemMap Timing)**
- Scenario: CUDA graph replay mid-stride, scheduler tick calls grow. design.md §5 claims "between batches, before backend fetches" but doesn't test this boundary.
- **Test case:** `test_grow_while_backward_active` (P2)
  - Mock backend that holds a ref to pool.req_to_token
  - Thread A: Simulates kernel execution (no actual kernel, just time.sleep)
  - Thread B: Calls pool.grow() → set_mapped_bytes() → cuMemMap
  - cuMemMap is host-side (no stream sync) per design, so should not interfere
  - Assertion: Backend can still read from grown range; no CUDA error
  - Captures: Validates the "safe between batches" claim with timing

**Gap 1.3: Multiple Grows Within Single Tick**
- Scenario: Mamba pool fires multiple times in one tick, Budgeter.tick() called multiple times, or _maybe_update_admission_cap called twice.
- **Test case:** `test_sequential_grows` (P1)
  - Call pool.grow(50), then grow(70), then grow(100) in sequence
  - Assertion: final size=100, free_slots has [0..100), data_ptr stable throughout
  - Captures: grow idempotency (line 227: `if new_size <= self.size: return`)

---

## 2. Boundary Conditions

### Current Test Coverage

- **Phase 1**: Grows from 2→4→6→8 rows (within chunk boundary)
- **Phase 2**: Grows from 2→4, shrinks to 2, re-grows to 6 (all well within bounds)
- **Phase 4**: Grows from 33 to various; never tests edge of max_running_requests_max

### Gaps

**Gap 2.1: Grow to Exactly max_size**
- Scenario: Pool initialized with max_size=100; grow repeatedly to reach exactly 100. Next grow should be no-op or raise.
- **Test case:** `test_grow_to_max_size_boundary` (P1)
  - Setup: Pool(size=10, max_size=100)
  - Call grow(100) — should succeed and fill to max
  - Call grow(101) — should raise ValueError (exceed max_size)
  - Call grow(100) again — should be no-op (already at 100)
  - Assertion: size==100 after first grow; raises on overshoot; second grow returns 100 (no-op)
  - Captures: boundary check at line 229

**Gap 2.2: Shrink to Exactly 0 or 1**
- Scenario: Pool with 10 slots, all free. Shrink to 1, then to 0. Edge of addressable space.
- **Test case:** `test_shrink_to_zero_boundary` (P1)
  - Setup: Pool(size=10, max_size=10), all slots free
  - Call shrink(1) — should succeed
  - Assertion: size==1, free_slots==[0]
  - Call shrink(0) — should succeed if allowed, or raise with clear error if not
  - Captures: minimum viable pool size (design.md §5 says "never below 1")

**Gap 2.3: Grow then Shrink then Grow Cycle**
- Scenario: Pool grows 10→20, shrinks 20→5, then grows 5→25. Verifies arena handle/chunk allocation is idempotent.
- **Test case:** `test_grow_shrink_grow_cycle` (P2)
  - Setup: Pool(size=10, max_size=50)
  - Grow to 25, write pattern A to [10:25]
  - Shrink to 10, verify rows [10:25] unmapped
  - Grow to 25 again, write pattern B to [10:25]
  - Assertion: Pattern A lost (re-mapped fresh), pattern B readable
  - Captures: arena reuses chunks after shrink (no leaks)

**Gap 2.4: Chunk-Boundary Edge Case (Small Grows)**
- Scenario: Pool with row_bytes=1 MiB, chunk_bytes=2 MiB (2 rows/chunk). Grow by 1 row when already at N=odd.
- **Test case:** `test_grow_sub_chunk` (P2)
  - Setup: Pool(size=1, max_size=10), row_bytes=1 MiB, chunk_bytes=2 MiB
  - Call grow(2) — maps 2 rows = 1 full chunk
  - Call grow(3) — maps 3 rows = needs 2 chunks (1 chunk + 1 row)
  - Assertion: arena.mapped_chunks==2 after grow(3); both rows 2 writable
  - Captures: rounding logic in set_mapped_bytes (line 91: `(n_bytes + chunk_bytes - 1) // chunk_bytes`)

---

## 3. Failure Modes

### Current Test Coverage

- **Phase 1**: No OOM simulation; assumes cuMemMap always succeeds
- **Phase 2**: shrink() with held slot raises RuntimeError (test 5), but no recovery test
- **Phase 5**: shrink blocked by held slot defers (test 5), but doesn't test cascading hold release

### Gaps

**Gap 3.1: cuMemMap Physical Memory Exhaustion**
- Scenario: Grow requested but GPU out of physical pages (rare but possible). cuMemMap fails.
- **Test case:** `test_grow_cumemmmap_oom` (P2)
  - Mock ChunkArena.grow() to return fewer chunks than requested
  - Call pool.grow(new_size) — should NOT raise but log + carry on
  - Assertion: pool.size unchanged; admission cap stays conservative
  - Captures: Graceful degradation per design.md §5 "cuMemMap failure"

**Gap 3.2: VA Exhaustion at Initialization**
- Scenario: cuMemAddressReserve fails at ReqTokenVAArena.__init__ (VA exhausted). ReqToTokenPool must handle.
- **Test case:** `test_pool_init_va_reserve_fail` (P2)
  - Mock tensor_from_va to raise RuntimeError("VA exhausted")
  - Call ReqToTokenPool(size=10, max_size=100)
  - Assertion: Raises with clear error message; doesn't silently fall back to torch.zeros
  - Captures: Init-time failure scenario; doesn't block D8 but good defensive programming

**Gap 3.3: Shrink Called on free_slots with Duplicates**
- Scenario: free_slots somehow contains duplicate ids (corruption, race condition). shrink() iterates and may miscalculate.
- **Test case:** `test_shrink_defensive_duplicates` (P2)
  - Setup: Pool with size=10, manually insert duplicates in free_slots (e.g., [0,1,2,2,3,4])
  - Call shrink(5) — should detect duplicates or handle robustly
  - Assertion: Either raises RuntimeError + lists duplicates, or cleans up gracefully
  - Captures: Defensive check (not strictly needed if Python list ops are sound, but good defensive)

**Gap 3.4: grow() Partial Failure Mid-Method**
- Scenario: set_mapped_bytes succeeds, but zero_() fails (CUDA error). Pool left in inconsistent state.
- **Test case:** `test_grow_partial_failure` (P2)
  - Mock req_to_token[self.size:new_size].zero_() to raise RuntimeError
  - Call pool.grow(new_size) — should either succeed fully or revert
  - Assertion: If raised, pool.size unchanged; free_slots unchanged; no partial state
  - Captures: Atomicity of grow; cleanup on failure

---

## 4. Integration Scenarios (Post-D8 Sweeps)

### Current Test Coverage

- **Phase 5**: Mocks only, no real scheduler/pool integration
- **No tests** with multiple pools (KV + mamba) growing simultaneously

### Gaps

**Gap 4.1: Alternating Grow/Shrink Under Sustained Load**
- Scenario: D8c-style: workload holds 80 reqs, mamba bounces between 240 (80×3) and 180 (60×3). Pool shrink defers, retries.
- **Test case:** `test_grow_shrink_sustained_load` (P1 for D8 confidence)
  - Setup: Pool(size=80, max_size=256); hold slots [0..80]
  - Call shrink(60) — should raise (slots held)
  - Free slots [0..20], call shrink(60) — should succeed
  - Assertion: Pool resized to 60, future grow works
  - Captures: Real admission under resize pressure

**Gap 4.2: Both Pools Grow Simultaneously**
- Scenario: Fire transfers between KV and mamba; both pool sizes increase. Budgeter calls grow on both in same tick.
- **Test case:** `test_both_pools_grow_cascade` (P1)
  - Setup: Create mock scheduler with both req_to_token_pool and future_map
  - Simulate mamba_pool.size increase
  - Call _maybe_update_admission_cap()
  - Assertion: pool.grow() and future_map.grow() both called with same new_cap
  - Captures: Cascading grows (lines 318–330 in agent.py)

**Gap 4.3: Very Tight Mamba Pool → Zero Admittable Reqs**
- Scenario: Mamba pool shrinks dramatically; mamba_per_req_ratio=3 means 1 mamba slot → 0 reqs (floor division). What happens?
- **Test case:** `test_mamba_shrink_below_one_req` (P2)
  - Setup: mamba_size=1, ratio=3 → new_cap = 1//3 = 0
  - Design says "never drop below 1" (line 314 in agent.py: `new_cap = max(1, new_cap)`)
  - Call _maybe_update_admission_cap()
  - Assertion: Pool doesn't shrink below size=1; admission stays at min 1
  - Captures: Floor clause (not tested explicitly in Phase 5)

---

## 5. CUDA-Graph Capture Validation

### Current Test Coverage

- **Phase 1–4**: Pure Python unit tests; no CUDA graph involved
- **Phase 5**: Mock scheduler, no real graph capture
- **Zero tests** that capture a graph, grow the pool, and replay

### Gaps

**Gap 5.1: Graph Capture Pre-Grow, Replay Post-Grow**
- Scenario: This is the **core claim of audit_cuda_graphs.md Option B**. Capture a Triton kernel that reads req_to_token_ptr (baked in). Grow the pool. Replay the graph. Verify it reads from newly-mapped rows.
- **Test case:** `test_cuda_graph_grows_va_stable` (P0 — **CRITICAL**)
  - Setup: Pool(size=2, max_size=4), write pattern A to rows 0–1
  - Create a simple Triton kernel that reads from req_to_token at indices provided
  - Capture CUDA graph calling that kernel with indices=[0,1]
  - Grow pool to 4, write pattern B to rows 2–3
  - Replay the graph (indices still [0,1])
  - Assertion: Graph reads pattern A (unchanged rows); no fault or stale read
  - **Why critical:** If this fails, the whole design (VA-stable wrapping) breaks. We claim graphs stay valid post-grow, but haven't proven it.
  - **Implementation note:** Mock/simplified version acceptable for Phase 5 (can't easily capture real Triton kernel without full sglang model). Even a torch.cuda.graph() with simple CUDA operations would validate the VA property.

**Gap 5.2: Multi-Capture Session: Pre-Capture Pool Size Changes**
- Scenario: Boot with size=33. Admission admits reqs. Grow pool to 50. New reqs admitted. Another capture round happens at size=50. Verify captured graphs work for both old and new batch sizes.
- **Test case:** `test_graph_capture_across_pool_resizes` (P1)
  - Setup: Simulate capture at size=33, then grow to 50, then capture again
  - (Defer to integration test; no pure unit test for this)
  - Captures: Multi-epoch capture safety

---

## 6. End-to-End Pre-D8 Sanity

### Current Test Coverage

- **Phase 5**: All 6 tests use mocks; no environment variables
- **No tests** for SGLANG_ADMISSION_MAX_FACTOR env var path

### Gaps

**Gap 6.1: Env Var SGLANG_ADMISSION_MAX_FACTOR**
- Scenario: User sets SGLANG_ADMISSION_MAX_FACTOR=4 at startup. Pool should init with max_size = 4 × init_size.
- **Test case:** `test_env_var_max_factor_respected` (P2)
  - Set env SGLANG_ADMISSION_MAX_FACTOR=4
  - Construct pool with size=25
  - Assertion: pool.max_size == 100 (or as computed per server_args logic)
  - Captures: Environment variable path (not tested; hardcoded in Phase 5)

**Gap 6.2: Budgeter Tick Actually Calls Grow on Real Scheduler**
- Scenario: Phase 5 mocks the scheduler; real integration test with actual LoRA/Budgeter/Scheduler trio.
- **Test case:** `test_budgeter_tick_calls_grow_real` (P1 for D8)
  - Setup: Real Budgeter instance with real scheduler (mock pools is OK)
  - Simulate mamba_pool.size change via allocator
  - Call budgeter.tick() once
  - Assertion: pool.grow() was invoked (can check via mock pool.grow_calls)
  - Captures: Real Budgeter integration (not just Phase 5 mock)

**Gap 6.3: Max Running Requests Updated Synchronously**
- Scenario: After grow, scheduler.max_running_requests updated (line 331 in agent.py). Verify admission layer sees new value on next alloc.
- **Test case:** `test_max_running_requests_scalar_updated` (P1)
  - Setup: Pool with live scheduler
  - Call pool.grow(new_size)
  - Assertion: sched.max_running_requests == new_size
  - Captures: Scalar coupling (line 331)

---

## 7. Summary: Test Additions Before D8 Re-Run

### P0 — CRITICAL (blocks D8 validity claim)

| Test | Why | File |
|------|-----|------|
| `test_cuda_graph_grows_va_stable` | Proves VA-stable wrapping works under graph capture/replay — the **entire design claim**. Without this, we don't know if captured graphs remain valid post-grow. | `test_phase1_cuda_graph.py` (new) |

### P1 — MUST ADD (before D8 rerun, or D8 results are not valid for production)

| Test | Why | File |
|------|-----|------|
| `test_grow_during_alloc_interleave` | Race: alloc slicing free_slots while grow extends it. Could corrupt admission logic. | `test_phase2_concurrency.py` (new) |
| `test_grow_to_max_size_boundary` | Boundary: grow to exactly max_size; next grow should reject. Validates line 229 bounds check. | `test_phase2.py` (add) |
| `test_shrink_to_zero_boundary` | Boundary: can pool shrink to 1? Is 1 the minimum? | `test_phase2.py` (add) |
| `test_sequential_grows` | Grows 50→70→100 with idempotent check. Validates line 227 no-op logic. | `test_phase2.py` (add) |
| `test_grow_shrink_sustained_load` | Integrates: hold slots, shrink defers, free, shrink succeeds. Real admission pressure. | `test_phase5.py` (add) |
| `test_both_pools_grow_cascade` | Integrates: grow both pool and future_map in same tick. Validates cascade (lines 318–330). | `test_phase5.py` (add) |
| `test_mamba_shrink_below_one_req` | Edge: floor(1 / 3) = 0, but design says min=1. Validates line 314. | `test_phase5.py` (add) |
| `test_budgeter_tick_calls_grow_real` | Integration: real Budgeter (not mock) invokes pool.grow. | `test_phase5_integration.py` (new) |
| `test_max_running_requests_scalar_updated` | Integration: sched.max_running_requests updated synchronously. | `test_phase5_integration.py` (new) |

### P2 — STRONGLY RECOMMENDED (add before Phase 6, after D8 preliminary)

| Test | Why | File |
|------|-----|------|
| `test_grow_while_backward_active` | Race: grow during kernel execution. Validates "safe between batches" claim. | `test_phase2_concurrency.py` (new) |
| `test_shrink_preserve_kept_range_multipass` | Grows 4→6→4→6; verifies data is preserved each cycle. | `test_phase2.py` (add) |
| `test_grow_sub_chunk` | Boundary: grow by odd rows when chunk boundary is even. Validates rounding (line 91). | `test_phase1.py` (add) |
| `test_grow_cumemmmap_oom` | Failure: mock arena.grow() return partial; pool should degrade gracefully. | `test_phase2_failures.py` (new) |
| `test_pool_init_va_reserve_fail` | Failure: VA reserve fails at init; should NOT silently fall back. | `test_phase2_failures.py` (new) |
| `test_grow_partial_failure` | Failure: grow succeeds partway then zero_() fails; pool must revert. | `test_phase2_failures.py` (new) |
| `test_env_var_max_factor_respected` | Sanity: SGLANG_ADMISSION_MAX_FACTOR=4 sets max_size correctly. | `test_phase5_sanity.py` (new) |

### After D8 (acceptance gate)

- `test_graph_capture_across_pool_resizes` — Multi-epoch capture safety (integration test; defer to post-D8 if D8 itself exercises this)

---

## 8. Concrete Example Test Implementations

### Example P0: `test_cuda_graph_grows_va_stable` (pseudocode)

```python
def test_cuda_graph_grows_va_stable():
    """Capture graph, grow pool, replay, verify no fault."""
    pool = ReqToTokenPool(size=2, max_size=4, max_context_len=1024, device="cuda:0")
    
    # Write pattern A to rows 0–1
    pattern_a = torch.full((2, 1024), 111, dtype=torch.int32, device="cuda:0")
    pool.req_to_token[:2] = pattern_a
    torch.cuda.synchronize()
    
    # Capture graph: simple kernel that reads from pool.req_to_token[batch_ids]
    stream = torch.cuda.Stream()
    with torch.cuda.graph(stream) as g:
        # Pseudo-kernel: copy req_to_token[0:2] to output
        output = pool.req_to_token[:2].clone()
    stream.synchronize()
    
    # Grow pool to 4, write pattern B to rows 2–3
    pool.grow(4)
    pattern_b = torch.full((2, 1024), 222, dtype=torch.int32, device="cuda:0")
    pool.req_to_token[2:4] = pattern_b
    torch.cuda.synchronize()
    
    # Replay graph (should still read from rows 0–1, which have pattern_a)
    g.replay()
    output = pool.req_to_token[:2].cpu()
    assert torch.equal(output, pattern_a.cpu()), \
        "graph replay must read from same VA rows despite pool.grow"
    print("  PASS  cuda-graph-grows-va-stable")
```

### Example P1: `test_grow_to_max_size_boundary` (in test_phase2.py)

```python
def test_grow_to_max_size_boundary():
    """Grow to exactly max_size; next grow should reject."""
    p = ReqToTokenPool(
        size=10, max_context_len=MAX_CONTEXT_LEN,
        device=DEVICE, enable_memory_saver=False,
        max_size=100,
    )
    # Grow to max
    new_size = p.grow(100)
    assert new_size == 100
    assert p.size == 100
    
    # Next grow above max should raise
    try:
        p.grow(101)
        raise AssertionError("grow(101) should have raised ValueError")
    except ValueError as e:
        assert "max_size" in str(e)
    
    # grow(100) again should be no-op
    new_size = p.grow(100)
    assert new_size == 100
    print("  PASS  7  grow to max_size boundary + rejection on overshoot")
```

---

## 9. Risk Assessment

**High Risk if Tests Not Added:**

1. **P0 (`test_cuda_graph_grows_va_stable`)**: If this fails, the VA-stable design is broken. D8 would incorrectly increase throughput via admission, but captured graphs would read stale memory or fault under real kernel execution.

2. **P1 (`test_grow_during_alloc_interleave`, `test_budgeter_tick_calls_grow_real`)**: Race conditions in admission path. Under sustained load, could corrupt req slot allocation or cause deadlocks.

3. **P1 (`test_grow_to_max_size_boundary`, boundary tests)**: Edge cases in shrink/grow logic. Could allow pool to grow beyond VA reservation, causing silent memory corruption.

**Medium Risk:**

4. Failure-mode tests (P2): OOM/VA exhaustion during grow. Production systems may hit these; graceful degradation is required.

---

## 10. Recommended Test Execution Order

1. **Now (before D8 re-run):** Add all P0 + P1 tests. Run locally to validate.
2. **D8 Re-Run:** With new tests passing, re-run D8. Target: Δ >= +10% throughput.
3. **Phase 6 Validation:** If D8 passes, add P2 tests + integration tests.
4. **Regression:** D7 / D8b must still pass (no behavior change at small caps).

