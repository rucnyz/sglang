# Test depth audit — Phase 4 (sync fire path)

Subagent audit of `test_sync_fire.py` (initial 8 tests). Identified 7
gaps; landed fixes for the CRITICAL gap (Gap 1) and 3 of the
IMPORTANT/NICE-TO-HAVE ones (Gaps 3, 4, 7). Two remaining
(Gap 2, Gap 5/6) are deferred — see "Deferred" below.

## Gaps identified

### Gap 1 — CRITICAL: The "real lock" contract was not exercised

The original `test_3_concurrent_serialization_with_fake_worker` uses
`FakeActuator.execute()`, which has its own lock around the body. It
proves the Admitter's orchestration is safe given any actuator that
holds a lock during execute(). But it does NOT prove that the REAL
`XPoolActuator._fire_inflight` is wired up correctly — that's a
separate claim ("the real actuator holds the lock during the body of
`_execute_async_locked`").

**Risk:** if a future refactor of `xpool_actuator.py` releases the lock
before the SharedHandlePool ops finish — or fails to acquire it at all
in a renamed method — the FakeActuator-based test would not catch it.
The fix in #4 would silently stop working at runtime.

**Fix landed: `test_9_real_xpool_actuator_lock_contract`** —
constructs a real `XPoolActuator` object via `object.__new__`
(bypasses MultiTensorArena requirement), monkey-patches
`_execute_async_locked` to record in-flight count + sleep 5 ms, then
spins 4 threads × 10 calls each. Asserts `max_in_flight == 1` over
all 40 invocations. Proves the real `with self._fire_inflight:`
serializes correctly.

### Gap 2 — IMPORTANT: execute→alloc gap atomicity

Between `actuator.execute()` returning and `dst_allocator.alloc()`
being called, `_fire_inflight` is released. If a different scheduler
thread or worker action drains the freshly-bumped dst capacity before
the Admitter's reservation `alloc()` runs, the triggering arrival
loses its reservation.

**Safety argument by construction:** NULL-disagg sglang has ONE
scheduler thread, and the worker only does dst_cap bumps (never
allocates). So the race is impossible in the current architecture.
The implementation also falls back to `defer` if reservation alloc
returns None (now tested as Gap 7 fix below) — graceful failure.

**Deferred to Phase 5.** A scheduler-integration test (Phase 5 work)
will pin this end-to-end. A pure-Admitter unit test for it would have
to fake too much of the scheduler structure to be informative.

### Gap 3 — IMPORTANT: `planner.build()` returning None untested

Lines 280-285 of `admitter.py::execute_decision` handle the case
where the planner can't build a feasible plan (insufficient free
pages, etc.) by falling back to `defer`. Zero coverage.

**Risk:** a refactor that drops this branch would crash on
`result = self.actuator.execute(None)` instead of degrading gracefully.

**Fix landed: `test_10_planner_returns_none_falls_back_to_defer`** —
sets `FakePlanner.force_return = None`, asserts decision becomes
`defer` with no actuator/allocator calls and reason mentions planner.

### Gap 4 — IMPORTANT: `execute_decision` should NOT re-gate warm-up

Phase 2's cold-start protocol is implemented in `decide()`. Phase 4's
`execute_decision()` deliberately does NOT re-check `is_warmed_up()`;
it trusts decide()'s action label. This is the right contract (single
source of truth) but never pinned. A future "extra safety" refactor
that adds warm-up re-checking would short-circuit valid
post-decision fires.

**Fix landed: `test_11_execute_decision_trusts_action_label`** —
freshly reset cost model (cold), hand in `cross_free` label, assert
fire still proceeds. Pairs with Phase 2 tests confirming `decide()`
itself wouldn't produce `cross_free` cold.

### Gap 5 — NICE-TO-HAVE: LCM edge cases

Missing: x_tokens=0, already-aligned x_tokens, n_src=n_dst=1 (lcm=1
degenerate), very large x_tokens. The math is simple ceil-divide; the
risk is low.

**Deferred.** Phase 2 already covers tokens_per_page boundary
(test_9). The interesting edge for production is `x_tokens > lcm
* tokens_per_page` which is the common case and already covered by
test_2.

### Gap 6 — NICE-TO-HAVE: Realistic-latency P99

Test 7 measures 0-latency orchestration overhead (33 µs P99). A
realistic test with actuator latency_s=0.002 + contending worker
would verify the spec target of P99 ≤ 5 ms wall.

**Deferred.** The spec target ≤ 5 ms is for the REAL XPoolActuator
on real hardware — it's covered by Phase 6 D6 live workload tests,
not unit tests.

### Gap 7 — NICE-TO-HAVE: Reservation underflow

Lines 302-310 handle `dst_allocator.alloc(x_tokens) == None` by
falling back to defer. Untested.

**Fix landed: `test_12_reservation_underflow_falls_back_to_defer`** —
`FakeAllocator(available=100)` with x_tokens=2048 → asserts
action='defer', no `reserved_token_ids`, reason mentions
reservation/alloc.

## Phase 4 final status

**12/12 tests PASS.**

| Category | Tests | Result |
|---|---|---|
| Cross-* fire + reservation happy path | 1, 4, 8 | PASS |
| LCM page rounding | 2 | PASS (2 pages → 12 pages w/ lcm=12) |
| Concurrent serialization (fake) | 3 | PASS (max in-flight = 1) |
| **Real `_fire_inflight` contract** | **9** | **PASS (40 calls, max in-flight = 1)** |
| Abort / fallback to defer | 5 | PASS |
| Planner.None fallback | 10 | PASS |
| Own/defer no-op | 6 | PASS |
| Orchestration P99 < 1 ms | 7 | PASS (33 µs) |
| Decision-label trust (no re-gate) | 11 | PASS |
| Reservation underflow fallback | 12 | PASS |

## Subagent verdict

> **Soft go for Phase 5** after landing CRITICAL Gap 1.

Gap 1 + 3 + 4 + 7 fixes landed. Gaps 2, 5, 6 deferred per rationale
above. Phase 5 (scheduler hook) unblocked.

## Action taken

- Added `test_9_real_xpool_actuator_lock_contract` (Gap 1 CRITICAL).
- Added `test_10_planner_returns_none_falls_back_to_defer` (Gap 3).
- Added `test_11_execute_decision_trusts_action_label` (Gap 4).
- Added `test_12_reservation_underflow_falls_back_to_defer` (Gap 7).
- Total: 8 initial tests → 12 final.
