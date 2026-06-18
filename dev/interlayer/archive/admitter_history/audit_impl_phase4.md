# Implementation audit — Phase 4 (sync fire path)

Code-review audit of the production implementation (separate from test
depth, covered in `audit_test_depth_phase4.md`). Subagent reviewed
`admitter.py::execute_decision`, `xpool_actuator.py` lock placement,
the `SharedHandlePool` thread-safety contract, and adjacent allocator
ordering.

## Severity summary

| Concern | Severity | Status |
|---|---|---|
| 1. SharedHandlePool grow() racing fires | HIGH (latent) | ✅ Fixed via `freeze()` + RuntimeError |
| 2. Lock-order deadlock between `_fire_inflight` ↔ `_alloc_lock` | OK | (proven by audit; no inversion) |
| 3. execute → alloc gap atomicity | MEDIUM | ✅ Documented as caller-serial contract |
| 4. LCM rounding correctness | OK (one micro) | (no fix needed) |
| 5. `reserved_token_ids` silent leak if Phase 5 forgets | HIGH | ✅ Fixed via consume/release + GC ERROR log |
| 6. Direction-string strictness on unknown pool name | LOW | ✅ Fixed — try/except → graceful defer |
| 7. Reservation underflow silently changes cap | MEDIUM | ✅ Fixed — `logger.warning` on underflow |
| 8. Read of `n_kv_subpools` / `n_mamba_subpools` lock-free | OK | (immutable post-init) |
| 9. `object.__new__(XPoolActuator)` in test_9 | LOW | (acceptable; no `__slots__`) |
| 10. Misc (logs, TOCTOU, GIL, monotonic_ns) | LOW | (no fix needed) |

## Fixes landed

### Concern 1 — SharedHandlePool freeze

`chunk_arena.py::SharedHandlePool`:
- Added `_frozen: bool` flag + `freeze()` method.
- `grow()` raises `RuntimeError` if called after `freeze()`.
- Docstring expanded to spell out the lock-ownership contract.

`xpool_actuator.py::XPoolActuator.__init__`: calls `self.shared.freeze()`
immediately after `self._fire_inflight = threading.Lock()`, so any future
call to `SharedHandlePool.grow()` post-init fails loudly. Today no
production path makes such a call — but a future "dynamic shared pool
growth" feature would race the actuator fires without this guard.

Audit recommended moving the lock onto `SharedHandlePool` itself.
**Rejected** because `_fire_inflight` must serialize the WHOLE fire
body (shrink_explicit + grow + cap_bump), not just pool ops. The body
spans both arenas' interactions with the shared pool, so the lock
properly lives on `XPoolActuator`. `freeze()` is the right minimal
guard against the latent risk.

Test: `test_17_freeze_blocks_post_init_grow`.

### Concern 5 — `reserved_token_ids` lifecycle

`AdmitterDecision`:
- New `_reservation_consumed: bool` field.
- `consume_reservation() -> Any`: returns token-ids, flips the flag.
  Idempotent (second call returns None).
- `release(dst_allocator)`: calls `dst_allocator.free(...)` and flips
  the flag. Useful when the request is retracted between admission and
  scheduling.
- `__del__`: logs ERROR if reserved_token_ids set AND consumed flag is
  False. Names the leaked token count for triage.

This converts silent leaks (Phase 5 forgets to call) into noisy ERRORs
visible in production logs. The slots STILL leak — they're off the
allocator's free list — but the loud log identifies the bug fast.

Tests: 13 (consume silences log), 14 (no-consume triggers log),
15 (release calls allocator.free and flips flag).

### Concern 3 — execute→alloc gap

Documented as a contract in `execute_decision` docstring:

> Caller MUST serialize `execute_decision` calls per scheduler instance.
> Between `actuator.execute()` returning and `dst_allocator.alloc()`
> running, `_fire_inflight` is released — another concurrent Admitter
> call would race the reservation. In NULL-disagg sglang today this is
> guaranteed by `_add_request_to_queue` being serial on a single
> scheduler thread. Phase 5's hook relies on this.

No code change needed; the invariant is real and the doc preserves it.

### Concern 6 — Direction strictness

`execute_decision` wraps `self.planner.build(direction, ...)` in
`try/except ValueError`. On exception, falls back to `defer` with a
reason citing the rejected direction string.

Today's pool labels are `'kv'` and `'mamba'`; the planner only accepts
`'kv_to_mamba'` and `'mamba_to_kv'`. When SWA or other pools land, a
typo or new label won't crash the scheduler.

Test: `test_16_unknown_direction_falls_back_to_defer`.

### Concern 7 — Reservation underflow

When `dst_allocator.alloc(x_tokens)` returns None after a successful
fire (e.g. planner+actuator granted fewer pages than the Admitter
budgeted), the decision falls back to `defer` AND emits
`logger.warning(...)` naming x_tokens, granted_pages, and the
mapped-but-unowned slot count. Ops teams should grep for this; it
indicates either:
- The actuator's stricter LCM clamp in `_execute_async_locked`
  reduced the grant (legitimate)
- A serial-scheduler invariant was violated (Concern 3 race; bug)

## Tests added

| # | Concern | Asserts |
|---|---|---|
| 13 | 5 | `consume_reservation()` flips flag → no leak log on `__del__` |
| 14 | 5 | No consume → ERROR-level "leaked" log on `__del__` with token count |
| 15 | 5 | `release(alloc)` calls `alloc.free(ids)` once + flips flag |
| 16 | 6 | Bad direction → graceful defer, no actuator call |
| 17 | 1 | `SharedHandlePool.grow()` raises after `freeze()` |

## Final Phase 4 status

**17/17 tests PASS.** Zero regression across Phase 1/2/3 + dyn_admission_cap.

| Phase | Tests | Result |
|---|---|---|
| Phase 1 (CostModel facade) | 6 | PASS |
| Phase 2 (no-cross admitter) | 10 | PASS |
| Phase 3 (EvictCostIndex) | 12 | PASS |
| **Phase 4 (sync fire, post-audit)** | **17** | **PASS** |
| Total Admitter | **45** | **PASS** |
| dyn_admission_cap owner_map | 6 | PASS |
| dyn_admission_cap mark_no_realloc | 4 | PASS |
| dyn_admission_cap balanced_atomic | 7 | PASS |

## Audit verdict

> "Ship after fixes."

All 2 HIGH and 1 MEDIUM issues addressed. LOW issues either addressed
(Concerns 6) or accepted (Concerns 4, 8, 9, 10 require no change).
Phase 5 unblocked. The reservation-lifecycle hand-off (consume vs
release) is now an explicit contract Phase 5 must follow.
