# T10 — integration / concurrency / recovery / GC

## WHAT WE PROMISED

This is the cross-component stress + recovery suite. T1-T8 verify each
layer in isolation. T10 verifies they compose correctly under the
adversarial conditions a real long-running deployment will see.

**Capability**
* **Inline scorer ↔ daemon do not produce conflicting decisions** under
  steady load: the rate of "daemon migrated X to HBM, inline evicted
  X within 100 ms" is < 1 % of all migrate actions.
* **`/aginfer/state` is consistent under live mutation**: 50 concurrent
  pollers + 32 trials in flight, every response is internally
  consistent (no dup hashes, sum-of-unit-bytes ≈ tier_usage.used).
* **Daemon survives restart mid-Run-K**: kill daemon at minute 10 of a
  full harbor run, restart, observe that paused programs are correctly
  identified on resume (no permanent stall, no all-resume-storm).
* **program_tracker GC bounds memory**: 24 h synthetic load with 10 k
  unique program_ids → memory < 100 MB.

**Cost ceiling**
* No additional sglang code beyond T1-T5; T10 is purely test code.
* T10 suite runtime: < 3 h (24-h GC test runs accelerated).

## HOW WE VERIFY

E2E and stress combo. `verify/t10_integration.sh`:

```
=== Stress A: inline vs daemon coexistence ===
1. Launch full stack (sglang + daemon, all 3 layers).
2. Run harbor swebenchpro -n 8 for 5 minutes.
3. Instrument sglang to log every (hash, action) eviction with
   timestamp; instrument daemon to log every (hash, target_tier)
   migrate.
4. Post-process: for each daemon migrate(X, HBM) at time t,
   check if sglang evicted X within [t, t+100ms]. Count "conflict"
   events.
5. Assert: conflict_rate = conflicts / total_migrates < 1%.

=== Stress B: /aginfer/state under live load ===
1. Same stack. Start a background script: 5 threads each calling
   `GET /aginfer/state` at 10 Hz for 60 s.
2. While that runs, send 32 concurrent chat completions with varied
   prefixes (forces tree mutation).
3. Validate each of the ~3000 responses:
   - JSON parses
   - No `hash` appears twice
   - Σ unit.n_bytes for tier == X ≈ tier_usage[X].used_bytes (±2 pages)
   - p99 latency < 50 ms
4. Assert: validation_pass_rate = 100% (zero corrupt responses).

=== Stress C: daemon restart mid-Run-K ===
1. Start a full Run K (32 trials). At t=10 minutes:
   - Capture daemon's program_tracker.dump() via debug endpoint.
   - `kill -9` daemon. Wait 2 s.
   - Restart daemon (it does startup fetch_state + synthesise
     memory_pressure if needed).
2. Observe:
   - Programs that were PAUSED before kill: are they still effectively
     paused (their next request blocks) or do they resume immediately?
     Either is OK as long as the system reaches consistency within 30 s.
   - No harbor trial errors caused by the kill itself (well, the
     in-flight trial may error; subsequent ones must proceed).
3. Run K continues to completion.
4. Compare K-with-restart vs clean K from T9: per-trial mean within
   ± 10 %.
5. Assert: harbor exits cleanly; at most 1 extra errored trial
   beyond Run K baseline.

=== Stress D: program_tracker GC under churn ===
1. Synthetic load: 10 k unique program_ids, each issues 5 chat
   completions over 30 s, then goes silent.
2. Run for 24 h (or speed-up via mock-clock: simulate 24 h in 30 min
   by manipulating program_tracker's idle-timeout to be in 1 s units).
3. psutil sample daemon RSS at t=1h, 6h, 12h, 24h.
4. Assert: RSS < 100 MB at all sample points; program_tracker.size()
   stays < 200 (= idle programs GC'd).

=== Stress E: webhook flood + idempotency ===
1. Stub sglang fires 10 k memory_pressure events in 10 s (i.e. 1000/s),
   half of which are duplicates of the previous payload.
2. Assert:
   - Daemon event_queue does not exceed 10 k items at any point.
   - All events processed within 20 s total.
   - migrate POSTs to sglang are ≤ 10 k (one per event, idempotent).
   - No exception in handler.
```

## WORST CASE (forced; same as the rows above — they ARE the worst case)

T10 IS the worst-case test suite. The "predicted floor" per row:

| Stress | Predicted floor |
|---|---|
| A inline ↔ daemon | conflict rate < 1 % |
| B /aginfer/state under load | 100 % responses pass schema; p99 < 50 ms |
| C daemon restart | Run K mean within ± 10 % of clean Run K; ≤ 1 extra errored trial |
| D program_tracker GC | RSS < 100 MB at 24 h; tracker.size() < 200 |
| E webhook flood | All events handled within 20 s; no exception; ≤ 10 k migrate POSTs |

If any row exceeds the floor, that floor row's component owner reopens
their TODO.

## RESULTS
* date: _pending_
* A inline-daemon conflict rate: _pending_
* B /aginfer/state validation pass rate: _pending_
* C daemon restart Run K mean: _pending_
* D GC RSS @ 24 h: _pending_
* E flood handling time: _pending_
* raw log: `verify/results/t10_<datetime>.log`
