# T5 — sglang→daemon webhook + daemon event router

## WHAT WE PROMISED

**Capability — sglang side**
* At the end of every scheduler step, compute HBM occupancy state ∈
  `{OK, HIGH, CRITICAL}` from `token_to_kv_pool_allocator`.
* Fire `POST <notify_url>/aginfer/event` (fire-and-forget) when EITHER:
  - `state != last_state` (transition); `kind = "memory_pressure"` for
    OK→HIGH/CRITICAL or HIGH→CRITICAL, `kind = "pressure_resolved"`
    for the reverse direction; OR
  - `state ∈ {HIGH, CRITICAL}` AND it has been ≥ `aginfer_heartbeat_s`
    (default 5 s) since the last fire for this state — `kind =
    "still_high"`. This is the plateau heartbeat that lets admission
    re-pick pause victims as the workload mix shifts.
* Payload always includes `{state, occ, used_bytes, last_fire_kind}`.
* Retry up to 3 × on network error with exponential backoff (0.1 s, 0.4 s, 1.6 s).
* If `--aginfer-notify-url` is empty/unset, the detector is bypassed
  entirely (stock sglang behavior).
* If the HTTP send raises, the scheduler step is **never** blocked
  (try/except + `asyncio.create_task`).

**Capability — daemon side**
* `POST /aginfer/event` enqueues into a single `asyncio.Queue`.
* A single `event_worker` consumes the queue serially; no two event
  handlers run concurrently (`asyncio.Lock` around the dispatch
  body).
* Each handler calls `fetch_state()` at entry — never trusts the
  payload's snapshot.
* Handlers are **idempotent**: receiving the same `memory_pressure`
  event twice produces the same final state.
* At daemon startup, perform one `/aginfer/state` fetch and, if
  `HBM_occ > θ_hi`, synthesise a local `memory_pressure` event so
  cold-start doesn't sit oblivious.

**Cost ceiling**
* Daemon idle CPU (no traffic, no events for 60 s): < 1 % of one core.
* Event-handler latency, event arrival → `client.migrate` returns: < 80 ms p99.
* sglang webhook fire overhead per scheduler step: < 50 μs (the state-
  check is a cheap int compare; POST only fires on transitions, which
  are rare).
* No new periodic timer anywhere in the daemon. Verified by grep.

## HOW WE VERIFY

Mechanism. `verify/t5_event_router.py`:

```
1. sglang side
   1a. Launch sglang with --aginfer-notify-url=http://stub:9999/aginfer/event,
       pointing at a stub HTTP server that captures POST bodies.
   1b. Drive a workload that fluctuates HBM occupancy between OK and HIGH
       (cap=64K, several rounds of (insert distinct prefixes, evict half)).
   1c. Assert the stub received webhooks on:
       (a) state transitions ({OK→HIGH, HIGH→CRITICAL, CRITICAL→HIGH, HIGH→OK}); AND
       (b) `still_high` heartbeats while in {HIGH, CRITICAL}, one every
           ~5 s (= aginfer_heartbeat_s).
       During OK plateau: ZERO additional webhooks (no spam).
       During a 60-s HIGH plateau: exactly 12 ± 1 still_high webhooks
       (60/5).
   1d. With the stub returning 500 for the first 2 attempts on each call,
       verify retry-on-error works (final call succeeds; total ~2 s backoff).
   1e. With the stub down entirely, verify sglang's scheduler does NOT
       block, hang, or crash. Workload throughput unaffected.

2. daemon side
   2a. Launch daemon + stub-sglang that fires fake events.
   2b. Send 100 fake events of mixed kinds in quick succession.
       Assert: handler invocations happen serially (instrumented to
       check no two are in flight simultaneously).
   2c. Send the same `memory_pressure` event twice with identical
       payload. Assert: total migrate-POSTs sent to sglang is 1× the
       count for a single send (idempotent).
   2d. Kill daemon, restart with stub-sglang reporting HBM_occ=0.9.
       Assert daemon synthesises a `memory_pressure` on startup and
       calls migrate.
   2e. Idle daemon for 60 s, sample CPU via psutil. Assert mean < 1 %.
   2f. Grep daemon source: assert no `asyncio.sleep`, no `time.sleep`,
       no `loop.call_later` in the scheduler / policy / admission paths
       (event-driven only).

3. Latency micro-bench
   3a. Time arrival-to-migrate for 100 synthetic events at low load.
       Report p50, p99. Assert p99 < 80 ms.
```

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Webhook receiver completely down | Launch sglang with `--aginfer-notify-url=http://127.0.0.1:1` (port closed); run full harbor swebenchpro -n 32 | per-trial mean ≤ Run F' × 1.10 = 960 s (degrades to inline scorer only, the safety net) | log-grep `failed` count + harbor result |
| Daemon receives event but handler raises | Patch handler to raise on every 5th event; run 100 events | other 4/5 succeed; bad event logged + counted; queue does not stall | event log count |
| Burst of 1000 events in 100 ms | fastapi `/aginfer/event` slammed with 1000 concurrent POSTs | All eventually processed serially; no dropped; total drain time < 5 s | metric: events_handled == 1000 |
| Webhook for transition that never occurs (idle workload) | Boot sglang + daemon, send no traffic for 60 s | Daemon CPU < 1 %; no events fire | psutil sample |
| Daemon process killed mid-handler | `kill -9` daemon while handler running; restart | On restart, sees current sglang state via 1× fetch; any in-flight migrate either committed or dropped (no half-state). Run K trial-in-flight may error; subsequent trials proceed | docker container survival + harbor stats |
* date: _pending_
* sglang sha:
* daemon sha:
* webhook fired only on transitions: _pending_
* retry works on 500: _pending_
* sglang unaffected when stub down: _pending_
* handlers serialised: _pending_
* idempotent (dup event = 1× action): _pending_
* startup synthesises memory_pressure: _pending_
* idle CPU < 1 % over 60 s: _pending_
* no periodic timers in code (grep): _pending_
* event-to-migrate p99 latency: _pending_
* raw log: `verify/results/t5_<datetime>.log`
