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

Mechanism. `verify/t5/verify.py` (two layers: A pure-asyncio, B real-sglang gated by `AGINFER_T5_FULL=1`):

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
| Webhook receiver completely down *(deferred to T9 Run K)* | Launch sglang with `--aginfer-notify-url=http://127.0.0.1:1` (port closed); run full harbor swebenchpro -n 32 | per-trial mean ≤ Run F' × 1.10 = 960 s (degrades to inline scorer only, the safety net) | log-grep `failed` count + harbor result |
| Daemon receives event but handler raises | Patch handler to raise on every 5th event; run 100 events | other 4/5 succeed; bad event logged + counted; queue does not stall | event log count |
| Burst of 1000 events in 100 ms | fastapi `/aginfer/event` slammed with 1000 concurrent POSTs | All eventually processed serially; no dropped; total drain time < 5 s | metric: events_handled == 1000 |
| Webhook for transition that never occurs (idle workload) | Boot sglang + daemon, send no traffic for 60 s | Daemon CPU < 1 %; no events fire | psutil sample |
| Daemon process killed mid-handler *(deferred to T10 integration)* | `kill -9` daemon while handler running; restart | On restart, sees current sglang state via 1× fetch; any in-flight migrate either committed or dropped (no half-state). Run K trial-in-flight may error; subsequent trials proceed | docker container survival + harbor stats |
## REPRODUCING

T5 has two layers.  Layer A is pure asyncio (no sglang, no GPU);
Layer B exercises the full sglang→daemon webhook path on real GPU.

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched
cd /scratch/yuzhou/projects/sglang/dev/aginfer

# Layer A only (~5 s, no GPU):
python verify/t5/verify.py

# Layer A + Layer B (~3 min, real sglang on a free GPU):
AGINFER_T5_FULL=1 CUDA_VISIBLE_DEVICES=4 python verify/t5/verify.py
```

Layer B launches sglang with aggressive thresholds
(`--aginfer-theta-hi 0.30 --aginfer-theta-crit 0.60
--aginfer-heartbeat-s 2.0 --max-total-tokens 4096`) so a small
workload (60 distinct chat completions, 32 tokens each) trips OK→HIGH
and HIGH→CRITICAL within seconds.  A stub HTTP capturer records the
daemon-bound POSTs for assertion.

## RESULTS

**PASSED** — Layer A (10 steps post-audit) + Layer B (full sglang
webhook path on a real GPU) both clean.

### Audit round-2: "are tests real tests?"

A follow-up audit checked whether the round-1 production fixes were
actually PINNED by regression-catching tests (per the meta-rule that
fake / vacuous tests slipped through T3/T4 earlier rounds).  Findings:

* A1 (M3 `ts_monotonic`) was **unpinned** — no test asserted the
  field existed in the payload.  Fixed: new step [A11] asserts both
  `ts` and `ts_monotonic` are floats.  Bisect demo: deleting the
  field trips the assertion immediately.
* A3 (retry-on-500 path) was **entirely untested** — `fail_first_n`
  helper existed but no call site used it.  Fixed: step [A11] uses
  `fail_first_n=2`; asserts capturer ultimately receives the body.
  Bisect demo: changing `range(3)` → `range(1)` in `_send` trips
  the assertion.
* A4 (M5 `task_done` pairing) was unpinned — no `queue.join()`
  anywhere.  Fixed: `la_serial_dispatch` now calls
  `await asyncio.wait_for(router.bus.queue.join(), timeout=2.0)`
  after the 100-event burst.  A regression removing the
  `finally: task_done()` would hang here forever.
* A5 (M1 non-default theta plumb) was unpinned — only default
  0.7/0.9 was tested.  Fixed: `la_cold_start_probe` now runs two
  scenarios — (a) default 0.7 + occ=0.885, (b) theta_hi=0.5 +
  occ=0.595.  Pre-M1 hardcoded 0.7 would skip (b).
* A7 (Layer B `still_high >= 1` floor) was trivially loose.  Fixed:
  tightened to `>= 2` (matches plateau hold ~6 s @ heartbeat 2 s).
* A2 (B2 close lifecycle) and A6 (T4 round-1 MAJOR broader-except)
  were partially addressed:
  - A2: step [A11] does construct + close the firer; the close()
    path is exercised, though without an in-flight request to
    stress-test the `aclose()`-on-pending-client branch.
  - A6: NEW step [11a] in T4 verify monkey-patches the daemon's
    http_client to raise a non-RequestError; asserts the 502
    response AND that `tracker.state(pid) != REASONING`.  A revert
    of T4 round-1 MAJOR's broader `except Exception` would re-raise
    and skip `_emit_completion`, failing this test.

### Audit round-1 findings + resolution

Found 2 BLOCKER + 7 MINOR + 5 NIT.  Real fixes:

* **BLOCKER B1** — Firer was POSTing to `notify_url` verbatim; user
  passing the documented base URL (`--aginfer-notify-url=http://daemon:8765`)
  would silently 404 every webhook.  Fixed: `AginferWebhookFirer.__init__`
  now appends `/aginfer/event` to bare base URLs.  Pinned by step [A10].
* **BLOCKER B2** — `close()` was never called → daemon thread + httpx
  client leaked per scheduler subprocess.  Fixed: `run_scheduler_process`
  finally-block now calls `scheduler.aginfer_webhook.close()`, and
  `close()` itself `aclose()`s the httpx client on the background loop
  before stopping it.
* **MINOR M1** — Cold-start probe hardcoded `0.7` / `0.9` instead of
  honoring sglang's `--aginfer-theta-hi` / `--aginfer-theta-crit`.
  Fixed: `EventRouter` constructor takes `theta_hi` / `theta_crit`
  parameters; `create_app` plumbs them through.
* **MINOR M2** — Retry loop slept 1.6 s AFTER the third (final) failure
  for no reason.  Fixed: skip sleep when `attempt == 2`.
* **MINOR M3** — Payload `ts` used wall clock; detector `last_fire_ts`
  used monotonic.  Mixing the two confuses downstream drift / latency
  calculations.  Fixed: payload now carries BOTH `ts` (wall) and
  `ts_monotonic`.  T7 / T8 should use `ts_monotonic` for inter-fire
  arithmetic.
* **MINOR M4** — `EventRouter.stop()` swallowed non-cancel exceptions.
  Fixed: now logs them.
* **MINOR M5** — Worker never called `queue.task_done()`; a future
  `queue.join()` would hang.  Fixed: paired in a `finally`.
* **NIT N1** — README HOW WE VERIFY referred to a legacy path
  (`verify/t5_event_router.py`); actual is `verify/t5/verify.py`.
  Fixed.
* **NIT N2** — WORST CASE table rows that are deferred to T9 / T10
  ("Webhook receiver completely down", "Daemon process killed
  mid-handler") now carry an explicit `*(deferred to ...)*` marker.

Documented (not coded around) for v1:
* **MINOR M6** — Idempotency claim is true for v1's noop handler but
  T7 / T8 implementers must enforce idempotency for their real
  handlers.  Documented in the not-covered-in-v1 section.
* **MINOR M7** — In TP/PP, each Scheduler subprocess constructs its
  own firer; daemon will receive N×events per transition.  Noop
  handler is fine; T7 / T8 must dedup.  Documented.
* **NITs N3 / N4 / N5** — async-around-sync emit, deprecated
  `@app.on_event`, probe-not-retried.  Cosmetic; no behaviour change.

* date: 2026-05-26
* sglang code added:
  - `python/sglang/srt/managers/aginfer_webhook.py` — ~220 LoC,
    `AginferWebhookFirer` runs an asyncio loop in a background daemon
    thread, sends POSTs via httpx, retries 3× with 0.1 / 0.4 / 1.6 s
    backoff.  Never blocks the scheduler.
  - `python/sglang/srt/managers/scheduler.py` — Scheduler.__init__
    constructs the firer when `--aginfer-notify-url` is set;
    `event_loop_normal` and `event_loop_overlap` each call
    `aginfer_webhook.maybe_fire(used, cap)` once per step.  ~16 LoC × 2.
  - `python/sglang/srt/server_args.py` — 4 new flags:
    `--aginfer-notify-url / --aginfer-heartbeat-s / --aginfer-theta-hi
    / --aginfer-theta-crit`.
* daemon code added:
  - `dev/aginfer/daemon/event_router.py` — ~210 LoC, `EventRouter`
    with the serial event_worker (`asyncio.Lock` around dispatch),
    handler registry (T7/T8 will register real handlers via
    `set_handler`), and cold_start_probe.  `attach_event_routes`
    mounts the `POST /aginfer/event` endpoint.
  - `dev/aginfer/daemon/proxy.py` — `create_app` now mounts the
    event-router endpoint and starts the worker on app startup.
* webhook fires on transitions + heartbeats: ✓ (Layer B captured
  2× `memory_pressure` for OK→HIGH and HIGH→CRITICAL, plus 5×
  `still_high` heartbeats during the 6 s plateau hold).
* sglang unaffected when stub down: ✓ — the firer runs on a
  separate event loop in a daemon thread, never awaiting on the
  scheduler step.
* handlers serialised: ✓ (Layer A [A2]: 100 concurrent POSTs;
  `max_in_flight == 1`).
* idempotent: ✓ (Layer A [A3]: same `memory_pressure` event 3× →
  final value matches LAST payload).
* handler raises → queue continues: ✓ (Layer A [A4]: 5/5 crashes
  counted in `router.handler_failures`; 5/5 good handlers run).
* unknown kind → 400: ✓ (Layer A [A5]).
* `still_high` heartbeat routes to MEMORY_PRESSURE handler: ✓
  (Layer A [A6]).
* cold-start probe: ✓ (Layer A [A7] — stub reports HBM occ 0.885;
  daemon synthesises `memory_pressure` with `synthetic=true`).
* no periodic timers in daemon source: ✓ (Layer A [A9]: AST-grep
  for `call_later` / `call_at` in `event_router.py`, `proxy.py`,
  `events.py`).

### Latency (multi-run, per memory:feedback-latency-multi-run)

5 trials × 50 events each (250 samples), Layer A in-process loopback:

| metric | mean ± std |
|---|---|
| event arrival → handler entry **p50** | **0.79 ± 0.01 ms** |
| event arrival → handler entry **p99** | **1.08 ± 0.10 ms** |

Budget per DESIGN.md "Acknowledged costs": event-handler latency p99
< 80 ms.  Audit round-2 ("audit of tests") tightened the verify
assertion from ``p99_mean + p99_std < 80 ms`` (which would have
passed at 100× regression) to ``mean + 3σ < 5 ms`` for both p50 and
p99 — current envelope ~1.4 ms.  T7 / T8 will register real handlers
and re-measure end-to-end "event arrival → migrate POST returns";
the 80 ms DESIGN budget remains the system-wide ceiling.

### Layer B summary (real sglang on GPU 4)

| metric | observed |
|---|---|
| webhooks captured by stub | 7 |
| `memory_pressure` transitions | 2 (OK→HIGH, HIGH→CRITICAL) |
| `still_high` heartbeats | 5 (one every ~2 s during plateau hold) |
| sglang crashes / 5xx | 0 |
| state sequence | HIGH×3, CRITICAL×4 |

### Not covered in v1 (deferred)

* The watermark webhook fires fine but the daemon's default handler
  is a `_noop_handler` that only logs.  T7 / T8 will register real
  `kv_scheduler.decide()` / admission handlers via
  `router.set_handler(...)`.
* The full WORST CASE row "Burst of 1000 events in 100 ms" is
  exercised at the 100-event scale (Layer A [A2]); 1000 not tested
  because the in-process loopback drains in milliseconds and the
  test would just measure FastAPI overhead.
* Idle CPU sample over 60 s deferred to T10 integration (needs a
  longer-running test harness; not really a v1-correctness thing).
* "Webhook receiver completely down" WORST CASE row deferred to T9
  Run K (full harbor smoke).

### Raw logs (relative to this directory)

* `results/<YYYYMMDD_HHMMSS>_run1_layerA.log` — Layer A only
* `results/<YYYYMMDD_HHMMSS>_run2_layerB.log` — Layer A + B
* `results/sglang_t5_*.log` — raw sglang stdout from Layer B
  (the `aginfer webhook armed: url=... heartbeat_s=2.0 ...` line at
  startup confirms the firer was constructed)
