# T30 + T39 — proxy-gate disconnect awareness (#183, DESIGN §10 F1)

DESIGN §10 invariant: *a request held in the proxy gate awaits BOTH
the gate condition AND `request.is_disconnected()`; whichever fires
first wins.  TCP disconnect deterministically signals the client
gave up — no timer, no fallback.*  On disconnect the proxy releases
the gated request locally (499), the daemon's
`program_tracker.client_disconnected(p)` transitions p to ENDED,
the proxy enqueues `PUT /aginfer/program_paused {ENDED}`, and p's
residence is reaped at the next state-dump.

## REUSES #185

The state-machine + outbound pieces already existed from T41 (#185):
`State.ENDED`, the `wait_if_paused` 499 verdict, and
`enqueue_program_paused`.  This task adds:

* `ProgramTracker.client_disconnected(pid)` — thin wrapper over
  `end()` with a distinct metric tag (so ops can tell disconnect-
  driven ENDs from harbor SESSION_END).  Returns the prior state.
* `proxy._gate_or_disconnect(wait_awaitable, disconnect_awaitable)`
  — races the gate-wait against client disconnect; returns
  `"proceed"` / `"ended"` / `"disconnect"`.  Pure w.r.t. its two
  awaitables (verify drives it with stubs).  Cancels the loser.
* `proxy._until_disconnected(raw)` — resolves when the client TCP
  drops.  Polls `is_disconnected()` at `_DISCONNECT_POLL_S` (0.1s).
  This is per-request disconnect DETECTION in the proxy coroutine,
  NOT policy polling (the cross-cutting "no polling in policy/
  scheduler/admission/event_worker" invariant does not cover the
  proxy's per-request disconnect race — Starlette exposes no pure
  await-until-disconnect).
* `chat_completions` gate: the disconnect race ONLY runs when the
  program is genuinely `PAUSED` (i.e. the request would actually
  park).  On the common non-gated path the handler takes a plain
  `await wait_if_paused(pid)` verdict-only fast path and NEVER calls
  `is_disconnected()`.  This is load-bearing: `_until_disconnected`
  polls Starlette's receive channel, and spinning it up on every
  request interferes with the normal request lifecycle (the #183
  audit-fix regression — T4's real-uvicorn requests hit
  `httpx.ReadTimeout` until the race was gated behind the PAUSED
  check).  When PAUSED: races the two; on `disconnect` →
  `client_disconnected(pid)` + `enqueue_program_paused(ENDED)` +
  499; on `ended` (F5) → 499; on `proceed` → forward.

## STAGES (14)

```
A. _gate_or_disconnect race (the F1 core, stub-driven)
  A0 gate True  → "proceed"
  A1 gate False → "ended" (F5 SESSION_END while gated)
  A2 disconnect resolves first → "disconnect"
  A3 the losing task is cancelled (no leak / no "exception never
     retrieved")
  A4 non-gated fast path: gate True wins < 50ms (disconnect poll
     never returns for a connected client)
B. ProgramTracker.client_disconnected
  B0 client_disconnected(PAUSED) → ENDED, prev=PAUSED
  B1 client_disconnected(release_gate=False) → no gate-release,
     no `_ended_while_gated` flag leak (the per-connection-vs-
     per-program #183 fix: a disconnecting connection must not
     poison a sibling parked on the same pid)
  B2 client_disconnected(unknown) → ENDED, prev=None
  B3 emits the distinct 'client_disconnected' metric
C. Proxy integration (real create_app proxy)
  C0 disconnected request → 499 + ENDED + PUT {ENDED} enqueued
  C1 ended-while-gated (F5) → 499, NO extra PUT from the proxy
     (the SESSION_END handler owns that PUT)
  C2 non-gated request → forwards (not 499), no disconnect PUT
  C3 MID-PARK disconnect (request connects, parks PAUSED, THEN the
     client drops) → 499 + ENDED + PUT {ENDED}
  C4 sibling NOT 499'd on per-connection disconnect: two requests
     parked on one pid, one drops → the dropping one 499s, the
     live sibling proceeds, no program-end (#183 audit bug)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t30/verify.py
```

Pure-Python (asyncio); ~0.5 s.  No GPU.  C0-C4 build the real
`create_app` proxy and call the chat handler with a fake Request
(controllable `is_disconnected()` + JSON body).

## RESULTS

**PASSED** — all 14 stages.

* date: 2026-06-02
* raw log: `results/20260602_t30_initial_pass.log`

## REGRESSION SANITY

* T4 daemon_proxy_events: PASS — the disconnect race is gated
  behind the PAUSED check, so non-gated requests keep the original
  fast path (an earlier always-race version timed out T4's real-
  uvicorn requests; see the `_until_disconnected` note above)
* T6 program_tracker: PASS (client_disconnected + release_gate
  bookkeeping additive)
* T41 SESSION_END: PASS (the `ended` verdict path unchanged)
* T36 outbound: PASS (enqueue_program_paused / method dispatch)

## DESIGN-vs-CODE NOTE

DESIGN §10 says the daemon's `client_disconnected(p)` enqueues the
PUT.  As-built, the **proxy** enqueues it (the proxy owns both the
Request object that detects disconnect AND `app.state.outbound`);
`tracker.client_disconnected` does only the state transition.  Same
net effect — ENDED + PUT {ENDED} on disconnect — just a cleaner
split of responsibilities (tracker = state, proxy = I/O).
