# T41 — F5 SESSION_END-for-PAUSED handler (#185, DESIGN §11 F5)

When harbor signals `SESSION_END` for a program whose next request
is parked in the proxy gate (PAUSED state), the daemon must:

1. Transition the program to **ENDED**.
2. **Release the gate with HTTP 499** — the client closed the
   session, so the in-flight parked request is implicitly
   cancelled; 499 ("client closed request") makes the client's
   framework treat it as their own cancellation.
3. Enqueue **PUT /aginfer/program_paused {state: ENDED}** so
   sglang clears the program's `per_program_usage[pid].state` on
   the next state-dump (T21 #181 stores it; #186 GC drops the
   ENDED-no-units entry once KV is gone).

## MECHANISM (the 499 gate-release)

There was no "release a parked gated request with a status code"
primitive before this.  Added:

* `ProgramTracker.State.ENDED` — terminal state.
* `ProgramTracker.end(pid)` — transition to ENDED; returns the
  prior state.  If the program was PAUSED, marks
  `_ended_while_gated[pid]` and sets the gate event (releasing the
  parked `wait_if_paused`).
* `ProgramTracker.wait_if_paused(pid) -> bool` — now returns a
  verdict: `True` = proceed (forward), `False` = the program was
  ENDED while this request was **parked** in the gate → proxy
  responds 499.  The verdict is honoured ONLY for requests that
  actually BLOCKED (gate event clear on entry → suspended on
  `e.wait()`).  A request that did not block — un-paused fast path,
  or an arrival AFTER `end()` set the event — is a fresh request
  and proceeds.  The `_ended_while_gated` flag is NOT popped on
  read (so the whole parked cohort sees it — #185 audit fix for
  the two-waiter case) and is cleared by the next lifecycle event
  (`observe_arrival` = new session, `pause` = new gate cycle).
* `proxy.py` — after `wait_if_paused`, `if not should_proceed:
  return Response(status_code=499)`.
* `OutboundBatch.method` (new, default "POST") + `_post_one`
  dispatch — `program_paused` is a PUT.  POST path is byte-
  identical to pre-#185 (existing stubs unaffected).
* `OutboundQueue.enqueue_program_paused(pid, state, pre_pause_state)`.
* `EventKind.SESSION_END` + `make_session_end_handler(tracker,
  outbound)` + `attach_session_end_handler` (wired in main.py
  after kv_scheduler so this F5 handler owns SESSION_END).

## SCOPE BOUNDARY

This handler owns the F5 **state-transition + gate-release + PUT**.
The migrate D_t for SESSION_END (DESIGN §7 table:
`session_scoped_units` demote/drop for the ending program's
exclusive units) is the **kv_scheduler's** concern — its
`_build_decision_set` returns `[]` for SESSION_END today.  Wiring
that decision_set is the "SESSION_END normal path" (a sibling of
F5), tracked separately.  See PLAN §4 T41 status note.

## STAGES (17 — 14 initial + 3 from #185 audit)

```
A. ProgramTracker.end() + ENDED state
  A0 end(REASONING) → ENDED, returns REASONING
  A1 end(ACTING) → ENDED, returns ACTING
  A2 end(PAUSED) → ENDED, returns PAUSED
  A3 end(unknown pid) → ENDED, returns None
  A4 end() idempotent (2nd → ENDED, no churn)
B. Gate verdict (the F5 499 mechanism)
  B0 un-paused program → wait_if_paused returns True
  B1 PAUSED + end() → the parked waiter wakes, returns False (→499)
  B2 arrival AFTER end() (not parked) proceeds + resurrects to
     REASONING (corrected contract; was the old "read-once" stage)
  B3 end() on non-paused → does NOT set the 499 verdict
  B5 TWO parked waiters, same pid → BOTH 499 (#185 audit: the old
     read-once flag let the 2nd leak a request for an ended session)
C. Outbound PUT
  C0 enqueue_program_paused: {pid, state, pre_pause_state,
     batch_id} body, endpoint=program_paused, method=PUT
  C1 OutboundBatch rejects an invalid method (DELETE)
  C2 the EXACT PUT body passes sglang's _validate_program_paused_
     body AND set_aginfer_program_state (#185 audit: wire round-trip
     — catches a pid/state vs program_id/transition field mismatch)
D. SESSION_END handler
  D0 handler calls tracker.end(pid) + enqueues the PUT
  D1 handler with no session id → no-op (no enqueue)
  D2 attach_session_end_handler registers on EventKind.SESSION_END
  D3 COMPOSED router (real EventRouter + attach_kv_scheduler then
     attach_session_end_handler, main.py's order) routes a real
     SESSION_END event to F5, NOT kv_scheduler.handle (#185 audit:
     catches the attach-order shadowing bug that every isolated
     stage would miss)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t41/verify.py
```

Pure-Python (asyncio); ~0.3 s.  No GPU, no sglang launch.

## RESULTS

**PASSED** — all 17 stages (14 initial + 3 from #185 audit).

* date: 2026-06-02
* raw logs: `results/20260602_t41_initial_pass.log` (14 stages),
  `results/20260602_t41_post_audit_pass.log` (17 stages)

## REGRESSION SANITY

* T6 program_tracker: PASS (ENDED state + end() additive)
* T36 outbound queue: PASS (POST path byte-identical; `_post_one`
  dispatch only changes the PUT branch)
* T164 sustained-escalate: PASS
* T21 program_paused: PASS
* T24 hash_collision: PASS
* T4 daemon_proxy_events: PASS (499 verdict additive to the gate)
