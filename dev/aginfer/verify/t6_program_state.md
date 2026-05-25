# T6 — program_tracker state machine

## WHAT WE PROMISED

**Capability**
* Tracks per-program state ∈ {REASONING, ACTING, PAUSED}.
* Transitions are driven by **observable HTTP events** at the daemon's
  proxy layer, not wall-clock timing:
  - Request arrives for program p → p enters REASONING.
  - Response stream ends (or unary response returned) → p enters ACTING.
  - admission_controller calls `pause(p)` → p enters PAUSED.
  - admission_controller calls `resume(p)` + next request arrives → p enters REASONING.
* PAUSED programs' requests block at the proxy on an `asyncio.Event`
  until resumed.

**Cost ceiling**
* < 5 ms per program update.
* No timing-based heuristic in the code path (no `if now() - last > X`
  driving state transitions); only HTTP-event-driven.

## HOW WE VERIFY

Mechanism. `verify/t6_program_state.py`:

```
1. Stub a fake "sglang" that returns a chat-completion stream after
   a fixed delay. Spin up daemon pointing at it.
2. Issue a non-streaming request for program "p1". Assert state(p1)
   == REASONING during the inflight period, ACTING after.
3. Issue a streaming request for "p2". Assert state(p2) flips to ACTING
   only AFTER the stream's "data: [DONE]" sentinel, not earlier.
4. Manually invoke admission_controller.pause("p1"). Issue a new
   request for "p1". Assert the request hangs (state PAUSED).
   resume("p1"); assert the request unblocks within 100 ms.
5. Verify, by grep-ing daemon code, that the state-transition path
   does NOT contain any `time.time() - last > threshold` logic.
```

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Lost `tool_call_start` (stream-end signal swallowed) | Drop proxy's stream-end hook for 10 % of requests | Program stuck in REASONING; admission_controller may still pause it via watermark fallback; not catastrophic | event log: stuck-in-REASONING count vs ground truth |
| **State-drift recovery** (audit #9) | Patch proxy to silently drop EVERY `tool_call_end` for 30 s, then unpatch | After unpatch + next observed completion for that program, `program_tracker` re-derives state correctly; no permanent stuck-in-REASONING entries; run K-style mini-traffic doesn't error out | record state at t=30 s (during drift), t=60 s (after recovery); assert recovered state ∈ {REASONING, ACTING} for all programs that issued ≥1 request after t=30 s |
| Bogus `pause(p)` for unknown p | Call `program_tracker.pause("never_seen")` | logs warning; creates a placeholder PAUSED entry; if request later arrives, it waits and resumes correctly | tracker dump + send-after-pause test |
| Program churn (10 k unique program_ids in 60 s) | Synthetic stream of distinct ids, one request each, then never again | program_tracker memory bounded ≤ 50 MB after 60 s; GC runs | psutil + tracker.size() |
| Concurrent transitions for same program | Spam 100 arrival+completion pairs for one program in 100 ms | All transitions correctly recorded; no lost state; final state matches expected | event count + final state check |
* date: _pending_
* daemon sha:
* state on inflight non-streaming: _pending_
* state on streaming chunks (DONE):
* PAUSED hangs requests: _pending_
* resume unblocks: _pending_
* event-only transitions (grep check):
* raw log: `verify/results/t6_<datetime>.log`
