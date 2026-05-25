# T4 — aginfer-daemon HTTP proxy + paper §4 event emission

## WHAT WE PROMISED

**Capability**
* OpenAI-compat `POST /v1/chat/completions` forwards verbatim to sglang.
* `extra_body.program_id` (or `X-Session-Id` header) drives program tracking
  but is otherwise transparent to the request body.
* Streaming responses pass through chunk-by-chunk (SSE) without buffering.
* Emits paper §4 events to the daemon's event queue as the request flows:
  - **On request arrival** for a previously-unseen program_id →
    `Event("session_arrival", session=pid)`.
  - **On request arrival** (always) → `Event("llm_prefill", session=pid)`.
  - **On response stream end** (SSE `[DONE]` or unary response returned)
    → `Event("tool_call_start", session=pid)` (this is paper §2.4's
    "cursor advances onto V^tool" — the proxy can't distinguish this
    from "agent decided to stop" without parsing tool-call markers in
    the response; the conservative interpretation is "agent is about
    to do something off-GPU"). On the **next** arrival for the same
    program → `Event("tool_call_end", session=pid)` is emitted first,
    then `llm_prefill`.
  - **On request body** containing a sub-agent dispatch marker (looked
    up by harbor / terminus-2's convention; if convention unknown,
    skip) → `Event("sub_dispatch_blocking" | "sub_dispatch_async",
    session=pid, child_session_id=...)`.
* If `program_tracker.is_paused(pid)`, the request awaits
  `program_tracker.resume_event(pid)` **before** forwarding (this is
  the load-bearing piece that gives us TA-style back-pressure).
* Non-`/v1/chat/completions` paths are forwarded too (health checks).

**Cost ceiling**
* Added latency vs hitting sglang directly: < 2 ms p50, < 5 ms p99
  (small request, 200 prompt tokens / 16 generation).
* Streaming throughput: ≥ 95 % of direct sglang throughput.
* Event emission overhead: < 0.1 ms per event (just an
  `asyncio.Queue.put_nowait`).

## HOW WE VERIFY

Mechanism. `verify/t4_proxy.py`:

```
1. Launch sglang on :30000 and daemon on :9100, pointing at sglang.
2. Response equivalence:
   - Send 100 small non-streaming chat completions to BOTH paths.
   - Assert response bodies semantically identical (mod timing fields).
3. Latency:
   - Same 100 reqs through each path; report (p50, p99) per path;
     assert proxied - direct < 2 ms p50, < 5 ms p99.
4. Streaming parity:
   - Send a streaming request through both, count chunks; assert
     equal count and proxied chunks arrive +1 ms each of direct ones.
5. Event emission:
   - Wire a stub for the daemon's event_queue that records every
     `put`. Send a sequence:
       turn1: chat completion (req → daemon → sglang → response).
              Expect events: session_arrival, llm_prefill,
              tool_call_start.
       turn2: another chat completion same program_id.
              Expect events: tool_call_end, llm_prefill,
              tool_call_start.
6. Pause back-pressure:
   - Call program_tracker.pause("p1"). Issue request for "p1".
   - Assert: request blocks (not forwarded) within 20 ms.
   - program_tracker.resume("p1"). Assert request forwards within
     100 ms after resume.
```

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| sglang dies mid-stream | Kill sglang during a 30-s streaming request | Proxy returns 502 to harbor; harbor trial errors with clear "upstream gone"; daemon does NOT crash; other in-flight requests fail gracefully | aggregated harbor stats: errored trial = the one in flight; daemon proc still up |
| sglang slow first-token (10 s TTFT) | Inject 10 s sleep in stub-sglang's response | proxy holds connection open; client receives keep-alive / data within timeout; per-request latency rises 10 s but no proxy crash | client-side timeit |
| Malformed extra_body | Send `extra_body=null`, `extra_body=[]`, `extra_body={"program_id": 42}` | Each request still forwards; program_id either coerced to "default" or skipped tagging; never crashes proxy | response 200; daemon log shows graceful coerce |
| Resume races request | `program_tracker.pause("p")`, send request for "p", immediately `program_tracker.resume("p")` | request unblocks within 100 ms of resume; no double-emission of `tool_call_end` | event log inspection |
| 50 concurrent requests, all same program | Issue 50 requests with same `program_id` | program_tracker observes 50 arrivals; events emitted in order; no race condition causes lost state transition | event log invariants |

## RESULTS
* date: _pending_
* daemon sha:
* response equivalence: _pending_
* latency overhead p50 / p99: _pending_
* streaming chunk parity: _pending_
* event emission sequence correct: _pending_
* pause back-pressure works: _pending_
* raw log: `verify/results/t4_<datetime>.log`
