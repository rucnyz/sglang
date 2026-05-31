# daemon proxy + paper §4 event emission (daemon I/O surface)

> **Status: infrastructure regression-guard.**  Not a numbered PLAN.md
> task.  Guards the daemon proxy's two jobs:
>
> 1. **Pass-through of `/v1/chat/completions`** to sglang (transparent
>    forwarding; streaming SSE preserved).
> 2. **Event emission** to the daemon's event bus: session_arrival /
>    llm_prefill / tool_call_start / tool_call_end on each request,
>    plus the program-gate (pause/resume) handling on the request
>    path.
>
> Pause gating is currently daemon-internal (`tracker.pause()` →
> `program_tracker._states[pid] = PAUSED` + `asyncio.Event` wait
> inside the proxy).  Two follow-up Tn extend it:
>   - **T21** (PLAN §3) — sglang-side `PUT /aginfer/program_paused`
>     so `per_program_usage[p].state` is authoritative.  PENDING.
>   - **T39** (PLAN §4) — proxy gate awaits BOTH the pause condition
>     AND `request.is_disconnected()`, releasing on TCP disconnect
>     with HTTP 499 + transition the program to ENDED.  PENDING.
>
> When those Tn land, the corresponding probes will be added here
> (or split into a sibling verify dir).
>
> Webhook-driven event kinds (memory_pressure, PRESSURE_CRITICAL,
> APPLY_FAILED, HASH_COLLISION, SESSION_END) are covered by the
> webhook-router verify (rename of old t5; see #137 cleanup).  This
> verify covers the 4 request-path event kinds only.

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

Mechanism. `verify/t4/verify.py` (in-process FastAPI stub-sglang +
real daemon; no GPU; ~10 s total runtime):

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

## REPRODUCING

T4 is in-process: a FastAPI stub sglang + the real daemon both run
inside the verify's asyncio loop, talking over loopback HTTP.  No
GPU, no real sglang launch.

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched

cd /scratch/yuzhou/projects/sglang/dev/aginfer
python verify/t4/verify.py
# expected last line: "=== T4 PASSED ==="
```

Daemon code lives at `dev/aginfer/daemon/proxy.py` (~270 LoC),
`dev/aginfer/daemon/events.py` (~100 LoC); T6 dep:
`dev/aginfer/daemon/program_tracker.py`.

## RESULTS

**PASSED** — all 16 steps (post audit round-1 + audit-of-tests
round-2 + audit round-3 stream-robustness expansion) including a
5-run multi-trial latency benchmark.

### Audit round-3 additions (stream / header robustness)

* [2] step_streaming_chunks (N3): tightened from ``len >= 2`` to
  ``== 4`` (3 data deltas + DONE) + per-frame ordering assertion.
  Catches a buffer-then-flush regression that collapses chunks.
* [2b] step_header_forwarding (N2): new step pinning that
  ``Authorization`` / ``traceparent`` / ``x-request-id`` survive the
  proxy hop.  Stub records ``raw.headers``; sentinel values asserted.
* [10b] step_streaming_midstream_break (M1): new step — stub commits
  200 + SSE then raises mid-stream; proxy must emit in-band SSE
  error frame + ``[DONE]`` and recover tracker via ``finally``.
* [10c] step_streaming_client_disconnect (N5): new step — client
  drops connection after 1 chunk; assert generator ``finally`` runs
  ``_emit_completion`` and tracker exits REASONING.
* [10d] step_streaming_connect_non_request_error_recovers (N1):
  new step — monkey-patches ``http_client.stream`` to raise a
  non-RequestError on ``__aenter__``; assert 502 + tracker recovers
  (symmetric to unary's [11a]).

* date: 2026-05-26
* daemon code: ~370 LoC across `daemon/proxy.py` + `daemon/events.py`
  (T6 reused unchanged).
* response equivalence: ✓ unary JSON pass-through verbatim;
  non-JSON / non-200 bodies (e.g. text/html error pages) preserved
  with original content-type (round-1 BLOCKER 1 fix; pinned by step
  [9]).
* event emission sequence: ✓ first-arrival emits SESSION_ARRIVAL +
  LLM_PREFILL + TOOL_CALL_START; subsequent arrival after ACTING
  emits TOOL_CALL_END (BEFORE) + LLM_PREFILL + TOOL_CALL_START.
* streaming chunk parity: ✓ SSE chunks pass through; original
  content-type preserved; `[DONE]` terminator survives.  Dead
  upstream returns a real 502 *before* committing 200 + SSE
  headers (round-1 BLOCKER 2 fix; pinned by step [10]).
* pause back-pressure: ✓ request blocks on pause; resumes < 1 s.
* `stream` is strict bool: ✓ string `"false"` does NOT trigger
  streaming branch (round-1 MINOR fix; pinned by step [11]).
* malformed `program_id`: ✓ 8 shapes (None / "" / "   " / 42 /
  dict / list / "x"\*10k / `[None, "deep"]`) all forward cleanly.
* upstream dead (unary): ✓ 502; tracker recovers (not stuck in
  REASONING).
* unary non-RequestError recovery (audit round-2 pin for round-1
  MAJOR): ✓ step [11a] monkey-patches the http_client to raise a
  custom non-httpx exception; daemon returns 502 AND the program
  tracker exits REASONING (not stuck).  A revert of the broader
  `except Exception` to bare `except httpx.RequestError` re-raises
  and trips this assertion.

### Latency (multi-run, per memory:feedback-latency-multi-run)

5 independent trials × 50 requests/trial = 250 samples per side
(direct stub vs proxy).  In-process loopback, no GPU work.

| metric                | direct       | proxy        | overhead         |
|---|---|---|---|
| **p50** (mean ± std)  | 1.18 ± 0.02 ms | 2.68 ± 0.04 ms | **+1.49 ± 0.02 ms** |
| **p99** (mean ± std)  | 1.31 ± 0.09 ms | 3.25 ± 0.55 ms | **+1.94 ± 0.47 ms** |

Both well inside the design budget (<2 ms p50 / <5 ms p99 added
latency).  Audit round-2 tightened the floor: assertion now uses
``mean + 3σ < 5 ms`` (was ``mean + 1σ < 10 / 25 ms`` which would
have masked a 5–10× regression).  Current envelope ≈ 1.5 ms p50 /
~2 ms p99 — leaves ~2.5× headroom but catches a 3× regression in
the proxy hot path.

* raw logs (relative to this directory):
  * `results/<YYYYMMDD_HHMMSS>_run4_audit.log` — round-1 audit
    state (12 steps + 5-run latency)
  * `results/<YYYYMMDD_HHMMSS>_run5_audit2.log` — round-2
    "audit of tests" state (13 steps + 5-run latency)
