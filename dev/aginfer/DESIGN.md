# aginfer daemon — design (v3, pure reactive)

> **Δ from v2.** Polling is gone. The daemon has no timer / no tick loop.
> All decisions are triggered by events:
> - 6 of paper §4's 8 event kinds come from the daemon's proxy
>   (session_arrival, llm_prefill, tool_call_start, tool_call_end,
>   sub_dispatch, sub_return).
> - The remaining 2 (`memory_pressure`, `pressure_resolved` — the
>   tier-3-occupancy-watermark transitions) come from sglang via a
>   `POST /aginfer/event` webhook fired by sglang's own scheduler step.
>
> Tightness ratio vs ThunderAgent: TA polls every 5 s → average reaction
> time 2.5 s, worst case 5 s. Our reactive design reacts within one
> sglang scheduler step (≈ 10–50 ms) of the actual cache state changing.

## Layered architecture

```
                harbor (terminus-2 trial containers)
                  │
                  ▼  OpenAI-compat HTTP
       ┌───────────────────────────────────────────────────┐
       │  aginfer-daemon (pure reactive)                   │
       │                                                   │
       │  /v1/chat/completions       /aginfer/event        │
       │       │                          │                │
       │       │ proxy emits 6 paper §4   │ sglang pushes  │
       │       │ events: session_arrival, │ memory_pressure│
       │       │ tool_call_*, sub_*       │ pressure_resolved│
       │       │                          │                │
       │       └──────────┬───────────────┘                │
       │                  ▼                                │
       │           single-worker event_queue (asyncio)     │
       │                  │ serialised, idempotent         │
       │                  ▼                                │
       │           on_event(e):                            │
       │            1. fetch_state()  (always fresh)       │
       │            2. dispatch to kv_scheduler /          │
       │               admission_controller based on e.kind│
       │            3. action -> migrate / pause / resume  │
       │                                                   │
       │       program_tracker (REASONING/ACTING/PAUSED)   │
       │       state machine, also event-driven from proxy │
       └─────────┬──────────────────────────────┬──────────┘
                 │ proxied requests             │ /aginfer/* admin HTTP
                 ▼                              ▼
       ┌───────────────────────────────────────────────────┐
       │  sglang  (V4-Flash, TP=2, HiCache + Mooncake)     │
       │                                                   │
       │   UnifiedRadixCache.FullComponent.drive_eviction  │
       │     SGLANG_KV_POLICY_MODULE=baselines.sglang_     │
       │     adapter:ours_greedy_score  (inline scorer)    │
       │                                                   │
       │   Admin endpoints (read + write):                 │
       │     GET  /aginfer/state                           │
       │     POST /aginfer/migrate                         │
       │                                                   │
       │   Outbound webhook:                               │
       │     on every scheduler step, detect HBM occ       │
       │     state ∈ {OK, HIGH, CRITICAL}. Fire             │
       │     fire-and-forget POST                          │
       │       --aginfer-notify-url/aginfer/event          │
       │     when (a) state transitions, OR                │
       │     (b) state ∈ {HIGH, CRITICAL} AND               │
       │         last_fire was >= 5 s ago (heartbeat).     │
       │     Payload {kind: memory_pressure |              │
       │     pressure_resolved | still_high, occ, used}.   │
       │                                                   │
       │   session_id passthrough -> UnifiedTreeNode       │
       │     .session_ids: set[str]                         │
       └───────────────────────────────────────────────────┘
```

Three layers, three cadences (none of them periodic):

| Layer | Where | When it fires | What it decides |
|---|---|---|---|
| inline scorer | inside sglang's `drive_eviction` | every alloc-failure | eviction heap key for THIS evict |
| kv_scheduler | aginfer-daemon | on each paper §4 event (proxy or sglang-pushed) | batch of `(hash, target_tier)` for `D_t` of that event |
| admission_controller | aginfer-daemon | on `memory_pressure` / `pressure_resolved` events | pause / resume per program at proxy |

All three use the **same paper §7 value rule** (`baselines/ours_greedy.py`),
just with different action vocabularies, different state visibilities, and
different *event kinds* feeding `OursGreedyPolicy.decide(state)`. No new
`decide_periodic()` method; we always invoke `decide()` with an
`event_kind` and a `decision_set` per paper §4.

## Why pure reactive

| | 5 s polling (TA-style) | pure reactive (this design) |
|---|---|---|
| reaction latency for `memory_pressure` | 0–5 s (avg 2.5 s) | ≤ 1 sglang scheduler step (~10-50 ms) |
| daemon idle CPU | non-zero (timer + fetch every 5 s) | zero (event-driven) |
| interval tuning knob | yes (5 s is a guess) | none |
| matches paper §4 directly | partial (memory_pressure faked from polling) | yes (all 8 event kinds are real events) |
| "we subsume TA" narrative | "we mirror TA" | **"we react faster than TA, with paper §4 events"** |

## Reliability — explicit failure modes & mitigations

These are real risks of going fully reactive. Each verify file checks the mitigation.

| Risk | Mitigation | Verified in |
|---|---|---|
| `POST /aginfer/event` fails over network | sglang retries 3× with exponential backoff; handler is idempotent | T5 |
| State drift (event payload is stale by the time daemon reads it) | every event handler re-`fetch_state()` at entry; never trust event payload's snapshot | T6, T7, T8 |
| Concurrent event handlers race on migrate | single asyncio worker via `asyncio.Queue`; handlers serialised | T5 |
| sglang webhook fire path itself crashes | fire-and-forget `asyncio.create_task` + `try/except` wrapper; sglang scheduler never blocks on send | T5 |
| Daemon cold start: unknown initial cache state | daemon does a single `/aginfer/state` fetch at startup; from then on reactive | T5 |
| Missed event (e.g. daemon restarted mid-flight) | on startup, daemon scans `tier_usage` once; if `HBM_occ > θ_hi` synthesise a `memory_pressure` event locally | T5 |
| Debounce: state oscillates around θ_hi | sglang fires webhook on `OK↔HIGH↔CRITICAL` transitions; in `HIGH` or `CRITICAL` it ALSO fires a heartbeat at `interval=5 s` so the daemon can re-evaluate pause victims during plateau (matches TA's polling cadence at the only point it matters) | T5 |
| Inline scorer module fails to load (ImportError) | sglang logs a structured `kv_policy_loaded={module}` line at startup; T9 / T10 grep that line and fail the run if the configured module is not loaded | T9 |

## sglang surface — final, ≈ 130 lines

| Endpoint | Direction | Purpose | LoC |
|---|---|---|---|
| `GET /aginfer/state` | sglang ← daemon | snapshot `s_t` (per-unit + per-tier) | ~40 |
| `POST /aginfer/migrate` | sglang ← daemon | apply `a_t = {(u, τ_target)}` | ~40 |
| `POST <notify_url>/aginfer/event` | sglang → daemon | webhook on watermark transition | ~30 |
| `session_id` passthrough to `UnifiedTreeNode.session_ids` | internal | wire `extra_body.program_id` into tree node | ~20 |

No new core algorithms in sglang. The inline scorer is already shipped
on commit `c784e51ee`. Webhook is fire-and-forget; never blocks sglang's
scheduler step.

## Daemon entry points

```python
# Proxy → emits 6 of paper §4's 8 events
@app.post("/v1/chat/completions")
async def chat(req):
    pid = extract_program_id(req)
    await program_tracker.wait_if_paused(pid)
    if pid not in known_programs:
        await event_queue.put(Event("session_arrival", session=pid))
    await event_queue.put(Event("llm_prefill", session=pid))
    program_tracker.observe_arrival(pid)
    # forward to sglang; on response stream end emit tool_call_start
    async for chunk in proxy_stream(req):
        yield chunk
    await event_queue.put(Event("tool_call_start", session=pid))
    program_tracker.observe_completion(pid)

# Webhook from sglang → emits the remaining 2 events
@app.post("/aginfer/event")
async def on_sglang_event(payload):
    await event_queue.put(Event(payload.kind, payload))

# Single-worker handler, idempotent
async def event_worker():
    while True:
        e = await event_queue.get()
        async with action_lock:                   # serialise
            state = await client.fetch_state()     # always fresh
            if e.kind == "memory_pressure":
                action = policy.decide(state, e.kind, decision_set=top_k_by_regret(state))
                await client.migrate(action.assignments)
                admission.on_pressure(state)
            elif e.kind == "pressure_resolved":
                admission.maybe_resume(state)
            elif e.kind in ("session_arrival", "llm_prefill",
                            "tool_call_start", "tool_call_end", ...):
                action = policy.decide(state, e.kind, decision_set=paper_§4_table(e))
                await client.migrate(action.assignments)
```

The `decision_set` per event kind is exactly paper §4's table. No
`decide_periodic`, no new policy entry point.

## Worst-case argument (final)

O_TA = observable behaviors from { proxy queue gating, sglang's
default LRU eviction, 5 s polling latency }.

O_us = observable behaviors from { proxy queue gating, paper §7
per-unit migration via `/aginfer/migrate`, paper §7 reactive eviction
order via inline scorer, sub-50 ms reaction via webhook }.

O_TA ⊊ O_us pointwise *and* with better latency on the shared
behaviors. Floor argument holds.

## Acknowledged costs

| Cost | Estimate | Verified by |
|---|---|---|
| Extra HTTP hop on each request (proxy) | +0.5-2 ms/req | T4 |
| Daemon idle CPU (no traffic) | < 1 % of one core | T5 |
| Daemon event-handler latency | < 80 ms p99 from event arrival to migrate POST | T5, T7 |
| sglang webhook fire overhead | < 50 μs per scheduler step (state check + occasional POST) | T5 |
| sglang `/aginfer/state` walk | < 10 ms @ 10 k tree nodes | T1 |
| `/aginfer/migrate` per action | best-effort; HiCache backup is async; ack via re-poll | T2 |
| Code added to sglang | ≈ 130 lines incl. webhook; inline scorer already shipped | T1+T2+T3+T5 git diff |

## Aggregation correctness (from v2 audit)

`admission_controller` aggregates `V_u` over a program to pick pause victims.
Naive `sum(V_u for u ∈ program.units)` **double-counts** terminus-2's
shared system prompt (which lives in every program's session_ids set).
Fix: weight each unit by `1 / len(unit.session_ids)` when aggregating
to program p, so a shared 32-way-owned unit contributes 1/32 to each
program's score. Equivalent to summing the unit's marginal value to
program p. Verified in T8.

**PAUSED holders are included in the denominator** (audit #13). A unit
held by 30 active + 2 paused programs has `|session_ids| = 32` for
weighting purposes — pausing one active program would not free the bytes
the unit occupies (the 2 paused holders still hold it), so each active
program's marginal benefit from pausing remains 1/32, not 1/30. This
matches the semantics in `OursGreedyPolicy._net_value`.

## Done = Run K narrows the gap to TA (revised per audit)

Realistic acceptance — pre-committed before the run:

* `K.successful ≥ 28`
* `K.mean < 716 s` ( = Run G 666 s + 50 s slack — narrows the
  gap to TA, doesn't have to beat it)
* `K.p99 < 1336 s` ( = Run H' p99; tail-latency property preserved)
* zero sglang crashes / scheduler-subprocess exits
* startup-log invariant: sglang logs `kv_policy_loaded=…` and daemon
  logs `kv_scheduler=enabled, admission_controller=enabled`. If
  either is missing, **halt the run** (audit #11).

Stretch:
* `K.mean < 666 s` (beat TA on mean — aspirational, expected 720-790 s
  per audit prediction).
* `K.std < 280 s` (= Run H' std).

Ablation:
* **Run K-a** — daemon with kv_scheduler ON, admission_controller OFF,
  inline scorer ON. Expected ≈ Run H' (885 s); shows value rule alone.
* ~~Run K-b~~ **dropped** per v2 audit: replicating TA's pause-victim
  selection inside our admission_controller requires either a token-
  count BFD fallback or duplicating TA's heuristic. We already have
  Run G as a real TA measurement; use that directly rather than
  rebuild TA inside our daemon.
* **Run K** (full): all three layers active. Target < 666 s.

## TODO (revised)

| ID | Task | Verify | Estimate |
|---|---|---|---|
Implementation order (revised per audit #16: T6 must precede T4 because
T4's verify uses `program_tracker.pause/resume`):

| ID | Task | Verify | Estimate |
|---|---|---|---|
| T1 | `GET /aginfer/state` | [verify/t1/](verify/t1/) | 2-3 h |
| T2 | `POST /aginfer/migrate` | [verify/t2/](verify/t2/) | 2-3 h |
| T3 | session_id passthrough | [verify/t3/](verify/t3/) | 1 h |
| **T6** | program_tracker state machine *(moved up — T4 depends on it)* | [verify/t6/](verify/t6/) | 2-3 h |
| T4 | daemon proxy + emits 6 paper §4 events | [verify/t4/](verify/t4/) | half day |
| T5 | sglang→daemon webhook (transition + 5 s heartbeat in HIGH/CRITICAL) + daemon event router | [verify/t5/](verify/t5/) | 3-4 h |
| T7 | kv_scheduler event handlers | [verify/t7/](verify/t7/) | 2-3 h |
| T8 | admission_controller event handlers + correct aggregation | [verify/t8/](verify/t8/) | 2-3 h |
| T9 | Run K + K-a ablation | [verify/t9/](verify/t9/) | half day |
| T10 | integration / concurrency / restart / GC + forced-fault verifies | [verify/t10/](verify/t10/) | half day |

Total: ~2.5 days.

## Pre-committed worst-case floor

Every verify file (T1-T10) has a **WORST CASE (forced)** section that
*actually injects* the failure mode and asserts the system stays
within the documented floor. Headline floors:

| layer fails | system degrades to | floor (per-trial mean) |
|---|---|---|
| kv_scheduler stuck | inline scorer only | ≈ Run H' 885 s |
| admission_controller off | no program back-pressure | ≈ Run F' 873 s |
| daemon crashes | sglang alone with inline scorer | ≈ Run H' 885 s |
| inline scorer crashes (and daemon survives) | LRU + daemon migrate hints | ≈ Run F' to Run H' band |
| **all three layers fail** | bare sglang LRU | ≈ Run F' 873 s |

So Run K's absolute worst case (if every daemon-side mechanism is
broken) is bounded **above** at Run F' 873 s. Any number worse than
that indicates a regression in the inline scorer path — independently
verified by Run H' as the pre-existing baseline.
