# T164 — Sustained-load escalate-to-fatal (DESIGN §10 sustained tier)

The "running degraded forever" anti-pattern.  When sglang has been
unreachable AND the daemon's outbound queue has been backing up, the
daemon self-kills so the supervisor (systemd / k8s) restarts.  If
sglang is still down, the daemon CrashLoopBackOffs visibly to ops.
Crash-only software (Candea & Fox); silently eating memory + doing
no useful work is NOT a valid state.

Lives in #164.  DESIGN §10 "sustained-escalation tier" (added by
#163) names this contract.

## WHAT WE PROMISED

**Trigger criterion: BOTH must trip simultaneously**:
1. `OutboundQueue.consecutive_failures >= escalate_failures`
   (default 100 — ~3-5 min at typical fire-and-forget rates)
2. `oldest_pending_batch_age_ms >= escalate_oldest_age_s * 1000`
   (default 300 s = 5 min)

If only ONE crosses, escalation does NOT fire.  Rationale:

* Low-traffic dead-sglang: every POST fails fast (ConnectError) →
  consec rockets, but queue drains as fast as failures complete →
  `oldest_age` stays small.  Daemon survives, will reconverge when
  sglang returns.  No memory damage.
* All-success but slow POSTs: `oldest_age` can grow under burst but
  consec is 0 → no escalation.

The both-or-neither contract guards against false positives on
either axis alone.

**Post-#228 coalescing note.**  The worker now drains each wake's burst
and emits at most one POST/PUT per endpoint, so `consecutive_failures`
**accumulates across wakes** (≤ one increment per endpoint per wake, reset
only on a 2xx) rather than one-per-enqueued-batch, and the age in (2) is
the oldest `enqueue_ts` of the coalesced burst being dispatched.  The
escalation BEHAVIOR is unchanged (both conditions still required; the
streak still reaches the threshold under a sustained stall, over more
wakes).  Accordingly the tests changed: A0–A2 now exercise the consec
accounting at the per-dispatch unit (`_dispatch_one`) directly, and B0/C0
pack three endpoints into one wake (program_paused + migrate + hints, all
failing) so a single wake yields three failed POSTs that trip the
low-threshold fatal — a fixture convenience, not a weakening of the
production sustained-failure condition.

**Failure classes counted for consec**:
| HTTP / exception | counts as |
|---|---|
| 2xx | success → resets consec to 0 |
| 4xx | failure (plan-shape bug — deployment-class; crashloop reveals) |
| 5xx | failure (transient sglang stall) |
| Transport exception | failure (sglang unreachable) |

**CLI flags** (`daemon/main.py`):
- `--sustained-escalate-fails` (default 100)
- `--sustained-escalate-age-s` (default 300.0)

**fatal() call site** (inside `OutboundQueue._worker_loop`):
`fatal('sglang_sustained_unreachable', sglang_base_url=...,
consecutive_failures=N, oldest_age_ms=X, queue_depth=Q,
escalate_failures_threshold=..., escalate_oldest_age_s_threshold=
...)`.  Forensic JSON dumped to `<data>/forensic/
sglang_sustained_unreachable_*.json` per T43 contract.

**Exit primitive** (#166): `fatal()` uses `os._exit(1)`, NOT
`sys.exit(1)`.  The crash-only contract MUST NOT depend on asyncio
`Task.__step` re-raising `SystemExit` past uvicorn / `gather` /
`shield` wrappers.  Modern CPython (3.12) happens to route the
`SystemExit` correctly (verified by stage C0), but a future asyncio
change or a custom `loop.set_exception_handler` could silently
swallow it.  `os._exit` bypasses Python shutdown — process dies
immediately, supervisor restarts cleanly.

**/health observability** (`daemon/proxy.py`):
Daemon's `/health` body now includes:
- `outbound_consecutive_failures: int` — direct read of the counter
- `outbound_oldest_age_ms: float` — **LIVE** peek of the current
  in-queue head's age (#166).  Earlier draft cached the just-popped
  batch's age in a sticky `last_outbound_oldest_age_ms` field;
  bug: once the queue drained (sglang healed), the field stuck at
  the last large value forever, marking the daemon perma-NotReady
  to any k8s readiness probe scripted against it.  The live peek
  reads `asyncio.Queue._queue[0]` (GIL-atomic on CPython) and
  returns 0.0 when the queue is empty.

HTTP status stays 200 — the daemon process IS responsive; the
restart signal is the fatal() exit, not health failure.  k8s
readiness probes can grep the body fields and depool independently.

## WORST CASE

| Failure mode | How to force | Floor | Assertion |
|---|---|---|---|
| Consec counter doesn't reset on success | mix 2xx with failures | counter resets to 0 on 2xx | A0 |
| One failure flavor missed | 5xx / 4xx / ConnectError each | all four increment consec | A1 |
| False positive: low-traffic dead-sglang fatal | 10 fresh failures, age ≈ 0 | NO fatal | A2 |
| False positive: high-age but all-success | aged batches, all 2xx | NO fatal (consec=0) | A3 |
| Operator can't grep current state | hit /health | body has both counters | A4 |
| True positive: dead sglang + backlog | aged batches + always-fail stub | fatal + forensic dump + exit 1 | B0 |
| fatal() under uvicorn doesn't exit | aged + always-fail + uvicorn.run | subprocess exits 1 within 15 s AND stderr names forensic file | C0 |
| Sticky age after queue drain | drain to empty | /health age ~0, not last-popped value | C1 |
| /health misses live in-queue head | 3 aged batches, no worker | /health = age of OLDEST (5 s), not 0 | C2 |
| Live-peek crashes / freezes under concurrent worker pop | delay-stub worker + 10-batch backlog + 10 Hz /health polling | no exception; final age ≈ 0; age series shrinks | C3 |
| Default `enqueue_ts=0.0` footgun trips escalation | bare `OutboundBatch(...)` / explicit 0 / negative | TypeError / ValueError raised | C4 |

## HOW WE VERIFY

`verify/t164/verify.py` — 9 stages.  A0-A4 + C1-C2 in-process; B0
and C0 spawn real subprocesses to exercise the actual `fatal()`
path (which exits the process — can't test in the same process).

```
A0  counter resets on 2xx (3 batches: 5xx, 5xx, ok → consec=0)
A1  consec increments on 5xx / 4xx / ConnectError (mix of 4
    flavors, no 2xx → consec=4; thresholds high so no escalation)
A2  high consec alone does NOT escalate (always-503 stub, 10 fresh
    batches, consec >> threshold but oldest_age ≈ 0 → daemon
    survives, returns normally)
A3  high oldest_age alone does NOT escalate (always-200 stub,
    aged batches with old enqueue_ts → oldest_age >> threshold
    but consec stays 0 → no escalation)
A4  /health body carries outbound_consecutive_failures +
    outbound_oldest_age_ms (uvicorn server with a fresh-enqueued
    batch; age finite and small)
B0  subprocess: always-fail stub + aged batches + low thresholds
    → fatal('sglang_sustained_unreachable') → process exit 1
    → forensic JSON file landed with context fields
    (sglang_base_url, consecutive_failures, oldest_age_ms,
    queue_depth, escalate_failures_threshold,
    escalate_oldest_age_s_threshold)
C0  fatal() inside a UVICORN-hosted daemon ACTUALLY exits the
    process within 15 s (production code path — uvicorn.run owns
    the event loop; sys.exit(1) propagation through asyncio Task
    + uvicorn was the audit concern; os._exit(1) makes it
    deterministic)
C1  /health outbound_oldest_age_ms DECAYS to ~0 after the queue
    drains (sticky-cached-field bug: pre-#166 the field held the
    last-popped batch's large age forever)
C2  /health outbound_oldest_age_ms reports the LIVE in-queue
    oldest (3 batches at 5 s / 3 s / 1 s aged, no worker → /health
    must return ~5 s, not 0)
C3  live-peek under concurrent worker (#167): 10 aged batches +
    delay-stub worker drains at ~5/s; poll /health ~10/s for
    ≤ 4 s; assert (a) no exception during any /health call, (b)
    final age ≈ 0 (drained), (c) age series shrinks (last < first
    — peek must follow the moving head, not freeze).  Closes the
    audit's exact "live peek under concurrent pop" concern that
    A4/C1/C2 cannot exercise (they monkey-patch the worker off).
C4  OutboundBatch.enqueue_ts validation (#167 nit-3): bare
    construction without ``enqueue_ts`` raises ``TypeError``;
    explicit 0.0 or negative raises ``ValueError`` via
    ``__post_init__``.  Closes a footgun where the previous
    default of 0.0 would compute age ≈ time.time()*1000 and
    instantly trip the sustained-escalation fatal.
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t164/verify.py
```

No GPU / no sglang launch.  Runs in < 5 s (B0 spawns a brief
subprocess).

## RESULTS

**PASSED** — all 11 stages.

* date: 2026-06-01
* initial #164 close: ~50 daemon-side LoC (worker loop + `_post_one`
  + ctor; `main.py` 2 CLI flags; `proxy.py` /health body)
* #166 audit closure: drop sticky `last_outbound_oldest_age_ms`
  field; add live-peek `current_oldest_pending_age_ms()` method;
  switch `_fatal.py` `sys.exit(1)` → `os._exit(1)` with stdout/err
  flush; +3 verify stages (C0/C1/C2)
* #167 round-2 audit closure: C0 stderr assertion; +C3 concurrent-
  peek-under-worker; +C4 `enqueue_ts` validation (no 0.0 default
  footgun); `main.py` --help text accuracy

| Stage | Result |
|---|---|
| A0 consec resets on 2xx | PASS |
| A1 consec increments on 5xx / 4xx / transport-exc | PASS — final consec = 4 after mixed failures |
| A2 high consec alone does NOT escalate | PASS — 10 failures, no fatal |
| A3 high oldest_age alone does NOT escalate | PASS — 5 aged successes, consec stays 0 |
| A4 /health body carries outbound counters | PASS — fresh batch reports finite, small `outbound_oldest_age_ms` |
| B0 subprocess: both thresholds → fatal + forensic | PASS — subprocess exit 1, forensic file with all 6 context keys |
| C0 fatal() under uvicorn ACTUALLY exits + names forensic on stderr | PASS — uvicorn.run subprocess exits 1 within ~1 s |
| C1 oldest_age decays after queue drains | PASS — drain to empty → /health reports ~0 ms (pre-#166: sticky 100 s) |
| C2 /health reports LIVE in-queue oldest | PASS — 3 batches aged 5/3/1 s → /health returns ~5 s (pre-#166: 0 ms) |
| C3 live-peek under concurrent worker | PASS — 10-batch drain at ~5 Hz, /health polled at ~10 Hz; no exception; age series shrinks |
| C4 enqueue_ts validation | PASS — bare construction TypeError; 0.0 / negative ValueError |

* raw logs: `results/20260601_t164_initial_pass.log` (initial #164),
  `results/20260601_t164_post_166_audit_pass.log` (post-#166 closure),
  `results/20260601_t164_post_167_audit_pass.log` (post-#167 closure)
