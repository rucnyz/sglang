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

Mechanism. `verify/t6/verify.py` (pure asyncio, no sglang launch,
no GPU).  Earlier README drafts described a "stub sglang" + daemon
HTTP setup; that pattern moves to T4 (the actual HTTP proxy).  T6
tests the state machine in isolation by directly importing
``ProgramTracker``:

```
[1]  observe_arrival -> REASONING; observe_completion -> ACTING.
[2]  observe_completion without a preceding arrival is a no-op
     (defends against lost arrival events).
[3]  pause + wait_if_paused blocks; resume unblocks.
[4]  WORST CASE: pause on a previously-unseen program creates a
     placeholder; a late arrival blocks on wait_if_paused and
     resumes correctly.
[5]  resume on an unknown program is a no-op + warning log.
[5b] pause-mid-flight (pause while REASONING): in-flight
     completion is a no-op (state stays PAUSED, per paper §9
     gating semantics); next arrival after resume flips to
     REASONING.
[5c] double observe_completion is safe (proxy at-least-once
     event delivery).
[6]  WORST CASE: 100 concurrent arrival+completion pairs;
     deterministic interleave; final state matches whichever
     event ran LAST (asserted exactly, not just "is one of
     {REASONING, ACTING}").
[7]  WORST CASE: 10 k unique program_ids; size() matches; RSS
     delta < 50 MB (resource.getrusage check).
[8]  Contract: AST-grep the module's source.  Forbids ANY import
     of `time` or `datetime` modules.  Forbids attribute names
     `time / monotonic / now / perf_counter / time_ns /
     monotonic_ns / perf_counter_ns / today / utcnow` ANYWHERE
     in any method on ProgramTracker (public OR private).
```

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Lost `tool_call_start` (stream-end signal swallowed) | Drop proxy's stream-end hook for 10 % of requests | Program stuck in REASONING; admission_controller may still pause it via watermark fallback; not catastrophic | event log: stuck-in-REASONING count vs ground truth |
| **State-drift recovery** (audit #9) | Patch proxy to silently drop EVERY `tool_call_end` for 30 s, then unpatch | After unpatch + next observed completion for that program, `program_tracker` re-derives state correctly; no permanent stuck-in-REASONING entries; run K-style mini-traffic doesn't error out | record state at t=30 s (during drift), t=60 s (after recovery); assert recovered state ∈ {REASONING, ACTING} for all programs that issued ≥1 request after t=30 s |
| Bogus `pause(p)` for unknown p | Call `program_tracker.pause("never_seen")` | logs warning; creates a placeholder PAUSED entry; if request later arrives, it waits and resumes correctly | tracker dump + send-after-pause test (step [4]) |
| Program churn (10 k unique program_ids in 60 s) | Synthetic stream of distinct ids, one request each, then never again | tracker survives 10 k without crash; RSS delta < 50 MB; v1 has NO GC / cap — programs accumulate.  T8 / T9 may add LRU if profiling shows churn-driven growth | step [7]: resource.getrusage RSS delta + size() check |
| Concurrent transitions for same program | Spam 100 arrival+completion pairs for one program in 100 ms | All transitions correctly recorded; final state matches the LAST observed event (exact, not "in some set") | step [6] |
| pause-mid-flight (paper §9 gating) | observe_arrival; pause; observe_completion | completion is a no-op on PAUSED state; recovery on next arrival after resume | step [5b] |
| double-completion (proxy at-least-once delivery) | observe_arrival; observe_completion; observe_completion | second call is a no-op (state stays ACTING) | step [5c] |

**Deferred to T4** (HTTP proxy): "Lost `tool_call_start`" and
"State-drift recovery" require an actual proxy + sglang interaction.
T6 (this task) ships the underlying state machine; T4 will add the
proxy-layer tests for those rows.

## REPRODUCING

T6 is the FIRST daemon-side code; the verify is pure asyncio
in-process (no sglang launch needed, no GPU).

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched

cd /scratch/yuzhou/projects/sglang/dev/aginfer
python verify/t6/verify.py
# expected last line: "=== T6 PASSED in <NNN> ms ==="
```

Daemon code lives at `dev/aginfer/daemon/program_tracker.py` —
~50 LoC, pure Python.

## RESULTS

**PASSED** — all 10 steps (post audit round-1 + audit-of-tests
round-2) in ~60 ms on the agsched env.

### Audit round-2 "audit of tests"

Two steps had previously-loose assertions that would not catch a
realistic regression in the production code:

* Step [6] (concurrent arrival/completion) used to derive its
  prediction from ``history[-1]`` and then assert the same quantity
  — a regression silently dropping every other transition would
  also drop matching entries from ``history`` and pass.  Now snapshots
  ``pt.state(PID)`` immediately after EACH ``observe_*`` and
  asserts all N arrival snapshots are ``REASONING`` and all N
  completion snapshots are ``ACTING``.  A silent no-op observe_*
  would now trip the per-call assertion.
* Step [7] memory cap was 50 MB with ``ru_maxrss`` (high-watermark,
  not delta).  Docstring claims ~300 B/program, so a 15× regression
  to 5 KB/program would silently pass.  Switched to ``tracemalloc``
  for a true delta and tightened cap to 5 MB; now catches a 1.7×
  regression instead of letting a 15× one slip through.

* date: 2026-05-26
* daemon code: `dev/aginfer/daemon/program_tracker.py` (~140 LoC
  incl. docstrings; the actual transition logic is ~30 LoC)
* state on inflight non-streaming: ✓ (step [1] — arrival → REASONING,
  completion → ACTING, re-arrival → REASONING)
* state on streaming chunks (DONE): same path covered by step [1]
  (the proxy will call `observe_completion` on the
  `data: [DONE]` sentinel; verified at the API level here)
* PAUSED hangs requests: ✓ (step [3] — wait_if_paused blocks for
  ≥100 ms while paused; resume releases within ≤100 ms)
* resume unblocks: ✓ (step [3])
* state stays PAUSED after resume until next arrival: ✓ (step [3]
  asserts state(p) is still PAUSED post-resume; next observe_arrival
  flips to REASONING)
* event-only transitions (grep / AST check): ✓ (step [8] — AST-walks
  every public method and refuses any `time.*` / `loop.time` /
  `monotonic` / `now` reference)
* defensive contract checks: ✓ (step [2] completion-without-arrival
  is a no-op; step [4] pause on unknown program creates a
  placeholder that a late arrival correctly waits on; step [5]
  resume on unknown program is a no-op with a warning log)
* memory: ✓ (step [7] — 10 k unique program_ids tracked, size()
  matches, no crash; far under the 50 MB target)
* raw logs (relative to this directory):
  * `results/20260526_042029_run1.log` — initial 8-step run
  * `results/<YYYYMMDD_HHMMSS>_run2_postaudit.log` — round-1 audit
    additions (steps [5b], [5c]; tighter step [6] + step [7] RSS
    check + step [8] AST guard upgraded to ban any `time` /
    `datetime` import + cover ALL methods incl. private)

### Audit-round-1 findings + resolution

The round-1 adversarial audit found 3 BLOCKER + 7 MINOR + 4 NIT.
Two of the three BLOCKERs (pause-mid-flight + REASONING-overwrite)
turned out to be **undocumented correct behaviour, not bugs** —
the paper §9 design says pause is gating-only, NOT abort.  The
in-flight request finishes; its `observe_completion` is a no-op
on the now-PAUSED state.  Documented explicitly in the module
docstring AND pinned by new step [5b].

Remaining audit items addressed:
* BLOCKER 3 (resume docstring drift): fixed.  Docstring now matches
  code — resume releases waiters, does NOT mutate state map.
* MINOR 5 (wait_if_paused defensive read): now creates a cleared
  event if state is PAUSED but no event yet (handles direct-state-
  mutation tests).
* MINOR 6 (memory claim unverified): step [7] now measures RSS via
  `resource.getrusage` and asserts <50 MB delta.  Also documented
  that v1 has no GC; T8/T9 may add LRU.
* MINOR 7 (thread-safety claim too strong): docstring tightened —
  "single-event-loop-safe; NOT thread-safe; do NOT call from
  FastAPI sync handlers or off-loop threads".
* MINOR 8 (AST guard holes): now bans imports of `time` / `datetime`
  AND every wall-clock attribute name (incl. `time_ns`,
  `perf_counter_ns`, etc.) in ALL methods (incl. private `_event`).
  Verified the guard fires by injecting `import time` — caught
  immediately with a specific error.
* MINOR 9 (timing test fragile): replaced `wait_for(timeout=0.1)`
  patterns with `sleep + is_set()` checks; race-free on a loaded box.
* MINOR 10 (step [6] weak invariant): now asserts the EXACT final
  state based on the deterministic interleave's last event, not
  "in {REASONING, ACTING}".
* NIT 11 (stale HOW WE VERIFY): rewritten to match the actual
  pure-asyncio in-process design (no fake-sglang stub; that's T4).
* NIT 13 (double-completion undocumented): now in module docstring
  AND pinned by step [5c].
* NIT 14 (resume edge-trigger semantics undocumented): added to
  module docstring.

### Not covered in v1 (deferred to T4 integration)

* `tool_call_start` / `tool_call_end` events live in T4's HTTP proxy
  layer; T6 only verifies the underlying state machine.
* WORST CASE rows "Lost tool_call_start" and "State-drift recovery"
  require an actual proxy + sglang round-trip; T4 verify owns them.
* Thread-safety hardening (locks / per-pid serialisation) — not
  needed for asyncio-only daemon; revisit if FastAPI sync handlers
  are introduced.
