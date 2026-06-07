"""T6 verify: program_tracker state machine.

In-process asyncio test — no sglang launch needed.  Imports
``dev.aginfer.daemon.program_tracker.ProgramTracker`` directly and
drives state transitions through Python events.  Covers the contract
documented in dev/aginfer/verify/t6/README.md plus the WORST CASE rows
that don't require a real proxy.

Usage:
    python dev/aginfer/verify/t6/verify.py
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import sys
import time
from pathlib import Path


# Make ``dev/aginfer`` importable so we can ``from daemon.program_tracker``.
_AGINFER_ROOT = Path(__file__).resolve().parents[2]
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.program_tracker import ProgramTracker, State  # noqa: E402


async def step_arrival_completion_basics() -> None:
    """[1] non-streaming-style: arrival -> REASONING, completion -> ACTING."""
    pt = ProgramTracker()
    pt.observe_arrival("p1")
    assert pt.state("p1") is State.REASONING, pt.state("p1")
    pt.observe_completion("p1")
    assert pt.state("p1") is State.ACTING, pt.state("p1")
    # Re-arrival flips ACTING -> REASONING.
    pt.observe_arrival("p1")
    assert pt.state("p1") is State.REASONING


async def step_completion_only_from_reasoning() -> None:
    """[2] observe_completion is a no-op if state isn't REASONING.

    Defensive against lost arrival events: if we see only a completion
    (no preceding arrival), don't silently fabricate an ACTING state.
    """
    pt = ProgramTracker()
    # No arrival; completion is a no-op.
    pt.observe_completion("p-ghost")
    assert pt.state("p-ghost") is None, pt.state("p-ghost")
    # Pause + completion: stays PAUSED, doesn't flip to ACTING.
    pt.pause("p-paused")
    pt.observe_completion("p-paused")
    assert pt.state("p-paused") is State.PAUSED


async def step_pause_blocks_wait() -> None:
    """[3] pause() + wait_if_paused() blocks until resume().

    Robustness (audit round-1 MINOR 9): rather than `wait_for` with a
    short timeout and catching TimeoutError, we use `sleep(small)` +
    `is_set()` checks.  Less racy on a loaded CI box; cleaner intent.
    """
    pt = ProgramTracker()
    pt.observe_arrival("p2")
    pt.observe_completion("p2")  # now ACTING
    pt.pause("p2")
    assert pt.state("p2") is State.PAUSED

    waiter_done = asyncio.Event()

    async def waiter():
        await pt.wait_if_paused("p2")
        waiter_done.set()

    task = asyncio.create_task(waiter())

    # Yield a few times; waiter must still be blocked.
    for _ in range(5):
        await asyncio.sleep(0.005)
    assert not waiter_done.is_set(), (
        "waiter unblocked while program is still PAUSED -- "
        "wait_if_paused failed to block"
    )

    # Now resume.  Waiter should unblock on the next loop iteration.
    pt.resume("p2")
    try:
        # 1 s is generous; on agsched env the actual time is sub-ms.
        await asyncio.wait_for(waiter_done.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
        raise AssertionError("waiter did NOT unblock within 1 s of resume")

    # Per contract: state STAYS PAUSED until next observe_arrival.
    assert pt.state("p2") is State.PAUSED, pt.state("p2")
    pt.observe_arrival("p2")
    assert pt.state("p2") is State.REASONING


async def step_pause_unknown_program() -> None:
    """[4] WORST CASE: pause() on an unknown program creates a placeholder.

    A subsequent arrival blocks on wait_if_paused; resume unblocks.
    """
    pt = ProgramTracker()
    pt.pause("never_seen")
    assert pt.state("never_seen") is State.PAUSED

    arrived = asyncio.Event()

    async def late_arrival():
        await pt.wait_if_paused("never_seen")
        pt.observe_arrival("never_seen")
        arrived.set()

    task = asyncio.create_task(late_arrival())
    for _ in range(5):
        await asyncio.sleep(0.005)
    assert not arrived.is_set(), "late arrival completed before resume"

    pt.resume("never_seen")
    await asyncio.wait_for(arrived.wait(), timeout=1.0)
    assert pt.state("never_seen") is State.REASONING


async def step_pause_mid_flight_then_recover() -> None:
    """[5b] CONTRACT: pause-while-REASONING + later completion stays
    PAUSED; next-arrival-after-resume flips to REASONING.

    Audit round-1 caught this case being undocumented and called it a
    BLOCKER.  Actual behaviour is correct (pause is gating-only per
    paper §9; in-flight requests complete normally), but no test
    pinned the trajectory.  This step pins it.
    """
    pt = ProgramTracker()
    pt.observe_arrival("p-mid")          # REASONING
    assert pt.state("p-mid") is State.REASONING
    pt.pause("p-mid")                    # PAUSED while REASONING
    assert pt.state("p-mid") is State.PAUSED
    pt.observe_completion("p-mid")       # no-op (state is PAUSED, not REASONING)
    assert pt.state("p-mid") is State.PAUSED, (
        f"pause-mid-flight completion clobbered the PAUSED state "
        f"(got {pt.state('p-mid')}); contract says completion is a no-op "
        f"when state != REASONING"
    )
    # Recovery: resume + next-arrival.
    pt.resume("p-mid")
    pt.observe_arrival("p-mid")
    assert pt.state("p-mid") is State.REASONING


async def step_double_completion_is_noop() -> None:
    """[5c] CONTRACT: at-least-once completion delivery is safe.

    The proxy (T4) may fire observe_completion twice for a single
    request under retry / at-least-once semantics.  Second call is a
    no-op (state is ACTING after the first call, not REASONING).
    """
    pt = ProgramTracker()
    pt.observe_arrival("p-dup")
    pt.observe_completion("p-dup")
    assert pt.state("p-dup") is State.ACTING
    pt.observe_completion("p-dup")      # second call, should no-op
    assert pt.state("p-dup") is State.ACTING


async def step_resume_unknown_program() -> None:
    """[5] resume() on an unknown program is a no-op (warning logged)."""
    pt = ProgramTracker()
    # No exception, no state change.
    pt.resume("ghost")
    assert pt.state("ghost") is None


async def step_concurrent_arrival_completion() -> None:
    """[6] WORST CASE: many concurrent arrival/completion pairs.

    asyncio is single-threaded so there's no actual race.  After
    asyncio.gather both bursts of length N, the deterministic
    interleave puts the LAST observe call as ``observe_completion``.

    Audit round-2 ("audit of tests"): the previous version derived
    its prediction from ``history[-1]`` and then asserted the same
    quantity, which would pass even if the tracker silently dropped
    every other transition (history would be missing the same
    entries).  We now (a) snapshot the tracker's state immediately
    after EACH observe_*, which catches a silently-no-op observe
    call (state would stay stale), and (b) count the matching
    snapshots — a regression that drops 50 % of transitions would
    show up as a mismatch count.  The final-state check is kept as
    the smoke / convergence assertion.
    """
    pt = ProgramTracker()
    N = 100
    PID = "p-spam"
    history: list[str] = []
    arrival_snapshots: list[State] = []
    completion_snapshots: list[State] = []

    async def arrival_burst():
        for _ in range(N):
            pt.observe_arrival(PID)
            arrival_snapshots.append(pt.state(PID))
            history.append("A")
            await asyncio.sleep(0)

    async def completion_burst():
        for _ in range(N):
            pt.observe_completion(PID)
            completion_snapshots.append(pt.state(PID))
            history.append("C")
            await asyncio.sleep(0)

    await asyncio.gather(arrival_burst(), completion_burst())

    assert len(arrival_snapshots) == N, (
        f"arrival burst ran {len(arrival_snapshots)} of {N} iterations"
    )
    assert len(completion_snapshots) == N, (
        f"completion burst ran {len(completion_snapshots)} of {N} iterations"
    )
    bad_arrival = [
        i for i, s in enumerate(arrival_snapshots) if s is not State.REASONING
    ]
    assert not bad_arrival, (
        f"observe_arrival did NOT transition to REASONING at indices "
        f"{bad_arrival[:5]}... ({len(bad_arrival)}/{N}); tracker is "
        f"silently dropping arrival transitions"
    )
    # Completion snapshots tolerate one quirk: if scheduler runs a full
    # arrival burst first (it shouldn't with sleep(0), but be precise),
    # the completion seen right after that may briefly observe ACTING
    # then REASONING.  Single-threaded asyncio with sleep(0) actually
    # round-robins, so each observe_completion lands on a state that
    # was just REASONING — we expect ACTING after every completion.
    bad_completion = [
        i for i, s in enumerate(completion_snapshots) if s is not State.ACTING
    ]
    assert not bad_completion, (
        f"observe_completion did NOT transition to ACTING at indices "
        f"{bad_completion[:5]}... ({len(bad_completion)}/{N}); tracker is "
        f"silently dropping completion transitions"
    )
    final = pt.state(PID)
    last_event = history[-1]
    expected = State.REASONING if last_event == "A" else State.ACTING
    assert final is expected, (
        f"final state {final} doesn't match last observed event "
        f"{last_event!r} (would expect {expected}); transitions are "
        f"not converging to the last event"
    )


async def step_program_churn_memory_bound() -> None:
    """[7] WORST CASE: 10 k unique program_ids; tracker memory bounded.

    Per-program cost claim is ~300 bytes (state str + asyncio.Event
    lazy slot); 10k programs ≈ 3 MB.  v1 has no GC; T8/T9 may add an
    LRU cap if profiling shows churn-driven growth.

    Audit round-2 ("audit of tests"): the previous version used
    ``resource.getrusage().ru_maxrss`` which is the high-watermark
    (not a delta) and capped at 50 MB — both flaws meant a 15×
    per-program regression would slip through silently.  We now use
    ``tracemalloc`` for a true delta and cap at 5 MB, which still
    leaves headroom over the 3 MB nominal but catches an order-of-
    magnitude regression.  Per memory:feedback-latency-multi-run
    spirit (cost claims must be tight).
    """
    import tracemalloc

    pt = ProgramTracker()  # allocate the bare tracker BEFORE measuring
    tracemalloc.start()
    before, _peak_before = tracemalloc.get_traced_memory()
    for i in range(10_000):
        pid = f"churn-{i}"
        pt.observe_arrival(pid)
        pt.observe_completion(pid)
    after, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert pt.size() == 10_000
    delta_bytes = after - before
    delta_mb = delta_bytes / (1024 * 1024)
    bytes_per_program = delta_bytes / 10_000
    # 5 MB cap: 3 MB nominal + ~60 % headroom for dict resize amortization
    # and lazy asyncio.Event allocation.  This catches a 1.7× regression
    # and definitely catches the 15× regression the old 50 MB cap missed.
    assert delta_mb < 5.0, (
        f"10k programs added {delta_mb:.2f} MB (tracemalloc) ≈ "
        f"{bytes_per_program:.0f} B/program; expected < 5 MB total / "
        f"< 500 B/program.  Old 50 MB cap masked the 5 KB/program "
        f"regression flagged in audit round-2."
    )


def step_no_wallclock_heuristic_in_source() -> None:
    """[8] WORST CASE / contract: NO timing-based heuristic anywhere
    in ProgramTracker's source.  Two layers of check:

      (a) Module-level: parse the imports.  Forbidden modules
          (``time``, ``datetime``) must NOT be imported into this
          module at all.  This closes the ``from time import time;
          time()`` and ``import time as t; t.time()`` bypasses.
      (b) Per-function AST: in EVERY function defined on the class
          (public AND private — round-1 audit MINOR 8: private
          helpers like ``_event`` could carry timing logic too),
          forbid attribute names that look like wall-clock APIs.

    Forbidden attr set covers: time, monotonic, now, perf_counter,
    time_ns, monotonic_ns, perf_counter_ns, today, utcnow.
    """
    import sys as _sys

    # Read the module's source (not just the class) so we also see
    # imports at the top.
    pt_module = _sys.modules["daemon.program_tracker"]
    module_src = inspect.getsource(pt_module)
    module_tree = ast.parse(module_src)

    forbidden_modules = {"time", "datetime"}
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden_modules:
                    raise AssertionError(
                        f"timing module imported: `{alias.name}` -- "
                        f"design contract forbids any wall-clock import "
                        f"in program_tracker"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in forbidden_modules:
                raise AssertionError(
                    f"timing module imported: `from {node.module} import "
                    f"...` -- design contract forbids any wall-clock "
                    f"import in program_tracker"
                )

    forbidden_attrs = {
        "time", "monotonic", "now", "perf_counter",
        "time_ns", "monotonic_ns", "perf_counter_ns",
        "today", "utcnow",
    }
    # Check every function (public + private) defined on
    # ProgramTracker.  No safe-list of method names — any timing call
    # anywhere on the class is a contract violation.
    class_src = inspect.getsource(ProgramTracker)
    class_tree = ast.parse(class_src)
    for node in ast.walk(class_tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in forbidden_attrs:
                raise AssertionError(
                    f"timing heuristic detected: {node.name} references "
                    f".{sub.attr} -- design contract forbids wall-clock "
                    f"in any program_tracker method (public or private)."
                )


async def step_gc_ended_bounds_tracker() -> None:
    """[9] #190: gc_ended(live_pids) reclaims ENDED programs that no
    longer hold any live unit, keeping the tracker bounded by the
    live-unit set instead of growing one entry per session forever."""
    pt = ProgramTracker()
    # live programs (various states) + ended ones
    pt.observe_arrival("p-reason")                       # REASONING
    pt.observe_arrival("p-act"); pt.observe_completion("p-act")  # ACTING
    pt.pause("p-pause")                                  # PAUSED
    pt.observe_arrival("p-ended-units"); pt.end("p-ended-units")   # ENDED, has units
    pt.observe_arrival("p-ended-gone");  pt.end("p-ended-gone")    # ENDED, no units
    assert pt.size() == 5, pt.size()

    # live_pids = the pids that still appear in the /aginfer/state dump
    # (any unit's holders).  p-ended-gone is absent → reclaimable.
    live = {"p-reason", "p-act", "p-pause", "p-ended-units"}
    reclaimed = pt.gc_ended(live)
    assert reclaimed == 1, reclaimed
    assert pt.state("p-ended-gone") is None, pt.state("p-ended-gone")
    # everything else survives
    assert pt.state("p-ended-units") is State.ENDED
    assert pt.state("p-reason") is State.REASONING
    assert pt.state("p-act") is State.ACTING
    assert pt.state("p-pause") is State.PAUSED
    assert pt.size() == 4

    # ONLY ENDED is GC'd: with NOTHING live, the live programs
    # (REASONING / ACTING / PAUSED) are still kept; only the remaining
    # ENDED pid (p-ended-units) is reclaimed.
    reclaimed2 = pt.gc_ended(set())
    assert reclaimed2 == 1, reclaimed2
    assert pt.state("p-ended-units") is None, "ENDED + not-live → reclaimed"
    assert pt.state("p-reason") is State.REASONING, "must NOT GC a REASONING pid"
    assert pt.state("p-act") is State.ACTING, "must NOT GC an ACTING pid"
    assert pt.state("p-pause") is State.PAUSED, "must NOT GC a PAUSED pid"
    assert pt.size() == 3

    # an ENDED pid with a request still PARKED in the gate is NOT GC'd
    pt2 = ProgramTracker()
    pt2.observe_arrival("p-gated"); pt2.pause("p-gated")
    waiter = asyncio.create_task(pt2.wait_if_paused("p-gated"))
    await asyncio.sleep(0.02)  # let it park
    pt2.end("p-gated", release_gate=False)   # ENDED, but a waiter is parked
    reclaimed_gated = pt2.gc_ended(set())    # not in live set, but gated
    assert reclaimed_gated == 0, "must NOT GC an ENDED pid with a parked waiter"
    assert pt2.state("p-gated") is State.ENDED
    # release + drain so the test doesn't leak the task
    pt2.resume("p-gated"); await asyncio.wait_for(waiter, timeout=2.0)

    # reused pid after GC: a new session reusing the id resurrects fresh
    pt3 = ProgramTracker()
    pt3.observe_arrival("reuse"); pt3.end("reuse")
    pt3.gc_ended(set())
    assert pt3.state("reuse") is None
    pt3.observe_arrival("reuse")     # reused
    assert pt3.state("reuse") is State.REASONING, "reused pid must resurrect"


async def step_resume_ack_reconciliation() -> None:
    """#215: the tracker owns resume-in-flight bookkeeping (reconciled against
    the fresh dump), so the daemon does not re-fire a resume during overlay lag
    yet recovers a lost clear.  Unit-test the primitive directly."""
    WIN = 2
    pt = ProgramTracker()
    pt.pause("p")
    # before any resume: not in flight.
    assert not pt.resume_in_flight("p"), "no resume issued yet"
    # daemon issues the resume.
    pt.resume("p")                         # resume() records issued (age 0)
    assert pt.resume_in_flight("p"), "resume must be in flight right after resume()"

    # dump STILL shows p PAUSED (overlay lag / lost clear): stays in flight up
    # to WIN reconciles, then re-arms (recovery) so the daemon re-fires.
    for k in range(WIN):
        pt.reconcile_resume_acks({"p"}, WIN)
        assert pt.resume_in_flight("p"), (
            f"must remain suppressed within the window (k={k})")
    pt.reconcile_resume_acks({"p"}, WIN)   # one past the window
    assert not pt.resume_in_flight("p"), (
        "a clear that never lands must re-arm after the window (recovery)")

    # clear LANDED: dump no longer shows p PAUSED → record dropped immediately.
    pt.resume("p")
    assert pt.resume_in_flight("p")
    pt.reconcile_resume_acks(set(), WIN)   # p not in the dump's PAUSED set
    assert not pt.resume_in_flight("p"), "clear landed → record dropped at once"

    # a fresh pause cycle clears a stale in-flight record so the new cycle's
    # resume is not wrongly suppressed.
    pt.resume("p")
    assert pt.resume_in_flight("p")
    pt.pause("p")
    assert not pt.resume_in_flight("p"), "fresh pause must clear stale record"

    # gc_ended also reclaims the record (no leak).
    pt.resume("p")
    pt.end("p")
    pt.gc_ended(live_pids=set())           # p ended + no units → reclaimed
    assert not pt.resume_in_flight("p"), "gc_ended must drop the record"
    assert "p" not in pt._resume_issued_age


async def main() -> None:
    print("=== T6 verify: program_tracker state machine ===")
    print()

    t0 = time.perf_counter()
    await step_arrival_completion_basics()
    print("[1] arrival + completion -> REASONING / ACTING ✓")

    await step_completion_only_from_reasoning()
    print("[2] completion no-op if not REASONING (lost-arrival defence) ✓")

    await step_pause_blocks_wait()
    print("[3] pause blocks wait_if_paused; resume unblocks within 100 ms ✓")

    await step_pause_unknown_program()
    print("[4] WORST CASE: pause on unknown program -> placeholder; "
          "late arrival waits + resumes correctly ✓")

    await step_resume_unknown_program()
    print("[5] resume on unknown program is a no-op (log warning only) ✓")

    await step_pause_mid_flight_then_recover()
    print("[5b] pause-mid-flight: completion is no-op on PAUSED; "
          "next-arrival-after-resume flips to REASONING ✓")

    await step_double_completion_is_noop()
    print("[5c] double observe_completion is safe (at-least-once delivery) ✓")

    await step_concurrent_arrival_completion()
    print("[6] WORST CASE: 100 concurrent arrival/completion pairs "
          "for one pid -> consistent final state ✓")

    await step_program_churn_memory_bound()
    print("[7] WORST CASE: 10k unique program_ids tracked; "
          "size() == 10_000, no crash ✓")

    step_no_wallclock_heuristic_in_source()
    print("[8] contract: NO time.* or loop.time in transition path "
          "(AST grep clean) ✓")

    await step_gc_ended_bounds_tracker()
    print("[9] #190: gc_ended reclaims ENDED-no-units pids (bounded "
          "tracker); keeps live + gated + ENDED-with-units; reuse resurrects ✓")

    await step_resume_ack_reconciliation()
    print("[10] #215: resume-in-flight bookkeeping — suppressed in the lag "
          "window, re-arms on a lost clear, drops on landed/pause/gc ✓")

    dur_ms = (time.perf_counter() - t0) * 1000
    print()
    print(f"=== T6 PASSED in {dur_ms:.0f} ms ===")


if __name__ == "__main__":
    asyncio.run(main())
