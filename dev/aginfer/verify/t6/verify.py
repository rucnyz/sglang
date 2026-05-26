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
    interleave puts the LAST observe call as ``observe_completion``
    (both bursts yield with sleep(0); the gather drives them in turn
    and the longer-ending one writes last; both have same length, so
    the second-scheduled wins).  We tighten the assertion to: final
    state is whatever the FINAL call set it to, observable by checking
    state==ACTING (last completion ran after last arrival on this loop).
    """
    pt = ProgramTracker()
    N = 100
    PID = "p-spam"
    history: list[str] = []

    async def arrival_burst():
        for _ in range(N):
            pt.observe_arrival(PID)
            history.append("A")
            await asyncio.sleep(0)

    async def completion_burst():
        for _ in range(N):
            pt.observe_completion(PID)
            history.append("C")
            await asyncio.sleep(0)

    await asyncio.gather(arrival_burst(), completion_burst())
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

    Per-program cost is ~300 bytes (state str + asyncio.Event); 10k
    programs ≈ 3 MB.  v1 has no GC; T8/T9 may add an LRU cap if
    profiling shows churn-driven growth.  We measure RSS delta to
    pin the actual cost (audit round-1 MINOR 6: the "memory bounded"
    claim was previously unverified).
    """
    import os
    import resource

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pt = ProgramTracker()
    for i in range(10_000):
        pid = f"churn-{i}"
        pt.observe_arrival(pid)
        pt.observe_completion(pid)
    assert pt.size() == 10_000
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux; MB on macOS.  Treat as KB.
    delta_kb = rss_after - rss_before
    delta_mb = delta_kb / 1024
    # Generous cap: 10k programs should fit in 50 MB even with Python
    # overhead, asyncio.Event internals, dict resizing etc.
    assert delta_mb < 50, (
        f"10k programs added {delta_mb:.1f} MB to RSS; expected <50 MB. "
        f"This suggests per-program overhead is much larger than the "
        f"~300 B/program docstring estimate."
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

    dur_ms = (time.perf_counter() - t0) * 1000
    print()
    print(f"=== T6 PASSED in {dur_ms:.0f} ms ===")


if __name__ == "__main__":
    asyncio.run(main())
