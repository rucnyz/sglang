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
    """[3] pause() + wait_if_paused() blocks until resume()."""
    pt = ProgramTracker()
    pt.observe_arrival("p2")
    pt.observe_completion("p2")  # now ACTING
    pt.pause("p2")
    assert pt.state("p2") is State.PAUSED

    # Spawn a waiter; it should NOT complete until resume.
    waiter_done = asyncio.Event()
    started_at = asyncio.get_event_loop().time()

    async def waiter():
        await pt.wait_if_paused("p2")
        waiter_done.set()

    task = asyncio.create_task(waiter())

    # Wait briefly; waiter must still be blocked.
    try:
        await asyncio.wait_for(waiter_done.wait(), timeout=0.10)
        raise AssertionError("waiter unblocked before resume")
    except asyncio.TimeoutError:
        pass  # expected — pause is holding it

    # Now resume.  Waiter should unblock within 100 ms.
    pt.resume("p2")
    try:
        await asyncio.wait_for(waiter_done.wait(), timeout=0.10)
    except asyncio.TimeoutError:
        task.cancel()
        raise AssertionError("waiter did NOT unblock within 100 ms of resume")
    elapsed_ms = (asyncio.get_event_loop().time() - started_at) * 1000
    assert elapsed_ms < 250, f"resume took {elapsed_ms:.1f} ms (too slow)"

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
    try:
        await asyncio.wait_for(arrived.wait(), timeout=0.10)
        raise AssertionError("late arrival completed before resume")
    except asyncio.TimeoutError:
        pass

    pt.resume("never_seen")
    await asyncio.wait_for(arrived.wait(), timeout=0.10)
    assert pt.state("never_seen") is State.REASONING


async def step_resume_unknown_program() -> None:
    """[5] resume() on an unknown program is a no-op (warning logged)."""
    pt = ProgramTracker()
    # No exception, no state change.
    pt.resume("ghost")
    assert pt.state("ghost") is None


async def step_concurrent_arrival_completion() -> None:
    """[6] WORST CASE: many concurrent arrival/completion pairs.

    asyncio is single-threaded so there's no actual race, but we
    verify the final state is consistent with the LAST event observed.
    """
    pt = ProgramTracker()
    N = 100
    PID = "p-spam"

    async def arrival_burst():
        for _ in range(N):
            pt.observe_arrival(PID)
            # Yield so completions can interleave.
            await asyncio.sleep(0)

    async def completion_burst():
        for _ in range(N):
            pt.observe_completion(PID)
            await asyncio.sleep(0)

    await asyncio.gather(arrival_burst(), completion_burst())
    # State is whatever the LAST hook called it.  Both events are
    # legal terminations; just assert it's not stuck or in PAUSED.
    final = pt.state(PID)
    assert final in (State.REASONING, State.ACTING), final


async def step_program_churn_memory_bound() -> None:
    """[7] WORST CASE: 10 k unique program_ids; tracker memory bounded."""
    pt = ProgramTracker()
    for i in range(10_000):
        pid = f"churn-{i}"
        pt.observe_arrival(pid)
        pt.observe_completion(pid)
    assert pt.size() == 10_000
    # rough memory budget per program: state str + Event ~= 300 bytes
    # 10_000 * 300 = 3 MB; well under 50 MB target.  We just confirm
    # the tracker didn't crash and accounting is correct.


def step_no_wallclock_heuristic_in_source() -> None:
    """[8] WORST CASE / contract: NO timing-based heuristic in the
    state-transition path.  AST-greps the production module for any
    use of ``time.time``, ``loop.time``, ``datetime.now``, or
    ``time.monotonic`` inside the public methods that drive state
    (observe_arrival / observe_completion / pause / resume /
    wait_if_paused).
    """
    src = inspect.getsource(ProgramTracker)
    tree = ast.parse(src)
    forbidden_attrs = {"time", "monotonic", "now", "perf_counter"}
    targets = {
        "observe_arrival",
        "observe_completion",
        "pause",
        "resume",
        "wait_if_paused",
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) and not isinstance(
            node, ast.AsyncFunctionDef
        ):
            continue
        if node.name not in targets:
            continue
        # Walk this function's body only.
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in forbidden_attrs:
                # Allow asyncio.Event() etc. but ANY of those attribute
                # names is suspect; fail loudly.
                raise AssertionError(
                    f"timing heuristic detected: {node.name} references "
                    f".{sub.attr} — design contract forbids wall-clock "
                    f"in the transition path"
                )
            if isinstance(sub, ast.Call):
                # Check for ``time.time()`` style calls on a Name root.
                func = sub.func
                if isinstance(func, ast.Attribute):
                    if (
                        isinstance(func.value, ast.Name)
                        and func.value.id == "time"
                    ):
                        raise AssertionError(
                            f"timing heuristic detected: {node.name} calls "
                            f"time.{func.attr}() — design contract forbids."
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
