"""Aginfer daemon program-state tracker (T6).

Per-program state machine that drives admission_controller's pause /
resume decisions (paper §9 deployment architecture).  State transitions
are **only** event-driven — observable HTTP events at the proxy layer.
There is intentionally NO wall-clock heuristic in the code path
(no ``time.time() - last > X`` style guards): the verify
(verify/t6/verify.py) AST-greps this module to enforce that contract.

States (per program_id):

    REASONING  — a request is inflight for the program; the model is
                 producing tokens.
    ACTING     — the model finished; the client is now doing tool
                 calls / side effects / etc.
    PAUSED     — admission_controller pinned the program; subsequent
                 requests block at the proxy until ``resume`` is
                 called.

Transitions:

    observe_arrival(p):
        any -> REASONING.  If p was PAUSED, the caller MUST have first
        awaited ``wait_if_paused(p)`` (resumed by admission_controller).

    observe_completion(p):
        REASONING -> ACTING (only).  Other source states are no-ops
        (defensive: lost arrival event leaves state untouched).

        IMPORTANT — completion on a PAUSED program is a deliberate
        no-op.  This is the "pause-mid-flight" path: a program in
        REASONING is paused while its current request is still being
        decoded.  The pause is gating-only (per paper §9): it blocks
        NEW requests but does NOT abort the in-flight one.  When the
        in-flight request eventually completes, its observe_completion
        intentionally does NOT clobber the PAUSED state.  The program
        recovers on the next arrival after admission_controller's
        resume.

    observe_completion(p) called twice (e.g., at-least-once event
        delivery from the proxy):
        First call flips REASONING -> ACTING.  Second call is a no-op
        (state is ACTING, not REASONING).  Safe under at-least-once
        delivery.

    pause(p):
        any -> PAUSED.  ``wait_if_paused(p)`` now blocks subsequent
        proxy requests on an asyncio.Event.  pause()-while-REASONING
        is allowed: the in-flight request finishes normally (its
        proxy code is already past wait_if_paused), and its
        observe_completion is a no-op (see above).

    resume(p):
        clears the block (sets the event).  Edge-triggered: only
        currently-queued waiters are released; a subsequent pause()
        re-blocks new arrivals.  Resume on an unknown program is a
        no-op apart from logging; it does NOT create a state entry
        (the state map is untouched until the next observe_*).  The
        program's state STAYS PAUSED until the next
        ``observe_arrival(p)`` flips it to REASONING — this matches
        the design contract that resume + next-arrival together
        constitute "back to normal".

The tracker is single-event-loop-safe: all transitions happen in the
same asyncio event loop and dict mutations are atomic.  It is NOT
thread-safe — do NOT call from FastAPI sync handlers running in a
threadpool, or any other off-loop thread; ``asyncio.Event.set()``
from a non-loop thread is undefined.

v1 has NO GC / cap.  Each unique program_id occupies ~300 bytes
(state string + asyncio.Event).  10k programs ≈ 3 MB; 100k ≈ 30 MB.
T8 / T9 may add an LRU cap if profiling shows churn-driven memory
growth in production.
"""
from __future__ import annotations

import asyncio
import enum
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class State(str, enum.Enum):
    REASONING = "REASONING"
    ACTING = "ACTING"
    PAUSED = "PAUSED"


class ProgramTracker:
    """Per-program state machine.  See module docstring for the contract."""

    def __init__(self) -> None:
        self._states: Dict[str, State] = {}
        # Per-program asyncio.Event.  ``set()`` means "not blocked".
        # The event is lazily created on first reference; when freshly
        # created it is set (so wait_if_paused returns immediately for
        # untracked programs).
        self._events: Dict[str, asyncio.Event] = {}

    # ---- queries ----

    def state(self, pid: str) -> Optional[State]:
        """Return current state, or None if the program has never been
        observed."""
        return self._states.get(pid)

    def size(self) -> int:
        """Number of programs the tracker is currently holding."""
        return len(self._states)

    def __contains__(self, pid: str) -> bool:
        return pid in self._states

    # ---- event hooks (HTTP-driven; no timing) ----

    def observe_arrival(self, pid: str) -> None:
        """Mark a request arrival for ``pid``.  Transitions to REASONING.

        Caller is expected to have first awaited ``wait_if_paused(pid)``
        if the program might be PAUSED; otherwise this unconditionally
        overwrites the state (which is correct, because the caller
        successfully got past wait_if_paused).
        """
        prev = self._states.get(pid)
        self._states[pid] = State.REASONING
        from ._metrics import m as _m
        _m(
            "program_state",
            pid=pid,
            from_=prev.value if prev is not None else "NONE",
            to="REASONING",
        )

    def observe_completion(self, pid: str) -> None:
        """Mark the response stream end / unary completion for ``pid``.

        Only transitions REASONING -> ACTING; other source states are
        left untouched.  This makes the tracker resilient to a lost
        arrival event (the program stays in whatever state observed_*
        last put it in, rather than silently flipping to ACTING).
        """
        if self._states.get(pid) == State.REASONING:
            self._states[pid] = State.ACTING
            from ._metrics import m as _m
            _m("program_state", pid=pid, from_="REASONING", to="ACTING")

    # ---- admission_controller hooks ----

    def pause(self, pid: str) -> None:
        """Pin ``pid`` to PAUSED.  Subsequent ``wait_if_paused(pid)``
        calls will block on an asyncio.Event until ``resume(pid)``.

        Pausing an unknown program is allowed; it creates a placeholder
        PAUSED entry so a later arrival blocks correctly (worst-case
        row in T6 README).
        """
        prev = self._states.get(pid)
        self._states[pid] = State.PAUSED
        self._event(pid).clear()
        logger.info("program_tracker: paused %s", pid)
        from ._metrics import m as _m
        _m(
            "program_state",
            pid=pid,
            from_=prev.value if prev is not None else "NONE",
            to="PAUSED",
        )

    def resume(self, pid: str) -> None:
        """Release any waiter blocked on ``wait_if_paused(pid)``.

        The state STAYS PAUSED until the next ``observe_arrival(pid)``
        flips it to REASONING.

        Calling resume on a program with no state entry is a no-op
        apart from logging a warning — the state map is NOT
        modified, only the asyncio.Event for this pid is set so any
        future ``wait_if_paused(pid)`` returns immediately (until a
        later ``pause(pid)`` clears it again).
        """
        if pid not in self._states:
            logger.warning(
                "program_tracker: resume(%s) — unknown program; "
                "no state change, only releasing any future waiters",
                pid,
            )
        self._event(pid).set()
        logger.info("program_tracker: resumed %s (state stays %s until next arrival)",
                    pid, self._states.get(pid))

    # ---- proxy hook ----

    async def wait_if_paused(self, pid: str) -> None:
        """Async-block while ``pid`` is paused.  Returns immediately for
        un-paused programs (no event yet, or event is set).

        Defensive against direct-state-mutation tests: if a caller
        wrote ``_states[pid] = PAUSED`` without going through
        ``pause(pid)`` (which always creates the event), we still want
        to block.  Cheap belt-and-suspenders: if no event exists yet
        but state is PAUSED, create a cleared event and await it.
        """
        e = self._events.get(pid)
        if e is None:
            if self._states.get(pid) == State.PAUSED:
                e = asyncio.Event()  # default: cleared
                self._events[pid] = e
            else:
                return
        await e.wait()

    # ---- internal ----

    def _event(self, pid: str) -> asyncio.Event:
        """Lazily create a per-program asyncio.Event.  New events are
        ``set()`` so default behaviour is "not blocked"."""
        e = self._events.get(pid)
        if e is None:
            e = asyncio.Event()
            e.set()
            self._events[pid] = e
        return e
