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

    pause(p):
        any -> PAUSED.  ``wait_if_paused(p)`` now blocks subsequent
        proxy requests on an asyncio.Event.

    resume(p):
        clears the block (sets the event).  The program's state STAYS
        PAUSED until the next ``observe_arrival(p)`` flips it to
        REASONING.  This matches the design contract that resume +
        next-arrival together constitute "back to normal".

The tracker is async-safe by virtue of asyncio being single-threaded:
all transitions happen in the same event loop and dict mutations are
atomic.  No locks needed.
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
        self._states[pid] = State.REASONING

    def observe_completion(self, pid: str) -> None:
        """Mark the response stream end / unary completion for ``pid``.

        Only transitions REASONING -> ACTING; other source states are
        left untouched.  This makes the tracker resilient to a lost
        arrival event (the program stays in whatever state observed_*
        last put it in, rather than silently flipping to ACTING).
        """
        if self._states.get(pid) == State.REASONING:
            self._states[pid] = State.ACTING

    # ---- admission_controller hooks ----

    def pause(self, pid: str) -> None:
        """Pin ``pid`` to PAUSED.  Subsequent ``wait_if_paused(pid)``
        calls will block on an asyncio.Event until ``resume(pid)``.

        Pausing an unknown program is allowed; it creates a placeholder
        PAUSED entry so a later arrival blocks correctly (worst-case
        row in T6 README).
        """
        self._states[pid] = State.PAUSED
        self._event(pid).clear()
        logger.info("program_tracker: paused %s", pid)

    def resume(self, pid: str) -> None:
        """Release any waiter blocked on ``wait_if_paused(pid)``.

        The state STAYS PAUSED until the next ``observe_arrival(pid)``
        flips it to REASONING.  Calling resume on an unknown program is
        a no-op apart from creating a (set) placeholder event so a
        later pause works correctly.
        """
        if pid not in self._states:
            logger.warning(
                "program_tracker: resume(%s) — unknown program; "
                "creating a no-op placeholder",
                pid,
            )
        self._event(pid).set()
        logger.info("program_tracker: resumed %s (state stays %s until next arrival)",
                    pid, self._states.get(pid))

    # ---- proxy hook ----

    async def wait_if_paused(self, pid: str) -> None:
        """Async-block while ``pid`` is paused.  Returns immediately for
        un-paused programs (no event yet, or event is set).
        """
        e = self._events.get(pid)
        if e is not None:
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
