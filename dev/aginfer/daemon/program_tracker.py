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

Bounded by the live-unit set (#190): ``gc_ended(live_pids)`` reclaims
ENDED programs whose KV has fully cleared from the latest
``/aginfer/state`` snapshot (called by ``kv_scheduler.handle`` each
event).  So ``_states`` tracks (live programs) + (ENDED programs with
residual units) — NOT one entry per session ever seen.  Each unique
program_id occupies ~300 bytes (state string + asyncio.Event).  A
LIVE program that never signals SESSION_END (REASONING/ACTING/PAUSED
forever) is still kept — an LRU/idle cap for that case is future work,
gated on profiling.
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
    # T41 (#185, DESIGN §11 F5 / §4 SESSION_END): terminal state.
    # The client signalled SESSION_END; no more arrivals expected.
    # Its session-scoped units become demote/drop candidates and it
    # contributes 0 to future p_hat.  A program reaches ENDED only
    # via end() (harbor's out-of-band /aginfer/session_end signal).
    ENDED = "ENDED"


class ProgramTracker:
    """Per-program state machine.  See module docstring for the contract."""

    def __init__(self) -> None:
        self._states: Dict[str, State] = {}
        # Per-program asyncio.Event.  ``set()`` means "not blocked".
        # The event is lazily created on first reference; when freshly
        # created it is set (so wait_if_paused returns immediately for
        # untracked programs).
        self._events: Dict[str, asyncio.Event] = {}
        # T41 (#185): pids that were end()'d while a request sat in
        # the gate.  ``wait_if_paused`` checks this on wake so the
        # proxy can respond 499 (client closed) instead of forwarding
        # a request for a session the client already ended.  Consumed
        # (cleared) on read.
        self._ended_while_gated: set = set()
        # T30 (#183): per-pid count of requests CURRENTLY parked in the
        # gate (blocked inside wait_if_paused).  A per-connection
        # disconnect must abort only its OWN request, not force-end the
        # whole program while sibling connections are still parked
        # (the #183 audit "two-request, one-disconnects" bug).  The
        # proxy ends the program on disconnect only when this drops to
        # zero.
        self._gated_count: Dict[str, int] = {}

    # ---- queries ----

    def state(self, pid: str) -> Optional[State]:
        """Return current state, or None if the program has never been
        observed."""
        return self._states.get(pid)

    def has_gated_waiters(self, pid: str) -> bool:
        """T30 (#183): is any request still PARKED in the gate for
        ``pid``?  The proxy checks this on a client disconnect: it
        ends the program only when no sibling connection is parked."""
        return self._gated_count.get(pid, 0) > 0

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
        # T41 (#185): a new arrival starts a fresh session — clear any
        # stale ended-while-gated verdict so this request (and the
        # resurrected pid) is not spuriously 499'd.
        self._ended_while_gated.discard(pid)
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
        # T41 (#185): a new gate cycle — drop any stale end() verdict
        # from a prior cycle so the freshly-parked request is judged
        # on THIS cycle's outcome (resume → proceed, end → 499).
        self._ended_while_gated.discard(pid)
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

    # ---- SESSION_END hook (T41 #185, DESIGN §11 F5) ----

    def end(self, pid: str, *, release_gate: bool = True) -> Optional[State]:
        """Transition ``pid`` to ENDED (terminal) on SESSION_END.

        Returns the PRIOR state (so the caller can branch — the F5
        PAUSED path releases the gate with 499; other prior states
        just transition).

        ``release_gate`` (default True) controls the F5 PAUSED
        behaviour: when the program was PAUSED and ``release_gate``
        is True, ``end()`` marks ``_ended_while_gated`` and sets the
        gate event so the PARKED ``wait_if_paused`` wakes and
        returns the "abort with 499" verdict.

        The disconnect path (T30 #183) passes ``release_gate=False``:
        the disconnecting request already left via the gate-vs-
        disconnect race (it 499s itself), and the proxy only calls
        this when NO sibling is parked — so there is no waiter to
        release.  Skipping the flag also avoids leaking it in
        ``_ended_while_gated`` (the #183 audit leak).

        Idempotent: ending an already-ENDED program is a no-op
        returning ENDED.
        """
        prev = self._states.get(pid)
        if prev is State.ENDED:
            return State.ENDED
        was_paused = prev is State.PAUSED
        self._states[pid] = State.ENDED
        if was_paused and release_gate:
            # A request may be parked in the gate; tell wait_if_paused
            # to abort it (499) rather than forward, then release.
            self._ended_while_gated.add(pid)
            self._event(pid).set()
        from ._metrics import m as _m
        _m(
            "program_state",
            pid=pid,
            from_=prev.value if prev is not None else "NONE",
            to="ENDED",
        )
        logger.info("program_tracker: ended %s (prev=%s)", pid, prev)
        return prev

    def client_disconnected(self, pid: str) -> Optional[State]:
        """T39 (#183, DESIGN §10 F1): the client's TCP connection
        dropped.  The proxy calls this ONLY after the disconnecting
        request has left the gate-vs-disconnect race (it 499s
        itself) AND only when no sibling connection is still parked
        (``has_gated_waiters(pid)`` is False) — so there is no
        waiter to release.

        Transitions the program to ENDED via ``end(release_gate=
        False)``: a per-CONNECTION disconnect must NOT 499 sibling
        connections (the #183 audit "two-request, one-disconnects"
        bug) and must NOT leak the ``_ended_while_gated`` flag with
        no waiter to consume it.  The proxy enqueues the PUT
        /aginfer/program_paused {ENDED} (it owns the outbound queue).

        Returns the prior state.  Distinct metric tag so ops can
        tell client-disconnect-driven ENDs from harbor SESSION_END.
        """
        prev = self.end(pid, release_gate=False)
        from ._metrics import m as _m
        _m(
            "client_disconnected",
            pid=pid,
            prev_state=prev.value if prev is not None else "NONE",
        )
        logger.info("program_tracker: client_disconnected %s (prev=%s)",
                    pid, prev)
        return prev

    def gc_ended(self, live_pids) -> int:
        """#190 (bounded tracker): reclaim ENDED programs that no longer
        hold any live unit.  ``live_pids`` is the set of program_ids
        present in the current ``/aginfer/state`` snapshot (the union of
        every unit's holders).  An ENDED pid absent from it has no
        residual KV and no daemon bookkeeping left, so it is dropped
        from every tracker structure — mirroring sglang's ENDED-no-units
        GC (#186) so ``_states`` stays bounded by the live-unit set
        rather than growing one entry per session forever.

        Only ENDED is GC'd (REASONING / ACTING / PAUSED are LIVE
        programs and are kept even when they hold no radix units yet).
        An ENDED pid with a request still PARKED in the gate
        (``_gated_count > 0``) is kept — there is a waiter to release.

        Safe w.r.t. pid reuse: a new session reusing an old id is
        resurrected to REASONING by its next ``observe_arrival``
        regardless of whether the old entry was GC'd, so GC can never
        strand a live session.  Called by ``kv_scheduler.handle`` each
        event with the live pids from the fresh snapshot.

        Returns the number of pids reclaimed.
        """
        stale = [
            pid for pid, st in self._states.items()
            if st is State.ENDED
            and pid not in live_pids
            and self._gated_count.get(pid, 0) == 0
        ]
        for pid in stale:
            self._states.pop(pid, None)
            self._events.pop(pid, None)
            self._ended_while_gated.discard(pid)
            self._gated_count.pop(pid, None)
        if stale:
            from ._metrics import m as _m
            _m("program_gc_ended", reclaimed=len(stale))
        return len(stale)

    # ---- proxy hook ----

    async def wait_if_paused(self, pid: str) -> bool:
        """Async-block while ``pid`` is paused.  Returns ``True`` to
        proceed (forward the request) or ``False`` if the program was
        ENDED while THIS request was PARKED in the gate (T41 #185 F5:
        the proxy responds 499 — the client closed the session).

        The 499 verdict is honoured ONLY for requests that actually
        BLOCKED (the gate event was clear on entry and we suspended
        on ``e.wait()``).  A request that did NOT block — un-paused
        fast path, or an arrival AFTER end() already set the event —
        is treated as a fresh request and PROCEEDS.  Rationale
        (#185 audit):

          * Two concurrent requests for the same pid both parked
            when SESSION_END fires: BOTH blocked, BOTH wake, BOTH
            see the ended-while-gated flag → BOTH get 499.  (The
            old read-once flag let the 2nd proceed — a request for
            an ended session leaking to sglang.)
          * A request arriving AFTER SESSION_END is a new session
            reusing the pid (``observe_arrival`` resurrects
            ENDED→REASONING); it never parked, so it proceeds.

        The flag is NOT popped on read (so the whole parked cohort
        sees it); it is cleared by the next lifecycle event for the
        pid — ``observe_arrival`` (new session) or ``pause`` (new
        gate cycle).

        Defensive against direct-state-mutation tests: if a caller
        wrote ``_states[pid] = PAUSED`` without going through
        ``pause(pid)`` (which always creates the event), we still
        block; create a cleared event and await it.
        """
        e = self._events.get(pid)
        if e is None:
            if self._states.get(pid) == State.PAUSED:
                e = asyncio.Event()  # default: cleared
                self._events[pid] = e
            else:
                return True  # never gated → proceed
        if e.is_set():
            # Did not block → not a parked request → proceed even if
            # an end() flag lingers (this is a fresh arrival).
            return True
        # Genuinely parked: count this request as gated (T30 #183, so a
        # sibling's disconnect can tell whether the program still has
        # parked requests), suspend, then honour the verdict on wake.
        # The decrement is in ``finally`` so it runs on cancellation
        # too (the gate-vs-disconnect race cancels the loser).
        self._gated_count[pid] = self._gated_count.get(pid, 0) + 1
        try:
            await e.wait()
        finally:
            n = self._gated_count.get(pid, 0) - 1
            if n <= 0:
                self._gated_count.pop(pid, None)
            else:
                self._gated_count[pid] = n
        return pid not in self._ended_while_gated

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
