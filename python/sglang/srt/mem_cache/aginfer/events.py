"""Paper §4 event types emitted by the aginfer daemon.

The daemon's proxy (T4) emits 6 of the 8 paper §4 event kinds during
the request lifecycle.  sglang pushes the remaining 2
(``memory_pressure`` / ``pressure_resolved``) via a webhook (T5).
All events land on a single ``asyncio.Queue`` and are processed by
the event_worker (T7 / T8) serially with an action_lock.
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class EventKind(str, enum.Enum):
    """The eight paper §4 event kinds.

    Six are emitted by the proxy (T4): SESSION_ARRIVAL, LLM_PREFILL,
    TOOL_CALL_START, TOOL_CALL_END, SUB_DISPATCH_BLOCKING,
    SUB_DISPATCH_ASYNC.

    Two are pushed by sglang via webhook (T5): MEMORY_PRESSURE,
    PRESSURE_RESOLVED.
    """

    # ---- proxy-emitted (T4) ----
    SESSION_ARRIVAL = "session_arrival"
    LLM_PREFILL = "llm_prefill"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    SUB_DISPATCH_BLOCKING = "sub_dispatch_blocking"
    SUB_DISPATCH_ASYNC = "sub_dispatch_async"
    # P2c (in-engine belief plane, EXP_PLAN.md): a dispatched subagent
    # returned to its parent. Not one of the original paper §4 eight kinds
    # (those predate the fan-out/return distinction this project's replay
    # traces carry -- see agentreplay.driver._emit); belief-plane-only, no
    # webhook / decision-set wiring beyond ProgramTracker.observe_arrival on
    # the parent (mirrors TOOL_CALL_END: the parent is about to reason again).
    SUB_RETURN = "sub_return"
    # T41 (#185, DESIGN §4 / §11 F5): harbor signals end-of-session
    # out-of-band (harbor's /aginfer/session_end → daemon
    # /aginfer/event with kind=session_end).  Handler transitions
    # the program to ENDED, releases the proxy gate with 499 if a
    # request was parked (F5 PAUSED branch), and enqueues the
    # PUT /aginfer/program_paused {ENDED} to inform sglang.
    SESSION_END = "session_end"
    # ---- sglang webhook (T5) ----
    MEMORY_PRESSURE = "memory_pressure"
    PRESSURE_RESOLVED = "pressure_resolved"
    # T23+T37 (DESIGN §4 round-9 B4): sglang fires this when a daemon-
    # issued action can't be applied.  Payload {endpoint, action_id,
    # reason, hash?}.  Handler logs + bumps observability counter;
    # the next event's joint_decide re-evaluates.  Idempotent
    # invariants (§10) make any re-issue safe.
    APPLY_FAILED = "apply_failed"
    # T24 (#182, DESIGN §4 + §10 round-15/16/17): sglang fires this
    # when its apply_aginfer_migrations DFS detects two distinct
    # radix nodes mapping to the same hash key.  Probability <10⁻²²
    # at any tree size; if it ever fires it's a deployment-bug
    # class signal.  Handler calls fatal('hash_collision', ...).
    HASH_COLLISION = "hash_collision"


@dataclass(frozen=True, slots=True)
class Event:
    """A single paper §4 event.

    ``session`` is the program_id (paper §3 maps "session" → "program"
    in our terminology).  ``payload`` carries event-specific data
    (e.g. ``child_session_id`` for sub_dispatch events; ``state`` for
    memory_pressure).
    """

    kind: EventKind
    session: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    # T42 — monotonic-clock timestamp stamped by EventBus.emit at the
    # moment the event is put on the queue.  The worker reads this at
    # dispatch entry to compute time-in-queue.  Default 0.0 is the
    # "unstamped" sentinel that triggers emit() to fill it in.
    enqueue_time: float = 0.0


class EventBus:
    """Thin wrapper around ``asyncio.Queue`` that the proxy calls into.

    Lives as a singleton on the daemon's FastAPI app state.  v1 has
    no flow control / backpressure; events are queued unconditionally.
    The event_worker (T7 / T8) drains the queue.

    Verification (T4 verify) replaces the bus with a recording stub
    that captures every ``put`` call for invariant checks.
    """

    def __init__(self) -> None:
        self._q: asyncio.Queue[Event] = asyncio.Queue()
        self._known_programs: set[str] = set()

    @property
    def queue(self) -> asyncio.Queue[Event]:
        return self._q

    def mark_known(self, pid: str) -> bool:
        """Idempotent: returns True iff ``pid`` was newly added.

        Used by the proxy to decide whether to emit ``SESSION_ARRIVAL``
        (only on the FIRST request from a program).
        """
        if pid in self._known_programs:
            return False
        self._known_programs.add(pid)
        return True

    def forget(self, pid: str) -> None:
        """Used by the event_worker once the program is fully done
        (paper §9 program lifecycle); v1 doesn't call this yet."""
        self._known_programs.discard(pid)

    def known_size(self) -> int:
        return len(self._known_programs)

    async def emit(self, event: Event) -> None:
        """Best-effort: never blocks the proxy on a full queue.

        v1 queue is unbounded so this is just ``put_nowait``.  If a
        future maintainer adds bounding, this should fall back to
        drop-event-with-warning rather than block the proxy step.

        T42: stamps ``enqueue_time`` if the caller didn't already.
        Since Event is frozen + slots, we replace the instance via
        ``dataclasses.replace`` (caller's reference is unaffected;
        the queued copy carries the timestamp).
        """
        if event.enqueue_time == 0.0:
            event = dataclasses.replace(
                event, enqueue_time=time.perf_counter()
            )
        self._q.put_nowait(event)
