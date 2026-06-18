"""DESIGN §3 — the action-timeline plane (predictive promote, proactive room).

A decided action may be **scheduled for a future moment** when that is when it
should land — e.g. a predictive promote at ``T_start + tool_ETA −
load_back_latency`` (§7) so a tool-bound agent's idle tail is back in HBM exactly
as its resume prefill arrives, paying ~0 load_back on the TTFT critical path.

Two design rules from §3 are load-bearing here and shape the whole module:

1. **The event stream is the clock — not wall-clock, not an asyncio timer.**
   Scheduled actions are dispatched *when the event stream next advances past
   their due time*.  The serialized single-consumer event_worker drains this heap
   on every event using **that event's own ``enqueue_time``** (a monotonic
   ``perf_counter`` stamp) as "now".  Under the pressure where anticipation
   matters the event rate is high (small dispatch jitter); the §4 ``heartbeat_s``
   re-fire under HIGH/CRITICAL gives a floor cadence.  We deliberately do NOT add
   a wall-clock timer task: that would let time *drive* execution outside the one
   serialized consumer and reintroduce exactly the race the "belief is
   event-sourced" rule forbids.

2. **Belief-validated at fire (§3 / §10 idempotency).**  This module only
   *schedules* and *orders* — it never decides whether an action is still valid.
   At dispatch the registered fire-callback re-reads belief (fresh
   ``/aginfer/state`` + program lifecycle) and turns a stale action into an
   idempotent no-op (program already returned/ended, or the tail was meanwhile
   dropped).  So time paces *execution* but never drives *state*.

The heap holds opaque payloads; the daemon (kv_scheduler) defines what a payload
means and how to fire it.  This keeps the timeline a pure ordering primitive with
no daemon-domain knowledge.
"""
from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Any, List

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _HeapItem:
    """Min-heap entry ordered by ``(due_time, seq)``.

    ``seq`` is a monotonic insertion counter that breaks ties deterministically
    and keeps the opaque ``payload`` out of the comparison (payloads are not
    required to be orderable)."""
    due_time: float
    seq: int
    payload: Any = field(compare=False)


class ActionTimeline:
    """A small due-action min-heap drained by the event stream (DESIGN §3).

    Not thread-safe and not asyncio-aware by design: every mutation happens on
    the single serialized event_worker task, under its dispatch lock, so no
    internal locking is needed.  ``schedule`` enqueues; ``pop_due(now)`` returns
    every action whose due time has passed (caller fires them, belief-validated).
    """

    def __init__(self) -> None:
        self._heap: List[_HeapItem] = []
        self._seq: int = 0
        # Observability counters (read by main.py / _observability).
        self.scheduled: int = 0
        self.fired: int = 0          # popped + handed to the fire callback
        self.dropped: int = 0        # explicitly cancelled before firing

    def schedule(self, due_time: float, payload: Any) -> None:
        """Place ``payload`` on the heap to be drained once the event-stream
        clock passes ``due_time`` (same monotonic frame as ``Event.enqueue_time``
        — a ``perf_counter`` stamp).  A ``due_time`` already in the past simply
        fires on the next drain (jitter = one inter-event gap)."""
        self._seq += 1
        heapq.heappush(self._heap, _HeapItem(due_time, self._seq, payload))
        self.scheduled += 1

    def pop_due(self, now: float) -> List[Any]:
        """Pop and return every payload whose ``due_time <= now`` in due order.

        ``now`` is the current event's ``enqueue_time`` — the event stream IS the
        clock (§3).  Returns ``[]`` when nothing is due (the common case), so the
        per-event drain is cheap: one heap-peek when the timeline is idle."""
        due: List[Any] = []
        while self._heap and self._heap[0].due_time <= now:
            item = heapq.heappop(self._heap)
            due.append(item.payload)
        self.fired += len(due)
        return due

    def pending(self) -> int:
        """Number of not-yet-due actions still on the heap (observability)."""
        return len(self._heap)

    def next_due_time(self) -> float:
        """Due time of the earliest pending action, or ``+inf`` if empty.

        Lets the worker note how far ahead the soonest scheduled action is
        without draining it (diagnostics only — the drain is event-clocked)."""
        return self._heap[0].due_time if self._heap else float("inf")


@dataclass(frozen=True)
class PromoteAction:
    """A scheduled predictive-promote (§7): bring ``unit_hashes`` of program
    ``session`` back to HBM, due to fire just before that program's tool returns.

    ``from_tiers`` records the tiers the units were resident in when scheduled
    (the demote target) so the fire-time wire action can REMOVE the right lower
    tiers; the fire callback re-validates current residence regardless, so this
    is a hint, not a trusted fact.  ``eta_s`` / ``load_back_s`` / ``scheduled_at``
    are carried for observability and for the §7 ``T_start + ETA − load_back``
    audit trail."""
    session: str
    unit_hashes: tuple
    from_tiers: tuple = ()
    eta_s: float = 0.0
    load_back_s: float = 0.0
    scheduled_at: float = 0.0
    reason: str = "tool_eta"
