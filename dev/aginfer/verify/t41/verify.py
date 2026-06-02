"""T41 — F5 SESSION_END-for-PAUSED handler (#185, DESIGN §11 F5).

When harbor signals SESSION_END for a program parked in the proxy
gate (PAUSED), the daemon:
  1. transitions the program to ENDED,
  2. releases the gate so the parked request gets HTTP 499 (client
     closed the session — the in-flight request is implicitly
     cancelled),
  3. enqueues PUT /aginfer/program_paused {state: ENDED} so sglang
     clears the program's per_program_usage state.

This verify covers the daemon-side machinery end-to-end (no GPU):
tracker ENDED state + gate verdict, outbound PUT enqueue, and the
SESSION_END handler wiring.

Stages (13):

  A. ProgramTracker.end() + ENDED state
    A0 end() on REASONING → ENDED, returns prior REASONING
    A1 end() on ACTING → ENDED, returns prior ACTING
    A2 end() on PAUSED → ENDED, returns prior PAUSED
    A3 end() on unknown pid → ENDED, returns None
    A4 end() idempotent: 2nd call returns ENDED, no state churn

  B. Gate verdict (the F5 499 mechanism)
    B0 wait_if_paused returns True for un-paused program
    B1 PAUSED program: end() releases gate; the parked
       wait_if_paused wakes and returns False (→ 499)
    B2 verdict is read-once: a 2nd wait_if_paused after the same
       end() returns True (re-arrival must not be spuriously aborted)
    B3 end() on a NON-paused program does NOT set the 499 verdict
       (a REASONING program ending mid-flight just transitions)

  C. Outbound PUT
    C0 enqueue_program_paused builds {pid, state, pre_pause_state,
       batch_id} body with endpoint=program_paused, method=PUT
    C1 OutboundBatch rejects an invalid method

  D. SESSION_END handler
    D0 handler calls tracker.end(pid) + enqueues the PUT
    D1 handler with no session id → no-op (logs, no enqueue)
    D2 attach_session_end_handler registers on EventKind.SESSION_END
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.events import Event, EventKind  # noqa: E402
from daemon.program_tracker import ProgramTracker, State  # noqa: E402
from daemon.outbound import OutboundBatch, OutboundQueue  # noqa: E402
from daemon.event_router import (  # noqa: E402
    make_session_end_handler,
    attach_session_end_handler,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


class _DummyHttp:
    async def post(self, *a, **k): return _Resp()
    async def request(self, *a, **k): return _Resp()
    async def aclose(self): return None


class _Resp:
    status_code = 200
    text = ""
    def json(self): return {}


def _new_outbound() -> OutboundQueue:
    return OutboundQueue(
        sglang_base_url="http://unused", http_client=_DummyHttp(),
    )


# ============================================================ A. end() + ENDED


def stage_a0_end_reasoning() -> None:
    t = ProgramTracker()
    t.observe_arrival("p")  # REASONING
    prev = t.end("p")
    if prev is not State.REASONING:
        raise StageFail(f"prev should be REASONING; got {prev}")
    if t.state("p") is not State.ENDED:
        raise StageFail(f"state should be ENDED; got {t.state('p')}")


def stage_a1_end_acting() -> None:
    t = ProgramTracker()
    t.observe_arrival("p")
    t.observe_completion("p")  # ACTING
    prev = t.end("p")
    if prev is not State.ACTING:
        raise StageFail(f"prev should be ACTING; got {prev}")
    if t.state("p") is not State.ENDED:
        raise StageFail(f"state should be ENDED; got {t.state('p')}")


def stage_a2_end_paused() -> None:
    t = ProgramTracker()
    t.observe_arrival("p")
    t.pause("p")  # PAUSED
    prev = t.end("p")
    if prev is not State.PAUSED:
        raise StageFail(f"prev should be PAUSED; got {prev}")
    if t.state("p") is not State.ENDED:
        raise StageFail(f"state should be ENDED; got {t.state('p')}")


def stage_a3_end_unknown() -> None:
    t = ProgramTracker()
    prev = t.end("never-seen")
    if prev is not None:
        raise StageFail(f"prev for unknown should be None; got {prev}")
    if t.state("never-seen") is not State.ENDED:
        raise StageFail("unknown program should transition to ENDED")


def stage_a4_end_idempotent() -> None:
    t = ProgramTracker()
    t.observe_arrival("p")
    t.end("p")
    prev2 = t.end("p")
    if prev2 is not State.ENDED:
        raise StageFail(f"2nd end() should return ENDED; got {prev2}")
    if t.state("p") is not State.ENDED:
        raise StageFail("state should stay ENDED")


# ============================================================ B. gate verdict


def stage_b0_unpaused_proceeds() -> None:
    async def _go():
        t = ProgramTracker()
        return await t.wait_if_paused("fresh")
    proceed = asyncio.run(_go())
    if proceed is not True:
        raise StageFail(f"un-paused should proceed (True); got {proceed}")


def stage_b1_paused_then_end_aborts() -> None:
    """The F5 mechanism: a request is parked in wait_if_paused on a
    PAUSED program; SESSION_END calls end() which releases the gate;
    the parked wait_if_paused wakes and returns False → proxy 499."""
    async def _go():
        t = ProgramTracker()
        t.observe_arrival("p")
        t.pause("p")
        # Park a waiter.
        waiter = asyncio.create_task(t.wait_if_paused("p"))
        await asyncio.sleep(0.02)  # let it block
        if waiter.done():
            raise StageFail("waiter should be blocked while PAUSED")
        # SESSION_END → end() releases the gate.
        t.end("p")
        proceed = await asyncio.wait_for(waiter, timeout=2.0)
        return proceed
    proceed = asyncio.run(_go())
    if proceed is not False:
        raise StageFail(
            f"ended-while-gated waiter should return False (→499); "
            f"got {proceed}"
        )


def stage_b2_verdict_read_once() -> None:
    """After end() sets the 499 verdict and the parked waiter consumes
    it, a LATER wait_if_paused on the same pid must return True (the
    pid could be re-used for a new session; we must not abort it)."""
    async def _go():
        t = ProgramTracker()
        t.observe_arrival("p")
        t.pause("p")
        t.end("p")
        first = await t.wait_if_paused("p")   # consumes the verdict
        second = await t.wait_if_paused("p")  # must proceed
        return first, second
    first, second = asyncio.run(_go())
    if first is not False:
        raise StageFail(f"first wait should abort; got {first}")
    if second is not True:
        raise StageFail(
            f"second wait must proceed (verdict read-once); got {second}"
        )


def stage_b3_end_non_paused_no_499() -> None:
    """A REASONING program that ends mid-flight just transitions; it
    must NOT set the 499 verdict (no request is parked in the gate —
    the in-flight request is already past wait_if_paused)."""
    async def _go():
        t = ProgramTracker()
        t.observe_arrival("p")  # REASONING, not gated
        t.end("p")
        # A fresh wait (e.g. a re-arrival) should proceed.
        return await t.wait_if_paused("p")
    proceed = asyncio.run(_go())
    if proceed is not True:
        raise StageFail(
            f"ending a non-paused program must not set the 499 "
            f"verdict; got {proceed}"
        )


# ============================================================ C. outbound PUT


def stage_c0_enqueue_program_paused_shape() -> None:
    ob = _new_outbound()
    batch_id = ob.enqueue_program_paused(
        pid="p", state="ENDED", pre_pause_state=None,
    )
    if ob.queue.qsize() != 1:
        raise StageFail(f"queue size: {ob.queue.qsize()}")
    batch = ob.queue.get_nowait()
    if batch.endpoint != "program_paused":
        raise StageFail(f"endpoint: {batch.endpoint}")
    if batch.method != "PUT":
        raise StageFail(f"method should be PUT; got {batch.method}")
    body = batch.body
    if body.get("pid") != "p":
        raise StageFail(f"body pid: {body!r}")
    if body.get("state") != "ENDED":
        raise StageFail(f"body state: {body!r}")
    if "pre_pause_state" not in body:
        raise StageFail(f"body missing pre_pause_state: {body!r}")
    if body.get("batch_id") != batch_id:
        raise StageFail(f"batch_id mismatch: {body!r} vs {batch_id}")


def stage_c1_invalid_method_rejected() -> None:
    try:
        OutboundBatch(
            batch_id="x", endpoint="program_paused", body={},
            enqueue_ts=1.0, method="DELETE",
        )
    except ValueError:
        return
    raise StageFail("OutboundBatch should reject method='DELETE'")


# ============================================================ D. handler


def stage_d0_handler_ends_and_enqueues() -> None:
    async def _go():
        t = ProgramTracker()
        t.observe_arrival("p")
        t.pause("p")
        ob = _new_outbound()
        handler = make_session_end_handler(t, ob)
        await handler(Event(EventKind.SESSION_END, session="p"), router=None)
        return t, ob
    t, ob = asyncio.run(_go())
    if t.state("p") is not State.ENDED:
        raise StageFail(f"handler should end the program; got {t.state('p')}")
    if ob.queue.qsize() != 1:
        raise StageFail(
            f"handler should enqueue one PUT; queue={ob.queue.qsize()}"
        )
    batch = ob.queue.get_nowait()
    if batch.endpoint != "program_paused" or batch.method != "PUT":
        raise StageFail(f"wrong batch: {batch.endpoint}/{batch.method}")
    if batch.body.get("state") != "ENDED":
        raise StageFail(f"PUT state should be ENDED: {batch.body!r}")


def stage_d1_handler_no_session_noop() -> None:
    async def _go():
        t = ProgramTracker()
        ob = _new_outbound()
        handler = make_session_end_handler(t, ob)
        await handler(Event(EventKind.SESSION_END, session=None), router=None)
        return ob
    ob = asyncio.run(_go())
    if ob.queue.qsize() != 0:
        raise StageFail(
            f"no-session SESSION_END should not enqueue; "
            f"queue={ob.queue.qsize()}"
        )


def stage_d2_attach_registers_handler() -> None:
    """attach_session_end_handler must register on
    EventKind.SESSION_END so a daemon-wide SESSION_END routes here."""
    registered: dict = {}

    class _FakeRouter:
        def set_handler(self, kind, fn):
            registered[kind] = fn

    t = ProgramTracker()
    ob = _new_outbound()
    attach_session_end_handler(_FakeRouter(), t, ob)
    if EventKind.SESSION_END not in registered:
        raise StageFail(
            f"SESSION_END handler not registered; got {list(registered)}"
        )
    if not callable(registered[EventKind.SESSION_END]):
        raise StageFail("registered handler not callable")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 end(REASONING) → ENDED, prev=REASONING",   stage_a0_end_reasoning),
    ("A1 end(ACTING) → ENDED, prev=ACTING",         stage_a1_end_acting),
    ("A2 end(PAUSED) → ENDED, prev=PAUSED",         stage_a2_end_paused),
    ("A3 end(unknown) → ENDED, prev=None",          stage_a3_end_unknown),
    ("A4 end() idempotent",                         stage_a4_end_idempotent),
    ("B0 un-paused program proceeds (True)",        stage_b0_unpaused_proceeds),
    ("B1 PAUSED + end() → parked waiter aborts (499)", stage_b1_paused_then_end_aborts),
    ("B2 verdict read-once (re-arrival proceeds)",  stage_b2_verdict_read_once),
    ("B3 end() on non-paused → no 499 verdict",     stage_b3_end_non_paused_no_499),
    ("C0 enqueue_program_paused PUT shape",         stage_c0_enqueue_program_paused_shape),
    ("C1 OutboundBatch rejects invalid method",     stage_c1_invalid_method_rejected),
    ("D0 handler ends program + enqueues PUT",      stage_d0_handler_ends_and_enqueues),
    ("D1 handler no-session → no-op",               stage_d1_handler_no_session_noop),
    ("D2 attach registers on EventKind.SESSION_END", stage_d2_attach_registers_handler),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: "
                  f"unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT41 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT41 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
