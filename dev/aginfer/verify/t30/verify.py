"""T30 + T39 — proxy-gate disconnect awareness (#183, DESIGN §10 F1).

A request held in the proxy gate (program PAUSED) awaits BOTH the
gate condition AND the client's TCP disconnect; whichever fires
first wins.  On disconnect:
  * the proxy releases the gated request locally → HTTP 499,
  * `tracker.client_disconnected(p)` transitions p to ENDED,
  * the proxy enqueues PUT /aginfer/program_paused {ENDED},
  * p's residence is reaped at the next state-dump.

Reuses the #185 machinery (`end()` + the gate verdict +
`enqueue_program_paused`); the new piece is the gate-vs-disconnect
race (`_gate_or_disconnect`) and `client_disconnected`.

Stages (12):

  A. _gate_or_disconnect race (the F1 core, stub-driven)
    A0 gate resolves True first → "proceed"
    A1 gate resolves False first → "ended" (F5 SESSION_END while gated)
    A2 disconnect resolves first → "disconnect"
    A3 the LOSER task is cancelled (no leaked / dangling task)
    A4 gate-already-resolved (non-gated fast path) → "proceed"
       without waiting on the disconnect poll

  B. tracker.client_disconnected
    B0 client_disconnected(PAUSED) → ENDED, returns prior PAUSED
    B1 client_disconnected releases a PARKED waiter with the 499
       verdict (wait_if_paused returns False)
    B2 client_disconnected(unknown) → ENDED, returns None
    B3 emits the distinct client_disconnected metric (vs end())

  C. Proxy integration (verdict → action mapping)
    C0 disconnect verdict → client_disconnected + PUT {ENDED}
       enqueued + 499 (real create_app proxy, stubbed gate race)
    C1 ended verdict → 499, NO end() re-call, NO extra PUT
    C2 proceed verdict → forwards to sglang (no 499)
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

from daemon.program_tracker import ProgramTracker, State  # noqa: E402
from daemon.outbound import OutboundQueue  # noqa: E402
from daemon import proxy as proxy_mod  # noqa: E402
from daemon.proxy import _gate_or_disconnect  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


class _DummyHttp:
    async def post(self, *a, **k): return _Resp()
    async def request(self, *a, **k): return _Resp()
    async def aclose(self): return None


class _Resp:
    """httpx-like response for the proxy's unary forward path."""
    status_code = 200
    text = ""
    content = b'{"ok": true}'
    headers = {"content-type": "application/json"}
    def json(self): return {"ok": True}


def _new_outbound() -> OutboundQueue:
    return OutboundQueue(sglang_base_url="http://unused", http_client=_DummyHttp())


# ============================================================ A. race


async def _resolves_to(value, delay=0.0):
    if delay:
        await asyncio.sleep(delay)
    return value


async def _never():
    # Blocks forever (until cancelled).
    await asyncio.Event().wait()


def stage_a0_gate_true_proceed() -> None:
    async def _go():
        return await _gate_or_disconnect(_resolves_to(True), _never())
    if asyncio.run(_go()) != "proceed":
        raise StageFail("gate True should yield 'proceed'")


def stage_a1_gate_false_ended() -> None:
    async def _go():
        return await _gate_or_disconnect(_resolves_to(False), _never())
    if asyncio.run(_go()) != "ended":
        raise StageFail("gate False should yield 'ended'")


def stage_a2_disconnect_wins() -> None:
    async def _go():
        # Gate blocks forever; disconnect resolves quickly.
        return await _gate_or_disconnect(_never(), _resolves_to(None, 0.01))
    if asyncio.run(_go()) != "disconnect":
        raise StageFail("disconnect-first should yield 'disconnect'")


def stage_a3_loser_cancelled() -> None:
    """The losing awaitable must be cancelled so it doesn't leak as a
    pending task / 'exception never retrieved' warning."""
    async def _go():
        # Track whether the gate coroutine gets cancelled when
        # disconnect wins.
        cancelled = {"gate": False}

        async def _gate_blocks():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled["gate"] = True
                raise

        verdict = await _gate_or_disconnect(
            _gate_blocks(), _resolves_to(None, 0.01),
        )
        # Give the cancelled task a tick to run its except.
        await asyncio.sleep(0.02)
        return verdict, cancelled["gate"]
    verdict, gate_cancelled = asyncio.run(_go())
    if verdict != "disconnect":
        raise StageFail(f"verdict: {verdict}")
    if not gate_cancelled:
        raise StageFail("losing gate task should be cancelled")


def stage_a4_non_gated_fast_proceed() -> None:
    """A non-gated request: gate resolves True immediately; the race
    returns 'proceed' without waiting the disconnect poll interval."""
    import time
    async def _go():
        t0 = time.perf_counter()
        v = await _gate_or_disconnect(_resolves_to(True), _until_never_disc())
        return v, (time.perf_counter() - t0) * 1000.0
    # disconnect side polls but never returns; gate True wins fast.
    async def _until_never_disc():
        while True:
            await asyncio.sleep(proxy_mod._DISCONNECT_POLL_S)
    v, ms = asyncio.run(_go())
    if v != "proceed":
        raise StageFail(f"non-gated should proceed; got {v}")
    if ms > 50.0:
        raise StageFail(
            f"non-gated race should resolve fast (<50ms); took {ms:.1f}ms"
        )


# ============================================================ B. client_disconnected


def stage_b0_disconnect_paused_to_ended() -> None:
    t = ProgramTracker()
    t.observe_arrival("p")
    t.pause("p")
    prev = t.client_disconnected("p")
    if prev is not State.PAUSED:
        raise StageFail(f"prev should be PAUSED; got {prev}")
    if t.state("p") is not State.ENDED:
        raise StageFail(f"state should be ENDED; got {t.state('p')}")


def stage_b1_releases_parked_waiter_with_499() -> None:
    async def _go():
        t = ProgramTracker()
        t.observe_arrival("p")
        t.pause("p")
        waiter = asyncio.create_task(t.wait_if_paused("p"))
        await asyncio.sleep(0.02)
        if waiter.done():
            raise StageFail("waiter should block while PAUSED")
        t.client_disconnected("p")
        return await asyncio.wait_for(waiter, timeout=2.0)
    proceed = asyncio.run(_go())
    if proceed is not False:
        raise StageFail(
            f"disconnect should release the parked waiter with the "
            f"499 verdict (False); got {proceed}"
        )


def stage_b2_disconnect_unknown() -> None:
    t = ProgramTracker()
    prev = t.client_disconnected("ghost")
    if prev is not None:
        raise StageFail(f"prev for unknown should be None; got {prev}")
    if t.state("ghost") is not State.ENDED:
        raise StageFail("unknown should transition to ENDED")


def stage_b3_emits_distinct_metric() -> None:
    """client_disconnected emits a 'client_disconnected' metric line
    (distinct from end()'s program_state) so ops can tell disconnect-
    driven ENDs from harbor SESSION_END."""
    import daemon._metrics as _metrics
    captured: List[Tuple[str, dict]] = []
    orig = _metrics.m

    def _spy(event, **kw):
        captured.append((event, kw))
        return orig(event, **kw)

    _metrics.m = _spy
    try:
        t = ProgramTracker()
        t.observe_arrival("p")
        t.pause("p")
        t.client_disconnected("p")
    finally:
        _metrics.m = orig
    kinds = [e for e, _ in captured]
    if "client_disconnected" not in kinds:
        raise StageFail(
            f"expected a 'client_disconnected' metric; got {kinds}"
        )


# ============================================================ C. proxy integration


def _make_app(outbound: Optional[OutboundQueue]):
    from daemon.proxy import create_app
    app = create_app(
        sglang_base_url="http://unused", enable_event_router=False,
    )
    if outbound is not None:
        app.state.outbound = outbound
    return app


def _get_chat_handler(app):
    """Pull the chat_completions route handler off the app."""
    for r in app.routes:
        if getattr(r, "path", None) == "/v1/chat/completions":
            return r.endpoint
    raise StageFail("chat_completions route not found")


class _FakeRequest:
    """Minimal Request stand-in for the chat handler: a JSON body +
    a controllable is_disconnected()."""
    def __init__(self, body: dict, disconnected: bool = False):
        self._body = body
        self._disc = disconnected
        self.headers = {}
    async def json(self):
        return self._body
    async def is_disconnected(self):
        return self._disc


def stage_c0_disconnect_path() -> None:
    """Real create_app proxy + a request whose client is already
    disconnected → handler returns 499, transitions ENDED, enqueues
    PUT {ENDED}."""
    async def _go():
        ob = _new_outbound()
        app = _make_app(ob)
        tracker: ProgramTracker = app.state.program_tracker
        tracker.observe_arrival("pc0")
        tracker.pause("pc0")  # gated
        handler = _get_chat_handler(app)
        req = _FakeRequest(
            {"program_id": "pc0", "messages": []}, disconnected=True,
        )
        resp = await handler(req, x_aginfer_program=None)
        return resp, tracker, ob
    resp, tracker, ob = asyncio.run(_go())
    if getattr(resp, "status_code", None) != 499:
        raise StageFail(f"disconnect should yield 499; got {resp}")
    if tracker.state("pc0") is not State.ENDED:
        raise StageFail(
            f"disconnect should transition ENDED; got {tracker.state('pc0')}"
        )
    if ob.queue.qsize() != 1:
        raise StageFail(
            f"disconnect should enqueue one PUT; queue={ob.queue.qsize()}"
        )
    batch = ob.queue.get_nowait()
    if batch.endpoint != "program_paused" or batch.method != "PUT":
        raise StageFail(f"wrong batch: {batch.endpoint}/{batch.method}")
    if batch.body.get("state") != "ENDED":
        raise StageFail(f"PUT state should be ENDED: {batch.body!r}")


def stage_c1_ended_verdict_no_extra_put() -> None:
    """A SESSION_END-while-gated (ended verdict) → 499, but the proxy
    does NOT call client_disconnected / enqueue an extra PUT (the
    SESSION_END handler already did the PUT)."""
    async def _go():
        ob = _new_outbound()
        app = _make_app(ob)
        tracker: ProgramTracker = app.state.program_tracker
        tracker.observe_arrival("pc1")
        tracker.pause("pc1")
        handler = _get_chat_handler(app)
        # Park the request, then SESSION_END it from outside.
        req = _FakeRequest({"program_id": "pc1", "messages": []})
        task = asyncio.create_task(handler(req, x_aginfer_program=None))
        await asyncio.sleep(0.02)
        tracker.end("pc1")  # F5 SESSION_END while parked
        resp = await asyncio.wait_for(task, timeout=2.0)
        return resp, ob
    resp, ob = asyncio.run(_go())
    if getattr(resp, "status_code", None) != 499:
        raise StageFail(f"ended-while-gated should 499; got {resp}")
    # The proxy must NOT enqueue a PUT on the 'ended' path (the
    # SESSION_END handler owns that).
    if ob.queue.qsize() != 0:
        raise StageFail(
            f"'ended' verdict should NOT enqueue a PUT from the proxy; "
            f"queue={ob.queue.qsize()}"
        )


def stage_c2_proceed_forwards() -> None:
    """A non-gated request proceeds past the gate (no 499).  We stop
    at the forward step by pointing at a dead sglang — the contract
    here is just 'the gate let it through' (status != 499)."""
    async def _go():
        ob = _new_outbound()
        app = _make_app(ob)
        # Inject a stub http client that returns 200 so the forward
        # path completes without a real sglang.
        app.state.http_client = _DummyHttp()
        handler = _get_chat_handler(app)
        req = _FakeRequest({"program_id": "pc2", "messages": []})
        resp = await handler(req, x_aginfer_program=None)
        return resp, ob
    resp, ob = asyncio.run(_go())
    if getattr(resp, "status_code", None) == 499:
        raise StageFail("non-gated request must not be 499'd")
    # No disconnect PUT for a clean proceed.
    if ob.queue.qsize() != 0:
        raise StageFail(
            f"proceed path should not enqueue a disconnect PUT; "
            f"queue={ob.queue.qsize()}"
        )


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 race: gate True → proceed",                stage_a0_gate_true_proceed),
    ("A1 race: gate False → ended",                 stage_a1_gate_false_ended),
    ("A2 race: disconnect first → disconnect",      stage_a2_disconnect_wins),
    ("A3 race: losing task cancelled",              stage_a3_loser_cancelled),
    ("A4 race: non-gated fast-proceed (<50ms)",     stage_a4_non_gated_fast_proceed),
    ("B0 client_disconnected(PAUSED) → ENDED",      stage_b0_disconnect_paused_to_ended),
    ("B1 client_disconnected releases parked waiter (499)",
                                                    stage_b1_releases_parked_waiter_with_499),
    ("B2 client_disconnected(unknown) → ENDED",     stage_b2_disconnect_unknown),
    ("B3 distinct client_disconnected metric",      stage_b3_emits_distinct_metric),
    ("C0 proxy disconnect path → 499 + ENDED + PUT", stage_c0_disconnect_path),
    ("C1 proxy ended verdict → 499, no extra PUT",  stage_c1_ended_verdict_no_extra_put),
    ("C2 proxy proceed → not 499, no PUT",          stage_c2_proceed_forwards),
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
        print(_red(f"\nT30 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT30 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
