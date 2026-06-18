"""T164 verify — sustained-load escalate-to-fatal (DESIGN §10 sustained tier).

When sglang is down for long enough that the daemon's outbound
queue is genuinely doing damage (memory-eating + no useful work),
the daemon self-kills so the supervisor (systemd/k8s) restarts it.
Crash-only-software pattern — running forever-degraded is not a
valid state.

The criterion is BOTH:
  * `OutboundQueue.consecutive_failures >= escalate_failures`
  * `oldest_pending_batch_age >= escalate_oldest_age_s`

Both must trip on the SAME worker iteration.  Low-traffic dead-
sglang produces a high consec count but tiny oldest_age (queue
drains as fast as POSTs fail) — daemon survives, will reconverge
when sglang returns.  High-traffic dead-sglang grows the queue
too → both trip → fatal.

Phase A (in-process unit tests + integration with a stub HTTP):
  A0  counter resets on a 2xx success
  A1  counter increments on each failure flavor (5xx, 4xx, transport
      exception)
  A2  fatal does NOT fire when only consec is over threshold
      (oldest_age stays small under low-traffic)
  A3  fatal does NOT fire when only oldest_age is over threshold
      (consec stays at 0 because POSTs succeed)
  A4  /health body includes outbound_consecutive_failures +
      outbound_oldest_age_ms

Phase B (subprocess, real fatal()):
  B0  spawn a subprocess that wires an always-failing HTTP stub +
      low thresholds + bulk-enqueued batches with old enqueue_ts;
      assert subprocess exits 1 AND a forensic dump file lands
      under `<data>/forensic/sglang_sustained_unreachable_*.json`
      with the contract context fields.

Phase C (#166 audit closure):
  C0  fatal() inside a UVICORN-hosted daemon MUST exit the process.
      The naive ``sys.exit(1)`` raises ``SystemExit`` which the
      asyncio Task wrapper swallows — under real uvicorn.run, the
      worker dies silently and the daemon keeps running.  Crash-
      only-software contract REQUIRES ``os._exit(1)`` (or signal).
  C1  After the queue drains to empty (sglang heals), /health
      ``outbound_oldest_age_ms`` decays back to ~0.  Pre-fix the
      cached ``last_outbound_oldest_age_ms`` was sticky at the last-
      popped batch's age forever — k8s readiness probes scripted
      against the field would mark the daemon NotReady permanently.
  C2  /health reports the LIVE oldest in-queue batch's age (not the
      last-popped batch's age).  Enqueue 3 batches of varying age
      (5 s / 3 s / 1 s); /health should return ~5 s — the actual
      oldest pending.

Usage:
    python dev/aginfer/verify/t164/verify.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.outbound import OutboundBatch, OutboundQueue  # noqa: E402


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ----------------------------------------------------------------- helpers


class _ProgrammableHttpClient:
    """Stub for httpx.AsyncClient.post.  The ``script`` is a list of
    ``(kind, value)`` directives consumed in order; the worker can
    poll multiple times so we cycle if exhausted.

    kinds:
      ('ok', _)             -> 200
      ('fivexx', code)      -> code (5xx)
      ('fourxx', code)      -> code (4xx)
      ('exc', exc_instance) -> raise exc_instance
      ('delay', seconds)    -> sleep then 200
    """

    def __init__(self, script: List[Tuple[str, Any]]) -> None:
        self._script = list(script)
        self._idx = 0
        self.calls: List[Tuple[str, dict]] = []

    async def post(self, url: str, *, json=None):  # type: ignore[no-untyped-def]
        self.calls.append((url, json or {}))
        if self._script:
            kind, value = self._script[self._idx % len(self._script)]
            self._idx += 1
        else:
            kind, value = ("ok", None)
        if kind == "ok":
            return _StubResponse(200)
        if kind == "fivexx":
            return _StubResponse(int(value))
        if kind == "fourxx":
            return _StubResponse(int(value))
        if kind == "exc":
            raise value
        if kind == "delay":
            await asyncio.sleep(float(value))
            return _StubResponse(200)
        raise StageFail(f"unknown script directive: {kind!r}")

    async def aclose(self) -> None:
        return None


class _StubResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.text = ""

    def json(self):  # type: ignore[no-untyped-def]
        return {"applied": 0, "applied_hashes": [], "skipped": []}


def _enqueue(
    outbound: OutboundQueue,
    *,
    n: int = 1,
    enqueue_ts_override: Optional[float] = None,
) -> None:
    """Enqueue ``n`` batches.  When ``enqueue_ts_override`` is set we
    DON'T go through ``enqueue_migrate`` (which would stamp wall-
    clock).  Instead, construct ``OutboundBatch`` directly and
    ``put_nowait`` so the worker sees an artificially-aged head.

    Direct path avoids the double-``put_nowait`` / ``unfinished_
    tasks`` accounting drift that would deadlock ``queue.join()``.
    """
    import uuid as _uuid
    for i in range(n):
        if enqueue_ts_override is None:
            outbound.enqueue_migrate(
                [{"hash": f"h{i}", "add_tiers": [],
                  "remove_tiers": ["HBM"], "action_id": f"a{i}"}]
            )
        else:
            batch = OutboundBatch(
                batch_id=str(_uuid.uuid4()),
                endpoint="migrate",
                body={
                    "actions": [{"hash": f"h{i}", "add_tiers": [],
                                 "remove_tiers": ["HBM"],
                                 "action_id": f"a{i}"}],
                    "batch_id": "synthetic",
                },
                enqueue_ts=enqueue_ts_override,
            )
            outbound.queue.put_nowait(batch)


# ============================================================ Phase A


def _fresh_failing_batch(i: int) -> OutboundBatch:
    """A migrate batch the worker will try to dispatch and fail on.
    Fresh (age ≈ 0) so the #164 oldest_age threshold is NOT crossed —
    these stages isolate the consec-counter accounting from the age
    gate.  ``_dispatch_one`` reads ``batch.enqueue_ts`` for its own
    oldest_age computation, so a near-now stamp keeps age tiny."""
    return OutboundBatch(
        batch_id=f"a-{i}",
        endpoint="migrate",
        body={"actions": [{"hash": f"h{i}"}], "batch_id": f"a-{i}"},
        enqueue_ts=time.time(),
    )


def stage_a0_consecutive_failures_resets_on_success() -> None:
    """#228 coalescing made the per-wake POST count != enqueued-batch
    count, so the escalation accounting (consec counter) is now tested
    at the per-DISPATCH unit directly: call ``_dispatch_one`` per
    coalesced batch.  Two failed dispatches (5xx) climb consec to 2,
    then a 2xx dispatch resets it to 0.  Thresholds high so no
    escalation fires mid-test."""
    async def _go():
        stub = _ProgrammableHttpClient(
            [("fivexx", 503), ("fivexx", 503), ("ok", None)]
        )
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
            escalate_failures=1000, escalate_oldest_age_s=10_000,  # high
        )
        # Two failing dispatches → consec climbs.
        await outbound._dispatch_one(_fresh_failing_batch(0))
        await outbound._dispatch_one(_fresh_failing_batch(1))
        if outbound.consecutive_failures != 2:
            raise StageFail(
                f"consec should be 2 after two failed dispatches; "
                f"got {outbound.consecutive_failures}"
            )
        # Third dispatch succeeds (2xx) → reset.
        await outbound._dispatch_one(_fresh_failing_batch(2))
        return outbound.consecutive_failures
    final = asyncio.run(_go())
    if final != 0:
        raise StageFail(
            f"counter should reset to 0 after the 2xx dispatch; got {final}"
        )


def stage_a1_consec_increments_on_each_failure_flavor() -> None:
    """#228: consec accounting tested per-DISPATCH (one ``_dispatch_one``
    call per coalesced POST).  Four failed dispatches of mixed flavor
    (5xx, 4xx, transport-exception, transport-exception) → consec=4 (no
    2xx so it never resets).  Thresholds high so no escalation fires."""
    async def _go():
        stub = _ProgrammableHttpClient([
            ("fivexx", 503),
            ("fourxx", 400),
            ("exc", httpx.ConnectError("simulated")),
            ("exc", httpx.ConnectError("simulated")),
        ])
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
            escalate_failures=1000, escalate_oldest_age_s=10_000,
        )
        # One dispatch per scripted flavor; assert the counter ticks
        # up by exactly one each time.
        for i in range(4):
            await outbound._dispatch_one(_fresh_failing_batch(i))
            if outbound.consecutive_failures != i + 1:
                raise StageFail(
                    f"after {i + 1} failed dispatches consec should be "
                    f"{i + 1}; got {outbound.consecutive_failures}"
                )
        return outbound.consecutive_failures
    final = asyncio.run(_go())
    if final != 4:
        raise StageFail(
            f"consec should equal 4 after 4 failed dispatches of mixed "
            f"flavor; got {final}"
        )


def stage_a2_high_consec_alone_does_not_escalate() -> None:
    """consec >> threshold BUT oldest_age < threshold (low-traffic
    dead-sglang).  Fatal must NOT fire — daemon stays alive,
    waiting for sglang to come back.

    #228: tested per-DISPATCH — 10 failed dispatches of FRESH batches
    (age ≈ 0) push consec to 10 (> escalate_failures=3) but the age
    gate is never crossed, so ``_dispatch_one`` must NOT call fatal().
    If it did, sys.exit/os._exit would kill this process mid-loop."""
    async def _go():
        stub = _ProgrammableHttpClient(
            [("fivexx", 503)]  # always fails (cycles)
        )
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
            escalate_failures=3,             # low
            escalate_oldest_age_s=10_000,    # impossibly high
        )
        # 10 fresh failing dispatches — consec rockets to 10 but each
        # batch's oldest_age stays ≈ 0.  Surviving the loop proves the
        # age gate held fatal back.
        for i in range(10):
            await outbound._dispatch_one(_fresh_failing_batch(i))
        return outbound.consecutive_failures
    # If fatal fired, the process would die and we'd never reach here.
    final = asyncio.run(_go())
    if final < 10:
        raise StageFail(
            f"consec should be at least 10 (all dispatches failed); "
            f"got {final}"
        )


def stage_a3_high_age_alone_does_not_escalate() -> None:
    """oldest_age >> threshold BUT consec stays 0 because POSTs
    succeed.  Fatal must NOT fire."""
    async def _go():
        stub = _ProgrammableHttpClient([("ok", None)])  # always 200
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
            escalate_failures=3,
            escalate_oldest_age_s=0.001,  # 1 ms — every batch crosses
        )
        # Inject batches BEFORE starting the worker so the
        # ``_enqueue(..., enqueue_ts_override=...)`` pop-and-replace
        # doesn't race the worker's own pop.
        old_ts = time.time() - 10.0
        _enqueue(outbound, n=5, enqueue_ts_override=old_ts)
        await outbound.start()
        try:
            await outbound.queue.join()
        finally:
            await outbound.stop()
        return outbound.consecutive_failures
    final = asyncio.run(_go())
    if final != 0:
        raise StageFail(
            f"consec should stay 0 on all-success; got {final}"
        )


def _spawn_health_server(
    outbound: OutboundQueue, *, start_worker: bool = False,
):
    """Start a uvicorn /health server backed by ``outbound`` on a free
    port.  Returns ``(port, server, thread)`` — caller stops via
    ``server.should_exit = True; thread.join()``.

    By default ``start_worker=False`` no-ops ``outbound.start()`` so
    the test fixture's pre-populated queue stays intact (otherwise
    uvicorn's startup hook would drain it before /health is hit).
    Pass ``start_worker=True`` when the test specifically needs the
    worker to run inside uvicorn's thread loop."""
    import socket
    import threading
    import uvicorn
    from daemon.proxy import create_app

    app = create_app(
        sglang_base_url="http://unused",
        enable_event_router=False,
    )
    if not start_worker:
        async def _noop_start() -> None:  # type: ignore[no-redef]
            return None
        outbound.start = _noop_start  # type: ignore[method-assign]
    app.state.outbound = outbound

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(40):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        raise StageFail("uvicorn /health server didn't start in time")
    return port, server, t


def stage_a4_health_body_carries_outbound_counters() -> None:
    """The daemon's `/health` endpoint includes the two outbound
    counters so the operator's alerting (k8s readiness, dashboard)
    can grep them before fatal threshold fires.

    #166: ``outbound_oldest_age_ms`` is now computed LIVE from the
    in-queue head (previously was a sticky cached field — see C1/C2).
    Enqueue one fresh batch + preset consec so we can grep both
    fields with one /health call.  No worker — we only test the
    /health serialisation here, not worker semantics."""
    outbound = OutboundQueue(
        sglang_base_url="http://unused",
        http_client=_ProgrammableHttpClient([("ok", None)]),
        escalate_failures=100, escalate_oldest_age_s=300.0,
    )
    outbound.consecutive_failures = 5
    # Single fresh batch in the queue — age should be small (< 100 ms).
    _enqueue(outbound, n=1)

    port, server, t = _spawn_health_server(outbound)
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"http://127.0.0.1:{port}/health")
        if r.status_code != 200:
            raise StageFail(f"health status {r.status_code}")
        body = r.json()
        if body.get("status") != "ok":
            raise StageFail(f"status not 'ok': {body}")
        if body.get("outbound_consecutive_failures") != 5:
            raise StageFail(
                f"outbound_consecutive_failures: {body!r}"
            )
        # Fresh batch → age should be < 100 ms (just-enqueued).  The
        # field MUST be present and finite.
        age = body.get("outbound_oldest_age_ms")
        if age is None:
            raise StageFail(f"outbound_oldest_age_ms missing: {body!r}")
        if not isinstance(age, (int, float)):
            raise StageFail(
                f"outbound_oldest_age_ms wrong type: {age!r}"
            )
        if not (0.0 <= float(age) < 1000.0):
            raise StageFail(
                f"fresh batch should have small age; got {age} ms"
            )
    finally:
        server.should_exit = True
        t.join(timeout=3.0)


# ============================================================ Phase C
# #166 audit closure: fatal-under-uvicorn + sticky-age + live-head.


_SUBPROCESS_SCRIPT_UVICORN = """\
import os
import socket
import sys
import time
sys.path.insert(0, {aginfer_root!r})

os.environ['AGINFER_DATA_DIR'] = {data_dir!r}

import httpx
import uvicorn
from daemon.outbound import OutboundQueue, OutboundBatch
from daemon.proxy import create_app


class _AlwaysFail:
    async def post(self, url, *, json=None):
        raise httpx.ConnectError('simulated unreachable')

    async def request(self, method, url, *, json=None):
        raise httpx.ConnectError('simulated unreachable')

    async def aclose(self):
        return None


app = create_app(
    sglang_base_url='http://unused',
    enable_event_router=False,
)

ob = OutboundQueue(
    sglang_base_url='http://unused',
    http_client=_AlwaysFail(),
    escalate_failures=3,
    escalate_oldest_age_s=0.001,
)
# #228: a wake coalesces to AT MOST one POST/PUT per endpoint, so 10
# aged migrate batches would collapse to ONE failed POST (consec=1) —
# never reaching escalate_failures=3.  Instead enqueue across THREE
# endpoints in one wake (program_paused + migrate + hints), all aged
# 5 s and all failing.  That yields 3 distinct coalesced dispatches
# in the single wake → consec reaches 3 (= threshold) AND oldest_age
# (5 s) >> 0.001 s → fatal fires.  The aged stamps are injected by
# directly constructing OutboundBatch (enqueue_* would stamp now()).
old_ts = time.time() - 5.0
ob.queue.put_nowait(OutboundBatch(
    batch_id='pp0', endpoint='program_paused',
    body={{'pid': 'p0', 'state': 'ENDED',
           'pre_pause_state': None, 'batch_id': 'pp0'}},
    enqueue_ts=old_ts, method='PUT',
))
ob.queue.put_nowait(OutboundBatch(
    batch_id='mg0', endpoint='migrate',
    body={{'actions': [{{'hash': 'h0'}}], 'batch_id': 'mg0'}},
    enqueue_ts=old_ts, method='POST',
))
ob.queue.put_nowait(OutboundBatch(
    batch_id='hn0', endpoint='hints',
    body={{'hints': [{{'hash': 'h', 'p_hat': 0.1, 'lambda': 0.01,
                       'stamp': 1}}], 'batch_id': 'hn0'}},
    enqueue_ts=old_ts, method='PUT',
))
app.state.outbound = ob

s = socket.socket()
s.bind(('127.0.0.1', 0))
port = s.getsockname()[1]
s.close()

# This is the production code path: uvicorn.run owns the event loop.
# fatal() raised from inside the worker Task MUST exit the process.
uvicorn.run(app, host='127.0.0.1', port=port, log_level='critical')
"""


def stage_c0_fatal_under_uvicorn_actually_exits() -> None:
    """fatal() called from the outbound worker MUST terminate the
    daemon process when running under uvicorn.run (not just under
    bare asyncio.run as B0 tests).  Crash-only-software contract.

    Pre-fix: `_fatal.py` uses ``sys.exit(1)`` → ``SystemExit`` raised
    inside the worker ``asyncio.Task`` is swallowed by the Task
    wrapper.  Uvicorn keeps running.  The subprocess hangs until our
    timeout, returncode != 1, → FAIL.

    Post-fix: ``os._exit(1)`` bypasses Python's normal shutdown; the
    process dies immediately regardless of asyncio loop state."""
    with tempfile.TemporaryDirectory(prefix="aginfer_t164_c0_") as td:
        data_dir = Path(td)
        body = _SUBPROCESS_SCRIPT_UVICORN.format(
            aginfer_root=str(_AGINFER_ROOT),
            data_dir=str(data_dir),
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_AGINFER_ROOT)
        env["AGINFER_DATA_DIR"] = str(data_dir)
        try:
            result = subprocess.run(
                [sys.executable, "-c", body],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired as exc:
            raise StageFail(
                "uvicorn-hosted daemon did NOT exit within 15 s after "
                "sustained-fatal trigger — process is hanging.  This is "
                "the production scenario; sys.exit(1) is being swallowed "
                "by the asyncio Task wrapper. "
                f"stderr_tail={(exc.stderr or '')[-400:]!r}"
            )
        if result.returncode != 1:
            raise StageFail(
                f"expected uvicorn subprocess exit=1; got "
                f"{result.returncode}; stderr={result.stderr[-600:]!r}"
            )
        # Forensic file should still land.
        forensic_dir = data_dir / "forensic"
        matches = sorted(forensic_dir.glob(
            "sglang_sustained_unreachable_*.json"
        )) if forensic_dir.exists() else []
        if not matches:
            raise StageFail(
                f"forensic dump missing under {forensic_dir} after "
                f"fatal-under-uvicorn"
            )
        # #167 round-2 audit: assert the CRITICAL log line names the
        # forensic file path on stderr.  Without this assert, a
        # regression where os._exit runs BEFORE the flush completes
        # (someone removes the flush loop in _fatal.py) would still
        # silently pass C0 — supervisor would see exit 1 but no
        # context.
        if str(matches[0]) not in result.stderr:
            raise StageFail(
                f"stderr does not name forensic file path ({matches[0]}) "
                f"— flush-before-os._exit regression?  stderr_tail="
                f"{result.stderr[-800:]!r}"
            )


def stage_c1_oldest_age_decays_when_queue_drains() -> None:
    """After the queue drains to empty (sglang heals), /health
    ``outbound_oldest_age_ms`` decays back to ~0.

    Pre-fix: ``last_outbound_oldest_age_ms`` is a sticky cached field
    only updated at pop time → after drain, it holds the last-popped
    batch's (large) age forever.  Post-fix: /health peeks the live
    in-queue head, returns 0 when queue is empty."""
    async def _drain():
        stub = _ProgrammableHttpClient([("ok", None)])  # always 200
        ob = OutboundQueue(
            sglang_base_url="http://unused",
            http_client=stub,
            escalate_failures=1000,
            escalate_oldest_age_s=10_000,
        )
        old_ts = time.time() - 100.0  # 100 s aged
        _enqueue(ob, n=3, enqueue_ts_override=old_ts)
        await ob.start()
        try:
            await ob.queue.join()  # drain to empty
            await asyncio.sleep(0.05)
        finally:
            await ob.stop()
        return ob
    ob = asyncio.run(_drain())
    # Queue is now empty; /health MUST report a tiny age, not the
    # sticky last-popped 100_000 ms value.
    port, server, t = _spawn_health_server(ob)
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"http://127.0.0.1:{port}/health")
        body = r.json()
        age = float(body.get("outbound_oldest_age_ms", -1.0))
        if age > 1000.0:
            raise StageFail(
                f"queue drained to empty; /health should report ~0 ms, "
                f"got {age} ms (sticky-cached-field bug). body={body!r}"
            )
    finally:
        server.should_exit = True
        t.join(timeout=3.0)


def stage_c2_health_reports_live_in_queue_oldest() -> None:
    """/health ``outbound_oldest_age_ms`` reports the LIVE oldest
    in-queue batch's age, not the last-popped batch's age.

    Pre-fix: cached field starts at 0.0 and only updates on pop.  No
    pops yet → /health returns 0 even though 5 s-aged batches are
    backed up.  Post-fix: /health peeks the queue head live."""
    # No worker started — items just sit in the queue.
    ob = OutboundQueue(
        sglang_base_url="http://unused",
        http_client=_ProgrammableHttpClient([("ok", None)]),
        escalate_failures=100, escalate_oldest_age_s=300.0,
    )
    now = time.time()
    # Enqueue oldest first (FIFO head): 5 s, 3 s, 1 s aged.
    _enqueue(ob, n=1, enqueue_ts_override=now - 5.0)
    _enqueue(ob, n=1, enqueue_ts_override=now - 3.0)
    _enqueue(ob, n=1, enqueue_ts_override=now - 1.0)

    port, server, t = _spawn_health_server(ob)
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"http://127.0.0.1:{port}/health")
        body = r.json()
        age_ms = float(body.get("outbound_oldest_age_ms", -1.0))
        # Should reflect the OLDEST head ≈ 5_000 ms.  Allow generous
        # tolerance (test fixture timing slack).
        if not (4_000.0 <= age_ms <= 7_000.0):
            raise StageFail(
                f"/health should report the oldest in-queue batch's age "
                f"(~5_000 ms); got {age_ms} ms — the sticky-last-popped "
                f"field reads 0 because no pops have happened. body={body!r}"
            )
    finally:
        server.should_exit = True
        t.join(timeout=3.0)


# ============================================================ Phase B
# Subprocess: real fatal() exits process; assert exit + forensic dump.


_SUBPROCESS_SCRIPT = """\
import asyncio
import os
import sys
import time
sys.path.insert(0, {aginfer_root!r})

# Import AFTER setting AGINFER_DATA_DIR so fatal() writes to our temp.
os.environ['AGINFER_DATA_DIR'] = {data_dir!r}

import httpx
from daemon.outbound import OutboundQueue, OutboundBatch


class _AlwaysFail:
    async def post(self, url, *, json=None):
        raise httpx.ConnectError('simulated unreachable')
    async def request(self, method, url, *, json=None):
        raise httpx.ConnectError('simulated unreachable')
    async def aclose(self): return None


async def _go():
    outbound = OutboundQueue(
        sglang_base_url='http://unused',
        http_client=_AlwaysFail(),
        # Low thresholds so escalation fires fast.
        escalate_failures=3,
        escalate_oldest_age_s=0.001,
    )
    # #228: a wake coalesces to AT MOST one POST/PUT per endpoint, so
    # N migrate batches collapse to ONE failed POST (consec=1).  To
    # reach escalate_failures=3 in a single wake, enqueue across THREE
    # endpoints (program_paused + migrate + hints), all aged 5 s and
    # all failing → 3 distinct coalesced dispatches → consec=3 AND
    # oldest_age (5 s) >> 0.001 s → fatal.  Aged stamps injected by
    # constructing OutboundBatch directly (enqueue_* would stamp now()).
    old_ts = time.time() - 5.0
    outbound.queue.put_nowait(OutboundBatch(
        batch_id='pp0', endpoint='program_paused',
        body={{'pid': 'p0', 'state': 'ENDED',
               'pre_pause_state': None, 'batch_id': 'pp0'}},
        enqueue_ts=old_ts, method='PUT',
    ))
    outbound.queue.put_nowait(OutboundBatch(
        batch_id='mg0', endpoint='migrate',
        body={{'actions': [{{'hash': 'h0'}}], 'batch_id': 'mg0'}},
        enqueue_ts=old_ts, method='POST',
    ))
    outbound.queue.put_nowait(OutboundBatch(
        batch_id='hn0', endpoint='hints',
        body={{'hints': [{{'hash': 'h', 'p_hat': 0.1, 'lambda': 0.01,
                           'stamp': 1}}], 'batch_id': 'hn0'}},
        enqueue_ts=old_ts, method='PUT',
    ))
    await outbound.start()
    await outbound.queue.join()  # will not return; fatal exits


asyncio.run(_go())
"""


def stage_b0_subprocess_escalates_to_fatal_with_forensic_dump() -> None:
    """Drive a real subprocess that triggers the fatal() inside the
    OutboundQueue worker.  Assert:
      - subprocess exit code == 1
      - forensic JSON file written under <data_dir>/forensic/
        with reason 'sglang_sustained_unreachable' + the expected
        context fields (consecutive_failures, oldest_age_ms, etc.)
    """
    with tempfile.TemporaryDirectory(prefix="aginfer_t164_") as td:
        data_dir = Path(td)
        body = _SUBPROCESS_SCRIPT.format(
            aginfer_root=str(_AGINFER_ROOT),
            data_dir=str(data_dir),
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_AGINFER_ROOT)
        env["AGINFER_DATA_DIR"] = str(data_dir)
        result = subprocess.run(
            [sys.executable, "-c", body],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 1:
            raise StageFail(
                f"expected exit=1; got {result.returncode}; "
                f"stderr={result.stderr[:600]!r}"
            )
        # Forensic file exists with the reason slug.
        forensic_dir = data_dir / "forensic"
        if not forensic_dir.exists():
            raise StageFail(
                f"forensic dir not created at {forensic_dir}"
            )
        matches = sorted(forensic_dir.glob(
            "sglang_sustained_unreachable_*.json"
        ))
        if not matches:
            raise StageFail(
                f"no forensic file for "
                f"sglang_sustained_unreachable_* in {forensic_dir} "
                f"(present: {[f.name for f in forensic_dir.iterdir()]})"
            )
        payload = json.loads(matches[0].read_text())
        if payload.get("reason") != "sglang_sustained_unreachable":
            raise StageFail(
                f"reason mismatch: {payload.get('reason')!r}"
            )
        ctx = payload.get("context", {})
        for key in (
            "sglang_base_url",
            "consecutive_failures",
            "oldest_age_ms",
            "queue_depth",
            "escalate_failures_threshold",
            "escalate_oldest_age_s_threshold",
        ):
            if key not in ctx:
                raise StageFail(
                    f"context missing {key!r}; keys={list(ctx)}"
                )
        if ctx["consecutive_failures"] < 3:
            raise StageFail(
                f"consecutive_failures < threshold at fatal time: "
                f"{ctx['consecutive_failures']}"
            )
        # CRITICAL log line on stderr names the forensic file.
        if str(matches[0]) not in result.stderr:
            raise StageFail(
                f"stderr does not name forensic file path "
                f"({matches[0]}); stderr={result.stderr[:800]!r}"
            )


def stage_c3_live_peek_under_concurrent_worker() -> None:
    """#167 round-2 audit: demonstrate that `current_oldest_pending_
    age_ms()` is safe AND accurate while a worker drains+dispatches the
    queue concurrently from another thread's event loop.

    The "GIL-atomic peek under concurrent drain" claim is asserted in
    the method docstring + DESIGN §10 but A4/C1/C2 all monkeypatch
    the worker off — they cannot distinguish "live peek" from "field
    set once at enqueue and never touched again".

    #228: a wake coalesces to AT MOST one POST/PUT per endpoint and the
    asyncio.Queue reads EMPTY the instant the worker drains it — so the
    in-flight backlog age now lives in ``_draining_oldest_ts``, which
    ``current_oldest_pending_age_ms()`` folds in.  To keep that window
    observable across multiple /health polls we (a) make the stub slow
    (0.2 s per dispatch) and (b) enqueue across all THREE endpoints
    (program_paused + migrate + hints) so one wake = three sequential
    ~0.2 s dispatches ≈ 0.6 s of in-flight time, all stamped old so the
    peeked age is large and clearly nonzero.

    Assertions:
      (a) No exception during any /health call.
      (b) At least one sample reports a NONZERO in-flight age (the
          live peek surfaces the draining burst, not 0).
      (c) Final reported age ≈ 0 (burst fully dispatched → idle).
      (d) Series shrinks: LAST observation < the max observation (the
          peek tracks the draining window, not a frozen value).
    """
    # Slow stub: 0.2 s per dispatch.  Both POST and PUT route through
    # .post()/.request(); add a .request shim that reuses .post.
    stub = _ProgrammableHttpClient([("delay", 0.2)])  # cycles 200 ms

    async def _request(method, url, *, json=None):
        return await stub.post(url, json=json)
    stub.request = _request  # type: ignore[attr-defined]

    ob = OutboundQueue(
        sglang_base_url="http://unused",
        http_client=stub,
        escalate_failures=10_000, escalate_oldest_age_s=10_000,  # high
    )
    now = time.time()
    # Three endpoints, all aged ~5 s, enqueued in ONE burst.  One wake
    # coalesces to three dispatches (program_paused → migrate → hints),
    # each ~0.2 s, so the draining window stays observable ~0.6 s.
    ob.queue.put_nowait(OutboundBatch(
        batch_id="c3-pp", endpoint="program_paused",
        body={"pid": "p0", "state": "ENDED",
              "pre_pause_state": None, "batch_id": "c3-pp"},
        enqueue_ts=now - 5.0, method="PUT",
    ))
    ob.queue.put_nowait(OutboundBatch(
        batch_id="c3-mg", endpoint="migrate",
        body={"actions": [{"hash": "h0"}], "batch_id": "c3-mg"},
        enqueue_ts=now - 5.0, method="POST",
    ))
    ob.queue.put_nowait(OutboundBatch(
        batch_id="c3-hn", endpoint="hints",
        body={"hints": [{"hash": "h", "p_hat": 0.1, "lambda": 0.01,
                         "stamp": 1}], "batch_id": "c3-hn"},
        enqueue_ts=now - 5.0, method="PUT",
    ))
    # Start the worker via _spawn_health_server — but this time we
    # WANT the worker to actually run, so pass start_worker=True.
    port, server, t = _spawn_health_server(ob, start_worker=True)

    samples: list = []  # list of (elapsed_s, age_ms)
    start = time.time()
    last_err: Optional[BaseException] = None
    try:
        with httpx.Client(timeout=2.0) as client:
            # Poll for up to 4 s OR until queue drained for ≥1 sample.
            deadline = start + 4.0
            drained_seen_at = None
            seen_nonzero = False
            while time.time() < deadline:
                try:
                    r = client.get(f"http://127.0.0.1:{port}/health")
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    break
                body = r.json()
                age = float(body.get("outbound_oldest_age_ms", -1.0))
                samples.append((time.time() - start, age))
                # Only treat a 0-age reading as "drained" AFTER we've
                # observed the in-flight burst (nonzero age); otherwise
                # the very first poll (worker hasn't grabbed the burst
                # yet) would trip an early stop before any work runs.
                if age > 0.0:
                    seen_nonzero = True
                if seen_nonzero and age == 0.0 and drained_seen_at is None:
                    drained_seen_at = time.time()
                elif (drained_seen_at is not None
                      and time.time() - drained_seen_at > 0.2):
                    break
                time.sleep(0.05)
    finally:
        server.should_exit = True
        t.join(timeout=3.0)

    if last_err is not None:
        raise StageFail(
            f"/health raised under concurrent worker: "
            f"{type(last_err).__name__}: {last_err}"
        )
    if len(samples) < 4:
        raise StageFail(
            f"too few /health samples ({len(samples)}); test fixture "
            f"may have completed before polling started.  samples="
            f"{samples!r}"
        )
    ages = [a for _, a in samples]
    max_age = max(ages)
    last_age = samples[-1][1]
    # (b) the live peek must surface the in-flight draining burst:
    # at least one sample reports a NONZERO age while the worker is
    # mid-dispatch (proves _draining_oldest_ts is folded in, not a
    # field frozen at 0 the instant the queue drained).
    if max_age <= 0.0:
        raise StageFail(
            f"no nonzero in-flight age observed; /health never saw the "
            f"draining burst (live-peek not folding _draining_oldest_ts). "
            f"samples={samples!r}"
        )
    # (c) eventually drained → idle → age decays to ~0.
    if last_age > 500.0:
        raise StageFail(
            f"burst should have fully dispatched by the end; final age="
            f"{last_age} ms.  samples_tail={samples[-5:]!r}"
        )
    # (d) live peek tracks the draining window — series must shrink
    # from its peak.
    if last_age >= max_age:
        raise StageFail(
            f"age series did not shrink (max={max_age} ms, last="
            f"{last_age} ms).  /health is not following the live "
            f"draining window.  samples={samples!r}"
        )
    # (a) implicit: no exceptions during polling — already verified.


def stage_c4_enqueue_ts_validation() -> None:
    """#167 round-2 audit nit-3: `OutboundBatch.enqueue_ts` previously
    defaulted to 0.0 — silent footgun where `(time.time() - 0.0) *
    1000 ≈ 1.7e15 ms` would instantly trip sustained-escalation.
    The fix removes the default + adds a `__post_init__` guard.

    Test contract: bare construction without ``enqueue_ts`` raises
    ``TypeError`` (Python dataclass enforces required field);
    construction with non-positive enqueue_ts raises ``ValueError``
    via ``__post_init__``."""
    # (1) Missing required arg → TypeError from dataclass.
    try:
        OutboundBatch(
            batch_id="x", endpoint="migrate", body={},
        )  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise StageFail(
            "OutboundBatch(...) without enqueue_ts must raise TypeError "
            "— default 0.0 footgun still present"
        )

    # (2) Explicit zero → ValueError from __post_init__.
    try:
        OutboundBatch(
            batch_id="x", endpoint="migrate", body={},
            enqueue_ts=0.0,
        )
    except ValueError:
        pass
    else:
        raise StageFail(
            "OutboundBatch(..., enqueue_ts=0.0) must raise ValueError"
        )

    # (3) Explicit negative → ValueError.
    try:
        OutboundBatch(
            batch_id="x", endpoint="migrate", body={},
            enqueue_ts=-1.0,
        )
    except ValueError:
        pass
    else:
        raise StageFail(
            "OutboundBatch(..., enqueue_ts=-1.0) must raise ValueError"
        )

    # (4) Sane positive value works.
    ok = OutboundBatch(
        batch_id="x", endpoint="migrate", body={},
        enqueue_ts=time.time(),
    )
    if ok.enqueue_ts <= 0.0:
        raise StageFail(f"ok.enqueue_ts not preserved: {ok.enqueue_ts}")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 consec resets on 2xx success",         stage_a0_consecutive_failures_resets_on_success),
    ("A1 consec increments on 5xx / 4xx / transport-exc",
                                                stage_a1_consec_increments_on_each_failure_flavor),
    ("A2 high consec alone does NOT escalate (low-traffic safe)",
                                                stage_a2_high_consec_alone_does_not_escalate),
    ("A3 high oldest_age alone does NOT escalate (success path)",
                                                stage_a3_high_age_alone_does_not_escalate),
    ("A4 /health body carries outbound counters", stage_a4_health_body_carries_outbound_counters),
    ("B0 subprocess: both thresholds trip → fatal + forensic dump",
                                                stage_b0_subprocess_escalates_to_fatal_with_forensic_dump),
    ("C0 fatal() under uvicorn ACTUALLY exits the process",
                                                stage_c0_fatal_under_uvicorn_actually_exits),
    ("C1 oldest_age decays after queue drains (sticky-cache bug)",
                                                stage_c1_oldest_age_decays_when_queue_drains),
    ("C2 /health reports LIVE in-queue oldest, not last-popped",
                                                stage_c2_health_reports_live_in_queue_oldest),
    ("C3 /health live-peek is safe + accurate under concurrent worker",
                                                stage_c3_live_peek_under_concurrent_worker),
    ("C4 OutboundBatch.enqueue_ts validation (no 0.0 footgun)",
                                                stage_c4_enqueue_ts_validation),
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
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(f"\nT164 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    print(_green(f"\nT164 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
