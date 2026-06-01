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


def stage_a0_consecutive_failures_resets_on_success() -> None:
    """Counter increments on a 5xx, then resets on a 2xx."""
    async def _go():
        stub = _ProgrammableHttpClient(
            [("fivexx", 503), ("fivexx", 503), ("ok", None)]
        )
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
            escalate_failures=1000, escalate_oldest_age_s=10_000,  # high
        )
        await outbound.start()
        try:
            _enqueue(outbound, n=3)
            await outbound.queue.join()
        finally:
            await outbound.stop()
        return outbound.consecutive_failures
    final = asyncio.run(_go())
    if final != 0:
        raise StageFail(
            f"counter should reset to 0 after the final 2xx; got {final}"
        )


def stage_a1_consec_increments_on_each_failure_flavor() -> None:
    """5xx, 4xx, transport-exception, transport-exception → consec=4
    (no 2xx in the script so the counter never resets).  Thresholds
    set high so no escalation fires."""
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
        await outbound.start()
        try:
            _enqueue(outbound, n=4)
            await outbound.queue.join()
        finally:
            await outbound.stop()
        return outbound.consecutive_failures
    final = asyncio.run(_go())
    if final != 4:
        raise StageFail(
            f"consec should equal 4 after 4 failures of mixed flavor; "
            f"got {final}"
        )


def stage_a2_high_consec_alone_does_not_escalate() -> None:
    """consec >> threshold BUT oldest_age < threshold (low-traffic
    dead-sglang).  Fatal must NOT fire — daemon stays alive,
    waiting for sglang to come back."""
    async def _go():
        stub = _ProgrammableHttpClient(
            [("fivexx", 503)]  # always fails (cycles)
        )
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
            escalate_failures=3,             # low
            escalate_oldest_age_s=10_000,    # impossibly high
        )
        await outbound.start()
        try:
            # 10 fresh batches (each just enqueued, age ≈ 0) — consec
            # rockets to 10 but oldest_age stays at ~0.  If fatal
            # fired, this process would die mid-drain.
            _enqueue(outbound, n=10)
            await outbound.queue.join()
        finally:
            await outbound.stop()
        return outbound.consecutive_failures
    # If fatal fired, sys.exit propagates and we'd never reach here.
    final = asyncio.run(_go())
    if final < 10:
        raise StageFail(
            f"consec should be at least 10 (all failed); got {final}"
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


def stage_a4_health_body_carries_outbound_counters() -> None:
    """The daemon's `/health` endpoint includes the two outbound
    counters so the operator's alerting (k8s readiness, dashboard)
    can grep them before fatal threshold fires."""
    import socket
    import threading
    import uvicorn
    from daemon.proxy import create_app

    # Build a minimal daemon app w/ an outbound queue already in a
    # known "5 consecutive failures" state.  We DON'T start the
    # outbound worker (we just want to read /health while the
    # counter is preset).
    app = create_app(
        sglang_base_url="http://unused",
        enable_event_router=False,
    )
    # Attach an outbound queue manually since we disabled the router.
    outbound = OutboundQueue(
        sglang_base_url="http://unused",
        http_client=_ProgrammableHttpClient([("ok", None)]),
        escalate_failures=100, escalate_oldest_age_s=300.0,
    )
    outbound.consecutive_failures = 5
    outbound.last_outbound_oldest_age_ms = 123.4
    app.state.outbound = outbound

    # Bind a free port + spin uvicorn in a thread.
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        # Wait briefly for the server to start.
        for _ in range(40):
            if server.started:
                break
            time.sleep(0.05)
        if not server.started:
            raise StageFail("uvicorn /health server didn't start in time")
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
        if abs(
            float(body.get("outbound_oldest_age_ms", 0.0)) - 123.4
        ) > 1e-3:
            raise StageFail(f"outbound_oldest_age_ms: {body!r}")
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
    async def aclose(self): return None


async def _go():
    outbound = OutboundQueue(
        sglang_base_url='http://unused',
        http_client=_AlwaysFail(),
        # Low thresholds so escalation fires fast.
        escalate_failures=3,
        escalate_oldest_age_s=0.001,
    )
    # Pre-enqueue 10 batches stamped 5 s in the past so oldest_age
    # is huge for every pop.  Worker fails on each → consec climbs;
    # at consec >= 3 AND oldest_age >> threshold → fatal.
    old_ts = time.time() - 5.0
    for i in range(10):
        bid = outbound.enqueue_migrate(
            [{{'hash': f'h{{i}}', 'add_tiers': [],
              'remove_tiers': ['HBM'], 'action_id': f'a{{i}}'}}]
        )
        batch = outbound.queue.get_nowait()
        outbound.queue.put_nowait(OutboundBatch(
            batch_id=batch.batch_id,
            endpoint=batch.endpoint,
            body=batch.body,
            enqueue_ts=old_ts,
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
