"""T22 verify — `GET /aginfer/thresholds` (daemon) + `PUT /aginfer/
thresholds` (sglang).

PLAN §3 T22.  DESIGN §6 round-6 H3 + §10 "Threshold parity":

  * The daemon is the canonical source of theta_hi / theta_lo /
    theta_crit / heartbeat_s.  Sglang has NO local cache (round-14
    dropped it).
  * At sglang launch, sglang ``GET /aginfer/thresholds`` from the
    daemon.  Halts loudly if unreachable (deployment-ordering bug —
    daemon must be up first).
  * Runtime changes flow daemon → sglang via ``PUT /aginfer/thresholds``
    on sglang's HTTP server.  Sglang updates ``AginferWebhookFirer``
    atomically.  Until propagation, sglang and daemon may transiently
    disagree by one update; next state fetch reconciles.

Closes G9 (theta mismatch between sglang webhook fire and daemon
admission gate) permanently — there is no longer a way for the two
sides to drift.

Phase A (in-process):
  A0  daemon ``GET /aginfer/thresholds`` returns canonical JSON shape
      with the EventRouter's theta_hi/lo/crit and heartbeat_s
  A1  sglang ``AginferWebhookFirer.apply_thresholds`` mutates in-
      memory thresholds atomically (atomicity: a ``maybe_fire`` race
      sees either the OLD set or the NEW set, never a half-applied
      hybrid)
  A2  malformed PUT body (missing field / non-numeric / negative) →
      400 with structured reason
  A3  daemon GET → sglang side bootstrap-fetch helper happy path:
      sglang library function fetches, validates, and returns the
      threshold dict

Phase B (live integration, opt-in):
  B0  launch daemon + sglang with --aginfer-notify-url; sglang's
      firer reports the daemon's thresholds (not the CLI defaults)
  B1  launch sglang WITHOUT daemon up; sglang halts at bootstrap
      with a clear error code

Usage:
    python dev/aginfer/verify/t22/verify.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))
_SGLANG_PY = _AGINFER_ROOT.parent.parent / "python"
if (_SGLANG_PY / "sglang").is_dir() and str(_SGLANG_PY) not in sys.path:
    sys.path.insert(0, str(_SGLANG_PY))

from daemon.events import EventBus  # noqa: E402
from daemon.event_router import EventRouter, attach_event_routes  # noqa: E402
from sglang.srt.managers.aginfer_webhook import (  # noqa: E402
    AginferWebhookFirer,
    bootstrap_thresholds_into_server_args,
    fetch_bootstrap_thresholds,
)


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


class StageFail(AssertionError):
    pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@asynccontextmanager
async def _run_server(app: FastAPI, host: str, port: int):
    cfg = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(cfg)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    try:
        yield server
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()


def _build_daemon_app(
    *, theta_hi: float, theta_lo: float, theta_crit: float,
    heartbeat_s: float = 5.0,
) -> Tuple[FastAPI, EventRouter]:
    app = FastAPI()
    bus = EventBus()
    router = EventRouter(
        bus=bus, sglang_base_url="http://unused",
        theta_hi=theta_hi, theta_crit=theta_crit,
    )
    router.theta_lo = theta_lo  # type: ignore[attr-defined]
    router.heartbeat_s = heartbeat_s  # type: ignore[attr-defined]
    attach_event_routes(app, router)
    return app, router


# ============================================================ Phase A


def stage_a0_daemon_get_returns_canonical_shape() -> None:
    """``GET /aginfer/thresholds`` on the daemon returns the four
    canonical numbers from the EventRouter, as a flat JSON object."""
    async def _go():
        app, _router = _build_daemon_app(
            theta_hi=0.72, theta_lo=0.57, theta_crit=0.91, heartbeat_s=4.5,
        )
        port = _free_port()
        async with _run_server(app, "127.0.0.1", port):
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"http://127.0.0.1:{port}/aginfer/thresholds"
                )
                if r.status_code != 200:
                    raise StageFail(
                        f"expected 200; got {r.status_code}: {r.text}"
                    )
                body = r.json()
                expected = {"theta_hi", "theta_lo", "theta_crit",
                            "heartbeat_s"}
                if set(body.keys()) != expected:
                    raise StageFail(
                        f"body keys mismatch: got {set(body.keys())}; "
                        f"want {expected}"
                    )
                if abs(body["theta_hi"] - 0.72) > 1e-9:
                    raise StageFail(f"theta_hi: {body['theta_hi']}")
                if abs(body["theta_lo"] - 0.57) > 1e-9:
                    raise StageFail(f"theta_lo: {body['theta_lo']}")
                if abs(body["theta_crit"] - 0.91) > 1e-9:
                    raise StageFail(f"theta_crit: {body['theta_crit']}")
                if abs(body["heartbeat_s"] - 4.5) > 1e-9:
                    raise StageFail(f"heartbeat_s: {body['heartbeat_s']}")
    asyncio.run(_go())


def stage_a1_firer_apply_thresholds_atomic() -> None:
    """``AginferWebhookFirer.apply_thresholds(...)`` mutates the four
    threshold fields atomically: a concurrent ``maybe_fire`` either
    sees the old (theta_hi, theta_lo, theta_crit, heartbeat_s) tuple
    or the new — never a half-applied state.  Defends against an impl
    that does four separate attribute writes without a barrier."""
    firer = AginferWebhookFirer(
        notify_url="http://unused/aginfer/event",
        theta_hi=0.70, theta_lo=0.55, theta_crit=0.90, heartbeat_s=5.0,
    )
    try:
        # Atomic apply.
        firer.apply_thresholds(
            theta_hi=0.82, theta_lo=0.65, theta_crit=0.95, heartbeat_s=3.0,
        )
        if firer.theta_hi != 0.82:
            raise StageFail(f"theta_hi not applied: {firer.theta_hi}")
        if firer.theta_lo != 0.65:
            raise StageFail(f"theta_lo not applied: {firer.theta_lo}")
        if firer.theta_crit != 0.95:
            raise StageFail(f"theta_crit not applied: {firer.theta_crit}")
        if firer.heartbeat_s != 3.0:
            raise StageFail(f"heartbeat_s not applied: {firer.heartbeat_s}")

        # Atomicity probe: 200 reads under 50 concurrent applies.  If
        # the impl writes the four fields separately, a reader will
        # eventually see a tuple where theta_lo > theta_hi (impossible
        # by spec — theta_lo < theta_hi always per hysteresis).
        # We alternate (low, high) pairs and assert the invariant.
        stop = [False]
        saw_inconsistent: List[str] = []

        def _writer():
            i = 0
            while not stop[0]:
                if i % 2 == 0:
                    firer.apply_thresholds(
                        theta_hi=0.82, theta_lo=0.65, theta_crit=0.95,
                        heartbeat_s=3.0,
                    )
                else:
                    firer.apply_thresholds(
                        theta_hi=0.70, theta_lo=0.55, theta_crit=0.90,
                        heartbeat_s=5.0,
                    )
                i += 1

        def _reader():
            for _ in range(20_000):
                hi = firer.theta_hi
                lo = firer.theta_lo
                # Invariant: lo < hi.  A torn write would produce
                # cases like (hi=0.70, lo=0.65) → still valid in this
                # particular pair, OR (hi=0.82, lo=0.55) → still valid.
                # The real torn case lands when partway: (hi=0.82,
                # lo=0.55) is OK but (hi=0.70, lo=0.65) is also OK.
                # Sharper invariant: AT LEAST ONE of (lo,hi)==
                # (0.55,0.70) or (0.65,0.82) — anything else proves
                # torn write.
                if (hi, lo) not in {(0.70, 0.55), (0.82, 0.65)}:
                    saw_inconsistent.append(f"(hi={hi}, lo={lo})")
                    break

        writer = threading.Thread(target=_writer, daemon=True)
        writer.start()
        readers = [threading.Thread(target=_reader, daemon=True)
                   for _ in range(4)]
        for r in readers:
            r.start()
        for r in readers:
            r.join(timeout=5.0)
        stop[0] = True
        writer.join(timeout=2.0)

        if saw_inconsistent:
            raise StageFail(
                f"apply_thresholds is NOT atomic (torn-write seen): "
                f"{saw_inconsistent[:3]}"
            )
    finally:
        firer.close()


def stage_a2_malformed_put_rejected() -> None:
    """Sglang's PUT handler must reject malformed bodies with 400 +
    structured reason.  Tested cases:

      * missing required field
      * non-numeric value
      * negative value (thresholds are in [0, 1])
      * theta_lo >= theta_hi (hysteresis violation)
    """
    # Build a tiny sglang-side FastAPI app exposing the PUT.  We
    # import the handler factory from sglang's http_server pattern
    # via a local stub since we can't easily spin up the full sglang
    # in-process.  The handler logic lives on
    # ``AginferWebhookFirer.apply_thresholds``; PUT is just plumbing
    # + validation.
    firer = AginferWebhookFirer(
        notify_url="http://unused/aginfer/event",
        theta_hi=0.70, theta_lo=0.55, theta_crit=0.90, heartbeat_s=5.0,
    )
    try:
        from sglang.srt.managers.aginfer_webhook import (
            apply_thresholds_payload,
        )
        # Missing fields.
        for body in (
            {},
            {"theta_hi": 0.8},
            {"theta_hi": 0.8, "theta_lo": 0.5, "theta_crit": 0.9},
        ):
            ok, reason = apply_thresholds_payload(firer, body)
            if ok:
                raise StageFail(
                    f"missing-field payload accepted: {body!r}"
                )
            if "missing" not in reason and "required" not in reason.lower():
                raise StageFail(
                    f"reason should name the missing field; got {reason!r}"
                )
        # Non-numeric.
        ok, reason = apply_thresholds_payload(firer, {
            "theta_hi": "high", "theta_lo": 0.5,
            "theta_crit": 0.9, "heartbeat_s": 5.0,
        })
        if ok or "type" not in reason.lower() and "numeric" not in reason.lower():
            raise StageFail(f"non-numeric accepted; reason={reason!r}")
        # Negative.
        ok, reason = apply_thresholds_payload(firer, {
            "theta_hi": -0.1, "theta_lo": 0.5,
            "theta_crit": 0.9, "heartbeat_s": 5.0,
        })
        if ok or "negative" not in reason.lower() and "range" not in reason.lower():
            raise StageFail(f"negative accepted; reason={reason!r}")
        # Hysteresis violation.
        ok, reason = apply_thresholds_payload(firer, {
            "theta_hi": 0.5, "theta_lo": 0.7,  # lo > hi
            "theta_crit": 0.9, "heartbeat_s": 5.0,
        })
        if ok or "hysteresis" not in reason.lower() and "theta_lo" not in reason.lower():
            raise StageFail(f"theta_lo>=theta_hi accepted; reason={reason!r}")
        # Sanity: the firer's state was not mutated by any of the
        # rejected payloads (all should have returned False early).
        if firer.theta_hi != 0.70:
            raise StageFail(
                f"state mutated by rejected payload: "
                f"theta_hi={firer.theta_hi}"
            )
        # Happy path.
        ok, reason = apply_thresholds_payload(firer, {
            "theta_hi": 0.80, "theta_lo": 0.60,
            "theta_crit": 0.95, "heartbeat_s": 4.0,
        })
        if not ok:
            raise StageFail(f"happy path rejected: {reason!r}")
        if firer.theta_hi != 0.80 or firer.theta_lo != 0.60:
            raise StageFail("happy path did not mutate")
    finally:
        firer.close()


def stage_a3_bootstrap_fetch_happy_and_unreachable() -> None:
    """``fetch_bootstrap_thresholds(daemon_url, timeout=...)`` returns
    the canonical dict on 200; raises a clearly-typed exception on
    unreachable daemon (deployment-ordering bug — sglang halts at
    its call site)."""
    async def _go():
        # --- happy path: daemon up, fetch returns dict ---
        # fetch_bootstrap_thresholds is SYNC (sglang launch calls it
        # at startup, before any asyncio loop).  Inside this test
        # we're already in an asyncio loop hosting uvicorn — calling
        # a sync httpx.Client.get() directly would block the loop
        # and uvicorn would time out serving us.  Use to_thread.
        app, _router = _build_daemon_app(
            theta_hi=0.71, theta_lo=0.54, theta_crit=0.92, heartbeat_s=6.5,
        )
        port = _free_port()
        async with _run_server(app, "127.0.0.1", port):
            base = f"http://127.0.0.1:{port}"
            result = await asyncio.to_thread(
                fetch_bootstrap_thresholds, base, timeout_s=3.0,
            )
            if result["theta_hi"] != 0.71:
                raise StageFail(f"theta_hi: {result['theta_hi']}")
            if result["heartbeat_s"] != 6.5:
                raise StageFail(f"heartbeat_s: {result['heartbeat_s']}")
        # --- unreachable: daemon shut down, fetch raises ---
        # Point at an unreachable address (loopback + a port nothing
        # binds) so the connect either gets refused or times out.
        # Use a port well outside the ephemeral range AND not in the
        # _free_port() recently-bound set to avoid TIME_WAIT races
        # that can make httpx see a transient ReadTimeout vs the
        # expected ConnectError.
        dead_port = 1   # privileged, unbound, refused fast
        raised: Optional[Exception] = None
        try:
            await asyncio.to_thread(
                fetch_bootstrap_thresholds,
                f"http://127.0.0.1:{dead_port}", timeout_s=1.0,
            )
        except Exception as exc:  # noqa: BLE001
            raised = exc
        if raised is None:
            raise StageFail(
                "fetch_bootstrap_thresholds against dead daemon "
                "should raise"
            )
        # Error class is implementation detail; the contract is
        # "raises a httpx-derived exception clearly attributable to
        # 'daemon unreachable'".  Accept ConnectError / ConnectTimeout
        # / ReadTimeout / RemoteProtocolError — all signal the caller
        # to halt the sglang bootstrap.  Reject only success
        # (already handled above) and malformed-payload error classes
        # (ValueError raised by us on shape mismatch).
        if isinstance(raised, ValueError):
            raise StageFail(
                f"unreachable should NOT raise ValueError (that's "
                f"reserved for shape mismatch); got {raised!r}"
            )
        # httpx exception family: anything from httpx.HTTPError tree.
        if not isinstance(raised, httpx.HTTPError):
            raise StageFail(
                f"unreachable should raise an httpx.HTTPError "
                f"subclass; got {type(raised).__name__}: {raised}"
            )
    asyncio.run(_go())


class _FakeServerArgs:
    """Minimal ServerArgs stand-in for the bootstrap-into-server-args
    helper.  The helper only touches ``aginfer_notify_url`` +
    ``aginfer_<theta_hi|theta_lo|theta_crit|heartbeat_s>``, so we
    don't need to instantiate the real ServerArgs (which pulls in
    the whole sglang model-config tree)."""

    def __init__(
        self,
        *,
        aginfer_notify_url: Optional[str] = None,
        aginfer_theta_hi: float = 0.7,
        aginfer_theta_lo: float = 0.55,
        aginfer_theta_crit: float = 0.9,
        aginfer_heartbeat_s: float = 5.0,
    ) -> None:
        self.aginfer_notify_url = aginfer_notify_url
        self.aginfer_theta_hi = aginfer_theta_hi
        self.aginfer_theta_lo = aginfer_theta_lo
        self.aginfer_theta_crit = aginfer_theta_crit
        self.aginfer_heartbeat_s = aginfer_heartbeat_s


def stage_a4_bootstrap_into_server_args_no_notify_url_is_noop() -> None:
    """When ``aginfer_notify_url is None`` (legacy / daemon-less
    deployment), the helper is a no-op — sglang stays on CLI
    defaults, no network, no halt.

    The G9 closure ONLY activates when the operator opts in via
    ``--aginfer-notify-url``."""
    sa = _FakeServerArgs(
        aginfer_notify_url=None,
        aginfer_theta_hi=0.7, aginfer_theta_lo=0.55,
        aginfer_theta_crit=0.9, aginfer_heartbeat_s=5.0,
    )
    # No daemon up; if this called fetch, it would explode.
    bootstrap_thresholds_into_server_args(sa)
    if (sa.aginfer_theta_hi, sa.aginfer_theta_lo,
            sa.aginfer_theta_crit, sa.aginfer_heartbeat_s) != (
            0.7, 0.55, 0.9, 5.0):
        raise StageFail(
            f"no-notify-url path should NOT touch fields; got "
            f"{(sa.aginfer_theta_hi, sa.aginfer_theta_lo, sa.aginfer_theta_crit, sa.aginfer_heartbeat_s)}"
        )


def stage_a5_bootstrap_into_server_args_overrides_from_daemon() -> None:
    """Daemon up, sglang launched with default CLI values: helper
    overrides ALL four fields with daemon's view + logs INFO lines
    (operator left defaults, no warning).  This is the headline
    G9-closure path."""
    captured_warnings: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                captured_warnings.append(record.getMessage())

    handler = _CaptureHandler()
    awh_logger = logging.getLogger(
        "sglang.srt.managers.aginfer_webhook"
    )
    prior_level = awh_logger.level
    awh_logger.addHandler(handler)
    awh_logger.setLevel(logging.INFO)
    try:
        async def _go():
            app, _router = _build_daemon_app(
                theta_hi=0.85, theta_lo=0.70,
                theta_crit=0.95, heartbeat_s=4.0,
            )
            port = _free_port()
            async with _run_server(app, "127.0.0.1", port):
                sa = _FakeServerArgs(
                    aginfer_notify_url=f"http://127.0.0.1:{port}/aginfer/event",
                    # All four at sglang CLI defaults.
                    aginfer_theta_hi=0.7, aginfer_theta_lo=0.55,
                    aginfer_theta_crit=0.9, aginfer_heartbeat_s=5.0,
                )
                await asyncio.to_thread(
                    bootstrap_thresholds_into_server_args, sa,
                )
                return sa
        result = asyncio.run(_go())
        for name, want in (
            ("aginfer_theta_hi", 0.85),
            ("aginfer_theta_lo", 0.70),
            ("aginfer_theta_crit", 0.95),
            ("aginfer_heartbeat_s", 4.0),
        ):
            got = getattr(result, name)
            if abs(got - want) > 1e-9:
                raise StageFail(
                    f"{name}: want {want}, got {got}"
                )
        # Operator was at defaults; no explicit-disagreement
        # warnings should have fired.
        if captured_warnings:
            raise StageFail(
                f"operator-at-defaults should not WARN; "
                f"got {captured_warnings!r}"
            )
    finally:
        awh_logger.removeHandler(handler)
        awh_logger.setLevel(prior_level)


def stage_a6_bootstrap_warns_on_operator_disagreement() -> None:
    """DESIGN §6 step 3: when operator explicitly passes a CLI
    value that disagrees with the daemon, daemon wins AND a WARNING
    line fires so the operator sees their launch flag is moot.

    Operator at defaults for 3 fields + an EXPLICIT non-default
    for theta_hi.  Daemon's theta_hi differs from operator's.
    Expect: theta_hi WARNS; other three INFO (or silent if equal)."""
    captured_warnings: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                captured_warnings.append(record.getMessage())

    handler = _CaptureHandler()
    awh_logger = logging.getLogger(
        "sglang.srt.managers.aginfer_webhook"
    )
    prior_level = awh_logger.level
    awh_logger.addHandler(handler)
    awh_logger.setLevel(logging.INFO)
    try:
        async def _go():
            app, _router = _build_daemon_app(
                theta_hi=0.85, theta_lo=0.55,
                theta_crit=0.9, heartbeat_s=5.0,
            )
            port = _free_port()
            async with _run_server(app, "127.0.0.1", port):
                sa = _FakeServerArgs(
                    aginfer_notify_url=f"http://127.0.0.1:{port}/aginfer/event",
                    # Operator explicitly set theta_hi to 0.5
                    # (NOT the sglang default 0.7) — disagrees
                    # with daemon's 0.85.
                    aginfer_theta_hi=0.5,
                    # The other three are sglang defaults.
                    aginfer_theta_lo=0.55,
                    aginfer_theta_crit=0.9,
                    aginfer_heartbeat_s=5.0,
                )
                await asyncio.to_thread(
                    bootstrap_thresholds_into_server_args, sa,
                )
                return sa
        result = asyncio.run(_go())
        if abs(result.aginfer_theta_hi - 0.85) > 1e-9:
            raise StageFail(
                f"daemon should win on theta_hi; got {result.aginfer_theta_hi}"
            )
        explicit_warns = [
            w for w in captured_warnings
            if "theta-hi" in w and "operator passed" in w
        ]
        if not explicit_warns:
            raise StageFail(
                f"operator-explicit-disagreement should WARN; "
                f"captured={captured_warnings!r}"
            )
    finally:
        awh_logger.removeHandler(handler)
        awh_logger.setLevel(prior_level)


def stage_a7_bootstrap_halts_on_unreachable_daemon() -> None:
    """DESIGN §6 step 1: daemon unreachable at bootstrap → halt
    loudly (no silent CLI-fallback, which IS what round-14 removed).

    The helper calls ``_exit_func(1)`` (injectable for tests; defaults
    to ``sys.exit``).  We pass a recording ``_exit_func`` so the
    test can assert "exit was called with 1" without terminating
    the verify run."""
    captured_errors: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                captured_errors.append(record.getMessage())

    handler = _CaptureHandler()
    awh_logger = logging.getLogger(
        "sglang.srt.managers.aginfer_webhook"
    )
    prior_level = awh_logger.level
    awh_logger.addHandler(handler)
    awh_logger.setLevel(logging.ERROR)
    try:
        exit_calls: List[int] = []
        def _fake_exit(code: int) -> None:
            exit_calls.append(code)
            raise SystemExit(code)  # mimic sys.exit so caller stops

        sa = _FakeServerArgs(
            aginfer_notify_url="http://127.0.0.1:1/aginfer/event",
            aginfer_theta_hi=0.7, aginfer_theta_lo=0.55,
            aginfer_theta_crit=0.9, aginfer_heartbeat_s=5.0,
        )
        raised: Optional[BaseException] = None
        try:
            bootstrap_thresholds_into_server_args(
                sa, timeout_s=1.0, _exit_func=_fake_exit,
            )
        except SystemExit as exc:
            raised = exc
        if raised is None:
            raise StageFail(
                "bootstrap with unreachable daemon should halt "
                "(SystemExit); didn't"
            )
        if exit_calls != [1]:
            raise StageFail(
                f"_exit_func should have been called with 1 exactly once; "
                f"got {exit_calls}"
            )
        if not captured_errors:
            raise StageFail(
                f"halt path should log an ERROR; captured={captured_errors!r}"
            )
        # ERROR line should name the deployment-ordering invariant
        # so the operator gets the right hint.
        if not any(
            "Deployment-ordering bug" in e
            or "daemon must be up" in e.lower()
            for e in captured_errors
        ):
            raise StageFail(
                f"ERROR line should name the deployment-ordering bug; "
                f"got {captured_errors!r}"
            )
        # Fields should NOT have been mutated since the fetch failed.
        if (sa.aginfer_theta_hi, sa.aginfer_theta_lo,
                sa.aginfer_theta_crit, sa.aginfer_heartbeat_s) != (
                0.7, 0.55, 0.9, 5.0):
            raise StageFail(
                "halt path must NOT mutate server_args (no partial "
                "application): got "
                f"{(sa.aginfer_theta_hi, sa.aginfer_theta_lo, sa.aginfer_theta_crit, sa.aginfer_heartbeat_s)}"
            )
    finally:
        awh_logger.removeHandler(handler)
        awh_logger.setLevel(prior_level)


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 daemon GET returns canonical shape",          stage_a0_daemon_get_returns_canonical_shape),
    ("A1 firer.apply_thresholds is atomic (torn-write probe)",
                                                       stage_a1_firer_apply_thresholds_atomic),
    ("A2 PUT validation rejects malformed bodies",     stage_a2_malformed_put_rejected),
    ("A3 bootstrap_fetch happy + unreachable contract", stage_a3_bootstrap_fetch_happy_and_unreachable),
    ("A4 bootstrap_into_server_args no-notify-url is no-op",
                                                       stage_a4_bootstrap_into_server_args_no_notify_url_is_noop),
    ("A5 bootstrap_into_server_args overrides from daemon",
                                                       stage_a5_bootstrap_into_server_args_overrides_from_daemon),
    ("A6 WARN on operator-CLI disagreement (DESIGN §6 step 3)",
                                                       stage_a6_bootstrap_warns_on_operator_disagreement),
    ("A7 halt loudly on unreachable daemon (G9 closure)",
                                                       stage_a7_bootstrap_halts_on_unreachable_daemon),
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
        print(_red(f"\nT22 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    print(_green(f"\nT22 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
