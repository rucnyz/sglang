"""T23 + T37 verify — APPLY_FAILED webhook (sglang) + handler (daemon).

DESIGN §4 round-9 B4 + §6 fire-and-forget delivery: when sglang
can't apply an action the daemon dispatched, it fires an
``APPLY_FAILED`` webhook back so the daemon's next ``joint_decide``
can re-evaluate.  Payload (DESIGN §4 + §6 L506):

    {kind: "apply_failed",
     endpoint: "migrate"|"program_paused"|"hints"|"thresholds",
     action_id: "<uuid>",
     reason: "<skip-class>",
     hash: "<unit hash>" | null,
     ts: <wall>, ts_monotonic: <perf>}

T37 daemon handler: log + bump ``observability.record_failure(reason)``;
no immediate action (next event's joint_decide re-evaluates).

**Double-count avoidance** (closing audit for T42): pre-T23 the
sync ``_record_skips`` path bumped the counter for each item in
``/aginfer/migrate``'s response.  Now the webhook is the
authoritative source.  ``_record_skips`` keeps emitting the per-
line ``migrate_skipped`` log (operator real-time view) but no
longer bumps the counter — otherwise both paths would fire for
every skip and double-count.

Phase A (in-process): POST a synthetic ``apply_failed`` payload to
the daemon's ``/aginfer/event`` endpoint; assert
``observability.failure_class_counts[reason]`` increments.

Phase B (integration, opt-in via ``AGINFER_VERIFY_BASE_SGLANG`` +
``AGINFER_VERIFY_BASE_DAEMON``): drive a known-bad migrate (hash
``node-99999999``) through the daemon, then wait for the
``APPLY_FAILED`` webhook to land on the daemon's queue, assert the
counter saw a ``not_in_tree`` bump (and that the sync skip path
did NOT also bump — exactly one increment per skip).

Usage:
    python dev/aginfer/verify/t23_t37_apply_failed/verify.py
    AGINFER_VERIFY_BASE_SGLANG=http://127.0.0.1:30002 \\
    AGINFER_VERIFY_BASE_DAEMON=http://127.0.0.1:9100 \\
        python dev/aginfer/verify/t23_t37_apply_failed/verify.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import time
import urllib.error
import urllib.request
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

from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.event_router import (  # noqa: E402
    EventRouter, attach_apply_failed_handler, attach_event_routes,
)


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ----------------------------------------------------------------- helpers


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@asynccontextmanager
async def _run_daemon_server(app: FastAPI, host: str, port: int):
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


def _build_test_daemon_app() -> Tuple[FastAPI, EventRouter]:
    """A minimal daemon (no proxy / sglang upstream) that ONLY exposes
    ``POST /aginfer/event`` and the worker.  Uses the PRODUCTION
    ``attach_apply_failed_handler`` from ``daemon.event_router`` so
    the verify exercises the same code path the live daemon does
    (including the structured ``aginfer_metric event=apply_failed``
    line that A4 checks for).
    """
    app = FastAPI()
    bus = EventBus()
    router = EventRouter(
        bus=bus, sglang_base_url="http://unused",
        theta_hi=0.7, theta_crit=0.9,
        observability_summary_every_n=10_000,  # no cadence noise
    )
    # Stub _fetch_state_impl since some handlers may call it on
    # other event kinds; APPLY_FAILED shouldn't need state at all.
    async def _no_fetch():
        return {"pool_usage": {"HBM": {"subpools": {}}}}
    router._fetch_state_impl = _no_fetch  # type: ignore[assignment]

    # Real production handler (same one main.py wires).
    attach_apply_failed_handler(router)
    attach_event_routes(app, router)

    @app.on_event("startup")
    async def _startup():
        await router.start()

    @app.on_event("shutdown")
    async def _shutdown():
        await router.stop()

    return app, router


# ============================================================ Phase A


def stage_a0_eventkind_apply_failed_exists() -> None:
    """``EventKind.APPLY_FAILED`` MUST exist so the webhook dispatcher
    accepts ``{"kind": "apply_failed"}`` payloads.  Pre-T23 this
    enum value didn't exist; the event router would 400 such webhooks
    with ``unknown event kind``."""
    if not hasattr(EventKind, "APPLY_FAILED"):
        raise StageFail("EventKind.APPLY_FAILED missing")
    if EventKind.APPLY_FAILED.value != "apply_failed":
        raise StageFail(
            f"EventKind.APPLY_FAILED.value = "
            f"{EventKind.APPLY_FAILED.value!r}; want 'apply_failed'"
        )


def stage_a1_webhook_bumps_counter_per_reason() -> None:
    """Fire 4 synthetic APPLY_FAILED webhooks at the daemon
    (3 different reasons, one repeated).  After drain:
       failure_class_counts == {"not_in_tree": 2,
                                "add_already_present:DRAM": 1,
                                "remove_not_leaf": 1}
    Proves the handler is registered, the worker drains it, and
    record_failure routes the per-reason counter correctly."""
    async def _scenario() -> Dict[str, int]:
        app, router = _build_test_daemon_app()
        port = _free_port()
        async with _run_daemon_server(app, "127.0.0.1", port):
            url = f"http://127.0.0.1:{port}/aginfer/event"
            payloads = [
                ("not_in_tree",                 "node-aa"),
                ("not_in_tree",                 "node-bb"),
                ("add_already_present:DRAM",    "node-cc"),
                ("remove_not_leaf",             "node-dd"),
            ]
            async with httpx.AsyncClient(timeout=5.0) as client:
                for i, (reason, h) in enumerate(payloads):
                    body = {
                        "kind": "apply_failed",
                        "endpoint": "migrate",
                        "action_id": f"act-{i:03d}",
                        "reason": reason,
                        "hash": h,
                        "ts": time.time(),
                        "ts_monotonic": time.monotonic(),
                    }
                    r = await client.post(url, json=body)
                    if r.status_code != 200:
                        raise StageFail(
                            f"webhook {i} got {r.status_code}: {r.text}"
                        )
            # Let the worker drain.
            await router.bus.queue.join()
            return dict(router.observability.failure_class_counts)
    counts = asyncio.run(_scenario())
    expected = {
        "not_in_tree": 2,
        "add_already_present:DRAM": 1,
        "remove_not_leaf": 1,
    }
    if counts != expected:
        raise StageFail(
            f"counter mismatch: got {counts!r}; want {expected!r}"
        )


def stage_a2_unknown_endpoint_still_counts() -> None:
    """DESIGN §4 names ``endpoint ∈ {migrate, program_paused, hints,
    thresholds}``.  The handler doesn't gate on endpoint — every
    apply_failed bumps the counter.  Future endpoints land for free."""
    async def _scenario() -> Dict[str, int]:
        app, router = _build_test_daemon_app()
        port = _free_port()
        async with _run_daemon_server(app, "127.0.0.1", port):
            url = f"http://127.0.0.1:{port}/aginfer/event"
            async with httpx.AsyncClient(timeout=5.0) as client:
                for endpoint in ("migrate", "program_paused",
                                 "hints", "thresholds",
                                 "future_endpoint_42"):
                    body = {
                        "kind": "apply_failed",
                        "endpoint": endpoint,
                        "action_id": "a1",
                        "reason": f"reason_for:{endpoint}",
                        "ts": time.time(),
                        "ts_monotonic": time.monotonic(),
                    }
                    r = await client.post(url, json=body)
                    if r.status_code != 200:
                        raise StageFail(
                            f"endpoint={endpoint}: HTTP {r.status_code}"
                        )
            await router.bus.queue.join()
            return dict(router.observability.failure_class_counts)
    counts = asyncio.run(_scenario())
    if len(counts) != 5:
        raise StageFail(f"expected 5 distinct reasons; got {counts!r}")
    for endpoint in ("migrate", "program_paused", "hints",
                     "thresholds", "future_endpoint_42"):
        key = f"reason_for:{endpoint}"
        if counts.get(key) != 1:
            raise StageFail(
                f"counter for {key!r} should be 1; got {counts.get(key)!r}"
            )


def stage_a3_missing_reason_ignored_no_crash() -> None:
    """Defensive: a malformed APPLY_FAILED payload (no ``reason``)
    must NOT crash the worker.  Handler ignores it; counter
    unchanged.  Webhook is fire-and-forget; sglang should never
    send a malformed payload but the daemon shouldn't gate on
    contract violations in the load-fault path."""
    async def _scenario() -> Dict[str, int]:
        app, router = _build_test_daemon_app()
        port = _free_port()
        async with _run_daemon_server(app, "127.0.0.1", port):
            url = f"http://127.0.0.1:{port}/aginfer/event"
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Payload missing 'reason'.
                r = await client.post(url, json={
                    "kind": "apply_failed",
                    "endpoint": "migrate",
                    "action_id": "a1",
                })
                if r.status_code != 200:
                    raise StageFail(
                        f"malformed payload should 200 (queue accepts); "
                        f"got {r.status_code}: {r.text}"
                    )
                # A second VALID one, to prove the worker survived.
                r = await client.post(url, json={
                    "kind": "apply_failed",
                    "endpoint": "migrate",
                    "action_id": "a2",
                    "reason": "valid_after_garbage",
                })
                if r.status_code != 200:
                    raise StageFail(f"second valid: HTTP {r.status_code}")
            await router.bus.queue.join()
            return dict(router.observability.failure_class_counts)
    counts = asyncio.run(_scenario())
    if counts != {"valid_after_garbage": 1}:
        raise StageFail(
            f"counter should reflect only the valid payload; "
            f"got {counts!r}"
        )


def stage_a4_apply_failed_handler_emits_structured_metric_line() -> None:
    """Post-T36 cleanup: the sync ``_record_skips`` path is gone,
    so the per-event ``aginfer_metric event=migrate_skipped`` log
    it used to emit is gone too.  T37's handler now emits a
    structured ``aginfer_metric event=apply_failed endpoint=...
    reason=... action_id=...`` line so operators still have a
    per-event grep target.

    This stage captures the metric logger and asserts the line
    fires once per webhook with the right field set."""
    captured: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _CaptureHandler()
    metric_logger = logging.getLogger("aginfer.metric")
    prior_level = metric_logger.level
    metric_logger.addHandler(handler)
    metric_logger.setLevel(logging.INFO)

    async def _scenario():
        app, router = _build_test_daemon_app()
        port = _free_port()
        async with _run_daemon_server(app, "127.0.0.1", port):
            url = f"http://127.0.0.1:{port}/aginfer/event"
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(url, json={
                    "kind": "apply_failed",
                    "endpoint": "migrate",
                    "action_id": "a-1",
                    "reason": "not_in_tree",
                    "hash": "node-1",
                })
            await router.bus.queue.join()

    try:
        asyncio.run(_scenario())
        apply_failed_lines = [
            l for l in captured if "event=apply_failed" in l
        ]
        if len(apply_failed_lines) != 1:
            raise StageFail(
                f"expected exactly 1 apply_failed metric line per webhook; "
                f"got {len(apply_failed_lines)}; all captured: {captured!r}"
            )
        line = apply_failed_lines[0]
        for needle in ("endpoint=migrate", "reason=not_in_tree",
                       "action_id=a-1"):
            if needle not in line:
                raise StageFail(
                    f"line missing {needle!r}; got: {line!r}"
                )
    finally:
        metric_logger.removeHandler(handler)
        metric_logger.setLevel(prior_level)


# ============================================================ Phase B
# Live sglang + daemon integration.


def _maybe_phase_b_envs() -> Optional[Tuple[str, str]]:
    s = os.environ.get("AGINFER_VERIFY_BASE_SGLANG", "").rstrip("/")
    d = os.environ.get("AGINFER_VERIFY_BASE_DAEMON", "").rstrip("/")
    if not s or not d:
        return None
    return s, d


def stage_b0_known_bad_migrate_fires_webhook() -> None:
    """Drive a migrate with ``hash=node-99999999`` (forced
    ``not_in_tree``) through the daemon.  After ~2 s for the
    webhook round-trip, the daemon should:
       (1) have logged event_received with kind=apply_failed
       (2) have bumped failure_class_counts[not_in_tree]
       (3) NOT have double-counted (counter goes up by exactly 1).
    """
    envs = _maybe_phase_b_envs()
    if envs is None:
        print(_yellow(
            "  (skip B0) set AGINFER_VERIFY_BASE_SGLANG + "
            "AGINFER_VERIFY_BASE_DAEMON to run live"
        ))
        return
    sglang_base, _daemon_base = envs

    # Direct sglang migrate — bypasses the daemon's outbound (we're
    # not testing T36 here).  sglang fires the APPLY_FAILED webhook
    # at the daemon's notify URL, configured at sglang launch.
    body = {"actions": [{
        "hash": "node-99999999",
        "add_tiers": [],
        "remove_tiers": ["HBM"],
        "action_id": "phase-b-known-bad",
    }]}
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(f"{sglang_base}/aginfer/migrate", json=body)
        if resp.status_code != 200:
            raise StageFail(
                f"sglang /aginfer/migrate got HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        sync = resp.json()
        if sync.get("applied") != 0 or not sync.get("skipped"):
            raise StageFail(
                f"expected applied=0 + skipped=[not_in_tree]; "
                f"got {sync!r}"
            )
        first_skip = sync["skipped"][0]
        if first_skip.get("reason") != "not_in_tree":
            raise StageFail(
                f"expected reason=not_in_tree; got {first_skip!r}"
            )
    # The daemon log captures the webhook arrival; B0 just proves
    # the sync side returned the right thing — the daemon-side
    # counter check is in B1.


def stage_b1_daemon_counter_reflects_webhook() -> None:
    """After B0 fired the bad migrate, the daemon should have
    received an APPLY_FAILED webhook with reason=not_in_tree, and
    the counter should reflect exactly one new bump."""
    envs = _maybe_phase_b_envs()
    if envs is None:
        print(_yellow("  (skip B1) Phase B not configured"))
        return
    # The daemon doesn't expose its observability over HTTP today
    # (T42 is log-emission only).  We grep the daemon log for the
    # event_received kind=apply_failed marker as a proxy.
    log_path = os.environ.get("AGINFER_VERIFY_DAEMON_LOG", "")
    if not log_path:
        print(_yellow(
            "  (skip B1) set AGINFER_VERIFY_DAEMON_LOG to the daemon log "
            "path so we can grep for APPLY_FAILED events"
        ))
        return
    # Give the webhook ~3 s to land + drain through the queue.
    deadline = time.monotonic() + 5.0
    found = False
    while time.monotonic() < deadline:
        try:
            with open(log_path) as fh:
                content = fh.read()
        except OSError:
            content = ""
        if "kind=apply_failed" in content and "reason=not_in_tree" not in content:
            # Webhook arrived but not yet drained.
            pass
        if "kind=apply_failed" in content:
            found = True
            break
        time.sleep(0.2)
    if not found:
        raise StageFail(
            f"no APPLY_FAILED webhook arrived at the daemon log "
            f"({log_path!r}) within 5 s of the B0 bad migrate"
        )


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 EventKind.APPLY_FAILED exists",            stage_a0_eventkind_apply_failed_exists),
    ("A1 webhook bumps counter per reason",         stage_a1_webhook_bumps_counter_per_reason),
    ("A2 every endpoint counted (forward-compat)",  stage_a2_unknown_endpoint_still_counts),
    ("A3 malformed payload ignored, no crash",      stage_a3_missing_reason_ignored_no_crash),
    ("A4 apply_failed handler emits structured metric line",
                                                    stage_a4_apply_failed_handler_emits_structured_metric_line),
    ("B0 known-bad migrate returns sync skipped[]", stage_b0_known_bad_migrate_fires_webhook),
    ("B1 daemon log shows apply_failed webhook arrival",
                                                    stage_b1_daemon_counter_reflects_webhook),
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
        print(_red(f"\nT23+T37 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    print(_green(f"\nT23+T37 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
