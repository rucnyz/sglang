"""T5 verify: sglang→daemon webhook + daemon event router.

Two layers:

  Layer A (daemon side, pure asyncio + stub-sglang): tests the
  ``POST /aginfer/event`` endpoint, the event_worker's serial-
  dispatch / idempotent / cold-start-probe contract, and the
  noop handler stats.

  Layer B (sglang side, full sglang launch): tests the watermark
  detector + outbound POST + retry / backoff / fire-and-forget
  via a stub HTTP capturer.  This is gated behind a flag because
  it needs a free GPU and ~30 s of real sglang traffic.

Usage:
    # Layer A only (default; ~5 s, no GPU):
    python verify/t5/verify.py

    # Layer A + Layer B (requires CUDA_VISIBLE_DEVICES + a free port):
    AGINFER_T5_FULL=1 CUDA_VISIBLE_DEVICES=5 python verify/t5/verify.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


_AGINFER_ROOT = Path(__file__).resolve().parents[2]
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.event_router import EventRouter, attach_event_routes  # noqa: E402
from daemon.program_tracker import ProgramTracker  # noqa: E402
from daemon.proxy import create_app  # noqa: E402


# ---------------------------------------------------------------- helpers


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@asynccontextmanager
async def run_server(app: FastAPI, host: str, port: int):
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.025)
    if not server.started:
        raise RuntimeError(f"server on :{port} failed to start within 5 s")
    try:
        yield server
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()


def make_stub_sglang(state_hbm_used: int = 0, state_hbm_cap: int = 65536) -> FastAPI:
    """Stub /aginfer/state returning whatever HBM usage you set."""
    app = FastAPI()
    app.state.hbm_used = state_hbm_used
    app.state.hbm_cap = state_hbm_cap

    @app.get("/aginfer/state")
    async def _state() -> Any:
        # DESIGN §5 schema (post-T17).  Stub emits the FULL shape so
        # daemon's cold_start_probe + build_paper_state can parse it;
        # most fields are zero placeholders since this test only
        # cares about HBM occ.
        return {
            "time_counter": 0,
            "throughput_ema": {"prefill_bps": 0.0,
                               "decode_per_program": {}},
            "pool_usage": {
                "HBM":  {"subpools": {"full": {
                    "used_bytes": app.state.hbm_used,
                    "cap_bytes": app.state.hbm_cap,
                    "available_bytes": max(0, app.state.hbm_cap - app.state.hbm_used),
                    "evictable_bytes": 0,
                    "page_bytes": 1,
                }}},
                "DRAM": {"subpools": {"full": {
                    "used_bytes": 0, "cap_bytes": 0,
                    "available_bytes": 0, "evictable_bytes": 0,
                    "page_bytes": 1}}},
                "DISK": {"subpools": {"full": {
                    "used_bytes": 0, "cap_bytes": 0,
                    "available_bytes": 0, "evictable_bytes": 0,
                    "page_bytes": 1}}},
            },
            "per_program_usage": {},
            "units": [],
            "link_stats": {
                link: {"peak_bw_bps": 1, "recent_throughput_bps": 0,
                       "time_since_last_sample_s": 1.0e12}
                for link in ("HBM->DRAM", "DRAM->HBM",
                             "DRAM->DISK", "DISK->DRAM")
            },
            "tier_holding_cost": {
                t: {"full": {"h_max_per_byte_sec": 0.0}}
                for t in ("HBM", "DRAM", "DISK")
            },
        }

    return app


def make_stub_webhook_capturer(*, fail_first_n: int = 0) -> FastAPI:
    """Captures POSTs to /aginfer/event for assertion.  Optionally
    returns 500 for the first ``fail_first_n`` requests to exercise
    the retry/backoff path.
    """
    app = FastAPI()
    app.state.captured: List[Dict[str, Any]] = []
    app.state.fail_count = 0
    app.state.fail_first_n = fail_first_n

    @app.post("/aginfer/event")
    async def _evt(raw: Request) -> Any:
        body = await raw.json()
        if app.state.fail_count < app.state.fail_first_n:
            app.state.fail_count += 1
            return JSONResponse({"err": "stub-fail"}, status_code=500)
        app.state.captured.append(body)
        return {"ok": True}

    return app


# ================================================================ Layer A


async def la_basic_enqueue_and_handle(daemon_url: str, router: EventRouter) -> None:
    """[A1] POST /aginfer/event enqueues; worker drains via noop handler."""
    body = {
        "kind": "memory_pressure",
        "state": "HIGH",
        "prev_state": "OK",
        "occ": 0.8,
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(f"{daemon_url}/aginfer/event", json=body)
    assert r.status_code == 200, r.text
    # Yield until the worker drains it (it just runs noop_handler).
    for _ in range(20):
        if router.events_handled >= 1:
            break
        await asyncio.sleep(0.02)
    assert router.events_handled >= 1, router.events_handled


async def la_serial_dispatch(daemon_url: str, router: EventRouter) -> None:
    """[A2] 100 events enqueued in a burst; handlers run serially.

    Replace the noop handler with one that records concurrency.
    Also pins M5 (task_done): after drain, asserts the queue's join
    completes within 2 s -- a regression that removes the
    ``finally: task_done()`` pairing in event_router would make
    queue.join() hang forever.
    """
    in_flight = 0
    max_in_flight = 0
    handled = 0
    lock = asyncio.Lock()

    async def _instrumented(evt: Event, _r: EventRouter) -> None:
        nonlocal in_flight, max_in_flight, handled
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.001)  # simulate handler work
        async with lock:
            in_flight -= 1
            handled += 1

    # Override handler for memory_pressure.
    router.set_handler(EventKind.MEMORY_PRESSURE, _instrumented)

    body = {"kind": "memory_pressure", "state": "HIGH", "prev_state": "OK", "occ": 0.8}
    async with httpx.AsyncClient(timeout=10.0) as client:
        await asyncio.gather(
            *[client.post(f"{daemon_url}/aginfer/event", json=body) for _ in range(100)]
        )
    # Wait for drain.
    for _ in range(500):
        if handled >= 100:
            break
        await asyncio.sleep(0.02)
    assert handled == 100, handled
    assert max_in_flight == 1, (
        f"observed max_in_flight={max_in_flight} -- event_worker not serial"
    )
    # Round-1-of-tests audit A4 -- M5 fix (task_done pairing) is
    # unpinned without a queue.join() somewhere.  Without the fix,
    # this hangs forever.
    await asyncio.wait_for(router.bus.queue.join(), timeout=2.0)


async def la_idempotent(daemon_url: str, router: EventRouter) -> None:
    """[A3] Same memory_pressure event twice -> same final state.

    With the noop handler, "same final state" reduces to: handler
    called twice, no exceptions, downstream side-effect count is
    expected.  Use a custom handler that counts calls AND records
    "last-seen occ"; the final value must equal the LAST payload's
    occ, regardless of duplicate POSTs.
    """
    calls = 0
    last_occ = -1.0

    async def _h(evt: Event, _r: EventRouter) -> None:
        nonlocal calls, last_occ
        calls += 1
        last_occ = evt.payload.get("occ", -1)

    router.set_handler(EventKind.MEMORY_PRESSURE, _h)

    payload = {
        "kind": "memory_pressure", "state": "HIGH", "prev_state": "OK", "occ": 0.85,
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        for _ in range(3):
            await client.post(f"{daemon_url}/aginfer/event", json=payload)
    for _ in range(50):
        if calls >= 3:
            break
        await asyncio.sleep(0.02)
    assert calls == 3
    assert last_occ == 0.85, last_occ  # idempotent final state


async def la_handler_raises_doesnt_stall(daemon_url: str, router: EventRouter) -> None:
    """[A4] WORST CASE: handler raises -> queue continues; failure counted."""
    failures_before = router.handler_failures
    good_calls = 0

    async def _raises(evt: Event, _r: EventRouter) -> None:
        raise RuntimeError("intentional handler crash")

    async def _good(evt: Event, _r: EventRouter) -> None:
        nonlocal good_calls
        good_calls += 1

    router.set_handler(EventKind.MEMORY_PRESSURE, _raises)
    router.set_handler(EventKind.PRESSURE_RESOLVED, _good)

    async with httpx.AsyncClient(timeout=5.0) as client:
        # 5 bad + 5 good interleaved.
        for _ in range(5):
            await client.post(
                f"{daemon_url}/aginfer/event",
                json={"kind": "memory_pressure", "occ": 0.9, "state": "HIGH", "prev_state": "OK"},
            )
            await client.post(
                f"{daemon_url}/aginfer/event",
                json={"kind": "pressure_resolved", "occ": 0.5, "state": "OK", "prev_state": "HIGH"},
            )
    for _ in range(100):
        if good_calls >= 5 and router.handler_failures - failures_before >= 5:
            break
        await asyncio.sleep(0.02)
    assert good_calls == 5
    assert router.handler_failures - failures_before == 5


async def la_unknown_kind_400(daemon_url: str) -> None:
    """[A5] WORST CASE: unknown event kind returns 400, not 5xx."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(
            f"{daemon_url}/aginfer/event",
            json={"kind": "WAT_IS_THIS", "occ": 0.5},
        )
    assert r.status_code == 400, r.status_code


async def la_still_high_routes_as_memory_pressure(
    daemon_url: str, router: EventRouter
) -> None:
    """[A6] sglang's heartbeat ``still_high`` kind routes to
    MEMORY_PRESSURE handler (same handler, daemon doesn't care
    whether it's a transition or a plateau heartbeat)."""
    saw = 0

    async def _h(evt: Event, _r: EventRouter) -> None:
        nonlocal saw
        if evt.payload.get("kind") == "still_high":
            saw += 1

    router.set_handler(EventKind.MEMORY_PRESSURE, _h)

    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(
            f"{daemon_url}/aginfer/event",
            json={"kind": "still_high", "occ": 0.85, "state": "HIGH", "prev_state": "HIGH"},
        )
    assert r.status_code == 200
    for _ in range(50):
        if saw >= 1:
            break
        await asyncio.sleep(0.02)
    assert saw == 1


async def la_cold_start_probe() -> None:
    """[A7] Cold-start probe synthesises memory_pressure when stub
    sglang already reports HBM_occ > theta_hi.

    Runs TWO scenarios:
      (a) default thresholds (theta_hi=0.7); occ=0.885 -> synth.
      (b) non-default thresholds (theta_hi=0.5); occ=0.6 -> synth.
          A regression that hardcodes 0.7 would silently miss (b)
          (audit A5 / M1 fix).
    """
    async def _scenario(*, used, cap, theta_hi, theta_crit, expect_synth):
        stub = make_stub_sglang(state_hbm_used=used, state_hbm_cap=cap)
        stub_port = _free_port()
        bus = EventBus()
        saw_synthetic = 0

        async def _h(evt: Event, _r: EventRouter) -> None:
            nonlocal saw_synthetic
            if evt.payload.get("synthetic"):
                saw_synthetic += 1

        daemon = create_app(
            sglang_base_url=f"http://127.0.0.1:{stub_port}",
            event_bus=bus,
            program_tracker=ProgramTracker(),
            theta_hi=theta_hi,
            theta_crit=theta_crit,
        )
        daemon.state.event_router.set_handler(EventKind.MEMORY_PRESSURE, _h)
        daemon_port = _free_port()
        async with run_server(stub, "127.0.0.1", stub_port):
            async with run_server(daemon, "127.0.0.1", daemon_port):
                for _ in range(50):
                    if saw_synthetic >= 1:
                        break
                    await asyncio.sleep(0.02)
        return saw_synthetic

    # (a) default 0.7/0.9 — 0.885 trips HIGH.
    a = await _scenario(used=58000, cap=65536, theta_hi=0.7, theta_crit=0.9,
                        expect_synth=True)
    assert a == 1, f"default-threshold scenario: expected 1 synth, got {a}"

    # (b) non-default 0.5/0.7 — occ=0.6 trips ONLY if probe honors the
    # plumbed threshold (audit M1 fix).  Pre-fix (hardcoded 0.7) would
    # NOT synth here.
    b = await _scenario(used=39000, cap=65536, theta_hi=0.5, theta_crit=0.7,
                        expect_synth=True)
    assert b == 1, (
        f"non-default-threshold scenario: expected 1 synth at occ=0.595 "
        f"with theta_hi=0.5, got {b}.  Cold-start probe is hardcoding "
        f"thresholds (M1 fix regressed)."
    )


async def la_event_handler_latency(
    daemon_url: str, router: EventRouter
) -> dict:
    """[A8] COST: 5-run multi-trial event-arrival -> handler latency.

    Per memory:feedback-latency-multi-run.  Replace handler with one
    that records receive time; measure (handler-start - POST-end).
    """
    N_RUNS = 5
    N_PER_RUN = 50
    run_p50: list[float] = []
    run_p99: list[float] = []

    last_post_ts = [0.0]
    handler_lats = []

    async def _timed(evt: Event, _r: EventRouter) -> None:
        # Recorded at handler entry.
        lat = (time.perf_counter() - last_post_ts[0]) * 1000
        handler_lats.append(lat)

    router.set_handler(EventKind.MEMORY_PRESSURE, _timed)
    body = {"kind": "memory_pressure", "state": "HIGH", "prev_state": "OK", "occ": 0.8}

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Warmup
        for _ in range(5):
            last_post_ts[0] = time.perf_counter()
            await client.post(f"{daemon_url}/aginfer/event", json=body)
            await asyncio.sleep(0.005)
        handler_lats.clear()

        for _ in range(N_RUNS):
            run_lats: list[float] = []
            for _ in range(N_PER_RUN):
                last_post_ts[0] = time.perf_counter()
                await client.post(f"{daemon_url}/aginfer/event", json=body)
                # Wait until the corresponding handler-call latency is in.
                start = time.perf_counter()
                while (
                    len(handler_lats) == 0
                    and time.perf_counter() - start < 1.0
                ):
                    await asyncio.sleep(0.001)
                if handler_lats:
                    run_lats.append(handler_lats.pop(0))
            run_lats.sort()
            run_p50.append(run_lats[N_PER_RUN // 2])
            run_p99.append(run_lats[max(0, int(N_PER_RUN * 0.99) - 1)])

    stats = {
        "N_runs": N_RUNS,
        "N_per_run": N_PER_RUN,
        "p50_mean": statistics.mean(run_p50),
        "p50_std": statistics.stdev(run_p50),
        "p99_mean": statistics.mean(run_p99),
        "p99_std": statistics.stdev(run_p99),
    }
    print(
        f"    arrival->handler latency ({N_RUNS} runs × {N_PER_RUN}): "
        f"p50 {stats['p50_mean']:.2f} ± {stats['p50_std']:.2f} ms; "
        f"p99 {stats['p99_mean']:.2f} ± {stats['p99_std']:.2f} ms"
    )
    # Audit round-2 ("audit of tests"): previous floor was p99<80 ms
    # but actual is ~0.7 ms — a 100× regression would slip through.
    # Tighten to mean+3σ < 5 ms (covers ~99.7 % of trials under normal
    # noise; catches a 5–7× regression).  In-process loopback FastAPI
    # over an asyncio.Queue: even a cold first run should land well
    # under 5 ms.  If a future GIL/loop change pushes us > 5 ms, the
    # docstring claim is also broken and we want this to fail.
    p50_envelope = stats["p50_mean"] + 3.0 * stats["p50_std"]
    p99_envelope = stats["p99_mean"] + 3.0 * stats["p99_std"]
    assert p50_envelope < 5.0, (
        f"p50 mean+3σ = {p50_envelope:.2f} ms exceeds 5 ms "
        f"(p50_mean={stats['p50_mean']:.2f}, p50_std={stats['p50_std']:.2f})"
    )
    assert p99_envelope < 5.0, (
        f"p99 mean+3σ = {p99_envelope:.2f} ms exceeds 5 ms "
        f"(p99_mean={stats['p99_mean']:.2f}, p99_std={stats['p99_std']:.2f})"
    )
    return stats


async def la_firer_retry_and_payload() -> None:
    """[A11] BLOCKER A3 (test-audit): firer retries on 500 + payload
    includes both ``ts`` (wall) and ``ts_monotonic``.

    Drive AginferWebhookFirer directly with a stub that returns 500
    twice then 200.  Assert the capturer ultimately recorded the
    payload AND that the body has both time fields (audit M3 fix).
    """
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.managers.aginfer_webhook import AginferWebhookFirer

    capturer = make_stub_webhook_capturer(fail_first_n=2)
    cap_port = _free_port()
    async with run_server(capturer, "127.0.0.1", cap_port):
        firer = AginferWebhookFirer(
            notify_url=f"http://127.0.0.1:{cap_port}",  # bare URL: tests B1 too
            heartbeat_s=5.0,
            theta_hi=0.7,
            theta_crit=0.9,
        )
        try:
            firer.maybe_fire(58000 / 65536)  # ~0.885 -> HIGH (< 0.9 crit)
            # The firer's background thread will retry 0.1 + 0.4 s = 0.5 s.
            # Plus some slack.
            for _ in range(50):  # up to 5 s
                if capturer.state.captured:
                    break
                await asyncio.sleep(0.1)
        finally:
            firer.close()

    assert capturer.state.captured, (
        "retry-on-500 BROKEN: capturer never received the body after 2 "
        "stub-failures.  Check the firer's retry loop in "
        "aginfer_webhook.py:_send."
    )
    assert capturer.state.fail_count == 2, capturer.state.fail_count
    body = capturer.state.captured[0]
    # Audit A1 (M3 ts_monotonic pin):
    assert "ts" in body and isinstance(body["ts"], (int, float)), body.keys()
    assert "ts_monotonic" in body and isinstance(body["ts_monotonic"], (int, float)), (
        f"M3 fix is unpinned: payload missing ts_monotonic; got keys {sorted(body)}"
    )
    # Verify the firer's classification of 0.885 with default
    # theta_hi=0.7 / theta_crit=0.9 is HIGH (0.7 <= 0.885 < 0.9).
    assert body["state"] == "HIGH", body["state"]


def la_firer_url_append() -> None:
    """[A10] BLOCKER B1 (round-1 audit): firer appends /aginfer/event
    if user passes a bare base URL.

    Pure unit test on AginferWebhookFirer.__init__ — no actual HTTP
    needed.  Verifies the firer normalises both shapes to the same
    final notify URL.
    """
    # The import here avoids needing the sglang source on path during
    # other Layer A tests.
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.managers.aginfer_webhook import AginferWebhookFirer

    cases = [
        ("http://daemon:8765", "http://daemon:8765/aginfer/event"),
        ("http://daemon:8765/", "http://daemon:8765/aginfer/event"),
        ("http://daemon:8765/aginfer/event", "http://daemon:8765/aginfer/event"),
        ("http://daemon:8765/aginfer/event/", "http://daemon:8765/aginfer/event"),
    ]
    for inp, want in cases:
        f = AginferWebhookFirer(notify_url=inp)
        try:
            assert f.notify_url == want, (
                f"firer URL mismatch: notify_url={inp!r} -> "
                f"firer.notify_url={f.notify_url!r}, expected {want!r}"
            )
        finally:
            f.close()


async def la_no_periodic_timer_in_source() -> None:
    """[A9] Contract: NO `time.sleep`, `asyncio.sleep`, or
    `loop.call_later` / `loop.call_at` in the daemon's event-router
    / proxy / event-bus modules (event-driven only).  The watermark
    *heartbeat* lives on sglang side; daemon receives events purely
    reactively.

    Audit round-2 ("audit of tests"): the previous version's
    docstring claimed sleep/timing primitives were forbidden, but
    the actual ``forbidden`` tuple only included ``call_later`` and
    ``call_at`` — so a regression introducing ``asyncio.sleep(0.1)``
    polling in the worker would have been invisible.  We now forbid:

      - .sleep           (asyncio.sleep / time.sleep / loop.sleep)
      - .call_later      (delayed callback)
      - .call_at         (absolute-time callback)

    Note (post-T42): ``perf_counter`` was previously banned as a
    heuristic proxy for "running a timer-driven loop", but it is the
    standard high-res measurement primitive — T42's state-fetch
    latency timer and ``Event.enqueue_time`` stamp legitimately use
    it for measurement, not control flow.  Polling manifests as
    sleep/call_later/call_at; those bans remain.

    Anywhere these appear in the daemon's event-driven hot path is a
    contract violation.
    """
    import ast
    import inspect

    from daemon import event_router, events as events_mod, proxy

    # Audit round-3: previously scanned `inspect.getsource(EventBus)`
    # which only returns the class body — a module-level `import time;
    # _LAST_FIRE = time.time()` in daemon/events.py would slip past the
    # AST scan entirely.  Scan the WHOLE module for each daemon source
    # so module-level state can't bypass the contract.
    sources = [
        ("daemon/event_router.py", inspect.getsource(event_router)),
        ("daemon/proxy.py", inspect.getsource(proxy)),
        ("daemon/events.py", inspect.getsource(events_mod)),
    ]
    forbidden = ("sleep", "call_later", "call_at")
    for name, src in sources:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                raise AssertionError(
                    f"polling/timer primitive `.{node.attr}` found in "
                    f"{name} -- event-driven contract forbids it"
                )
            # Defend against `from asyncio import sleep` + `sleep(...)`
            # bypass.
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in forbidden:
                    raise AssertionError(
                        f"polling/timer primitive `{node.func.id}(...)` "
                        f"found in {name} -- event-driven contract "
                        f"forbids it (covers `from asyncio import sleep`)"
                    )


# ================================================================ Layer B
# (sglang side)


async def lb_watermark_transitions_and_heartbeat() -> Dict[str, Any]:
    """[B1+B2] Full sglang launch + stub webhook capturer.

    Drive HBM occupancy past theta_hi by submitting distinct chat
    completions; observe state transitions + plateau heartbeat.

    This is gated by AGINFER_T5_FULL=1 because it needs a free GPU.
    """
    import subprocess

    cap_port = _free_port()
    sglang_port = _free_port()
    capturer = make_stub_webhook_capturer()

    async with run_server(capturer, "127.0.0.1", cap_port):
        notify_url = f"http://127.0.0.1:{cap_port}/aginfer/event"
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "5")
        log_path = (
            Path(__file__).parent
            / "results"
            / f"sglang_t5_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        # Aggressive thresholds so a small workload trips HIGH quickly.
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", "Qwen/Qwen3-0.6B",
            "--host", "127.0.0.1",
            "--port", str(sglang_port),
            "--tp", "1",
            "--mem-fraction-static", "0.15",
            "--max-total-tokens", "4096",
            "--trust-remote-code",
            "--attention-backend", "flashinfer",
            "--aginfer-notify-url", notify_url,
            "--aginfer-heartbeat-s", "2.0",
            "--aginfer-theta-hi", "0.30",
            "--aginfer-theta-crit", "0.60",
        ]
        env = os.environ.copy()
        env["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            # Wait for sglang.
            up = False
            for _ in range(100):
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        r = await client.get(f"http://127.0.0.1:{sglang_port}/health")
                        if r.status_code == 200:
                            up = True
                            break
                except Exception:
                    pass
                await asyncio.sleep(3.0)
            assert up, "sglang failed to start within 5 minutes"

            # Drive workload to push HBM up.  Distinct prompts so cache
            # doesn't dedupe; max_tokens small.
            async with httpx.AsyncClient(timeout=60.0) as client:
                for i in range(60):
                    try:
                        await client.post(
                            f"http://127.0.0.1:{sglang_port}/v1/chat/completions",
                            json={
                                "model": "Qwen/Qwen3-0.6B",
                                "messages": [
                                    {"role": "user", "content": f"distinct prompt {i}: tell me about prime {i}."}
                                ],
                                "max_tokens": 32,
                                "temperature": 0.0,
                            },
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)

                # Plateau hold (let heartbeats fire).
                await asyncio.sleep(6.0)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        kinds = [c.get("kind") for c in capturer.state.captured]
        states = [c.get("state") for c in capturer.state.captured]
        # We expect to see at least ONE memory_pressure (OK -> HIGH/CRITICAL)
        # and >=2 still_high heartbeats (plateau hold is ~6 s with
        # heartbeat_s=2.0 -> expect ~2-3; round-tests audit A7 tightens
        # this from the original >=1 floor).
        print(
            f"    captured {len(kinds)} webhooks; kinds={kinds[:10]}{'...' if len(kinds) > 10 else ''}"
        )
        print(
            f"    states sequence: {states[:10]}{'...' if len(states) > 10 else ''}"
        )
        return {
            "kinds": kinds,
            "states": states,
            "raw": capturer.state.captured,
        }


# ================================================================ main


async def main() -> None:
    print("=== T5 verify: sglang->daemon webhook + daemon event router ===")
    print()

    # ---- Layer A: daemon side, no sglang launch ----
    stub = make_stub_sglang()  # default: 0 / 65536, no synthesised event
    stub_port = _free_port()
    bus = EventBus()
    tracker = ProgramTracker()
    daemon = create_app(
        sglang_base_url=f"http://127.0.0.1:{stub_port}",
        event_bus=bus,
        program_tracker=tracker,
    )
    daemon_port = _free_port()
    daemon_url = f"http://127.0.0.1:{daemon_port}"

    async with run_server(stub, "127.0.0.1", stub_port):
        async with run_server(daemon, "127.0.0.1", daemon_port):
            router = daemon.state.event_router
            assert router is not None
            await la_basic_enqueue_and_handle(daemon_url, router)
            print("[A1] POST /aginfer/event enqueues; worker drains via noop ✓")

            await la_serial_dispatch(daemon_url, router)
            print("[A2] 100 events; handlers strictly serial (max_in_flight==1) ✓")

            await la_idempotent(daemon_url, router)
            print("[A3] same memory_pressure event 3× → final state matches "
                  "LAST payload (idempotent) ✓")

            await la_handler_raises_doesnt_stall(daemon_url, router)
            print("[A4] WORST CASE: handler raises → queue continues; "
                  "5/5 failures + 5/5 successes ✓")

            await la_unknown_kind_400(daemon_url)
            print("[A5] WORST CASE: unknown event kind → 400 (not 5xx) ✓")

            await la_still_high_routes_as_memory_pressure(daemon_url, router)
            print("[A6] sglang's `still_high` heartbeat routes to "
                  "MEMORY_PRESSURE handler ✓")

            lat = await la_event_handler_latency(daemon_url, router)
            print("[A8] COST: arrival->handler p99 within 80 ms budget "
                  "(5-run mean ± std) ✓")
            global _T5_STATS  # noqa: PLW0603
            _T5_STATS = lat

    await la_cold_start_probe()
    print("[A7] cold_start_probe: stub reports HBM occ 0.885 → daemon "
          "synthesises memory_pressure on startup ✓")

    await la_no_periodic_timer_in_source()
    print("[A9] contract: no `call_later` / `call_at` in daemon "
          "event-router source (event-driven only) ✓")

    la_firer_url_append()
    print("[A10] BLOCKER B1 fix: AginferWebhookFirer appends "
          "/aginfer/event when user passes a bare base URL ✓")

    await la_firer_retry_and_payload()
    print("[A11] BLOCKER A3 + A1 fixes: firer retries 2× on 500 + "
          "payload carries `ts_monotonic` (audit-round-1 M3) ✓")

    # ---- Layer B: sglang side, full launch (gated) ----
    if os.environ.get("AGINFER_T5_FULL") == "1":
        print()
        print("[B] full sglang launch + watermark webhook test")
        result = await lb_watermark_transitions_and_heartbeat()
        kinds = result["kinds"]
        assert "memory_pressure" in kinds, (
            f"never saw OK->HIGH transition; kinds={kinds!r}"
        )
        # During the plateau hold (~6 s with heartbeat 2 s) we expect
        # 2-3 still_high heartbeats.  Round-tests audit A7: tightened
        # from >=1 to >=2 so a heartbeat-throttle regression that
        # fires exactly once would be caught.
        n_still = kinds.count("still_high")
        assert n_still >= 2, (
            f"plateau hold ~6 s @ heartbeat_s=2 expected >=2 still_high; "
            f"got {n_still}; kinds={kinds!r}"
        )
        print(f"[B] PASS: {len(kinds)} webhooks, "
              f"{kinds.count('memory_pressure')} memory_pressure, "
              f"{n_still} still_high heartbeats")
    else:
        print()
        print("[B] (skipped — set AGINFER_T5_FULL=1 + CUDA_VISIBLE_DEVICES "
              "to run the full sglang+watermark test)")

    if _T5_STATS:
        print()
        print("Latency summary (record in RESULTS):")
        print(
            f"  arrival->handler p50 = {_T5_STATS['p50_mean']:.2f} "
            f"± {_T5_STATS['p50_std']:.2f} ms   "
            f"(N={_T5_STATS['N_runs']} × {_T5_STATS['N_per_run']})"
        )
        print(
            f"  arrival->handler p99 = {_T5_STATS['p99_mean']:.2f} "
            f"± {_T5_STATS['p99_std']:.2f} ms"
        )

    print()
    print("=== T5 PASSED ===")


_T5_STATS: dict = {}


if __name__ == "__main__":
    asyncio.run(main())
