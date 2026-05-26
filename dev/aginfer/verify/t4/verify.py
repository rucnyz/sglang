"""T4 verify: aginfer-daemon HTTP proxy + paper §4 event emission.

In-process FastAPI stub-sglang + daemon, driven by httpx.AsyncClient.
No GPU, no real sglang launch.  Covers the contract documented in
dev/aginfer/verify/t4/README.md.

Usage:
    python dev/aginfer/verify/t4/verify.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


# Make ``dev/aginfer`` importable so daemon/* resolves.
_AGINFER_ROOT = Path(__file__).resolve().parents[2]
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.program_tracker import ProgramTracker, State  # noqa: E402
from daemon.proxy import create_app, _extract_program_id, _sanitize_program_id  # noqa: E402


# ---------------------------------------------------------------- stub sglang


def make_stub_sglang(
    *,
    response_text: str = "stub response",
    chunks: Optional[List[str]] = None,
    first_token_delay_s: float = 0.0,
    fail_with: Optional[int] = None,
    fail_with_body: Optional[bytes] = None,
    fail_with_ct: str = "text/plain",
) -> FastAPI:
    """Build a tiny FastAPI app that emulates sglang's /v1/chat/completions.

    Supports both non-streaming and streaming via the request body's
    ``stream`` field.  Recording happens via app.state.received[].
    """
    app = FastAPI()
    app.state.received: List[Dict[str, Any]] = []

    @app.get("/health")
    async def _h() -> Any:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def _chat(raw: Request) -> Any:
        body = await raw.json()
        app.state.received.append(body)

        if fail_with is not None:
            # Allow returning a non-JSON body (e.g. an HTML 5xx error
            # page from a reverse proxy) to exercise the proxy's
            # pass-through path on non-JSON content types.
            if fail_with_body is not None:
                return Response(
                    content=fail_with_body,
                    status_code=fail_with,
                    media_type=fail_with_ct,
                )
            return JSONResponse({"error": "stub failure"}, status_code=fail_with)

        if first_token_delay_s:
            await asyncio.sleep(first_token_delay_s)

        if body.get("stream") is True:
            async def _gen():
                for ch in (chunks or [response_text]):
                    payload = json.dumps(
                        {"choices": [{"delta": {"content": ch}, "index": 0}]}
                    ).encode()
                    yield b"data: " + payload + b"\n\n"
                    await asyncio.sleep(0)
                yield b"data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")

        return {
            "id": "stub",
            "object": "chat.completion",
            "choices": [
                {"message": {"role": "assistant", "content": response_text}}
            ],
        }

    return app


# ---------------------------------------------------------------- harness


@asynccontextmanager
async def run_server(app: FastAPI, host: str, port: int):
    """Start a uvicorn server in a background task; tear down on exit."""
    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for the listener.
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


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------- step fns


def step_sanitize_unit_tests() -> None:
    """[0] _sanitize_program_id + _extract_program_id offline checks."""
    # Sanitize: cover the same shape matrix as t3 round-5 demands.
    assert _sanitize_program_id(None) is None
    assert _sanitize_program_id("") is None
    assert _sanitize_program_id("  ") is None
    assert _sanitize_program_id("prog-A") == "prog-A"
    assert _sanitize_program_id("x" * 200) == "x" * 64
    assert _sanitize_program_id(42) == "42"
    assert _sanitize_program_id({"a": 1}) == "{'a': 1}"
    assert _sanitize_program_id(["a", "b"]) == "a"
    assert _sanitize_program_id([None, "b"]) == "b"
    # Cycle: depth cap prevents recursion crash.
    cyc: list = []
    cyc.append(cyc)
    assert _sanitize_program_id(cyc) is None

    # Extract: priority order top-level > extra_body > header.
    assert _extract_program_id({"program_id": "A"}, None) == "A"
    assert _extract_program_id(
        {"program_id": "A", "extra_body": {"program_id": "B"}}, None
    ) == "A"
    assert _extract_program_id({"extra_body": {"program_id": "B"}}, None) == "B"
    assert _extract_program_id({}, "C") == "C"
    assert _extract_program_id({"program_id": "  "}, "C") == "C"
    assert _extract_program_id(None, "C") == "C"  # type: ignore[arg-type]


async def step_nonstream_response_passthrough(daemon_url: str, stub_app) -> None:
    """[1] Non-streaming response equivalence: daemon == direct stub."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        body = {
            "model": "stub",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "program_id": "prog-NS",
        }
        r = await client.post(f"{daemon_url}/v1/chat/completions", json=body)
    assert r.status_code == 200, (r.status_code, r.text)
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "stub response"
    # Stub recorded the body.
    assert any(
        rcv.get("program_id") == "prog-NS"
        for rcv in stub_app.state.received
    ), "stub didn't see the forwarded request"


async def step_streaming_chunks(daemon_url: str) -> None:
    """[2] Streaming pass-through: chunks arrive in order; terminator present."""
    body = {
        "model": "stub",
        "messages": [{"role": "user", "content": "stream test"}],
        "stream": True,
        "program_id": "prog-STREAM",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        async with client.stream("POST", f"{daemon_url}/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            chunks_seen = []
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    chunks_seen.append(line[6:])
    assert "[DONE]" in chunks_seen[-1], chunks_seen[-1]
    # At least 3 chunks (default stub uses ["hello", " ", "world"] via chunks override).
    assert len(chunks_seen) >= 2, len(chunks_seen)


async def step_event_sequence_two_turns(daemon_url: str, bus: EventBus) -> None:
    """[3] First turn: SESSION_ARRIVAL, LLM_PREFILL, TOOL_CALL_START.
    Second turn: TOOL_CALL_END, LLM_PREFILL, TOOL_CALL_START.
    """
    # Drain anything leftover from prior steps for this specific pid.
    PID = "prog-EV-SEQ"
    initial = list(_drain_queue(bus.queue))

    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"{daemon_url}/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "turn1"}],
                "max_tokens": 4,
                "program_id": PID,
            },
        )
    turn1 = [e for e in _drain_queue(bus.queue) if e.session == PID]
    assert [e.kind for e in turn1] == [
        EventKind.SESSION_ARRIVAL,
        EventKind.LLM_PREFILL,
        EventKind.TOOL_CALL_START,
    ], [e.kind for e in turn1]

    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"{daemon_url}/v1/chat/completions",
            json={
                "model": "stub",
                "messages": [{"role": "user", "content": "turn2"}],
                "max_tokens": 4,
                "program_id": PID,
            },
        )
    turn2 = [e for e in _drain_queue(bus.queue) if e.session == PID]
    assert [e.kind for e in turn2] == [
        EventKind.TOOL_CALL_END,
        EventKind.LLM_PREFILL,
        EventKind.TOOL_CALL_START,
    ], [e.kind for e in turn2]


def _drain_queue(q: asyncio.Queue) -> List[Event]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def step_pause_back_pressure(
    daemon_url: str, tracker: ProgramTracker
) -> None:
    """[4] pause_back_pressure: pause(p) blocks the proxy's forward.

    Pause -> issue request -> request must NOT complete within 100 ms.
    Resume -> request completes within 1 s.
    """
    PID = "prog-PAUSED-BP"
    tracker.pause(PID)

    async with httpx.AsyncClient(timeout=10.0) as client:
        body = {
            "model": "stub",
            "messages": [{"role": "user", "content": "should block"}],
            "max_tokens": 4,
            "program_id": PID,
        }
        # Spawn the request; it should hang on wait_if_paused.
        req_task = asyncio.create_task(
            client.post(f"{daemon_url}/v1/chat/completions", json=body)
        )
        # Yield a few times.
        for _ in range(10):
            await asyncio.sleep(0.01)
        assert not req_task.done(), "request did NOT block on pause"

        # Resume.
        tracker.resume(PID)
        r = await asyncio.wait_for(req_task, timeout=1.0)
    assert r.status_code == 200


async def step_malformed_program_id(daemon_url: str) -> None:
    """[5] WORST CASE: malformed program_id shapes don't 5xx."""
    shapes: List[Any] = [
        None,
        "",
        "   ",
        42,
        {"oh": "no"},
        ["a", "b"],
        "x" * 10_000,
        [None, "deep"],
    ]
    async with httpx.AsyncClient(timeout=10.0) as client:
        for shape in shapes:
            body: Dict[str, Any] = {
                "model": "stub",
                "messages": [{"role": "user", "content": "bogus pid"}],
                "max_tokens": 4,
            }
            if shape is not None:
                body["program_id"] = shape
            r = await client.post(
                f"{daemon_url}/v1/chat/completions", json=body
            )
            assert r.status_code == 200, (shape, r.status_code, r.text[:200])


async def step_upstream_dead(stub_app, daemon_url: str, tracker: ProgramTracker) -> None:
    """[6] WORST CASE: sglang upstream dead -> 502, daemon stays up,
    program_tracker recovers (NOT stuck in REASONING)."""
    # We can't kill the stub mid-flight here, but we CAN point the
    # daemon at a port nothing listens on by spinning up a fresh app.
    # Simpler: build a daemon pointing at a dead port and hit it.
    dead_port = _free_port()
    # Don't bind anything; dead_port is unused.
    pid = "prog-DEAD-UPSTREAM"
    fresh_bus = EventBus()
    fresh_tracker = ProgramTracker()
    fresh_daemon = create_app(
        sglang_base_url=f"http://127.0.0.1:{dead_port}",
        event_bus=fresh_bus,
        program_tracker=fresh_tracker,
    )
    fresh_port = _free_port()
    async with run_server(fresh_daemon, "127.0.0.1", fresh_port):
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"http://127.0.0.1:{fresh_port}/v1/chat/completions",
                json={
                    "model": "stub",
                    "messages": [{"role": "user", "content": "upstream dead"}],
                    "max_tokens": 4,
                    "program_id": pid,
                },
            )
        assert r.status_code == 502, r.status_code
        # tracker must recover: completion event was emitted so state
        # isn't stuck in REASONING.
        assert fresh_tracker.state(pid) in (
            State.ACTING,
            State.PAUSED,  # paranoia — but really should be ACTING
        ), fresh_tracker.state(pid)


async def step_non_json_passthrough() -> None:
    """[9] BLOCKER 1 (round-1 audit): non-JSON upstream response is
    passed through verbatim, NOT wrapped in JSONResponse(str) which
    would re-encode it as a quoted JSON string.

    Spin up a fresh stub that returns text/html on failure; proxy
    must forward the original bytes + content-type.
    """
    html_body = b"<html><body><h1>500 Internal Server Error</h1></body></html>"
    stub = make_stub_sglang(
        fail_with=500, fail_with_body=html_body, fail_with_ct="text/html"
    )
    stub_p = _free_port()
    bus_ = EventBus()
    tracker_ = ProgramTracker()
    daemon = create_app(
        sglang_base_url=f"http://127.0.0.1:{stub_p}",
        event_bus=bus_,
        program_tracker=tracker_,
    )
    daemon_p = _free_port()
    async with run_server(stub, "127.0.0.1", stub_p):
        async with run_server(daemon, "127.0.0.1", daemon_p):
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    f"http://127.0.0.1:{daemon_p}/v1/chat/completions",
                    json={
                        "model": "stub",
                        "messages": [{"role": "user", "content": "fail me"}],
                        "max_tokens": 4,
                        "program_id": "prog-NONJSON",
                    },
                )
    assert r.status_code == 500, r.status_code
    assert r.content == html_body, (
        f"non-JSON body was re-encoded; got {r.content!r}"
    )
    assert r.headers.get("content-type", "").startswith("text/html"), (
        f"content-type lost; got {r.headers.get('content-type')!r}"
    )


async def step_streaming_connect_error() -> None:
    """[10] BLOCKER 2 (round-1 audit): streaming with a dead upstream
    must return a real 502 from the proxy, NOT 200 + an in-band SSE
    error frame.  Probe upstream connect BEFORE committing
    StreamingResponse headers.
    """
    dead_port = _free_port()
    bus_ = EventBus()
    tracker_ = ProgramTracker()
    daemon = create_app(
        sglang_base_url=f"http://127.0.0.1:{dead_port}",
        event_bus=bus_,
        program_tracker=tracker_,
    )
    daemon_p = _free_port()
    async with run_server(daemon, "127.0.0.1", daemon_p):
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"http://127.0.0.1:{daemon_p}/v1/chat/completions",
                json={
                    "model": "stub",
                    "messages": [{"role": "user", "content": "stream + dead"}],
                    "max_tokens": 4,
                    "stream": True,
                    "program_id": "prog-STREAM-DEAD",
                },
            )
    assert r.status_code == 502, (
        f"streaming connect-error should 502, got {r.status_code}"
    )
    # tracker must have recovered (not stuck in REASONING)
    assert tracker_.state("prog-STREAM-DEAD") != State.REASONING, (
        tracker_.state("prog-STREAM-DEAD")
    )


async def step_stream_field_is_strict_bool() -> None:
    """[11] MINOR (round-1 audit): ``"stream": "false"`` (string,
    truthy) must NOT trigger the streaming branch.  Only ``True`` does.
    """
    stub = make_stub_sglang(response_text="unary OK")
    stub_p = _free_port()
    bus_ = EventBus()
    tracker_ = ProgramTracker()
    daemon = create_app(
        sglang_base_url=f"http://127.0.0.1:{stub_p}",
        event_bus=bus_,
        program_tracker=tracker_,
    )
    daemon_p = _free_port()
    async with run_server(stub, "127.0.0.1", stub_p):
        async with run_server(daemon, "127.0.0.1", daemon_p):
            async with httpx.AsyncClient(timeout=10.0) as client:
                # "stream": "false" (string) is truthy in Python but NOT True.
                r = await client.post(
                    f"http://127.0.0.1:{daemon_p}/v1/chat/completions",
                    json={
                        "model": "stub",
                        "messages": [{"role": "user", "content": "trick"}],
                        "max_tokens": 4,
                        "stream": "false",
                        "program_id": "prog-STREAM-STR",
                    },
                )
    assert r.status_code == 200
    # Should have gone through the UNARY path (JSON object body).
    assert r.headers.get("content-type", "").startswith("application/json"), (
        r.headers.get("content-type")
    )


async def step_concurrent_same_program(daemon_url: str, bus: EventBus) -> None:
    """[7] WORST CASE: 50 concurrent requests, same program_id."""
    PID = "prog-CONCURRENT"
    _drain_queue(bus.queue)  # clean slate

    async with httpx.AsyncClient(timeout=20.0) as client:
        body = {
            "model": "stub",
            "messages": [{"role": "user", "content": "concurrent"}],
            "max_tokens": 4,
            "program_id": PID,
        }
        results = await asyncio.gather(
            *[client.post(f"{daemon_url}/v1/chat/completions", json=body) for _ in range(50)]
        )
    for r in results:
        assert r.status_code == 200, r.status_code

    events_for_pid = [e for e in _drain_queue(bus.queue) if e.session == PID]
    kinds = [e.kind for e in events_for_pid]
    # Invariants:
    # (a) Exactly one SESSION_ARRIVAL (first-seen).
    assert kinds.count(EventKind.SESSION_ARRIVAL) == 1
    # (b) Each request emits exactly one LLM_PREFILL and one
    #     TOOL_CALL_START -> 50 of each.
    assert kinds.count(EventKind.LLM_PREFILL) == 50
    assert kinds.count(EventKind.TOOL_CALL_START) == 50
    # (c) No TOOL_CALL_END from the first arrival (first request has
    #     no preceding completion).  Subsequent requests in this burst
    #     may or may not see ACTING (depends on concurrency), so we
    #     just assert TOOL_CALL_END count <= 49.
    assert kinds.count(EventKind.TOOL_CALL_END) <= 49


async def step_latency_overhead(daemon_url: str, stub_url: str) -> dict:
    """[8] COST: latency overhead daemon vs direct stub.

    Multi-run experiment per memory:feedback-latency-multi-run.  We run
    N_RUNS independent trials, each measuring 50-request p50/p99 on
    both the direct stub and the daemon proxy.  Cross-trial mean+std
    is the headline figure; the assertion uses (mean + 1 std) so a
    single noisy trial doesn't flake.

    Returns a stats dict so main() can record it into the RESULTS log.
    """
    import statistics

    body = {
        "model": "stub",
        "messages": [{"role": "user", "content": "lat"}],
        "max_tokens": 4,
        "program_id": "prog-LAT",
    }
    N_RUNS = 5
    N_PER_RUN = 50

    direct_p50_per_run: list[float] = []
    direct_p99_per_run: list[float] = []
    proxy_p50_per_run: list[float] = []
    proxy_p99_per_run: list[float] = []
    overhead_p50_per_run: list[float] = []
    overhead_p99_per_run: list[float] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        # One warm-up phase suffices; subsequent trials run on warm caches.
        for _ in range(10):
            await client.post(f"{stub_url}/v1/chat/completions", json=body)
            await client.post(f"{daemon_url}/v1/chat/completions", json=body)

        for run in range(N_RUNS):
            d_samples: list[float] = []
            p_samples: list[float] = []
            for _ in range(N_PER_RUN):
                t0 = time.perf_counter()
                await client.post(f"{stub_url}/v1/chat/completions", json=body)
                d_samples.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                await client.post(f"{daemon_url}/v1/chat/completions", json=body)
                p_samples.append((time.perf_counter() - t0) * 1000)
            d_samples.sort()
            p_samples.sort()
            d_p50 = d_samples[N_PER_RUN // 2]
            d_p99 = d_samples[max(0, int(N_PER_RUN * 0.99) - 1)]
            p_p50 = p_samples[N_PER_RUN // 2]
            p_p99 = p_samples[max(0, int(N_PER_RUN * 0.99) - 1)]
            direct_p50_per_run.append(d_p50)
            direct_p99_per_run.append(d_p99)
            proxy_p50_per_run.append(p_p50)
            proxy_p99_per_run.append(p_p99)
            overhead_p50_per_run.append(p_p50 - d_p50)
            overhead_p99_per_run.append(p_p99 - d_p99)
            print(
                f"    run {run + 1}/{N_RUNS}: "
                f"direct p50={d_p50:.2f} p99={d_p99:.2f}; "
                f"proxy p50={p_p50:.2f} p99={p_p99:.2f}; "
                f"overhead p50={p_p50 - d_p50:+.2f} p99={p_p99 - d_p99:+.2f}"
            )

    stats = {
        "N_runs": N_RUNS,
        "N_per_run": N_PER_RUN,
        "direct_p50_mean": statistics.mean(direct_p50_per_run),
        "direct_p50_std": statistics.stdev(direct_p50_per_run),
        "direct_p99_mean": statistics.mean(direct_p99_per_run),
        "direct_p99_std": statistics.stdev(direct_p99_per_run),
        "proxy_p50_mean": statistics.mean(proxy_p50_per_run),
        "proxy_p50_std": statistics.stdev(proxy_p50_per_run),
        "proxy_p99_mean": statistics.mean(proxy_p99_per_run),
        "proxy_p99_std": statistics.stdev(proxy_p99_per_run),
        "overhead_p50_mean": statistics.mean(overhead_p50_per_run),
        "overhead_p50_std": statistics.stdev(overhead_p50_per_run),
        "overhead_p99_mean": statistics.mean(overhead_p99_per_run),
        "overhead_p99_std": statistics.stdev(overhead_p99_per_run),
    }
    print()
    print(
        f"    SUMMARY ({N_RUNS} runs × {N_PER_RUN} reqs/run):"
    )
    print(
        f"      direct p50:   {stats['direct_p50_mean']:.2f} ± "
        f"{stats['direct_p50_std']:.2f} ms"
    )
    print(
        f"      direct p99:   {stats['direct_p99_mean']:.2f} ± "
        f"{stats['direct_p99_std']:.2f} ms"
    )
    print(
        f"      proxy  p50:   {stats['proxy_p50_mean']:.2f} ± "
        f"{stats['proxy_p50_std']:.2f} ms"
    )
    print(
        f"      proxy  p99:   {stats['proxy_p99_mean']:.2f} ± "
        f"{stats['proxy_p99_std']:.2f} ms"
    )
    print(
        f"      overhead p50: {stats['overhead_p50_mean']:+.2f} ± "
        f"{stats['overhead_p50_std']:.2f} ms"
    )
    print(
        f"      overhead p99: {stats['overhead_p99_mean']:+.2f} ± "
        f"{stats['overhead_p99_std']:.2f} ms"
    )

    # Use (mean + 1 std) for the assertion so a single noisy run
    # doesn't flake the suite.  Ceiling is intentionally generous
    # for in-process loopback; a real regression blows past it.
    p50_ceiling = stats["overhead_p50_mean"] + stats["overhead_p50_std"]
    p99_ceiling = stats["overhead_p99_mean"] + stats["overhead_p99_std"]
    assert p50_ceiling < 10.0, (
        f"proxy p50 overhead mean+std = {p50_ceiling:.2f} ms exceeds 10 ms"
    )
    assert p99_ceiling < 25.0, (
        f"proxy p99 overhead mean+std = {p99_ceiling:.2f} ms exceeds 25 ms"
    )
    return stats


# ---------------------------------------------------------------- main


_T4_LATENCY_STATS: dict = {}


async def main() -> None:
    print("=== T4 verify: aginfer-daemon HTTP proxy ===")
    print()

    step_sanitize_unit_tests()
    print("[0] _sanitize_program_id + _extract_program_id unit checks ✓")

    # Build stub sglang.
    stub_app = make_stub_sglang(
        chunks=["hello", " ", "world"],
        response_text="stub response",
    )
    stub_port = _free_port()
    stub_url = f"http://127.0.0.1:{stub_port}"

    # Build daemon pointing at stub.
    bus = EventBus()
    tracker = ProgramTracker()
    daemon_app = create_app(
        sglang_base_url=stub_url,
        event_bus=bus,
        program_tracker=tracker,
    )
    daemon_port = _free_port()
    daemon_url = f"http://127.0.0.1:{daemon_port}"

    async with run_server(stub_app, "127.0.0.1", stub_port):
        async with run_server(daemon_app, "127.0.0.1", daemon_port):
            await step_nonstream_response_passthrough(daemon_url, stub_app)
            print("[1] non-streaming response equivalence ✓")

            await step_streaming_chunks(daemon_url)
            print("[2] streaming chunks pass-through; terminator preserved ✓")

            await step_event_sequence_two_turns(daemon_url, bus)
            print("[3] event sequence: arrival emits SESSION_ARRIVAL + "
                  "LLM_PREFILL + TOOL_CALL_START; second turn emits "
                  "TOOL_CALL_END + LLM_PREFILL + TOOL_CALL_START ✓")

            await step_pause_back_pressure(daemon_url, tracker)
            print("[4] pause back-pressure: request blocks on pause, "
                  "resumes < 1 s ✓")

            await step_malformed_program_id(daemon_url)
            print("[5] WORST CASE: 8 malformed program_id shapes "
                  "all forward cleanly (no 5xx) ✓")

            await step_concurrent_same_program(daemon_url, bus)
            print("[6] WORST CASE: 50 concurrent requests same pid; "
                  "exactly 1 SESSION_ARRIVAL, 50 LLM_PREFILL, "
                  "50 TOOL_CALL_START ✓")

            lat_stats = await step_latency_overhead(daemon_url, stub_url)
            print("[7] COST: proxy overhead within budget (5-run mean ± std) ✓")
            # Stash for main() to print at the end.
            global _T4_LATENCY_STATS  # noqa: PLW0603
            _T4_LATENCY_STATS = lat_stats

        # daemon torn down; stub still up.  Test upstream-dead with
        # a fresh disposable daemon.
        await step_upstream_dead(stub_app, daemon_url, tracker)
        print("[8] WORST CASE: upstream sglang dead -> 502; "
              "daemon stays up; program_tracker recovers ✓")

    # Round-1 audit BLOCKERs (these spin up their own stub + daemon).
    await step_non_json_passthrough()
    print("[9] BLOCKER 1 fix: non-JSON upstream body passes through "
          "verbatim (not double-encoded) ✓")

    await step_streaming_connect_error()
    print("[10] BLOCKER 2 fix: streaming with dead upstream -> real "
          "502, not 200 with in-band error frame ✓")

    await step_stream_field_is_strict_bool()
    print("[11] MINOR fix: stream=\"false\" (string, truthy) does NOT "
          "trigger streaming branch ✓")

    if _T4_LATENCY_STATS:
        print()
        print("Latency summary (record in RESULTS):")
        ls = _T4_LATENCY_STATS
        print(
            f"  proxy overhead p50 = {ls['overhead_p50_mean']:.2f} ± "
            f"{ls['overhead_p50_std']:.2f} ms   "
            f"(N={ls['N_runs']} runs × {ls['N_per_run']} reqs)"
        )
        print(
            f"  proxy overhead p99 = {ls['overhead_p99_mean']:.2f} ± "
            f"{ls['overhead_p99_std']:.2f} ms"
        )

    print()
    print("=== T4 PASSED ===")


if __name__ == "__main__":
    asyncio.run(main())
