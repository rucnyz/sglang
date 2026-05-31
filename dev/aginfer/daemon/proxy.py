"""Aginfer daemon HTTP proxy (T4).

Forwards OpenAI-style ``/v1/chat/completions`` to sglang, emits paper
§4 events to the event bus along the way, and gates new requests on
``program_tracker.wait_if_paused``.

Wire layout (paper §9 deployment):

    client  →  daemon (this module)  →  sglang
                │
                ├─→ event_bus  → event_worker (T7 / T8)
                └─→ program_tracker (T6)

Responsibilities (per t4/README.md):

  * Verbatim forward of /v1/chat/completions to the upstream sglang.
  * Streaming responses pass chunk-by-chunk (no buffering).
  * Extract ``program_id`` from ``extra_body.program_id`` OR top-level
    ``program_id`` OR ``X-Aginfer-Program`` header; sanitize to a
    short str via the same ``_sanitize_program_id`` used by sglang's
    Req constructor (so daemon and sglang agree on canonical form).
  * Emit 6 of paper §4's 8 event kinds:
      arrival:           SESSION_ARRIVAL (first-seen) + LLM_PREFILL
      response stream end: TOOL_CALL_START
      second arrival:    TOOL_CALL_END (emitted BEFORE the new
                         LLM_PREFILL for the same program_id)
      sub-dispatch:      SUB_DISPATCH_{BLOCKING,ASYNC} (NOT wired in
                         v1 because harbor / terminus-2 sub-dispatch
                         conventions are out of scope here; the proxy
                         exposes a hook for T7/T8 to populate later)

  * If ``program_tracker.is_paused(pid)``, the request awaits the
    resume event BEFORE forwarding.  This is the load-bearing piece
    that gives the daemon TA-style program-level back-pressure.

Cost ceiling (per t4/README.md):
  * Added latency vs direct sglang: < 2 ms p50, < 5 ms p99
  * Streaming throughput: ≥ 95 % of direct
  * Event emission: < 0.1 ms per event (put_nowait on unbounded queue)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


# Headers we forward upstream verbatim.  Body is forwarded verbatim; the
# rest of the request headers are mostly hop-by-hop and irrelevant to
# sglang, but a few semantic ones must survive (auth, tracing, request
# id, accept-encoding).
_FORWARD_HEADERS = {
    "authorization",
    "openai-organization",
    "openai-beta",
    "openai-project",
    "traceparent",
    "tracestate",
    "x-request-id",
    "accept-encoding",
}

from .events import Event, EventBus, EventKind
from .program_tracker import ProgramTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- helpers


def _sanitize_program_id(pid: Any) -> Optional[str]:
    """Mirror of sglang's Req-side sanitizer (schedule_batch.py).

    Coerces dict / int / list / very-long-str to a stable ≤64-char
    string, or returns None for empty / whitespace-only / unset.
    Daemon and sglang must agree on canonical form so the daemon's
    program_tracker key matches what sglang's session_ids contains.
    """
    _MAX_RECURSION = 8

    def _go(v: Any, depth: int = 0) -> Optional[str]:
        if depth > _MAX_RECURSION:
            return None
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            for elem in v:
                got = _go(elem, depth + 1)
                if got is not None:
                    return got
            return None
        if not isinstance(v, str):
            try:
                v = str(v)
            except Exception:
                return None
        v = v.strip()
        if not v:
            return None
        return v[:64]

    return _go(pid)


def _extract_program_id(
    body: Any,
    header_value: Optional[str],
) -> Optional[str]:
    """Pull program_id from (in priority order):
       1. body['program_id'] (top-level; what the OpenAI client unpacks
          ``extra_body`` to before sending)
       2. body['extra_body']['program_id'] (raw POST without OpenAI
          client; not the recommended path but a daemon may see it)
       3. X-Aginfer-Program header (escape hatch for clients that
          can't modify the body)

    Each source is sanitized; if a source sanitizes to None (e.g.
    whitespace-only body field), we fall through to the next source
    rather than locking in a None.
    """
    candidates: list = []
    if isinstance(body, dict):
        candidates.append(body.get("program_id"))
        extra = body.get("extra_body")
        if isinstance(extra, dict):
            candidates.append(extra.get("program_id"))
    candidates.append(header_value)
    for c in candidates:
        sanitized = _sanitize_program_id(c)
        if sanitized is not None:
            return sanitized
    return None


# ---------------------------------------------------------------- app factory


def create_app(
    *,
    sglang_base_url: str,
    event_bus: Optional[EventBus] = None,
    program_tracker: Optional[ProgramTracker] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    enable_event_router: bool = True,
    theta_hi: float = 0.7,
    theta_crit: float = 0.9,
    observability_summary_every_n: int = 200,
) -> FastAPI:
    """Build the daemon's FastAPI app.

    Dependency-injectable so the verify can pass in a stub sglang URL
    and a recording event_bus.  Defaults create fresh singletons.

    ``enable_event_router=True`` (default) mounts the T5
    ``POST /aginfer/event`` endpoint + spawns the event_worker.
    Set False for proxy-only tests.
    """
    app = FastAPI(title="aginfer-daemon", version="0.1")
    app.state.sglang_base_url = sglang_base_url.rstrip("/")
    app.state.event_bus = event_bus or EventBus()
    app.state.program_tracker = program_tracker or ProgramTracker()
    # The HTTP client is created on startup so it can be tied to the
    # event loop running the FastAPI app.  Tests may pass one in.
    app.state.http_client = http_client
    app.state.owns_http_client = http_client is None
    app.state.event_router = None
    app.state.enable_event_router = enable_event_router

    # Attach the event router's routes at create-time (routes must be
    # registered before app start).  The worker task itself is spawned
    # on startup so it ties to the running event loop.
    if enable_event_router:
        from .event_router import EventRouter, attach_event_routes

        router = EventRouter(
            bus=app.state.event_bus,
            sglang_base_url=app.state.sglang_base_url,
            http_client=None,  # router lazily creates its own on start()
            theta_hi=theta_hi,
            theta_crit=theta_crit,
            observability_summary_every_n=observability_summary_every_n,
        )
        attach_event_routes(app, router)
        app.state.event_router = router

    @app.on_event("startup")
    async def _startup() -> None:
        if app.state.http_client is None:
            # No bounded timeout on the upstream call: a long
            # generation legitimately takes minutes.  Connect timeout
            # IS bounded so a dead sglang fails fast.
            app.state.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)
            )
        if app.state.event_router is not None:
            await app.state.event_router.start()
            await app.state.event_router.cold_start_probe()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if app.state.event_router is not None:
            await app.state.event_router.stop()
            app.state.event_router = None
        if app.state.owns_http_client and app.state.http_client is not None:
            await app.state.http_client.aclose()

    @app.get("/health")
    async def health() -> Any:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        raw: Request,
        x_aginfer_program: Optional[str] = Header(default=None),
    ) -> Any:
        try:
            body = await raw.json()
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": f"invalid JSON: {exc!s}"}},
                status_code=400,
            )

        pid = _extract_program_id(body, x_aginfer_program)
        bus: EventBus = app.state.event_bus
        tracker: ProgramTracker = app.state.program_tracker
        client: httpx.AsyncClient = app.state.http_client

        # 1. Gate on pause/resume.
        if pid is not None:
            await tracker.wait_if_paused(pid)

            # 2. Emit arrival-side paper §4 events.
            #    - First-ever request for pid -> SESSION_ARRIVAL.
            #    - Returning program (last seen in ACTING after a
            #      prior TOOL_CALL_START) -> TOOL_CALL_END first,
            #      THEN LLM_PREFILL.
            #    - In all cases -> LLM_PREFILL.
            from .program_tracker import State

            prior_state = tracker.state(pid)
            if bus.mark_known(pid):
                await bus.emit(Event(EventKind.SESSION_ARRIVAL, session=pid))
            elif prior_state is State.ACTING:
                await bus.emit(Event(EventKind.TOOL_CALL_END, session=pid))
            await bus.emit(Event(EventKind.LLM_PREFILL, session=pid))

            # 3. State transition: -> REASONING.
            tracker.observe_arrival(pid)

        # 4. Forward to sglang.
        upstream_url = f"{app.state.sglang_base_url}/v1/chat/completions"
        # Audit round-1 MINOR: use ``is True`` rather than truthy so a
        # buggy client sending ``"stream": "false"`` doesn't trip into
        # the streaming branch.
        is_stream = (
            isinstance(body, dict) and body.get("stream") is True
        )
        forwarded_headers = {
            k: v
            for k, v in raw.headers.items()
            if k.lower() in _FORWARD_HEADERS
        }

        async def _emit_completion() -> None:
            """Run after the request completes (stream end / unary /
            failure).  bus.emit is put_nowait on an unbounded queue
            and tracker.observe_completion is a dict op — neither can
            raise under v1's contract, so we let exceptions propagate
            (handle()'s try/except catches if they ever do)."""
            if pid is None:
                return
            await bus.emit(Event(EventKind.TOOL_CALL_START, session=pid))
            tracker.observe_completion(pid)

        if not is_stream:
            try:
                resp = await client.post(
                    upstream_url, json=body, headers=forwarded_headers or None
                )
                # Always pass response body through verbatim with the
                # upstream content-type.  Round-1 audit BLOCKER 1: the
                # previous code wrapped non-JSON bodies in JSONResponse,
                # which serialised them as a quoted string.
                body_bytes = resp.content
                ct = resp.headers.get("content-type", "application/octet-stream")
                pass_resp: Response = Response(
                    content=body_bytes,
                    status_code=resp.status_code,
                    media_type=ct,
                )
            except httpx.RequestError as exc:
                pass_resp = JSONResponse(
                    {"error": {"message": f"upstream sglang error: {exc!s}"}},
                    status_code=502,
                )
            except Exception as exc:  # noqa: BLE001 -- round-1 MAJOR
                # Any other exception (httpx.HTTPError subclasses, JSON
                # decode failure on the upstream body, etc.) MUST NOT
                # leave the program stuck in REASONING.
                logger.warning(
                    "unary forwarding raised; returning 502 and recovering "
                    "program_tracker",
                    exc_info=True,
                )
                pass_resp = JSONResponse(
                    {
                        "error": {
                            "message": f"daemon proxy error: {type(exc).__name__}: {exc!s}"
                        }
                    },
                    status_code=502,
                )
            finally:
                await _emit_completion()
            return pass_resp

        # Streaming SSE pass-through.
        #
        # Round-1 audit BLOCKER 2: starting StreamingResponse before we
        # know the upstream connection is alive commits 200 + SSE
        # headers; a connect-time failure would then show up as a
        # synthetic in-band error frame rather than a real 502.  Probe
        # the connect by initiating the stream BEFORE returning.
        try:
            req_ctx = client.stream(
                "POST", upstream_url, json=body, headers=forwarded_headers or None
            )
            upstream_resp = await req_ctx.__aenter__()
        except httpx.RequestError as exc:
            await _emit_completion()
            return JSONResponse(
                {"error": {"message": f"upstream sglang error: {exc!s}"}},
                status_code=502,
            )
        except Exception as exc:  # noqa: BLE001
            await _emit_completion()
            logger.warning(
                "stream connect raised; returning 502", exc_info=True
            )
            return JSONResponse(
                {
                    "error": {
                        "message": f"daemon proxy error: {type(exc).__name__}: {exc!s}"
                    }
                },
                status_code=502,
            )

        async def _stream() -> Any:
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                # Mid-stream upstream break.  Emit an SSE error frame +
                # [DONE] so the client sees the truncation.
                err = (
                    b'data: {"error": {"message": "upstream sglang error: '
                    + str(exc).encode("utf-8", "backslashreplace").replace(b'"', b"'")
                    + b'"}}\n\n'
                )
                yield err
                yield b"data: [DONE]\n\n"
            finally:
                try:
                    await req_ctx.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    # Cleanup-path exception: log + continue.  Silent
                    # swallow would mask connection-pool leaks.
                    logger.exception("proxy: req_ctx cleanup raised")
                await _emit_completion()

        # Preserve upstream content-type if present (defaults to SSE).
        upstream_ct = upstream_resp.headers.get(
            "content-type", "text/event-stream"
        )
        return StreamingResponse(_stream(), media_type=upstream_ct)

    return app


__all__ = ["create_app"]
