"""Aginfer daemon event router (T5).

Receives the sglang webhook ``POST /aginfer/event`` AND drains the
shared event queue serially via a single ``event_worker`` task.

Design contract (verify/t5/README.md):

* ``POST /aginfer/event`` enqueues into the daemon's shared
  ``asyncio.Queue`` (the same EventBus the proxy emits to).
* A SINGLE ``event_worker`` consumes the queue serially.  No two
  handlers run concurrently; guarded by ``asyncio.Lock``.
* Each handler refetches ``/aginfer/state`` at entry; never trusts
  the payload's snapshot.  v1 ``handle()`` is a stub — T7/T8 will
  wire real kv_scheduler / admission_controller logic.
* Handlers are IDEMPOTENT: same event received twice produces the
  same final state.
* At daemon startup, perform one ``/aginfer/state`` fetch; if
  ``HBM_occ > theta_hi``, synthesise a local ``memory_pressure``
  event so cold-start doesn't sit oblivious.

No new periodic timer.  The watermark heartbeat lives on sglang's
side (managers/aginfer_webhook.py).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .events import Event, EventBus, EventKind

logger = logging.getLogger(__name__)


HandlerFn = Callable[[Event, "EventRouter"], Awaitable[None]]


class EventRouter:
    """Owns the single event_worker task + handler registry.

    v1 ships with a default ``noop_handler`` that just logs.  T7 / T8
    will register real handlers via ``set_handler(kind, fn)``.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        sglang_base_url: str,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.bus = bus
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        # Per-kind handler.  Defaults to noop; T7 / T8 override.
        self._handlers: dict[str, HandlerFn] = {}
        # Serialise handler execution.  paper §9: "no two handlers
        # run concurrently".
        self._dispatch_lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task[None]] = None
        # Stats for tests.
        self.events_received: int = 0
        self.events_handled: int = 0
        self.handler_failures: int = 0

    # ---- registration ----

    def set_handler(self, kind: EventKind, fn: HandlerFn) -> None:
        self._handlers[kind.value] = fn

    # ---- lifecycle ----

    async def start(self) -> None:
        """Start the worker.  Idempotent."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(5.0)
            )
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._event_worker(), name="aginfer-event-worker"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
            self._worker_task = None
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def cold_start_probe(self) -> None:
        """One-shot ``/aginfer/state`` fetch.  If HBM_occ > theta_hi,
        synthesise a local ``memory_pressure`` event so the daemon
        doesn't sit oblivious to pre-existing pressure.

        v1: theta_hi is hardcoded to 0.7 here (matches sglang default
        --aginfer-theta-hi); a future PR can plumb it through.
        """
        try:
            state = await self.fetch_state()
        except Exception as exc:  # noqa: BLE001
            logger.warning("cold_start_probe: /aginfer/state failed: %s", exc)
            return
        tier_usage = state.get("tier_usage", {}).get("HBM", {})
        used = tier_usage.get("used_bytes", 0)
        cap = tier_usage.get("cap_bytes", 0)
        if cap > 0 and (used / cap) > 0.7:
            logger.info(
                "cold_start_probe: HBM occ %.2f > 0.7; synthesising memory_pressure",
                used / cap,
            )
            await self.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    session=None,
                    payload={
                        "kind": "memory_pressure",
                        "state": "HIGH" if used / cap < 0.9 else "CRITICAL",
                        "prev_state": "OK",
                        "occ": used / cap,
                        "used_bytes": int(used),
                        "cap_bytes": int(cap),
                        "synthetic": True,
                    },
                )
            )

    # ---- helpers used by handlers ----

    async def fetch_state(self) -> dict:
        assert self._client is not None
        r = await self._client.get(f"{self.sglang_base_url}/aginfer/state")
        r.raise_for_status()
        return r.json()

    # ---- worker ----

    async def _event_worker(self) -> None:
        while True:
            event = await self.bus.queue.get()
            self.events_received += 1
            try:
                async with self._dispatch_lock:
                    handler = self._handlers.get(
                        event.kind.value, _noop_handler
                    )
                    await handler(event, self)
                self.events_handled += 1
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                self.handler_failures += 1
                logger.exception(
                    "aginfer event handler for %s raised; queue continues",
                    event.kind.value,
                )


async def _noop_handler(event: Event, router: "EventRouter") -> None:
    """Default handler used when T7 / T8 haven't registered one yet.

    Logs the event and (for memory_pressure kinds) does a state
    fetch so a future migration step has a fresh snapshot.
    """
    logger.info(
        "aginfer event (noop): %s session=%s payload-keys=%s",
        event.kind.value,
        event.session,
        sorted(event.payload.keys()),
    )
    if event.kind in (EventKind.MEMORY_PRESSURE, EventKind.PRESSURE_RESOLVED):
        try:
            await router.fetch_state()
        except Exception:
            logger.warning("state fetch in noop handler failed", exc_info=True)


def attach_event_routes(app: FastAPI, router: EventRouter) -> None:
    """Mount the ``POST /aginfer/event`` endpoint on the daemon app."""

    @app.post("/aginfer/event")
    async def aginfer_event(raw: Request) -> Any:
        try:
            payload = await raw.json()
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": f"invalid JSON: {exc!s}"}},
                status_code=400,
            )
        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": {"message": "payload must be a JSON object"}},
                status_code=400,
            )
        kind_str = payload.get("kind")
        try:
            kind = EventKind(kind_str)
        except ValueError:
            # Accept "still_high" as a memory_pressure heartbeat.
            if kind_str == "still_high":
                kind = EventKind.MEMORY_PRESSURE
            else:
                return JSONResponse(
                    {
                        "error": {
                            "message": f"unknown event kind: {kind_str!r}"
                        }
                    },
                    status_code=400,
                )
        evt = Event(
            kind=kind,
            session=payload.get("session"),
            payload=payload,
        )
        await router.bus.emit(evt)
        return {"status": "queued"}
