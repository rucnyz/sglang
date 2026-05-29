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
        theta_hi: float = 0.7,
        theta_crit: float = 0.9,
    ) -> None:
        self.bus = bus
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        # Watermark thresholds used by cold_start_probe ONLY (the
        # real-time detector lives on sglang side).  Should match
        # the sglang launch's --aginfer-theta-hi / --aginfer-theta-crit
        # to avoid the cold-start synth firing on a different
        # threshold than steady-state webhooks.  Audit-round-1 M1.
        self.theta_hi = float(theta_hi)
        self.theta_crit = float(theta_crit)
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

    def set_handler(
        self, kind: EventKind, fn: HandlerFn, *, force: bool = False
    ) -> None:
        """Register a handler for ``kind``.

        Audit T8 round-2 R2-M2: a future subsystem (T10 GC, T9
        observer) registering after T8's admission would SILENTLY
        replace the admission composite — admission stops firing,
        no log.  This is symmetric to the round-1 B2 ordering bug
        (which only caught the "before" direction).

        Now: if the existing handler is a wrapped composite
        (attribute ``_aginfer_wrap`` set by
        ``attach_admission_controller``), refuse the overwrite
        unless ``force=True``.  Pass ``force=True`` for legitimate
        test re-attach.
        """
        prev = self._handlers.get(kind.value)
        if (
            prev is not None
            and getattr(prev, "_aginfer_wrap", False)
            and not force
        ):
            raise RuntimeError(
                f"set_handler({kind.name}): refusing to overwrite a "
                f"wrapped composite handler.  Pass force=True if you "
                f"intend to bypass admission_controller's wrap, OR "
                f"call attach_<your_layer> BEFORE "
                f"attach_admission_controller (which must be last)."
            )
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
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                # Audit-round-1 M4: don't silently swallow real bugs at
                # shutdown.  Log + continue (we're shutting down anyway).
                logger.exception("event_worker raised during shutdown")
            self._worker_task = None
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def cold_start_probe(self) -> None:
        """One-shot ``/aginfer/state`` fetch.  If HBM_occ > theta_hi,
        synthesise a local ``memory_pressure`` event so the daemon
        doesn't sit oblivious to pre-existing pressure.

        Uses ``self.theta_hi`` / ``self.theta_crit`` — set these to
        match the sglang launch flags so cold-start synth and steady-
        state webhooks agree on the watermark thresholds.  Audit-round-1
        M1.
        """
        try:
            state = await self.fetch_state()
        except Exception as exc:  # noqa: BLE001
            logger.warning("cold_start_probe: /aginfer/state failed: %s", exc)
            return
        hbm = state["tier_usage"]["HBM"]
        used = hbm["used_bytes"]
        cap = hbm["cap_bytes"]
        if cap > 0 and (used / cap) > self.theta_hi:
            occ = used / cap
            state_label = "HIGH" if occ < self.theta_crit else "CRITICAL"
            logger.info(
                "cold_start_probe: HBM occ %.3f > theta_hi %.3f; "
                "synthesising memory_pressure (state=%s)",
                occ, self.theta_hi, state_label,
            )
            await self.bus.emit(
                Event(
                    kind=EventKind.MEMORY_PRESSURE,
                    session=None,
                    payload={
                        "kind": "memory_pressure",
                        "state": state_label,
                        "prev_state": "OK",
                        "occ": occ,
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
        from ._metrics import m as _m
        while True:
            event = await self.bus.queue.get()
            self.events_received += 1
            _m(
                "event_received",
                kind=event.kind.value,
                sid=event.session if event.session else "-",
            )
            try:
                async with self._dispatch_lock:
                    handler = self._handlers.get(
                        event.kind.value, _noop_handler
                    )
                    await handler(event, self)
                self.events_handled += 1
            except asyncio.CancelledError:
                # Pair the get() with task_done() so a future
                # ``queue.join()`` doesn't hang on a cancel-time event.
                self.bus.queue.task_done()
                raise
            except Exception:  # noqa: BLE001
                self.handler_failures += 1
                logger.exception(
                    "aginfer event handler for %s raised; queue continues",
                    event.kind.value,
                )
            finally:
                # Audit-round-1 M5: pair get() with task_done() so
                # ``bus.queue.join()`` works.
                try:
                    self.bus.queue.task_done()
                except ValueError:
                    # Already called (cancel path above).  Tolerate.
                    pass


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
