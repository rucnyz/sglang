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
import time
from typing import Any, Awaitable, Callable, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ._observability import DaemonObservability
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
        observability_capacity: int = 1024,
        observability_summary_every_n: int = 200,
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
        # T42 — observability aggregator.  Handlers should call
        # router.fetch_state() (which is now the instrumented wrapper)
        # to get the state-fetch latency metric.  The worker records
        # queue depth + time-in-queue at dispatch entry.
        self.observability = DaemonObservability(
            capacity=observability_capacity,
            summary_every_n=observability_summary_every_n,
        )

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
        # DESIGN §5: pool_usage.HBM.subpools is the allocator-truth view
        # admission gates on.  Occupancy = max over subpools (admission
        # acts when ANY subpool crosses theta_hi, not when the aggregate
        # does — DESIGN §5 "Why two views" clause).  used/cap report
        # the SUMS across subpools (the tier-aggregate view), for the
        # synth payload's informational fields.
        subpools = state["pool_usage"]["HBM"]["subpools"]
        if subpools:
            occ = max(
                (e["used_bytes"] / e["cap_bytes"]) if e["cap_bytes"] > 0
                else 0.0
                for e in subpools.values()
            )
            used = sum(e["used_bytes"] for e in subpools.values())
            cap = sum(e["cap_bytes"] for e in subpools.values())
        else:
            occ = 0.0
            used = 0
            cap = 0
        if occ > self.theta_hi:
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
        """Public entry point used by handlers.  T42: wraps
        ``_fetch_state_impl`` with a perf_counter to record state-
        fetch latency into the observability aggregator.  Tests that
        need to stub the network should override ``_fetch_state_impl``
        (the timer still fires); replacing ``fetch_state`` itself
        bypasses the instrumentation.
        """
        t0 = time.perf_counter()
        result = await self._fetch_state_impl()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.observability.record_state_fetch(elapsed_ms)
        return result

    async def _fetch_state_impl(self) -> dict:
        """The HTTP-only path; override in tests to skip the network."""
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
            # T42 — observability: queue depth at dispatch entry +
            # time-in-queue.  qsize() reports remaining backlog AFTER
            # this pop (operator-meaningful "how far behind are we").
            t_dispatch = time.perf_counter()
            time_in_queue_ms = (
                (t_dispatch - event.enqueue_time) * 1000.0
                if event.enqueue_time > 0.0 else 0.0
            )
            qdepth_after_pop = self.bus.queue.qsize()
            self.observability.record_dispatch(
                qdepth=qdepth_after_pop,
                time_in_queue_ms=time_in_queue_ms,
            )
            _m(
                "event_received",
                kind=event.kind.value,
                sid=event.session if event.session else "-",
                qdepth=qdepth_after_pop,
                time_in_queue_ms=time_in_queue_ms,
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


async def _apply_failed_handler(event: Event, router: "EventRouter") -> None:
    """T37 (DESIGN §4 round-9 B4 / §6 L506): bump the per-reason
    observability counter and log.  The next event's joint_decide
    re-evaluates state and may re-issue any superseding action
    (DESIGN §10 idempotency makes re-issue safe).

    No retry / no immediate action here — the design is fire-and-
    forget + always-fresh state read at the next handler entry.
    """
    payload = event.payload or {}
    reason = payload.get("reason")
    endpoint = payload.get("endpoint")
    action_id = payload.get("action_id")
    if isinstance(reason, str) and reason:
        router.observability.record_failure(reason)
    logger.info(
        "aginfer apply_failed received: endpoint=%s action_id=%s reason=%s "
        "hash=%s",
        endpoint, action_id, reason, payload.get("hash"),
    )


def attach_apply_failed_handler(router: "EventRouter") -> None:
    """Register the T37 default APPLY_FAILED handler on ``router``.

    Always wired by main.py at daemon startup; kept as a separate
    function so verify probes can opt-in selectively."""
    router.set_handler(EventKind.APPLY_FAILED, _apply_failed_handler)


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
