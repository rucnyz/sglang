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
from typing import Any, Awaitable, Callable, Dict, Optional

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
        theta_lo: float = 0.55,
        theta_crit: float = 0.9,
        heartbeat_s: float = 5.0,
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
        # T22 (#155): theta_lo + heartbeat_s also live on the router so
        # the canonical GET /aginfer/thresholds endpoint can serve all
        # four in one shot.  cold_start_probe only reads theta_hi /
        # theta_crit; theta_lo + heartbeat_s are owned here for the
        # threshold-parity contract (DESIGN §10).
        self.theta_lo = float(theta_lo)
        self.theta_crit = float(theta_crit)
        self.heartbeat_s = float(heartbeat_s)
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

        Generic guard: if the existing handler is marked as a wrapped
        composite (attribute ``_aginfer_wrap``), refuse the overwrite
        unless ``force=True``.  No layer sets ``_aginfer_wrap`` since
        #194 removed the admission composite (kv_scheduler is now the
        sole joint handler), but the guard is kept so any future
        compose-on-top layer is protected from silent clobbering.
        Pass ``force=True`` for legitimate test re-attach.
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
                f"intend to bypass it, OR attach your layer BEFORE the "
                f"wrapping layer."
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

    Also emits a structured ``aginfer_metric event=apply_failed``
    line so operators have a per-event grep target (post-T36
    cleanup removed the sync-path ``migrate_skipped`` line that
    used to serve this purpose).
    """
    payload = event.payload or {}
    reason = payload.get("reason")
    endpoint = payload.get("endpoint")
    action_id = payload.get("action_id")
    if isinstance(reason, str) and reason:
        router.observability.record_failure(reason)
        # #223: a dump-vs-apply leaf TOCTOU (remove_*_not_*_leaf /
        # remove_not_leaf) means a correctly-proposed remove was rejected
        # because the node gained a child before apply.  Cool the hash down
        # so the daemon stops re-proposing the doomed remove every event
        # (the systematic 956/cycle reject storm).  Keyed on hash; read by
        # kv_scheduler.handle before dispatch.
        h = payload.get("hash")
        if h and "leaf" in reason:
            import time as _time
            from .kv_scheduler import _EVICT_COOLDOWN_S
            cd = getattr(router, "evict_cooldown", None)
            if cd is None:
                cd = router.evict_cooldown = {}
            cd[str(h)] = _time.monotonic() + _EVICT_COOLDOWN_S
    logger.info(
        "aginfer apply_failed received: endpoint=%s action_id=%s reason=%s "
        "hash=%s",
        endpoint, action_id, reason, payload.get("hash"),
    )
    from ._metrics import m as _m
    _m(
        "apply_failed",
        endpoint=endpoint or "?",
        action_id=action_id or "?",
        reason=(reason or "?").replace(" ", "_")[:120],
    )


def attach_apply_failed_handler(router: "EventRouter") -> None:
    """Register the T37 default APPLY_FAILED handler on ``router``.

    Always wired by main.py at daemon startup; kept as a separate
    function so verify probes can opt-in selectively."""
    router.set_handler(EventKind.APPLY_FAILED, _apply_failed_handler)


async def _hash_collision_handler(event: Event, router: "EventRouter") -> None:
    """T24 (#182, DESIGN §4 + §10): sglang fired HASH_COLLISION.
    Deployment-bug class — fatal() with forensic dump and exit.

    Probability < 10⁻²² at any practical tree size, so if this
    handler ever runs it's either (a) a sglang hash-function
    regression, (b) a daemon re-keying bug, or (c) genuinely the
    1-in-10²² event (in which case the forensic dump captures
    enough context to recover).
    """
    from ._fatal import fatal
    payload = event.payload or {}
    fatal(
        "hash_collision",
        key=payload.get("key"),
        node_a_summary=payload.get("node_a_summary"),
        node_b_summary=payload.get("node_b_summary"),
        ts=payload.get("ts"),
        ts_monotonic=payload.get("ts_monotonic"),
    )


def attach_hash_collision_handler(router: "EventRouter") -> None:
    """Register the T24 HASH_COLLISION fatal handler on ``router``.

    Mirrors attach_apply_failed_handler — wired at daemon startup
    so any sglang-emitted collision triggers an immediate crash-only
    restart cycle."""
    router.set_handler(EventKind.HASH_COLLISION, _hash_collision_handler)


def make_session_end_handler(tracker, outbound, kv_scheduler=None):
    """T41 (#185, DESIGN §11 F5) + T187 (#187, DESIGN §4 / §7
    SESSION_END normal path): build the SESSION_END handler closure.

    On SESSION_END for program ``p`` the handler runs three steps,
    IN THIS ORDER:

      1. **State transition + gate release (F5)** — ``tracker.end(p)``
         transitions p to ENDED.  If p was PAUSED with a request
         parked in the proxy gate, end() releases the gate so
         ``wait_if_paused`` wakes and the proxy responds 499 (client
         closed the session).
      2. **Joint decision (T187 + #194)** — if a ``kv_scheduler`` is
         wired, run its ``handle(event, router)`` for the SESSION_END
         event.  Post-#194 ``handle`` runs the full ``joint_decide``, so
         on top of demoting p's exclusive units this can ALSO pause /
         resume *unrelated* live programs if HBM is pressured at this
         instant (admission's candidate generators iterate all programs,
         not just p; ended p itself is skipped).  That is intended — §9's
         entry point is uniform across all 13 event kinds; SESSION_END is
         not special-cased.
         Because step 1 already set p to ENDED, ``build_paper_state``
         scores p's units with the workload-prior p_hat (not 1.0),
         and ``_build_decision_set`` returns ``session_scoped_units(p)``
         (units held only by p) — so the policy demotes/drops p's
         exclusive units while units shared with live programs are
         untouched (DESIGN §7 table + "SESSION_END normal path").
         Ordering is load-bearing: end() MUST precede handle() or the
         scorer would see p still alive (p_hat=1.0) and keep the
         units.  The migrate is BEST-EFFORT: this step is wrapped in
         a try/except so a downstream error can NOT skip step 3.
         (``handle`` only guards its own ``fetch_state`` /
         ``build_paper_state``; ``decide`` / ``_dispatch_*`` propagate
         — #187 audit B1.)
      3. **PUT (F5)** — enqueue ``PUT /aginfer/program_paused
         {state: ENDED}`` so sglang clears p's per_program_usage
         state on the next dump (idempotent; ENDED-no-units entries
         GC'd at dump time per #186).  Enqueued AFTER the migrate
         batch, matching DESIGN's on_session_end (migrate then PUT).
         Runs even if step 2 raised — the F5 state-transition + PUT
         is the contract; the migrate is an optimisation on top.

    Closure holds ``tracker`` / ``outbound`` / ``kv_scheduler`` (the
    same pattern kv_scheduler/admission use) since the handler
    signature is ``(event, router)`` and the router doesn't expose
    them.  ``kv_scheduler=None`` keeps the pure F5 behaviour (tests
    that don't exercise the migrate path).
    """
    async def _session_end_handler(event: Event, router: "EventRouter") -> None:
        pid = event.session
        if pid is None:
            logger.warning("SESSION_END with no session id; ignoring")
            return
        # 1. F5 state transition + gate release.  MUST happen before
        #    the migrate decision so the scorer sees p as ENDED.
        prev = tracker.end(pid)
        # 2. T187 migrate D_t = session_scoped_units(p).  Reuses the
        #    full kv_scheduler.handle pipeline (fetch state → build →
        #    decide → dispatch migrate + hints), now that p is ENDED.
        #    BEST-EFFORT: guarded so a downstream policy/dispatch error
        #    (handle() only catches its own fetch/build) can't skip the
        #    F5 PUT below (#187 audit B1).
        if kv_scheduler is not None:
            try:
                await kv_scheduler.handle(event, router)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "SESSION_END migrate (kv_scheduler.handle) failed for "
                    "pid=%s; continuing to the F5 ENDED PUT", pid,
                )
        # 3. F5 PUT — after the migrate batch (DESIGN: migrate, then
        #    PUT), regardless of prior state.
        outbound.enqueue_program_paused(
            pid=pid, state="ENDED", pre_pause_state=None,
        )
        from ._metrics import m as _m
        _m(
            "session_end",
            pid=pid,
            prev_state=prev.value if prev is not None else "NONE",
            migrate=kv_scheduler is not None,
        )
        logger.info(
            "SESSION_END handled: pid=%s prev=%s → ENDED + migrate(%s) "
            "+ PUT enqueued",
            pid, prev, kv_scheduler is not None,
        )

    return _session_end_handler


def attach_session_end_handler(
    router: "EventRouter", tracker, outbound, kv_scheduler=None,
) -> None:
    """Register the SESSION_END handler.  Wired at daemon startup
    AFTER kv_scheduler's blanket attach so this composite OWNS
    SESSION_END (it internally invokes ``kv_scheduler.handle`` for
    the migrate D_t — T187 — rather than letting the blanket handler
    run on its own, which would skip the F5 state-transition + gate
    release + PUT)."""
    router.set_handler(
        EventKind.SESSION_END,
        make_session_end_handler(tracker, outbound, kv_scheduler),
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

    # ---- T22 (#155): canonical thresholds endpoint -----------------
    # DESIGN §6 / §10 "Threshold parity": the daemon is the source of
    # truth for theta_hi / theta_lo / theta_crit / heartbeat_s.
    # Sglang fetches once at bootstrap (halts loudly if unreachable);
    # daemon→sglang PUTs (out of scope here, lives in #164/#155
    # follow-up wiring) carry runtime updates.
    @app.get("/aginfer/thresholds")
    async def aginfer_thresholds_get() -> Dict[str, float]:
        return {
            "theta_hi":    float(router.theta_hi),
            "theta_lo":    float(router.theta_lo),
            "theta_crit":  float(router.theta_crit),
            "heartbeat_s": float(router.heartbeat_s),
        }
