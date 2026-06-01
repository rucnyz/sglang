"""T36 — fire-and-forget outbound action queue + worker (DESIGN §6 B4).

PLAN §4 T36.  Pre-T36 the daemon's event handler did
``await client.post(...)`` for each migrate dispatch, blocking the
event-worker coroutine for the full sglang round trip.  T42's stress
probe measured this directly: ``time_in_queue_p99 = 355.84 ms`` on
100 webhook events, 3.5× over PLAN's F3-revisit threshold (100 ms).

T36 decouples handler latency from sglang latency:

  - ``OutboundQueue.enqueue_migrate(...)`` is ``put_nowait`` + a
    ``uuid.uuid4()`` for ``batch_id``.  Returns < 50 µs.
  - ``OutboundWorker`` is a single background asyncio task that
    pops batches and POSTs them via ``httpx.AsyncClient``.
  - On 200, the response body is dropped on the floor — sglang's
    APPLY_FAILED webhook (T23+T37) is the authoritative failure
    path, so any per-item skip already flows through the daemon's
    observability counter via that webhook handler.  Reading sync
    skipped[] here would double-count.
  - On 5xx / connect-error, log + move on.  The next
    ``joint_decide`` re-converges; DESIGN §10 idempotency makes
    re-issue safe (no retry bookkeeping inside the worker).

Multi-endpoint design: the queue stores ``OutboundBatch`` records
keyed by endpoint (``migrate`` today; ``program_paused`` / ``hints``
/ ``thresholds`` plug in via ``enqueue_<endpoint>`` helpers as those
PLAN tasks land — DESIGN §6 L506 covers the full set).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- types


@dataclass
class OutboundBatch:
    """One outbound HTTP request the worker will issue.  Created at
    enqueue time so the worker doesn't need to know how to build the
    body — handlers shape it per-endpoint."""
    batch_id: str
    endpoint: str
    body: Dict[str, Any]
    # Wall-clock at enqueue (not perf_counter; this is for log
    # correlation across processes, not arithmetic).
    enqueue_ts: float = 0.0


# --------------------------------------------------------------- queue


class OutboundQueue:
    """In-memory ``asyncio.Queue[OutboundBatch]`` + dedicated worker.

    One instance per daemon process; created by ``main.py`` and
    injected into ``KvScheduler`` (and any other handler that needs
    to dispatch fire-and-forget).

    The worker is started by ``await start()`` (typically from the
    FastAPI startup hook) and stopped by ``await stop()`` (shutdown
    hook).  Both are idempotent.

    Failure semantics (DESIGN §6 / §10):
      * 200 → drop response; sglang's APPLY_FAILED webhook handles
        any per-item skips.
      * 4xx → log a structured warning and move on.  4xx is
        almost-always a deployment-bug payload shape (would have
        been caught at the daemon's plan stage); the next event's
        ``joint_decide`` re-evaluates.
      * 5xx / transport error → log a warning and move on.  Treated
        as transient backpressure that the next ``joint_decide``
        re-converges from.
      * No retry bookkeeping inside the worker.  Per DESIGN §10
        idempotency, re-issuing the same action is safe; the next
        event sees the unchanged state and emits the same plan.
    """

    def __init__(
        self,
        *,
        sglang_base_url: str,
        http_client: Optional[httpx.AsyncClient] = None,
        observability=None,  # daemon._observability.DaemonObservability
    ) -> None:
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        self.observability = observability
        self.queue: asyncio.Queue[OutboundBatch] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None

    # ---- enqueue (handler-facing) -----------------------------------

    def enqueue_migrate(
        self,
        actions: List[Dict[str, Any]],
    ) -> str:
        """Enqueue a ``POST /aginfer/migrate`` batch.  Returns the
        new ``batch_id`` (UUID4 string) for correlation with any
        future APPLY_FAILED webhook.  Each ``action`` dict is the
        residence-set form (DESIGN §6) — caller is responsible for
        having attached ``action_id`` per item if it wants per-item
        APPLY_FAILED correlation."""
        import time
        batch_id = str(uuid.uuid4())
        # DESIGN §6 L507: batch_id is written into the request body
        # envelope so sglang can echo it in APPLY_FAILED.
        body = {"actions": list(actions), "batch_id": batch_id}
        batch = OutboundBatch(
            batch_id=batch_id, endpoint="migrate",
            body=body, enqueue_ts=time.time(),
        )
        self.queue.put_nowait(batch)
        return batch_id

    # ---- lifecycle (daemon-facing) ----------------------------------

    async def start(self) -> None:
        """Spawn the worker task.  Idempotent."""
        if self._client is None:
            # No bounded read timeout: a sglang stall is the path
            # T36 is decoupling from; the worker can absorb 200 ms
            # CUDA-graph pauses just fine.  Connect+write are
            # bounded so a dead sglang fails fast on send.
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0, read=30.0, write=10.0, pool=5.0,
                ),
            )
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._worker_loop(), name="aginfer-outbound-worker",
            )

    async def stop(self) -> None:
        """Cancel the worker (any in-flight POST is awaited if
        possible) + close the owned httpx client."""
        task = self._worker_task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception(
                    "outbound worker raised during shutdown"
                )
            self._worker_task = None
        if self._owns_client and self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("outbound client aclose raised",
                               exc_info=True)
            self._client = None

    # ---- worker -----------------------------------------------------

    async def _worker_loop(self) -> None:
        while True:
            batch = await self.queue.get()
            try:
                await self._post_one(batch)
            except asyncio.CancelledError:
                # Re-raise so stop() sees clean termination.
                self.queue.task_done()
                raise
            except Exception:  # noqa: BLE001
                # Any uncaught exception in _post_one is a
                # programmer bug — log + continue so one bad batch
                # doesn't kill the worker.
                logger.exception(
                    "outbound worker: unexpected exception while "
                    "POSTing batch %s",
                    batch.batch_id,
                )
            finally:
                try:
                    self.queue.task_done()
                except ValueError:
                    # Already done on the cancel path; tolerate.
                    pass

    async def _post_one(self, batch: OutboundBatch) -> None:
        from ._metrics import m as _m
        assert self._client is not None
        url = f"{self.sglang_base_url}/aginfer/{batch.endpoint}"
        try:
            r = await self._client.post(url, json=batch.body)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "outbound %s batch %s: POST raised: %s",
                batch.endpoint, batch.batch_id, exc,
            )
            _m(
                "outbound_post",
                endpoint=batch.endpoint,
                batch_id=batch.batch_id,
                status="exception",
            )
            return
        if r.status_code >= 500:
            logger.warning(
                "outbound %s batch %s: HTTP %d (transient — next "
                "joint_decide re-converges)",
                batch.endpoint, batch.batch_id, r.status_code,
            )
            _m(
                "outbound_post",
                endpoint=batch.endpoint,
                batch_id=batch.batch_id,
                status=r.status_code,
            )
            return
        if r.status_code >= 400:
            # 4xx is plausibly a plan-shape bug from the daemon.  The
            # response body is structured; surface it once.
            logger.warning(
                "outbound %s batch %s: HTTP %d body=%s",
                batch.endpoint, batch.batch_id, r.status_code,
                r.text[:200] if hasattr(r, "text") else "<?>",
            )
            _m(
                "outbound_post",
                endpoint=batch.endpoint,
                batch_id=batch.batch_id,
                status=r.status_code,
            )
            return
        # 2xx — drop the body on the floor.  sglang's APPLY_FAILED
        # webhook (T23+T37) is the authoritative source for per-item
        # failures.  We still emit a single line so the operator
        # knows POST went out.
        _m(
            "outbound_post",
            endpoint=batch.endpoint,
            batch_id=batch.batch_id,
            status=r.status_code,
        )
