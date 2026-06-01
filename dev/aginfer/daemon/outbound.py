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

    **Queue is unbounded by design** (no ``maxsize``).  DESIGN §6
    makes the producer (event handler) non-blocking, and silently
    dropping actions would lose scheduling decisions.  Steady-state
    is bounded because arrival rate is itself capped by sglang's own
    throughput (webhook firer is sglang-side).  The unbounded queue
    is at risk only during multi-minute sglang HTTP stalls;
    ``DaemonObservability.outbound_queue_depth`` and
    ``outbound_oldest_age_ms`` let an operator alert before OOM,
    and the sustained-escalation fatal (DESIGN §10 / #164) is the
    hard backstop — running forever-degraded is not a valid state.
    """

    def __init__(
        self,
        *,
        sglang_base_url: str,
        http_client: Optional[httpx.AsyncClient] = None,
        observability=None,  # daemon._observability.DaemonObservability
        # T36/F3 #164 sustained-escalation thresholds.  BOTH must
        # trip simultaneously for the worker to fatal().  Defaults
        # picked per DESIGN §10 "sustained-escalation tier":
        # 100 consecutive failures = ~3-5 min at 1 POST per few
        # seconds, queue age 5 min = multi-minute stall regime.
        # Operator-tunable via main.py CLI flags.
        escalate_failures: int = 100,
        escalate_oldest_age_s: float = 300.0,
    ) -> None:
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        self.observability = observability
        self.queue: asyncio.Queue[OutboundBatch] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task[None]] = None
        # T36/F3 #164: counter resets on every 2xx success.  /health
        # exposes the current value so an operator can alert before
        # the fatal threshold.
        self.consecutive_failures: int = 0
        self.last_outbound_oldest_age_ms: float = 0.0
        self._escalate_failures = int(escalate_failures)
        self._escalate_oldest_age_s = float(escalate_oldest_age_s)
        if self._escalate_failures < 1:
            raise ValueError(
                f"escalate_failures must be >= 1; got {escalate_failures}"
            )
        if self._escalate_oldest_age_s < 0:
            raise ValueError(
                f"escalate_oldest_age_s must be >= 0; "
                f"got {escalate_oldest_age_s}"
            )

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
        import time
        while True:
            batch = await self.queue.get()
            # T36 audit (#163): sample outbound queue health BEFORE
            # POSTing the batch we just popped.  qsize() = "backlog
            # still waiting"; batch.enqueue_ts → wall-clock age of
            # the (was-) oldest pending batch.
            depth_after_pop = self.queue.qsize()
            oldest_age_ms = max(
                0.0, (time.time() - batch.enqueue_ts) * 1000.0,
            )
            self.last_outbound_oldest_age_ms = oldest_age_ms
            if self.observability is not None:
                self.observability.record_outbound(
                    queue_depth=depth_after_pop,
                    oldest_age_ms=oldest_age_ms,
                )
            try:
                success = await self._post_one(batch)
            except asyncio.CancelledError:
                # Re-raise so stop() sees clean termination.
                self.queue.task_done()
                raise
            except Exception:  # noqa: BLE001
                # Any uncaught exception in _post_one is a
                # programmer bug — log + continue so one bad batch
                # doesn't kill the worker.  Treat as failure for
                # the escalation counter.
                logger.exception(
                    "outbound worker: unexpected exception while "
                    "POSTing batch %s",
                    batch.batch_id,
                )
                success = False
                self.consecutive_failures += 1
            finally:
                try:
                    self.queue.task_done()
                except ValueError:
                    # Already done on the cancel path; tolerate.
                    pass
            # T36/F3 (#164): sustained-escalation fatal.  Both BOTH
            # conditions must trip: streak length AND queue
            # backlog age.  Low-traffic dead-sglang fails consec but
            # drains fast (oldest_age stays small) → no fatal,
            # daemon survives until sglang returns.  High-traffic
            # dead-sglang fails consec AND oldest_age grows → fatal
            # → supervisor restart.  DESIGN §10 "sustained tier".
            if (
                not success
                and self.consecutive_failures >= self._escalate_failures
                and oldest_age_ms >= self._escalate_oldest_age_s * 1000.0
            ):
                from ._fatal import fatal
                fatal(
                    "sglang_sustained_unreachable",
                    sglang_base_url=self.sglang_base_url,
                    consecutive_failures=self.consecutive_failures,
                    oldest_age_ms=oldest_age_ms,
                    queue_depth=depth_after_pop,
                    escalate_failures_threshold=self._escalate_failures,
                    escalate_oldest_age_s_threshold=(
                        self._escalate_oldest_age_s
                    ),
                )

    async def _post_one(self, batch: OutboundBatch) -> bool:
        """Issue one POST; update the sustained-escalation counter.

        Returns True iff sglang accepted (2xx).  Worker loop reads
        the return to decide whether to check the escalation
        threshold (success → reset counter; failure → check).

        Failure classes for the streak counter:
          * 2xx: success → reset
          * 4xx: failure (plan-shape bug; restart may not fix but
            crashloop reveals it — DESIGN §10 calls 4xx a
            deployment bug too)
          * 5xx: failure (transient sglang issue)
          * transport exception: failure (sglang unreachable)
        """
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
            self.consecutive_failures += 1
            return False
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
            self.consecutive_failures += 1
            return False
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
            self.consecutive_failures += 1
            return False
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
        if self.consecutive_failures > 0:
            logger.info(
                "outbound recovered after %d consecutive failures",
                self.consecutive_failures,
            )
        self.consecutive_failures = 0
        return True
