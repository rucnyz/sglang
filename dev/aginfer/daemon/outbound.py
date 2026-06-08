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
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- types


@dataclass
class OutboundBatch:
    """One outbound HTTP request the worker will issue.  Created at
    enqueue time so the worker doesn't need to know how to build the
    body — handlers shape it per-endpoint.

    ``enqueue_ts`` is a REQUIRED positive wall-clock time.time().  No
    default — #167 audit closed a footgun where a default of 0.0 would
    yield age ≈ ``time.time() * 1000`` ≈ 1.7e15 ms, instantly tripping
    the sustained-escalation fatal.  The ``__post_init__`` guard
    catches explicit ``0`` or negative values for the same reason."""
    batch_id: str
    endpoint: str
    body: Dict[str, Any]
    enqueue_ts: float
    # HTTP verb for the worker.  migrate is POST; program_paused +
    # thresholds are PUT (T41 #185 / T22).  Default POST keeps every
    # existing call site unchanged.
    method: str = "POST"

    def __post_init__(self) -> None:
        if self.method not in ("POST", "PUT"):
            raise ValueError(
                f"OutboundBatch.method must be POST or PUT; "
                f"got {self.method!r}"
            )
        if self.enqueue_ts <= 0.0:
            raise ValueError(
                f"OutboundBatch.enqueue_ts must be > 0 (wall-clock "
                f"time.time()); got {self.enqueue_ts!r}.  A zero or "
                f"negative value would compute age ≈ time.time()*1000 "
                f"and instantly trip the sustained-escalation fatal."
            )


# ------------------------------------------------------- coalesce (#228)


def _partition_and_coalesce(
    batches: List[OutboundBatch],
    *,
    now_ts: float,
    migrate_freshness_ms: float,
) -> Tuple[List[OutboundBatch], Dict[str, int]]:
    """Collapse a drained burst of queued batches into at most one dispatch
    per endpoint, honouring each endpoint's temporal semantics (#228).

    The outbound channel is single-flight at sglang (one communicator,
    serialised apply ≈ one scheduler iteration per POST).  The T40 design
    pushes a ``hints`` PUT EVERY event, so hints are ~99% of outbound
    traffic (≈11.7k vs ≈130 migrate per cycle observed) and the FIFO makes
    every time-sensitive ``migrate`` wait behind that idempotent flood —
    ageing it ~1.8 s until the tree diverges and sglang rejects it.

    Coalescing fixes the root cause:

      * ``hints``  — overwrite-by-stamp + idempotent ⇒ merge ALL pending
        hint batches into ONE PUT (latest value per hash wins).  Collapses
        the flood to one PUT per worker wake, which un-clogs the channel.
      * ``migrate`` — time-sensitive (a stale remove races the tree → reject).
        Drop any batch older than ``migrate_freshness_ms`` (#227 freshness
        bound — generous, a pathological-spike floor, NOT a tight storm
        suppressor; the coalescing is what removes the normal-operation
        latency), then dispatch survivors individually.  NOTE (audit #2): a
        stale-dropped migrate produces no dispatch and hence no failure
        signal, so the #164 sustained-unreachable backstop relies on the
        always-on ``hints`` traffic (pushed every event) as its stall canary,
        not on migrates.
      * ``program_paused`` — liveness-critical (a dropped resume starves a
        paused program, #211).  NEVER dropped; coalesced by pid (latest
        state wins) so a stale transition can't override a newer one.

    Dispatch order: program_paused (liveness) → migrate (eviction) → hints
    (idempotent), so time-sensitive intents never wait behind the flood.

    Pure — no I/O, no clock read (``now_ts`` injected) — so the latency and
    correctness tests drive it deterministically.
    """
    migrates: List[OutboundBatch] = []
    hints: List[OutboundBatch] = []
    paused: List[OutboundBatch] = []
    passthrough: List[OutboundBatch] = []
    for b in batches:
        if b.endpoint == "migrate":
            migrates.append(b)
        elif b.endpoint == "hints":
            hints.append(b)
        elif b.endpoint == "program_paused":
            paused.append(b)
        else:
            # Unknown endpoint (forward-compat): dispatch untouched, first.
            passthrough.append(b)

    stats: Dict[str, int] = {
        "migrate_in": len(migrates), "hints_in": len(hints),
        "paused_in": len(paused), "migrate_dropped_stale": 0,
        "migrate_out": 0, "hints_out": 0, "paused_out": 0,
    }
    out: List[OutboundBatch] = list(passthrough)

    # ---- program_paused: coalesce by pid (latest wins), never drop -------
    if paused:
        by_pid: Dict[Any, OutboundBatch] = {}
        for b in sorted(paused, key=lambda x: x.enqueue_ts):
            pid = b.body.get("pid") if isinstance(b.body, dict) else None
            by_pid[pid] = b  # later enqueue supersedes
        kept = list(by_pid.values())
        out.extend(kept)
        stats["paused_out"] = len(kept)

    # ---- migrate: stale-drop only; dispatch each survivor individually ---
    # Migrates are rare (≈130/cycle vs ≈11.7k hints) so they never clog the
    # channel — the latency win is entirely from collapsing the hint flood
    # they wait behind.  Keeping them per-batch preserves the FIFO order and
    # the per-POST sustained-escalation accounting (#164); only the #227
    # freshness bound applies here (drop a batch decided too long ago to
    # still be valid against the live tree).
    if migrates:
        for b in sorted(migrates, key=lambda x: x.enqueue_ts):
            age_ms = max(0.0, (now_ts - b.enqueue_ts) * 1000.0)
            if migrate_freshness_ms > 0.0 and age_ms > migrate_freshness_ms:
                stats["migrate_dropped_stale"] += 1
            else:
                out.append(b)
                stats["migrate_out"] += 1

    # ---- hints: coalesce ALL into one PUT, HIGHEST stamp per hash --------
    # Key by max(stamp), not enqueue order: sglang's hint table is itself
    # overwrite-by-stamp (§10), so forwarding the highest-stamp value is the
    # correct merge even if a burst enqueues stamps out of order (audit #4).
    if hints:
        by_hash_h: Dict[Any, Dict[str, Any]] = {}
        for b in hints:
            for h in (b.body.get("hints", []) if isinstance(b.body, dict)
                      else []):
                hsh = h.get("hash")
                prev = by_hash_h.get(hsh)
                if prev is None or h.get("stamp", -1) >= prev.get("stamp", -1):
                    by_hash_h[hsh] = h
        merged_h = list(by_hash_h.values())
        if merged_h:
            bid = str(uuid.uuid4())
            out.append(OutboundBatch(
                batch_id=bid, endpoint="hints",
                body={"hints": merged_h, "batch_id": bid},
                enqueue_ts=min(b.enqueue_ts for b in hints),
                method="PUT",
            ))
            stats["hints_out"] = 1

    return out, stats


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
        # #227/#228: per-dispatch freshness bound for the time-sensitive
        # ``migrate`` endpoint.  A migrate decided on a state snapshot is a
        # prediction with a validity horizon; the worker drops one aged past
        # this before dispatch (coalescing handles the normal-operation
        # latency, so this is a GENEROUS pathological-spike floor, not a
        # tight tuner).  0 disables.  Env AGINFER_MIGRATE_FRESHNESS_MS.
        migrate_freshness_ms: Optional[float] = None,
    ) -> None:
        self.sglang_base_url = sglang_base_url.rstrip("/")
        self._client = http_client
        self._owns_client = http_client is None
        self.observability = observability
        self.queue: asyncio.Queue[OutboundBatch] = asyncio.Queue()
        # #228: the worker drains the whole queue into a local burst before
        # dispatching (to coalesce), so the asyncio.Queue can read empty
        # while a large burst is still in-flight.  Track the oldest
        # enqueue_ts of the burst currently being dispatched so /health's
        # current_oldest_pending_age_ms() still reflects real backlog age
        # (the #166 live-peek contract) instead of decaying to 0 the instant
        # the worker drains.  GIL-atomic float assignment; None when idle.
        self._draining_oldest_ts: Optional[float] = None
        self._worker_task: Optional[asyncio.Task[None]] = None
        # T36/F3 #164: counter resets on every 2xx success.  /health
        # exposes the current value so an operator can alert before
        # the fatal threshold.
        self.consecutive_failures: int = 0
        self._escalate_failures = int(escalate_failures)
        self._escalate_oldest_age_s = float(escalate_oldest_age_s)
        if migrate_freshness_ms is None:
            import os
            # GENEROUS default (30 s): the hints COALESCING removes the
            # normal-operation latency, so this is purely a pathological-
            # spike floor — it drops only a migrate so old (sglang
            # catastrophically stalled) that its decision is certainly dead,
            # never a tight knob masking ordinary dispatch latency.
            raw = os.environ.get("AGINFER_MIGRATE_FRESHNESS_MS", "30000")
            try:
                migrate_freshness_ms = float(raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"AGINFER_MIGRATE_FRESHNESS_MS must be a float; got {raw!r}"
                )
        self._migrate_freshness_ms = float(migrate_freshness_ms)
        if self._migrate_freshness_ms < 0:
            raise ValueError(
                f"migrate_freshness_ms must be >= 0; "
                f"got {migrate_freshness_ms}"
            )
        if self._escalate_failures < 1:
            raise ValueError(
                f"escalate_failures must be >= 1; got {escalate_failures}"
            )
        if self._escalate_oldest_age_s < 0:
            raise ValueError(
                f"escalate_oldest_age_s must be >= 0; "
                f"got {escalate_oldest_age_s}"
            )

    # ---- observability (sync; safe to call from /health) -------------

    def current_oldest_pending_age_ms(self) -> float:
        """Live age in ms of the OLDEST batch still in the queue.
        Returns 0.0 if the queue is empty.

        Used by ``/health`` so k8s readiness probes / dashboards see
        the CURRENT backlog snapshot — not a sticky last-popped value
        (the #166 audit-found bug: cached field updated only at pop
        time stuck at large values forever once sglang healed).

        Implementation peeks ``asyncio.Queue._queue[0]`` (a
        ``collections.deque``).  Reading the head element is GIL-
        atomic on CPython; a concurrent worker pop is a benign race
        (snapshot either reflects the popped or the new head, both
        valid coarse observations).  We tolerate ``IndexError`` /
        ``AttributeError`` for the empty-queue race."""
        import time
        now = time.time()
        ages = []
        # In-flight drained burst (#228): counts as backlog until dispatched.
        drain_ts = self._draining_oldest_ts
        if drain_ts is not None:
            ages.append((now - drain_ts) * 1000.0)
        q_internal = getattr(self.queue, "_queue", None)
        if q_internal is not None:
            try:
                ages.append((now - q_internal[0].enqueue_ts) * 1000.0)
            except IndexError:
                pass
        return max(0.0, max(ages)) if ages else 0.0

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

    def enqueue_program_paused(
        self,
        *,
        pid: str,
        state: str,
        pre_pause_state: Optional[str] = None,
    ) -> str:
        """T41 (#185) / DESIGN §6: enqueue a ``PUT /aginfer/program_
        paused`` body.  Fire-and-forget like migrate.  Used by the
        SESSION_END handler (transition to ENDED) and F1 disconnect
        (#183).  Returns the batch_id for correlation.

        sglang's PUT handler stores (state, pre_pause_state) and
        echoes it in the next /aginfer/state dump (T21 #181).
        """
        import time
        batch_id = str(uuid.uuid4())
        body = {
            "pid": pid,
            "state": state,
            "pre_pause_state": pre_pause_state,
            "batch_id": batch_id,
        }
        batch = OutboundBatch(
            batch_id=batch_id, endpoint="program_paused",
            body=body, enqueue_ts=time.time(), method="PUT",
        )
        self.queue.put_nowait(batch)
        return batch_id

    def enqueue_hints(
        self,
        hints: List[Dict[str, Any]],
    ) -> str:
        """T40 (#184) / DESIGN §6 ``PUT /aginfer/hints``: enqueue the
        V_u-input hint batch.  Fire-and-forget like migrate.  Each
        ``hint`` is ``{"hash", "p_hat", "lambda", "stamp"}``; sglang's
        hint table is overwrite-by-stamp (DESIGN §10) so the daemon
        keeps NO shadow ``{hash: last_pushed}`` map — it re-scores the
        units in D_t and pushes them unconditionally every event.

        Returns the batch_id for APPLY_FAILED correlation.
        """
        import time
        batch_id = str(uuid.uuid4())
        body = {"hints": list(hints), "batch_id": batch_id}
        batch = OutboundBatch(
            batch_id=batch_id, endpoint="hints",
            body=body, enqueue_ts=time.time(), method="PUT",
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
            first = await self.queue.get()
            # #228: drain everything ELSE currently queued so a wake
            # coalesces to at most one POST/PUT per endpoint.  Under the
            # T40 every-event hints push (≈99% of traffic) on a single-
            # flight channel, this is what stops time-sensitive migrates
            # ageing behind the idempotent flood.  get_nowait() is the
            # whole burst that arrived while the prior dispatch was in
            # flight — bounded by how fast handlers enqueue.
            drained: List[OutboundBatch] = [first]
            while True:
                try:
                    drained.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            # #228: expose the burst's oldest wait to /health while in flight.
            self._draining_oldest_ts = min(b.enqueue_ts for b in drained)
            try:
                to_dispatch, stats = _partition_and_coalesce(
                    drained,
                    now_ts=time.time(),
                    migrate_freshness_ms=self._migrate_freshness_ms,
                )
                if (stats["migrate_dropped_stale"]
                        or stats["hints_in"] != stats["hints_out"]
                        or stats["migrate_in"] != stats["migrate_out"]
                        or stats["paused_in"] != stats["paused_out"]):
                    from ._metrics import m as _m
                    _m("outbound_coalesce", **stats)
                for batch in to_dispatch:
                    await self._dispatch_one(batch)
            finally:
                self._draining_oldest_ts = None
                # One task_done per DRAINED item (queue accounting is by
                # get, not by dispatch — coalescing emits fewer POSTs).
                for _ in drained:
                    try:
                        self.queue.task_done()
                    except ValueError:
                        pass

    async def _dispatch_one(self, batch: OutboundBatch) -> None:
        """Issue one (already-coalesced) batch + run the #164 sustained-
        escalation check.  Cancellation propagates so ``stop()`` sees clean
        termination; the worker loop's ``finally`` still drains task_done."""
        import time
        oldest_age_ms = max(0.0, (time.time() - batch.enqueue_ts) * 1000.0)
        if self.observability is not None:
            self.observability.record_outbound(
                queue_depth=self.queue.qsize(),
                oldest_age_ms=oldest_age_ms,
            )
        try:
            success = await self._post_one(batch)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "outbound worker: unexpected exception while POSTing "
                "batch %s", batch.batch_id,
            )
            success = False
            self.consecutive_failures += 1
        # T36/F3 (#164): both streak length AND backlog age must trip.
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
                queue_depth=self.queue.qsize(),
                escalate_failures_threshold=self._escalate_failures,
                escalate_oldest_age_s_threshold=self._escalate_oldest_age_s,
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
            if batch.method == "POST":
                # Keep the POST path on .post() so existing
                # POST-only test stubs (and the migrate hot path)
                # are byte-identical to pre-#185.
                r = await self._client.post(url, json=batch.body)
            else:
                r = await self._client.request(
                    batch.method, url, json=batch.body,
                )
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
