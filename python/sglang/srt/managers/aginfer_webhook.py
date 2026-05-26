"""Aginfer T5 outbound webhook from sglang's scheduler.

Watermark detector + fire-and-forget POST to the daemon's
``/aginfer/event`` endpoint.  Hooks into the end of each scheduler
step (``event_loop_overlap`` / ``event_loop_normal``).

Contract (from dev/aginfer/verify/t5/README.md):

* State ∈ {OK, HIGH, CRITICAL} from
  ``HBM_occ = used_tokens / cap_tokens``.
* Fire a POST when EITHER:
    (a) state changed since the last fire, OR
    (b) state ∈ {HIGH, CRITICAL} AND last fire was ≥ heartbeat_s ago.
* Payload: ``{kind, state, prev_state, occ, used_tokens, cap_tokens, ts}``.
* Retry up to 3× on network error with exponential backoff
  (0.1 s, 0.4 s, 1.6 s).
* If ``--aginfer-notify-url`` is unset, the detector is a no-op.
* If the send raises, the scheduler step is NEVER blocked
  (asyncio.create_task + try/except).

Design constraints:
* No new periodic timer.  The watermark is checked once per
  scheduler step; the heartbeat is "fire if state ∈ HIGH/CRITICAL
  and it's been >= heartbeat_s since the last fire" -- the
  scheduler step itself acts as the clock.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class WatermarkState(str, enum.Enum):
    OK = "OK"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class WebhookPayload:
    kind: str  # "memory_pressure" | "pressure_resolved" | "still_high"
    state: str
    prev_state: str
    occ: float
    used_tokens: int
    cap_tokens: int
    # Two time bases (round-1 audit M3): wall clock for human/log
    # correlation, monotonic for arithmetic on inter-fire intervals.
    # Downstream consumers (T7/T8 admission) should use ts_monotonic
    # when computing "time since last fire" — wall clock can step
    # backward on NTP adjustment.
    ts: float
    ts_monotonic: float


def classify(occ: float, theta_hi: float, theta_crit: float) -> WatermarkState:
    """Map HBM occupancy ∈ [0, 1] to a watermark state."""
    if occ >= theta_crit:
        return WatermarkState.CRITICAL
    if occ >= theta_hi:
        return WatermarkState.HIGH
    return WatermarkState.OK


def derive_kind(prev: WatermarkState, cur: WatermarkState) -> Optional[str]:
    """The event kind for a transition, or None if the transition is
    a no-op (same state).  Heartbeats are NOT covered here -- caller
    handles those separately.
    """
    if cur == prev:
        return None
    # Severity rank: OK=0, HIGH=1, CRITICAL=2.
    rank = {WatermarkState.OK: 0, WatermarkState.HIGH: 1, WatermarkState.CRITICAL: 2}
    if rank[cur] > rank[prev]:
        return "memory_pressure"
    return "pressure_resolved"


class AginferWebhookFirer:
    """Watermark detector + outbound fire-and-forget HTTP sender.

    One instance per scheduler subprocess; called once per scheduler
    step.  Internally maintains last-fired state and timestamp so the
    heartbeat decision is local (no periodic timer).

    sglang's scheduler is single-threaded; this class doesn't need
    locks for its own state.  The HTTP send is done via
    ``asyncio.create_task`` on a worker event-loop running in a
    background thread, so the scheduler step never awaits.
    """

    # Path the firer appends to a base URL.  README contract:
    # ``POST <notify_url>/aginfer/event``.  Audit-round-1 caught the
    # firer was posting to ``notify_url`` verbatim, so a user passing
    # the documented base URL (``--aginfer-notify-url=http://daemon:8765``)
    # would silently 404 every webhook.
    _EVENT_PATH = "/aginfer/event"

    def __init__(
        self,
        *,
        notify_url: str,
        heartbeat_s: float = 5.0,
        theta_hi: float = 0.7,
        theta_crit: float = 0.9,
    ) -> None:
        # Append the event path if the user passed a bare base URL.
        # Tolerate trailing slashes and either form ("base/" or
        # "base/aginfer/event") so the launch CLI is forgiving.
        url = notify_url.rstrip("/")
        if not url.endswith(self._EVENT_PATH):
            url = url + self._EVENT_PATH
        self.notify_url = url
        self.heartbeat_s = float(heartbeat_s)
        self.theta_hi = float(theta_hi)
        self.theta_crit = float(theta_crit)
        # Detector state.
        self._last_state: WatermarkState = WatermarkState.OK
        self._last_fire_ts: float = 0.0
        self._last_fire_kind: str = ""
        # Background event loop + thread for fire-and-forget HTTP.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client = None  # lazy httpx.AsyncClient
        self._start_loop()

    # ---- public API ----

    def maybe_fire(self, used_tokens: int, cap_tokens: int) -> Optional[str]:
        """Called once per scheduler step.  Returns the kind that was
        fired (for telemetry / tests), or None if no fire happened.

        Never raises; any error is logged.
        """
        if cap_tokens <= 0:
            return None
        occ = used_tokens / cap_tokens
        cur = classify(occ, self.theta_hi, self.theta_crit)
        now = time.monotonic()

        kind: Optional[str] = derive_kind(self._last_state, cur)
        if kind is None and cur != WatermarkState.OK:
            # Heartbeat: state is HIGH/CRITICAL and stable; fire if
            # last_fire was long enough ago.  Last_fire_ts starts at 0
            # so the FIRST tick in HIGH/CRITICAL fires immediately --
            # which is the desired memory_pressure transition firing
            # (already covered above).  Subsequent ticks will only
            # heartbeat after heartbeat_s.
            if now - self._last_fire_ts >= self.heartbeat_s:
                kind = "still_high"

        if kind is None:
            # No transition, no heartbeat due.  Update state in case
            # cur differs from last (it shouldn't, but defensive).
            self._last_state = cur
            return None

        payload = WebhookPayload(
            kind=kind,
            state=cur.value,
            prev_state=self._last_state.value,
            occ=occ,
            used_tokens=int(used_tokens),
            cap_tokens=int(cap_tokens),
            ts=time.time(),
            ts_monotonic=now,
        )
        self._last_state = cur
        self._last_fire_ts = now
        self._last_fire_kind = kind
        # Fire-and-forget on the background loop.
        try:
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(self._send(payload), self._loop)
        except Exception:  # noqa: BLE001 -- never block the scheduler
            logger.warning("aginfer webhook scheduling failed", exc_info=True)
        return kind

    def close(self) -> None:
        """Stop the background loop + close the httpx client on scheduler
        shutdown.  Audit-round-1 BLOCKER 2: without this, every scheduler
        restart leaked the daemon thread + open httpx connections."""
        loop = self._loop
        client = self._client
        if loop is not None and client is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    client.aclose(), loop
                )
                fut.result(timeout=2.0)
            except Exception:
                logger.warning("aginfer client.aclose() failed", exc_info=True)
        self._client = None
        if loop is not None:
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---- internals ----

    def _start_loop(self) -> None:
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()
            loop.close()

        t = threading.Thread(
            target=_run, name="aginfer-webhook", daemon=True
        )
        t.start()
        self._loop = loop
        self._thread = t

    async def _send(self, payload: WebhookPayload) -> None:
        # Lazy-init httpx client on the background loop.
        if self._client is None:
            import httpx  # noqa: WPS433

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        body = {
            "kind": payload.kind,
            "state": payload.state,
            "prev_state": payload.prev_state,
            "occ": payload.occ,
            "used_tokens": payload.used_tokens,
            "cap_tokens": payload.cap_tokens,
            "ts": payload.ts,
            "ts_monotonic": payload.ts_monotonic,
        }
        backoff = 0.1
        for attempt in range(3):
            try:
                resp = await self._client.post(self.notify_url, json=body)
                if resp.status_code < 500:
                    return
                logger.info(
                    "aginfer webhook %s -> %d (attempt %d/3); retrying",
                    payload.kind, resp.status_code, attempt + 1,
                )
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "aginfer webhook %s raised (attempt %d/3): %s",
                    payload.kind, attempt + 1, exc,
                )
            # Round-1 audit M2: skip sleep after the last attempt (we're
            # about to give up anyway).
            if attempt < 2:
                await asyncio.sleep(backoff)
                backoff *= 4
        logger.warning(
            "aginfer webhook %s gave up after 3 attempts",
            payload.kind,
        )
