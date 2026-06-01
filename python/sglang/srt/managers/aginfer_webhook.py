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
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class WatermarkState(str, enum.Enum):
    OK = "OK"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class ApplyFailedPayload:
    """T23 (DESIGN §4 round-9 B4): one webhook fire per failed action
    inside `apply_aginfer_migrations` / equivalent endpoint impls.
    Sent to the daemon's /aginfer/event with kind="apply_failed".
    The daemon's T37 handler bumps its observability counter and
    lets the next `joint_decide` re-evaluate.
    """
    endpoint: str   # "migrate" | "program_paused" | "hints" | "thresholds"
    action_id: str
    reason: str
    hash_: Optional[str]
    ts: float
    ts_monotonic: float


@dataclass(slots=True)
class HashCollisionPayload:
    """T24 (#182, DESIGN §4 + §10 round-15/16/17): fire when
    apply_aginfer_migrations' DFS detects two distinct radix nodes
    mapping to the same hash key.

    Probability is < 10⁻²² at any tree size aginfer encounters; if
    it ever fires it's a deployment-bug class signal (sglang side:
    hash-function regression; daemon side: re-keying bug).  Daemon
    handler calls fatal() on receipt.

    Both node summaries carry the structural info needed to
    identify which two nodes collided (residence, n_tokens,
    session_ids, hex hash_value).
    """
    key: str
    node_a_summary: dict
    node_b_summary: dict
    ts: float
    ts_monotonic: float


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


def classify(
    occ: float,
    prev: WatermarkState,
    theta_hi: float,
    theta_lo: float,
    theta_crit: float,
) -> WatermarkState:
    """Hysteretic mapping HBM occupancy ∈ [0, 1] → watermark state.

    Up-crossings use ``theta_hi`` / ``theta_crit``; down-crossings
    use ``theta_lo`` (HIGH↔OK) and ``theta_hi`` (CRITICAL↔HIGH).
    Without hysteresis the previous version could leave the daemon
    stuck in HIGH while admission's pause had already cleared
    pressure — no PRESSURE_RESOLVED ever fired, paused programs
    never resumed.
    """
    if prev == WatermarkState.OK:
        if occ >= theta_crit: return WatermarkState.CRITICAL
        if occ >= theta_hi:   return WatermarkState.HIGH
        return WatermarkState.OK
    if prev == WatermarkState.HIGH:
        if occ >= theta_crit: return WatermarkState.CRITICAL
        if occ <  theta_lo:   return WatermarkState.OK
        return WatermarkState.HIGH
    # prev == CRITICAL
    if occ <  theta_lo:   return WatermarkState.OK
    if occ <  theta_hi:   return WatermarkState.HIGH
    return WatermarkState.CRITICAL


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
        theta_lo: float = 0.55,
        theta_crit: float = 0.9,
    ) -> None:
        # Append the event path if the user passed a bare base URL.
        # Tolerate trailing slashes and either form ("base/" or
        # "base/aginfer/event") so the launch CLI is forgiving.
        url = notify_url.rstrip("/")
        if not url.endswith(self._EVENT_PATH):
            url = url + self._EVENT_PATH
        self.notify_url = url
        # T22 (#155): store the four hysteresis values as a tuple so
        # the runtime ``apply_thresholds`` PUT handler can swap all
        # four atomically (single tuple rebind = GIL-atomic on
        # CPython).  Read-access via @property below preserves the
        # existing ``firer.theta_hi`` etc. caller interface.
        self._theta_tuple: Tuple[float, float, float, float] = (
            float(theta_hi),
            float(theta_lo),
            float(theta_crit),
            float(heartbeat_s),
        )
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

    def maybe_fire(self, occ: float) -> Optional[str]:
        """Called once per scheduler step with an occupancy ratio
        sourced from the same view the daemon reads (typically the
        radix cache's `_aginfer_pool_usage()['HBM']['token_usage']`
        — see scheduler.py for the callsite).

        Returns the kind that was fired (for telemetry / tests), or
        None if no fire happened.  Never raises; any error is logged.
        """
        cur = classify(
            occ, self._last_state, self.theta_hi, self.theta_lo, self.theta_crit
        )
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
            # No transition, no heartbeat due.  derive_kind returned
            # None ⇒ cur == prev, so _last_state already equals cur.
            return None

        payload = WebhookPayload(
            kind=kind,
            state=cur.value,
            prev_state=self._last_state.value,
            occ=float(occ),
            used_tokens=0,        # kept for wire compat; sglang no longer
            cap_tokens=0,         # tracks (used, cap) at this site
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

    def fire_apply_failed(
        self,
        *,
        endpoint: str,
        action_id: str,
        reason: str,
        hash_: Optional[str] = None,
    ) -> None:
        """T23 — schedule a fire-and-forget APPLY_FAILED webhook.

        Called from the scheduler's endpoint handlers (e.g.,
        `migrate_aginfer`) once per skipped/failed action.  The
        daemon's T37 handler bumps the per-reason observability
        counter and the next `joint_decide` re-evaluates.

        Never raises; never blocks the scheduler step.  Network
        errors are logged via the same retry-then-give-up path the
        watermark webhook uses.
        """
        payload = ApplyFailedPayload(
            endpoint=endpoint,
            action_id=action_id,
            reason=reason,
            hash_=hash_,
            ts=time.time(),
            ts_monotonic=time.monotonic(),
        )
        try:
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(
                self._send_apply_failed(payload), self._loop,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "aginfer apply_failed webhook scheduling failed",
                exc_info=True,
            )

    def fire_hash_collision(
        self,
        *,
        key: str,
        node_a_summary: dict,
        node_b_summary: dict,
    ) -> None:
        """T24 (#182, DESIGN §4 + §10 round-15/16/17): schedule a
        fire-and-forget HASH_COLLISION webhook.

        Called from `scheduler.migrate_aginfer` once per pair the
        cache's DFS detected (deduped via the cache's
        `_aginfer_collision_seen` set so a persistent collision
        doesn't spam fires).  Daemon's handler calls fatal() on
        receipt — deployment-bug class.

        Never raises; never blocks the scheduler step.
        """
        payload = HashCollisionPayload(
            key=key,
            node_a_summary=node_a_summary,
            node_b_summary=node_b_summary,
            ts=time.time(),
            ts_monotonic=time.monotonic(),
        )
        try:
            assert self._loop is not None
            asyncio.run_coroutine_threadsafe(
                self._send_hash_collision(payload), self._loop,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "aginfer hash_collision webhook scheduling failed",
                exc_info=True,
            )

    def apply_thresholds(
        self,
        *,
        theta_hi: float,
        theta_lo: float,
        theta_crit: float,
        heartbeat_s: float,
    ) -> None:
        """T22 (#155): atomically replace the four hysteresis values.

        Caller is responsible for validating ranges + the
        ``theta_lo < theta_hi <= theta_crit`` ordering (use
        ``apply_thresholds_payload`` if the values came from an HTTP
        body).  Here we just commit the four with a single GIL-atomic
        rebind to a tuple — a concurrent ``maybe_fire`` either sees
        all-old or all-new, never a torn write.

        Sglang's scheduler is single-threaded so the atomicity
        argument is theoretical for the watermark path, but the PUT
        handler runs on the FastAPI event loop (different thread);
        the tuple swap is the cheapest correct primitive.
        """
        # Single rebind = atomic w.r.t. concurrent reads on CPython.
        # Storing as a tuple keeps "all four together" by construction.
        self._theta_tuple = (
            float(theta_hi),
            float(theta_lo),
            float(theta_crit),
            float(heartbeat_s),
        )

    # Read-through properties so callers keep ``firer.theta_hi`` etc.
    # The tuple swap above is the actual write site.
    @property
    def theta_hi(self) -> float:  # type: ignore[override]
        return self._theta_tuple[0]

    @theta_hi.setter
    def theta_hi(self, value: float) -> None:
        # Legacy attribute-style write path (some tests use it).
        # Preserve atomic-pair-swap by rebinding the whole tuple.
        _, lo, crit, hb = self._theta_tuple
        self._theta_tuple = (float(value), lo, crit, hb)

    @property
    def theta_lo(self) -> float:
        return self._theta_tuple[1]

    @theta_lo.setter
    def theta_lo(self, value: float) -> None:
        hi, _, crit, hb = self._theta_tuple
        self._theta_tuple = (hi, float(value), crit, hb)

    @property
    def theta_crit(self) -> float:  # type: ignore[override]
        return self._theta_tuple[2]

    @theta_crit.setter
    def theta_crit(self, value: float) -> None:
        hi, lo, _, hb = self._theta_tuple
        self._theta_tuple = (hi, lo, float(value), hb)

    @property
    def heartbeat_s(self) -> float:  # type: ignore[override]
        return self._theta_tuple[3]

    @heartbeat_s.setter
    def heartbeat_s(self, value: float) -> None:
        hi, lo, crit, _ = self._theta_tuple
        self._theta_tuple = (hi, lo, crit, float(value))

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

    async def _send_apply_failed(self, payload: ApplyFailedPayload) -> None:
        # Lazy-init httpx client on the background loop.
        if self._client is None:
            import httpx  # noqa: WPS433

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        body = {
            "kind": "apply_failed",
            "endpoint": payload.endpoint,
            "action_id": payload.action_id,
            "reason": payload.reason,
            "hash": payload.hash_,
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
                    "aginfer apply_failed[%s/%s] -> %d (attempt %d/3); "
                    "retrying",
                    payload.endpoint, payload.action_id,
                    resp.status_code, attempt + 1,
                )
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "aginfer apply_failed[%s/%s] raised (attempt %d/3): %s",
                    payload.endpoint, payload.action_id,
                    attempt + 1, exc,
                )
            if attempt < 2:
                await asyncio.sleep(backoff)
                backoff *= 4
        logger.warning(
            "aginfer apply_failed[%s/%s] gave up after 3 attempts",
            payload.endpoint, payload.action_id,
        )

    async def _send_hash_collision(self, payload: HashCollisionPayload) -> None:
        # Lazy-init httpx client on the background loop.
        if self._client is None:
            import httpx  # noqa: WPS433

            self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        body = {
            "kind": "hash_collision",
            "key": payload.key,
            "node_a_summary": payload.node_a_summary,
            "node_b_summary": payload.node_b_summary,
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
                    "aginfer hash_collision[%s] -> %d (attempt %d/3); "
                    "retrying",
                    payload.key, resp.status_code, attempt + 1,
                )
            except Exception as exc:  # noqa: BLE001
                logger.info(
                    "aginfer hash_collision[%s] raised (attempt %d/3): %s",
                    payload.key, attempt + 1, exc,
                )
            if attempt < 2:
                await asyncio.sleep(backoff)
                backoff *= 4
        logger.warning(
            "aginfer hash_collision[%s] gave up after 3 attempts",
            payload.key,
        )


# --------------------------------------------------------------- T22


def apply_thresholds_payload(
    firer: AginferWebhookFirer,
    body: Dict[str, object],
) -> Tuple[bool, str]:
    """Validate a ``PUT /aginfer/thresholds`` body and apply it.

    DESIGN §6 / §10 "Threshold parity": runtime updates flow daemon
    → sglang via this endpoint.  Caller (sglang HTTP route) wraps
    the (ok, reason) return into 200 / 400.

    Validation invariants:
      * Required keys: ``theta_hi``, ``theta_lo``, ``theta_crit``,
        ``heartbeat_s`` — no defaults, no partial updates (an
        operator never wants to set "just one of four" and end up
        with hysteresis inverted).
      * All four numeric (``int`` or ``float``).
      * ``theta_*`` in [0, 1]; ``heartbeat_s`` > 0.
      * Hysteresis: ``theta_lo < theta_hi <= theta_crit``.

    Mutates ``firer`` only on success (via atomic tuple swap).
    """
    required = ("theta_hi", "theta_lo", "theta_crit", "heartbeat_s")
    missing = [k for k in required if k not in body]
    if missing:
        return False, f"missing required field(s): {missing}"
    vals: Dict[str, float] = {}
    for k in required:
        v = body[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, f"field {k!r} must be numeric (type/numeric)"
        vals[k] = float(v)
    for k in ("theta_hi", "theta_lo", "theta_crit"):
        if vals[k] < 0.0 or vals[k] > 1.0:
            return False, (
                f"field {k!r} out of range [0, 1]: {vals[k]} (negative)"
                if vals[k] < 0.0
                else f"field {k!r} out of range [0, 1]: {vals[k]}"
            )
    if vals["heartbeat_s"] <= 0.0:
        return False, (
            f"field 'heartbeat_s' must be > 0 (range): {vals['heartbeat_s']}"
        )
    if not (vals["theta_lo"] < vals["theta_hi"] <= vals["theta_crit"]):
        return False, (
            f"hysteresis violation: theta_lo ({vals['theta_lo']}) < "
            f"theta_hi ({vals['theta_hi']}) <= theta_crit "
            f"({vals['theta_crit']}) required"
        )
    firer.apply_thresholds(
        theta_hi=vals["theta_hi"],
        theta_lo=vals["theta_lo"],
        theta_crit=vals["theta_crit"],
        heartbeat_s=vals["heartbeat_s"],
    )
    return True, "ok"


# Sglang's CLI defaults for the four aginfer thresholds.  Used by
# ``bootstrap_thresholds_into_server_args`` to distinguish
# "operator left this at the default" from "operator explicitly
# passed a non-default value" — the latter case logs a WARNING when
# it disagrees with the daemon (DESIGN §6 step 3 spirit).
_SGLANG_CLI_THRESHOLD_DEFAULTS: Dict[str, float] = {
    "theta_hi":    0.7,
    "theta_lo":    0.55,
    "theta_crit":  0.9,
    "heartbeat_s": 5.0,
}


def fetch_bootstrap_thresholds(
    daemon_base_url: str,
    *,
    timeout_s: float = 5.0,
) -> Dict[str, float]:
    """Sglang-side bootstrap GET.  Called at sglang launch when the
    operator passes ``--aginfer-notify-url`` (canonical daemon
    presence flag); the daemon is the source of truth for thresholds
    (DESIGN §10 "Threshold parity", round-14 dropped the cache).

    Returns the four-field dict on 200.  Raises a connect-error class
    (httpx-derived) on unreachable / timeout — caller (sglang
    launch) MUST halt loudly: running without canonical thresholds
    is a deployment-ordering bug (daemon must be up before sglang).
    """
    import httpx

    base = daemon_base_url.rstrip("/")
    # Tolerate the operator passing the bare base or the event-path
    # form (``http://daemon/aginfer/event``).  Strip a trailing
    # ``/aginfer/event`` to recover the base.
    suffix = "/aginfer/event"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    url = f"{base}/aginfer/thresholds"
    with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
        r = client.get(url)
    r.raise_for_status()
    body = r.json()
    required = {"theta_hi", "theta_lo", "theta_crit", "heartbeat_s"}
    if not isinstance(body, dict) or set(body.keys()) != required:
        raise ValueError(
            f"daemon /aginfer/thresholds returned unexpected shape: {body!r}"
        )
    return {k: float(body[k]) for k in required}


def bootstrap_thresholds_into_server_args(
    server_args,
    *,
    timeout_s: float = 10.0,
    _exit_func=None,  # injected for tests; defaults to sys.exit
) -> None:
    """T22 G9 closure (DESIGN §6 step 1 + step 3).

    Called from ``prepare_server_args`` AFTER CLI parse, BEFORE the
    scheduler subprocess spawn.  Behavior:

      1. If ``server_args.aginfer_notify_url`` is None: legacy /
         daemon-less mode.  No-op.  Sglang stays on CLI defaults.

      2. Otherwise (the canonical "daemon-managed" deployment):

         a. ``fetch_bootstrap_thresholds`` from the daemon.  On
            ANY failure (unreachable, timeout, malformed shape) →
            log ERROR + ``sys.exit(1)``.  Deployment-ordering bug:
            daemon must be up before sglang.  No silent fallback
            to CLI defaults (round-14 dropped the cache; this is
            the same principle).

         b. For each of the four canonical values, OVERWRITE
            ``server_args.aginfer_<k>`` with the daemon's view.
            If the CLI value differs from BOTH the daemon and
            sglang's hardcoded default, the operator explicitly
            passed a non-default that disagrees with the daemon —
            log a WARNING per DESIGN §6 step 3.  Daemon always
            wins; the warning is the operator-visible signal that
            their launch flag is moot.

    Idempotent: re-calling with the same daemon state produces
    the same server_args.
    """
    if _exit_func is None:
        _exit_func = sys.exit
    notify_url = getattr(server_args, "aginfer_notify_url", None)
    if not notify_url:
        return
    try:
        fetched = fetch_bootstrap_thresholds(notify_url, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "T22 (DESIGN §6 step 1): aginfer bootstrap threshold "
            "fetch from daemon %r failed: %s.  Deployment-ordering "
            "bug — daemon must be up before sglang.  Halting.",
            notify_url, exc,
        )
        _exit_func(1)
        return  # pragma: no cover — _exit_func usually doesn't return
    for key in ("theta_hi", "theta_lo", "theta_crit", "heartbeat_s"):
        attr = f"aginfer_{key}"
        current = float(getattr(server_args, attr))
        daemon_val = float(fetched[key])
        default = _SGLANG_CLI_THRESHOLD_DEFAULTS[key]
        if abs(current - daemon_val) < 1e-9:
            continue  # already in sync (could be coincidence; harmless)
        operator_explicit = abs(current - default) > 1e-9
        if operator_explicit:
            logger.warning(
                "T22 (DESIGN §6 step 3): operator passed "
                "--aginfer-%s=%g but daemon canonical value is %g.  "
                "Daemon wins.  Align the launch flag (or omit it) "
                "to silence this warning.",
                key.replace("_", "-"), current, daemon_val,
            )
        else:
            logger.info(
                "T22: aginfer %s seeded from daemon: %g (was sglang "
                "default %g)",
                key, daemon_val, current,
            )
        setattr(server_args, attr, daemon_val)
