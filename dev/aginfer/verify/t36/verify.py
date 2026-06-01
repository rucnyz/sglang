"""T36 verify — outbound action queue + worker (PLAN §4 / DESIGN §6 B4).

Before T36, ``KvScheduler._dispatch_migrate`` did
``await client.post(...)`` on the event-worker's coroutine — every
event blocked on sglang's 50–200 ms POST round trip.  T42's stress
probe measured this directly: ``time_in_queue_p99 = 355.84 ms``
under 100 webhook events (PLAN F3-revisit threshold is 100 ms p99).

T36 splits the path:
  * ``EventHandler`` (kv_scheduler) **enqueues** a batch and returns
    immediately.  No await on HTTP.
  * ``OutboundWorker`` (background asyncio task) consumes the queue
    and issues the POST.  Failures surface via APPLY_FAILED webhook
    (T23+T37) — the sync response body is dropped on success.

The key property test (A1): handler returns < 1 ms even if the
downstream POST takes 200 ms.

Phase A is in-process; Phase B is an opt-in live re-run of T42's
stress probe to measure how much time_in_queue p99 dropped under
the new path.

Usage:
    python dev/aginfer/verify/t36/verify.py
    AGINFER_VERIFY_BASE=http://127.0.0.1:9100 \\
        python dev/aginfer/verify/t36/verify.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import httpx


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.outbound import OutboundBatch, OutboundQueue  # noqa: E402


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


class StageFail(AssertionError):
    pass


_UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


# ----------------------------------------------------------------- helpers


class _StubHttpClient:
    """A drop-in replacement for httpx.AsyncClient whose .post sleeps
    a configurable number of milliseconds then returns a stub
    response.  Tracks per-URL call counts so tests can assert the
    worker actually issued the POST."""

    def __init__(
        self,
        *,
        post_delay_ms: float = 0.0,
        status_code: int = 200,
        raise_on_post: bool = False,
    ) -> None:
        self._post_delay_s = post_delay_ms / 1000.0
        self._status_code = status_code
        self._raise = raise_on_post
        self.posts: List[Tuple[str, dict]] = []

    async def post(self, url: str, *, json=None):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self._post_delay_s)
        self.posts.append((url, json or {}))
        if self._raise:
            raise httpx.ConnectError("stubbed connection error")
        return _StubResponse(self._status_code)

    async def aclose(self) -> None:
        return None


class _StubResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.text = ""

    def json(self):  # type: ignore[no-untyped-def]
        return {"applied": 0, "applied_hashes": [], "skipped": []}


# ============================================================ Phase A


def stage_a0_batch_id_is_uuid4() -> None:
    """``enqueue_migrate`` returns a UUID4 batch_id; ``_UUID4_RE``
    matches.  Defends against a regression that uses a counter or
    a non-UUID identifier — the APPLY_FAILED correlation contract
    (DESIGN §6 L507) requires a UUID."""
    async def _go():
        outbound = OutboundQueue(
            sglang_base_url="http://unused",
            http_client=_StubHttpClient(),
        )
        bid = outbound.enqueue_migrate([{"hash": "h", "add_tiers": [],
                                          "remove_tiers": ["HBM"]}])
        if not _UUID4_RE.match(bid):
            raise StageFail(f"batch_id is not UUID4: {bid!r}")
        # Two enqueues → two distinct batch_ids.
        bid2 = outbound.enqueue_migrate([])
        if bid == bid2:
            raise StageFail(f"non-unique batch_ids: {bid!r} {bid2!r}")
        await outbound.stop()
    asyncio.run(_go())


def stage_a1_handler_returns_under_1ms_regardless_of_post() -> None:
    """**The headline property.**  Even when the downstream POST
    sleeps 200 ms inside the stub, ``enqueue_migrate`` must return
    in well under 1 ms (it's just an ``asyncio.Queue.put_nowait``
    + ``uuid.uuid4()``).  The PRE-T36 sync code would block the
    handler for the full 200 ms."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=200.0)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
        )
        await outbound.start()
        try:
            # Time 50 enqueues; assert max < 1 ms (generous: typical
            # is < 50 µs).  Don't time a single shot — JIT / Python
            # noise can spike one call.
            timings_us = []
            for i in range(50):
                t0 = time.perf_counter()
                outbound.enqueue_migrate([
                    {"hash": f"h{i}", "add_tiers": [],
                     "remove_tiers": ["HBM"], "action_id": f"a{i}"}
                ])
                timings_us.append((time.perf_counter() - t0) * 1e6)
            # Let the worker drain.
            await outbound.queue.join()
        finally:
            await outbound.stop()
        max_us = max(timings_us)
        if max_us > 1000.0:
            raise StageFail(
                f"handler-side enqueue exceeded 1 ms: max={max_us:.1f} µs; "
                f"all={[f'{t:.0f}' for t in timings_us]}"
            )
        # Sanity: the worker DID issue 50 posts despite each taking
        # 200 ms — proves the worker is actually draining and the
        # handler isn't synchronously waiting.
        if len(stub.posts) != 50:
            raise StageFail(
                f"worker drain incomplete: {len(stub.posts)} of 50 posts"
            )
    asyncio.run(_go())


def stage_a2_worker_drains_queue_in_order() -> None:
    """Single worker drains FIFO.  Defends against a regression that
    swaps the queue for a stack."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=10.0)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
        )
        await outbound.start()
        try:
            for i in range(10):
                outbound.enqueue_migrate(
                    [{"hash": f"h-{i:02d}", "add_tiers": [],
                      "remove_tiers": ["HBM"], "action_id": f"a-{i:02d}"}]
                )
            await outbound.queue.join()
        finally:
            await outbound.stop()
        # Reconstruct order from posts.
        order = [
            p[1]["actions"][0]["hash"] for p in stub.posts
        ]
        expected = [f"h-{i:02d}" for i in range(10)]
        if order != expected:
            raise StageFail(f"FIFO violated: got {order!r}")
    asyncio.run(_go())


def stage_a3_worker_survives_5xx() -> None:
    """5xx from sglang must NOT crash the worker.  Surfaces as a
    log warning; the worker pops the next batch and continues.
    DESIGN §6: APPLY_FAILED webhook is the structured-failure path;
    a 5xx is a transient transport / overload that the next
    joint_decide re-converges from."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=2.0, status_code=503)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
        )
        await outbound.start()
        try:
            for i in range(5):
                outbound.enqueue_migrate(
                    [{"hash": f"h{i}", "add_tiers": [],
                      "remove_tiers": ["HBM"], "action_id": f"a{i}"}]
                )
            await outbound.queue.join()
        finally:
            await outbound.stop()
        # Each batch should still have hit the wire (5xx is observed,
        # not skipped).  Retry semantics live below (A4); A3 just
        # asserts the worker survived the cluster of 5xx.
        if len(stub.posts) != 5:
            raise StageFail(
                f"5xx made the worker drop posts; got {len(stub.posts)}/5"
            )
    asyncio.run(_go())


def stage_a4_worker_survives_connect_error() -> None:
    """``httpx.ConnectError`` (sglang down, daemon-attached-mode race
    at restart, network blip) must NOT crash the worker.  Same
    contract as A3: log + move on."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=1.0, raise_on_post=True)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
        )
        await outbound.start()
        try:
            for i in range(3):
                outbound.enqueue_migrate(
                    [{"hash": f"h{i}", "add_tiers": [],
                      "remove_tiers": ["HBM"], "action_id": f"a{i}"}]
                )
            await outbound.queue.join()
        finally:
            await outbound.stop()
        # The connection-error path also attempts the post; presence
        # in stub.posts confirms the call site reached.
        if len(stub.posts) != 3:
            raise StageFail(
                f"connect-error made the worker drop posts; got "
                f"{len(stub.posts)}/3"
            )
    asyncio.run(_go())


def stage_a5_stop_drains_inflight_then_exits() -> None:
    """``OutboundQueue.stop()`` must wait for the in-flight POST to
    finish (or be cancelled cleanly) before returning.  Without this
    a SIGTERM-fast-restart could lose actions silently."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=50.0)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
        )
        await outbound.start()
        for i in range(3):
            outbound.enqueue_migrate(
                [{"hash": f"h{i}", "add_tiers": [],
                  "remove_tiers": ["HBM"], "action_id": f"a{i}"}]
            )
        # Without queue.join(), stop() should still drain or cleanly
        # cancel within a bounded window.
        t0 = time.perf_counter()
        await outbound.stop()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # 3 batches × 50 ms each = 150 ms upper bound on full drain.
        # Allow generous headroom; reject infinite hang.
        if elapsed_ms > 2000.0:
            raise StageFail(
                f"stop() took too long: {elapsed_ms:.0f} ms"
            )
    asyncio.run(_go())


def stage_a6_kv_scheduler_uses_outbound_no_sync_post() -> None:
    """End-to-end wiring: KvScheduler._dispatch_migrate must enqueue
    onto outbound, NOT await client.post directly.  After T36 the
    `httpx_client` parameter on KvScheduler is no longer the dispatch
    client — it might still exist for backward compat but the new
    code path goes through ``self.outbound``.

    Verify: stub an OutboundQueue, hand it to KvScheduler, call
    `_dispatch_migrate(...)`; the queue should have exactly one
    OutboundBatch with the right actions; no HTTP client call."""
    from daemon.kv_scheduler import KvScheduler, assignments_to_wire
    from daemon.program_tracker import ProgramTracker
    from baselines.base import Tier

    async def _go():
        stub_http = _StubHttpClient()
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub_http,
        )
        sched = KvScheduler(
            tracker=ProgramTracker(),
            sglang_base_url="http://unused",
            outbound=outbound,
        )
        assignments = [
            ("h1", [Tier.HBM], []),
            ("h2", [Tier.DRAM], [Tier.HBM]),
        ]
        # Don't start() the outbound worker — we want to assert what
        # got enqueued, not what got posted.
        await sched._dispatch_migrate(assignments)
        if outbound.queue.qsize() != 1:
            raise StageFail(
                f"expected exactly one batch on the queue; got "
                f"{outbound.queue.qsize()}"
            )
        batch: OutboundBatch = outbound.queue.get_nowait()
        if batch.endpoint != "migrate":
            raise StageFail(f"endpoint={batch.endpoint!r}; want 'migrate'")
        if "actions" not in batch.body:
            raise StageFail(f"batch body missing actions: {batch.body!r}")
        if len(batch.body["actions"]) != 2:
            raise StageFail(
                f"actions count mismatch: {len(batch.body['actions'])}"
            )
        # No sync POST happened.
        if stub_http.posts:
            raise StageFail(
                f"unexpected sync POST(s): {stub_http.posts}"
            )
        await outbound.stop()
    asyncio.run(_go())


# ============================================================ Phase B
# Live re-measurement: how much did time_in_queue p99 drop?


def stage_b0_live_time_in_queue_under_threshold() -> None:
    """Optional integration probe: drive 50 webhook events through a
    live daemon (which uses the outbound queue), then grep the daemon
    log for a daemon_obs_summary line and assert
    ``time_in_queue_p99_ms < 100``.

    PRE-T36 baseline from T42 stress: 355.84 ms.
    POST-T36 expected: handler returns <1 ms regardless of POST
    latency, so queue depth shouldn't accumulate; p99 should drop
    by 1–2 orders of magnitude.
    """
    base = os.environ.get("AGINFER_VERIFY_BASE", "").rstrip("/")
    log_path = os.environ.get("AGINFER_VERIFY_DAEMON_LOG", "")
    if not base or not log_path:
        print(_yellow(
            "  (skip B0) set AGINFER_VERIFY_BASE + AGINFER_VERIFY_DAEMON_LOG"
        ))
        return
    import urllib.request
    import json
    # Fire 50 synthetic webhook events.
    for i in range(50):
        body = {
            "kind": "memory_pressure",
            "session": f"t36-obs-{i}",
            "state": "HIGH",
            "prev_state": "OK",
            "occ": 0.85,
        }
        req = urllib.request.Request(
            f"{base}/aginfer/event",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass
    # Wait for drain + a summary emission.
    time.sleep(5)
    try:
        content = Path(log_path).read_text()
    except OSError:
        content = ""
    summary_lines = [
        l for l in content.splitlines()
        if "event=daemon_obs_summary" in l
    ]
    if not summary_lines:
        raise StageFail(
            f"no daemon_obs_summary in {log_path!r}; "
            f"daemon may not have emitted yet"
        )
    last = summary_lines[-1]
    m = re.search(r"time_in_queue_p99_ms=([0-9.eE+-]+)", last)
    if not m:
        raise StageFail(
            f"time_in_queue_p99_ms not found in summary: {last!r}"
        )
    p99_ms = float(m.group(1))
    if p99_ms >= 100.0:
        raise StageFail(
            f"time_in_queue_p99_ms={p99_ms:.2f} >= 100 ms PLAN F3 threshold; "
            f"T36 should have dropped it 1-2 orders of magnitude"
        )
    print(f"      (live) time_in_queue_p99_ms = {p99_ms:.2f} ms")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 batch_id is UUID4 + unique",                 stage_a0_batch_id_is_uuid4),
    ("A1 handler enqueue returns <1ms regardless of POST latency",
                                                      stage_a1_handler_returns_under_1ms_regardless_of_post),
    ("A2 worker drains queue in FIFO order",          stage_a2_worker_drains_queue_in_order),
    ("A3 worker survives sglang 5xx",                 stage_a3_worker_survives_5xx),
    ("A4 worker survives httpx ConnectError",         stage_a4_worker_survives_connect_error),
    ("A5 stop() drains in-flight then exits bounded", stage_a5_stop_drains_inflight_then_exits),
    ("A6 KvScheduler._dispatch_migrate enqueues, no sync POST",
                                                      stage_a6_kv_scheduler_uses_outbound_no_sync_post),
    ("B0 live time_in_queue_p99 < 100 ms (post-T36)", stage_b0_live_time_in_queue_under_threshold),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(f"\nT36 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    print(_green(f"\nT36 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
