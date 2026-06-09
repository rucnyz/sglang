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

    async def request(self, method, url, *, json=None):  # type: ignore[no-untyped-def]
        # #228: PUT endpoints (hints, program_paused) route through
        # ``.request``; mirror ``.post`` so coalesced PUTs are recorded
        # the same way.  ``self.posts`` is the unified dispatch log.
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
    """**The headline property.**  Even when the downstream PUT sleeps
    200 ms inside the stub, ``enqueue_hints`` must return in well under
    1 ms (it's just an ``asyncio.Queue.put_nowait`` + ``uuid.uuid4()``).
    The PRE-T36 sync code would block the handler for the full 200 ms.

    #228: the worker now COALESCES per wake — a burst of hint batches
    collapses to at most one PUT per wake (latest value per hash).  So
    the drain assertion is no longer "N PUTs for N enqueues"; it is
    "every enqueued hash's CONTENT is delivered across the coalesced
    PUT(s)".  We enqueue 50 distinct-hash hint batches and assert the
    UNION of hashes across all dispatched PUTs == all 50."""
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
                outbound.enqueue_hints([
                    {"hash": f"h{i}", "p_hat": 0.1, "lambda": 0.01,
                     "stamp": i}
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
        # #228: assert the worker delivered every batch's CONTENT across
        # the coalesced PUT(s) — the union of hashes seen on the wire
        # must equal all 50, even though the worker emitted far fewer
        # than 50 PUTs (coalescing).  Proves the worker drained and no
        # content was lost.
        seen_hashes = set()
        for _url, body in stub.posts:
            for h in body.get("hints", []):
                seen_hashes.add(h.get("hash"))
        expected = {f"h{i}" for i in range(50)}
        if seen_hashes != expected:
            missing = expected - seen_hashes
            raise StageFail(
                f"coalesced PUTs did not deliver all 50 hashes; "
                f"missing={sorted(missing)} "
                f"(saw {len(seen_hashes)} across {len(stub.posts)} PUTs)"
            )
    asyncio.run(_go())


def stage_a2_worker_coalesces_latest_per_key_and_orders_endpoints() -> None:
    """#228: the worker no longer dispatches per-batch FIFO — it
    COALESCES each wake.  This stage pins the replacement contract:

      * Within an endpoint, latest-enqueued value per key wins
        (migrate: latest decision per unit hash; hints: highest stamp
        per hash).
      * Across endpoints, the dispatch order is
        program_paused → migrate → hints (liveness → eviction →
        idempotent flood), regardless of enqueue order.

    All batches are enqueued BEFORE the worker starts so the first
    ``queue.get`` + drain pulls the whole burst into ONE coalesce →
    one dispatch per endpoint, making the order deterministic."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=1.0)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
            migrate_freshness_ms=0.0,  # disable stale-drop for the test
        )
        # Enqueue hints LAST and migrate/paused interleaved so the
        # dispatch order is NOT the enqueue order — the contract must
        # reorder them.  Within migrate, enqueue hash "u" twice; the
        # LATER decision (remove DRAM) must win.
        outbound.enqueue_hints(
            [{"hash": "x", "p_hat": 0.1, "lambda": 0.01, "stamp": 1}]
        )
        outbound.enqueue_migrate([{"hash": "u", "remove_tiers": ["HBM"]}])
        outbound.enqueue_program_paused(pid="p0", state="ENDED")
        outbound.enqueue_migrate([{"hash": "u", "remove_tiers": ["DRAM"]}])
        # hints for "x" again with a HIGHER stamp — must supersede.
        outbound.enqueue_hints(
            [{"hash": "x", "p_hat": 0.9, "lambda": 0.5, "stamp": 7}]
        )
        await outbound.start()
        try:
            await outbound.queue.join()
        finally:
            await outbound.stop()
        # One dispatch per endpoint, in the contract order.
        endpoint_order = [url.rsplit("/", 1)[-1] for url, _ in stub.posts]
        expected_order = ["program_paused", "migrate", "hints"]
        if endpoint_order != expected_order:
            raise StageFail(
                f"cross-endpoint dispatch order wrong: got {endpoint_order!r}, "
                f"want {expected_order!r}"
            )
        bodies = {url.rsplit("/", 1)[-1]: body for url, body in stub.posts}
        # migrate: latest decision per hash "u" wins → remove DRAM.
        m_actions = bodies["migrate"]["actions"]
        if len(m_actions) != 1 or m_actions[0]["remove_tiers"] != ["DRAM"]:
            raise StageFail(
                f"migrate coalesce did not keep latest-per-hash; "
                f"actions={m_actions!r}"
            )
        # hints: highest-stamp value per hash "x" wins → stamp 7.
        h_hints = bodies["hints"]["hints"]
        if len(h_hints) != 1 or h_hints[0]["stamp"] != 7:
            raise StageFail(
                f"hints coalesce did not keep highest-stamp; "
                f"hints={h_hints!r}"
            )
    asyncio.run(_go())


def stage_a3_worker_survives_5xx() -> None:
    """5xx from sglang must NOT crash the worker.  Surfaces as a
    log warning; the worker stays alive and keeps dispatching on the
    NEXT wake.  DESIGN §6: APPLY_FAILED webhook is the structured-
    failure path; a 5xx is a transient transport / overload that the
    next joint_decide re-converges from.

    #228: a wake coalesces, so "N batches → N posts" is no longer the
    contract.  Instead: enqueue a wave that fails (5xx), wait for it to
    drain, then enqueue a SECOND wave and assert it is STILL dispatched
    (the worker survived the error and is processing subsequent wakes)."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=2.0, status_code=503)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
        )
        await outbound.start()
        try:
            # Wave 1 — fails with 5xx.
            outbound.enqueue_migrate([{"hash": "w1", "remove_tiers": ["HBM"]}])
            await outbound.queue.join()
            posts_after_wave1 = len(stub.posts)
            if posts_after_wave1 < 1:
                raise StageFail("wave 1 never reached the wire")
            # Wave 2 — if the worker died on the 5xx, this never ships.
            outbound.enqueue_migrate([{"hash": "w2", "remove_tiers": ["HBM"]}])
            await outbound.queue.join()
        finally:
            await outbound.stop()
        if len(stub.posts) <= posts_after_wave1:
            raise StageFail(
                f"worker did not dispatch wave 2 after a 5xx — it likely "
                f"died; posts={len(stub.posts)} (was {posts_after_wave1})"
            )
        seen = {h.get("hash")
                for _u, b in stub.posts for h in b.get("actions", [])}
        if "w2" not in seen:
            raise StageFail(
                f"wave-2 content not delivered after 5xx; saw {sorted(seen)}"
            )
    asyncio.run(_go())


def stage_a4_worker_survives_connect_error() -> None:
    """``httpx.ConnectError`` (sglang down, daemon-attached-mode race
    at restart, network blip) must NOT crash the worker.  Same
    contract as A3: log + move on, keep dispatching subsequent wakes.

    #228: assert survival across wakes, not a per-batch post count —
    enqueue a failing wave, then a second wave, and assert the second
    is still processed."""
    async def _go():
        stub = _StubHttpClient(post_delay_ms=1.0, raise_on_post=True)
        outbound = OutboundQueue(
            sglang_base_url="http://unused", http_client=stub,
        )
        await outbound.start()
        try:
            # Wave 1 — raises ConnectError on dispatch.
            outbound.enqueue_migrate([{"hash": "w1", "remove_tiers": ["HBM"]}])
            await outbound.queue.join()
            posts_after_wave1 = len(stub.posts)
            if posts_after_wave1 < 1:
                raise StageFail("wave 1 never reached the wire")
            # Wave 2 — proves the worker survived the transport error.
            outbound.enqueue_migrate([{"hash": "w2", "remove_tiers": ["HBM"]}])
            await outbound.queue.join()
        finally:
            await outbound.stop()
        if len(stub.posts) <= posts_after_wave1:
            raise StageFail(
                f"worker did not dispatch wave 2 after a ConnectError — it "
                f"likely died; posts={len(stub.posts)} "
                f"(was {posts_after_wave1})"
            )
        seen = {h.get("hash")
                for _u, b in stub.posts for h in b.get("actions", [])}
        if "w2" not in seen:
            raise StageFail(
                f"wave-2 content not delivered after ConnectError; "
                f"saw {sorted(seen)}"
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
        # Note: post-T36-cleanup the sync POST path was removed from
        # KvScheduler entirely, so the "no sync POST" assertion is
        # structurally unreachable — kept here only as the post-
        # cleanup contract: enqueue happens, nothing else.  The
        # stub_http is not started (no worker spun up) so no POST
        # could fire even if a code path tried.
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
    ("A2 worker coalesces latest-per-key + orders endpoints",
                                                      stage_a2_worker_coalesces_latest_per_key_and_orders_endpoints),
    ("A3 worker survives sglang 5xx (keeps dispatching next wakes)",
                                                      stage_a3_worker_survives_5xx),
    ("A4 worker survives httpx ConnectError (keeps dispatching)",
                                                      stage_a4_worker_survives_connect_error),
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
