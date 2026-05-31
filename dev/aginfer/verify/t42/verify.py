"""T42 verify — daemon-side observability (PLAN §4 T42).

Companion to T14 (sglang state-dump cost).  Daemon emits four metric
streams via the existing ``daemon/_metrics.py`` line format:

  * ``state_fetch`` latency p50 / p95 / p99
  * event-queue depth at handler entry
  * time-in-queue per event
  * cumulative failure-class counters (migrate-skip reasons today;
    APPLY_FAILED breakdown when T23+T37 lands)

PLAN's F3-revisit conditions explicitly key on this metric:
  * queue depth > 64 sustained → F3-revisit
  * time-in-queue > 100 ms p99 → F3-revisit

Phase A (in-process unit tests of ``DaemonObservability``): the
aggregator's ring buffer, summary contract, and per-N-events
emission cadence.

Phase B (integration with EventRouter + EventBus): drive synthetic
events through the bus, assert the observability records the
right counts and queue-depth + time-in-queue both show up.

Phase C (capture log emissions): a logging.Handler grabs every
``aginfer_metric`` line and asserts the ``daemon_obs_summary``
event lands with the contract field set.

Usage:
    python dev/aginfer/verify/t42/verify.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon._observability import (  # noqa: E402
    DaemonObservability,
    _DaemonMetricsRing,
)
from daemon.events import Event, EventBus, EventKind  # noqa: E402
from daemon.event_router import EventRouter  # noqa: E402


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ============================================================ Phase A


def stage_a0_ring_empty_summary() -> None:
    r = _DaemonMetricsRing(capacity=64)
    s = r.summary()
    expected = {"n", "n_recorded_total", "capacity",
                "p50", "p95", "p99", "max", "mean"}
    if set(s.keys()) != expected:
        raise StageFail(
            f"ring summary key mismatch: got {set(s.keys())}; "
            f"want {expected}"
        )
    if s["n"] != 0:
        raise StageFail(f"empty n != 0: {s}")
    for q in ("p50", "p95", "p99", "max", "mean"):
        if s[q] != 0.0:
            raise StageFail(f"empty {q} != 0.0: {s[q]}")


def stage_a1_ring_record_and_quantiles() -> None:
    r = _DaemonMetricsRing(capacity=128)
    for ms in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        r.record(float(ms))
    s = r.summary()
    if s["n"] != 10:
        raise StageFail(f"n=10 expected: {s}")
    if abs(s["mean"] - 5.5) > 0.01:
        raise StageFail(f"mean != 5.5: {s['mean']}")
    if abs(s["max"] - 10.0) > 0.01:
        raise StageFail(f"max != 10.0: {s['max']}")
    if not (s["p50"] <= s["p95"] <= s["p99"] <= s["max"]):
        raise StageFail(
            f"quantile ordering broken: "
            f"p50={s['p50']} p95={s['p95']} p99={s['p99']} max={s['max']}"
        )


def stage_a2_ring_wraps() -> None:
    r = _DaemonMetricsRing(capacity=32)
    for i in range(200):
        r.record(float(i))
    s = r.summary()
    if s["n"] != 32:
        raise StageFail(f"wrapped n != cap 32: {s['n']}")
    if s["n_recorded_total"] != 200:
        raise StageFail(
            f"n_recorded_total != 200: {s['n_recorded_total']}"
        )
    # The buffer holds samples 168..199; max should be 199.
    if abs(s["max"] - 199.0) > 0.001:
        raise StageFail(f"wrapped max != 199: {s['max']}")


def stage_a3_observability_empty_summary_shape() -> None:
    """Empty observability returns the full contract field set."""
    obs = DaemonObservability(capacity=64, summary_every_n=100)
    s = obs.summary_dict()
    required = {
        "state_fetch_lat_ms",
        "queue_depth",
        "time_in_queue_ms",
        "failure_class_counts",
        "events_handled_total",
    }
    if set(s.keys()) != required:
        raise StageFail(
            f"summary_dict keys mismatch: got {set(s.keys())}; "
            f"want {required}"
        )
    # Each ring sub-summary has the standard quantile fields.
    for ring_key in ("state_fetch_lat_ms", "queue_depth", "time_in_queue_ms"):
        sub = s[ring_key]
        if sub["n"] != 0 or sub["p99"] != 0.0:
            raise StageFail(f"empty {ring_key}: {sub}")
    if s["failure_class_counts"] != {}:
        raise StageFail(f"empty counts: {s['failure_class_counts']}")


def stage_a4_observability_failure_counter() -> None:
    """``record_failure(reason)`` increments the per-reason counter."""
    obs = DaemonObservability(capacity=64, summary_every_n=10_000)
    for _ in range(5):
        obs.record_failure("add_already_present:DRAM")
    for _ in range(3):
        obs.record_failure("remove_not_leaf")
    obs.record_failure("not_in_tree")
    counts = obs.summary_dict()["failure_class_counts"]
    if counts != {
        "add_already_present:DRAM": 5,
        "remove_not_leaf": 3,
        "not_in_tree": 1,
    }:
        raise StageFail(f"counter mismatch: {counts}")


def stage_a5_emit_summary_cadence() -> None:
    """``record_dispatch`` triggers ``emit_summary`` every
    ``summary_every_n`` events.  Capture via a logging.Handler that
    collects ``aginfer_metric`` lines."""
    captured: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _CaptureHandler()
    logger = logging.getLogger("aginfer.metric")
    prior_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        obs = DaemonObservability(capacity=64, summary_every_n=5)
        # 4 events: no summary yet
        for i in range(4):
            obs.record_dispatch(qdepth=i, time_in_queue_ms=float(i))
        n_summary_lines = sum(
            1 for line in captured
            if "event=daemon_obs_summary" in line
        )
        if n_summary_lines != 0:
            raise StageFail(
                f"summary emitted prematurely after 4 events: {n_summary_lines}"
            )
        # 5th event triggers a summary line
        obs.record_dispatch(qdepth=4, time_in_queue_ms=4.0)
        n_summary_lines = sum(
            1 for line in captured
            if "event=daemon_obs_summary" in line
        )
        if n_summary_lines != 1:
            raise StageFail(
                f"expected exactly 1 summary after 5 events; "
                f"got {n_summary_lines}; captured={captured!r}"
            )
        # Another 5 events → second summary
        for i in range(5):
            obs.record_dispatch(qdepth=i, time_in_queue_ms=float(i))
        n_summary_lines = sum(
            1 for line in captured
            if "event=daemon_obs_summary" in line
        )
        if n_summary_lines != 2:
            raise StageFail(
                f"expected 2 summary lines after 10 events; "
                f"got {n_summary_lines}"
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


def stage_a6_emit_summary_contract_fields() -> None:
    """``emit_summary`` line must carry the four PLAN T42 streams."""
    captured: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _CaptureHandler()
    logger = logging.getLogger("aginfer.metric")
    prior_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        obs = DaemonObservability(capacity=64, summary_every_n=5)
        for i in range(5):
            obs.record_state_fetch(elapsed_ms=float(10 + i))
            obs.record_dispatch(qdepth=i, time_in_queue_ms=float(5 + i))
        obs.record_failure("add_already_present:DRAM")
        obs.emit_summary()  # force emit
        summary_lines = [
            l for l in captured if "event=daemon_obs_summary" in l
        ]
        if not summary_lines:
            raise StageFail(
                f"no daemon_obs_summary line captured; got: {captured}"
            )
        line = summary_lines[-1]
        required_substrings = (
            "state_fetch_p99_ms=",
            "state_fetch_p50_ms=",
            "queue_depth_p99=",
            "time_in_queue_p99_ms=",
            "events_handled_total=",
            "n_failure_classes=",
        )
        for needle in required_substrings:
            if needle not in line:
                raise StageFail(
                    f"summary line missing {needle!r}; line={line!r}"
                )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)


# ============================================================ Phase B
# Integration with EventRouter + EventBus + Event.enqueue_time.


def stage_b0_enqueue_time_stamped_by_bus() -> None:
    """``EventBus.emit`` must stamp ``enqueue_time`` on the queued
    Event (the worker reads it to compute time-in-queue).  An Event
    constructed without enqueue_time and emitted should arrive on
    the queue with a populated, monotonic-clock-positive value."""
    bus = EventBus()
    evt = Event(kind=EventKind.MEMORY_PRESSURE)
    if evt.enqueue_time != 0.0:
        raise StageFail(
            f"unstamped event should have enqueue_time=0.0; "
            f"got {evt.enqueue_time}"
        )
    asyncio.run(bus.emit(evt))
    queued = bus.queue.get_nowait()
    if queued.enqueue_time <= 0.0:
        raise StageFail(
            f"emit() should stamp enqueue_time; got {queued.enqueue_time}"
        )
    if (queued.kind, queued.payload) != (evt.kind, evt.payload):
        raise StageFail("emit replaced more than enqueue_time")


def stage_b1_router_records_dispatch_and_time_in_queue() -> None:
    """Drive 5 events through a real EventRouter; observability should
    record 5 dispatch samples with time_in_queue > 0 and qdepth >= 0."""
    async def _scenario() -> Dict[str, Any]:
        bus = EventBus()
        router = EventRouter(
            bus=bus,
            sglang_base_url="http://unused",
            theta_hi=0.7,
            theta_crit=0.9,
        )
        # Stub the inner HTTP impl so handlers don't hit the network
        # (the public fetch_state path keeps the instrumentation timer).
        async def _fake_fetch():
            return {"pool_usage": {"HBM": {"subpools": {}}}}
        router._fetch_state_impl = _fake_fetch  # type: ignore[assignment]
        # Trivial handler that just touches state once.
        async def _h(evt, r):
            await r.fetch_state()
        for kind in (EventKind.MEMORY_PRESSURE,
                     EventKind.SESSION_ARRIVAL,
                     EventKind.LLM_PREFILL,
                     EventKind.TOOL_CALL_START,
                     EventKind.TOOL_CALL_END):
            router.set_handler(kind, _h)
        await router.start()
        try:
            for kind in (EventKind.MEMORY_PRESSURE,
                         EventKind.SESSION_ARRIVAL,
                         EventKind.LLM_PREFILL,
                         EventKind.TOOL_CALL_START,
                         EventKind.TOOL_CALL_END):
                await bus.emit(Event(kind=kind))
            # Wait for the worker to drain.
            await bus.queue.join()
            return router.observability.summary_dict()
        finally:
            await router.stop()

    summary = asyncio.run(_scenario())
    if summary["queue_depth"]["n"] != 5:
        raise StageFail(
            f"queue_depth ring should hold 5 samples after 5 events; "
            f"got {summary['queue_depth']['n']}"
        )
    if summary["time_in_queue_ms"]["n"] != 5:
        raise StageFail(
            f"time_in_queue ring should hold 5 samples; "
            f"got {summary['time_in_queue_ms']['n']}"
        )
    if summary["time_in_queue_ms"]["max"] <= 0.0:
        raise StageFail(
            f"time_in_queue max should be > 0 after real dispatch; "
            f"got {summary['time_in_queue_ms']['max']}"
        )


def stage_b2_router_records_state_fetch_latency() -> None:
    """``router.fetch_state()`` should bump ``state_fetch_lat_ms`` per
    call.  The PLAN F3-revisit trigger key on this metric directly."""
    async def _scenario() -> Dict[str, Any]:
        bus = EventBus()
        router = EventRouter(
            bus=bus,
            sglang_base_url="http://unused",
            theta_hi=0.7,
            theta_crit=0.9,
        )
        # A fake fetch with a fixed sleep so we know the latency is
        # measurable (>= 5 ms).
        async def _fake_fetch():
            await asyncio.sleep(0.005)
            return {"pool_usage": {"HBM": {"subpools": {}}}}
        router._fetch_state_impl = _fake_fetch  # type: ignore[assignment]

        # Call the public, instrumented entry point.  The 5 ms sleep
        # in the stub guarantees the timer reads a measurable value.
        for _ in range(4):
            await router.fetch_state()
        return router.observability.summary_dict()

    summary = asyncio.run(_scenario())
    if summary["state_fetch_lat_ms"]["n"] != 4:
        raise StageFail(
            f"state_fetch_lat ring should hold 4 samples; "
            f"got {summary['state_fetch_lat_ms']['n']}"
        )
    if summary["state_fetch_lat_ms"]["max"] < 4.0:
        raise StageFail(
            f"state_fetch_lat max should be ≥ ~5 ms (fake fetch sleeps); "
            f"got {summary['state_fetch_lat_ms']['max']}"
        )


def stage_b3_router_records_failure_class() -> None:
    """``router.observability.record_failure(reason)`` increments the
    per-reason counter; kv_scheduler's skip-loop should be the call
    site once it's wired.  We verify the path is callable here so the
    integration is exercised."""
    bus = EventBus()
    router = EventRouter(
        bus=bus, sglang_base_url="http://unused",
        theta_hi=0.7, theta_crit=0.9,
    )
    router.observability.record_failure("not_in_tree")
    router.observability.record_failure("not_in_tree")
    router.observability.record_failure("add_already_present:DRAM")
    counts = router.observability.summary_dict()["failure_class_counts"]
    if counts != {"not_in_tree": 2, "add_already_present:DRAM": 1}:
        raise StageFail(f"counter mismatch via router: {counts}")


def stage_b4_kv_scheduler_skips_bump_observability() -> None:
    """Wire KvScheduler with observability=router.observability and
    feed it a synthetic skipped_list; assert the per-reason counter
    in the observability instance reflects the skips.  This is the
    production wiring (main.py passes router.observability into
    KvScheduler) — verifying it here closes the loop without spinning
    up an HTTP server."""
    from daemon.kv_scheduler import KvScheduler
    from daemon.program_tracker import ProgramTracker

    bus = EventBus()
    router = EventRouter(
        bus=bus, sglang_base_url="http://unused",
        theta_hi=0.7, theta_crit=0.9,
    )
    sched = KvScheduler(
        tracker=ProgramTracker(),
        sglang_base_url="http://unused",
        observability=router.observability,
    )
    # Synthetic /aginfer/migrate response.skipped[]; matches the T20
    # skip-reason vocabulary.
    skipped = [
        {"hash": "h1", "reason": "add_already_present:DRAM",
         "action_id": "a1"},
        {"hash": "h2", "reason": "add_already_present:DRAM",
         "action_id": "a2"},
        {"hash": "h3", "reason": "remove_not_leaf",
         "action_id": "a3"},
        {"hash": "h4", "reason": "not_in_tree",
         "action_id": "a4"},
    ]
    captured: List[str] = []
    def _capture(event, **kv):
        captured.append(event)
    sched._record_skips(skipped, _m_func=_capture)
    counts = router.observability.summary_dict()["failure_class_counts"]
    if counts != {
        "add_already_present:DRAM": 2,
        "remove_not_leaf": 1,
        "not_in_tree": 1,
    }:
        raise StageFail(
            f"kv_scheduler-driven counter mismatch: {counts}"
        )
    if len(captured) != 4:
        raise StageFail(
            f"per-line `migrate_skipped` should fire once per skip; "
            f"captured={captured}"
        )


# ============================================================ Phase C
# log-line capture: confirms the end-to-end emission lands.


def stage_c0_summary_line_emitted_in_real_dispatch() -> None:
    """Run 12 events through a real EventRouter with summary_every_n=10;
    expect exactly one daemon_obs_summary line in the logs."""
    captured: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _CaptureHandler()
    metric_logger = logging.getLogger("aginfer.metric")
    prior_level = metric_logger.level
    metric_logger.addHandler(handler)
    metric_logger.setLevel(logging.INFO)
    try:
        async def _scenario():
            bus = EventBus()
            router = EventRouter(
                bus=bus,
                sglang_base_url="http://unused",
                theta_hi=0.7,
                theta_crit=0.9,
                observability_summary_every_n=10,
            )
            async def _fake_fetch():
                return {"pool_usage": {"HBM": {"subpools": {}}}}
            router._fetch_state_impl = _fake_fetch  # type: ignore[assignment]
            async def _h(evt, r):
                pass
            for kind in EventKind:
                router.set_handler(kind, _h)
            await router.start()
            try:
                for _ in range(12):
                    await bus.emit(Event(kind=EventKind.MEMORY_PRESSURE))
                await bus.queue.join()
            finally:
                await router.stop()
        asyncio.run(_scenario())
        summary_lines = [
            l for l in captured if "event=daemon_obs_summary" in l
        ]
        if len(summary_lines) != 1:
            raise StageFail(
                f"expected exactly 1 daemon_obs_summary line after "
                f"12 events at every-10 cadence; got {len(summary_lines)}; "
                f"all captured: {captured!r}"
            )
        line = summary_lines[0]
        # Sanity-check the key fields are present and not zero.
        if "events_handled_total=10" not in line:
            raise StageFail(
                f"summary should show events_handled_total=10; line={line!r}"
            )
    finally:
        metric_logger.removeHandler(handler)
        metric_logger.setLevel(prior_level)


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 _DaemonMetricsRing empty summary",        stage_a0_ring_empty_summary),
    ("A1 ring record + quantile monotonicity",     stage_a1_ring_record_and_quantiles),
    ("A2 ring wraps at capacity",                  stage_a2_ring_wraps),
    ("A3 observability empty summary shape",       stage_a3_observability_empty_summary_shape),
    ("A4 failure-class counter increments",        stage_a4_observability_failure_counter),
    ("A5 emit_summary cadence (every N events)",   stage_a5_emit_summary_cadence),
    ("A6 summary line carries contract fields",    stage_a6_emit_summary_contract_fields),
    ("B0 EventBus.emit stamps enqueue_time",       stage_b0_enqueue_time_stamped_by_bus),
    ("B1 router records dispatch + time-in-queue", stage_b1_router_records_dispatch_and_time_in_queue),
    ("B2 router records state-fetch latency",      stage_b2_router_records_state_fetch_latency),
    ("B3 router exposes failure-class recorder",   stage_b3_router_records_failure_class),
    ("B4 kv_scheduler skips bump observability",   stage_b4_kv_scheduler_skips_bump_observability),
    ("C0 real-dispatch summary line lands",        stage_c0_summary_line_emitted_in_real_dispatch),
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
        print(_red(f"\nT42 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    print(_green(f"\nT42 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
