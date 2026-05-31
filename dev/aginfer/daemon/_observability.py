"""T42 — daemon-side observability aggregator (PLAN §4).

Companion to T14 (sglang-side state-dump cost).  Aggregates four
metric streams the operator needs to spot daemon-side backpressure
empirically before designing the F3 fix:

  1. state-fetch latency (the daemon's GET /aginfer/state wall-clock)
  2. event-queue depth at handler entry (qsize after pop)
  3. time-in-queue per event (now - event.enqueue_time)
  4. cumulative failure-class counters (migrate-skip reasons today;
     APPLY_FAILED breakdown will plug in via the same recorder when
     T23+T37 lands)

Emission: every ``summary_every_n`` handled events the aggregator
fires one ``daemon_obs_summary`` line through the existing
``_metrics.m()`` log format (so an operator's metric-grep pipeline
needs no new ingester).  Per-call lines are NOT emitted here —
high-frequency events already get their own per-event log line
(``event_received``, ``migrate_skipped``); T42 adds the *aggregate*
view on top.

PLAN's F3-revisit conditions for the daemon side:
  * queue-depth > 64 sustained → revisit
  * time-in-queue p99 > 100 ms → revisit
both keyed on this aggregator's summary line.
"""
from __future__ import annotations

from typing import Dict, Optional


class _DaemonMetricsRing:
    """Single-valued bounded ring + on-demand quantile summary.

    Same shape as T14's ``_StateDumpMetrics`` but holds only a single
    float per sample (no ``dump_bytes`` companion).  Single-threaded
    by construction (daemon's event_worker is the only writer); no
    lock.

    The ``summary()`` quantile output uses the nearest-rank method
    (``round(p * (n - 1))``); operationally close enough to linear
    interpolation at n ≥ 32 and avoids the boundary-case bookkeeping.
    """

    __slots__ = ("_capacity", "_samples", "_total_count")

    def __init__(self, capacity: int = 1024) -> None:
        self._capacity = int(capacity)
        self._samples: list[float] = []
        self._total_count = 0

    def record(self, value: float) -> None:
        self._samples.append(float(value))
        if len(self._samples) > self._capacity:
            del self._samples[0]
        self._total_count += 1

    def summary(self) -> Dict[str, float]:
        n = len(self._samples)
        if n == 0:
            return {
                "n": 0,
                "n_recorded_total": 0,
                "capacity": self._capacity,
                "p50": 0.0, "p95": 0.0, "p99": 0.0,
                "max": 0.0, "mean": 0.0,
            }
        sorted_samples = sorted(self._samples)

        def _q(p: float) -> float:
            if n == 1:
                return sorted_samples[0]
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return sorted_samples[idx]

        return {
            "n": n,
            "n_recorded_total": self._total_count,
            "capacity": self._capacity,
            "p50": _q(0.50),
            "p95": _q(0.95),
            "p99": _q(0.99),
            "max": sorted_samples[-1],
            "mean": sum(sorted_samples) / n,
        }


class DaemonObservability:
    """Owns the four T42 metric streams.  One instance lives on the
    ``EventRouter`` and is exposed to handlers via ``router.observability``.

    Sized for ~5 minutes of recent history at the daemon's typical
    3.4 Hz event rate (capacity=1024 ≈ 5 min).  Deep enough for a
    stable p99, shallow enough that a stale outlier ages out within
    the window so the F3-revisit trigger doesn't stick on one past
    spike.
    """

    __slots__ = (
        "state_fetch_lat_ms",
        "queue_depth",
        "time_in_queue_ms",
        "failure_class_counts",
        "events_handled_total",
        "_summary_every_n",
        "_events_since_summary",
    )

    def __init__(
        self,
        *,
        capacity: int = 1024,
        summary_every_n: int = 200,
    ) -> None:
        self.state_fetch_lat_ms = _DaemonMetricsRing(capacity)
        self.queue_depth = _DaemonMetricsRing(capacity)
        self.time_in_queue_ms = _DaemonMetricsRing(capacity)
        self.failure_class_counts: Dict[str, int] = {}
        self.events_handled_total: int = 0
        self._summary_every_n = int(summary_every_n)
        self._events_since_summary: int = 0

    # ---- recorders ---------------------------------------------------

    def record_state_fetch(self, elapsed_ms: float) -> None:
        self.state_fetch_lat_ms.record(elapsed_ms)

    def record_dispatch(self, qdepth: int, time_in_queue_ms: float) -> None:
        """Worker calls this AT handler entry (after queue.get()).
        Bumps ``events_handled_total`` and emits a summary line every
        ``summary_every_n`` events."""
        self.queue_depth.record(float(qdepth))
        self.time_in_queue_ms.record(time_in_queue_ms)
        self.events_handled_total += 1
        self._events_since_summary += 1
        if self._events_since_summary >= self._summary_every_n:
            self.emit_summary()
            self._events_since_summary = 0

    def record_failure(self, reason: str) -> None:
        """Cumulative per-reason counter.  Reason should already be a
        short ``snake_case`` slug — the value is logged verbatim in
        the summary line's ``failure_class_counts`` JSON, so spaces
        and equals-signs in reasons would break the metric format."""
        self.failure_class_counts[reason] = (
            self.failure_class_counts.get(reason, 0) + 1
        )

    # ---- output ------------------------------------------------------

    def summary_dict(self) -> Dict[str, object]:
        return {
            "state_fetch_lat_ms": self.state_fetch_lat_ms.summary(),
            "queue_depth":        self.queue_depth.summary(),
            "time_in_queue_ms":   self.time_in_queue_ms.summary(),
            "failure_class_counts": dict(self.failure_class_counts),
            "events_handled_total": self.events_handled_total,
        }

    def emit_summary(self) -> None:
        """Fire one ``daemon_obs_summary`` line via the existing
        structured-metric format.  Quantile fields are flattened into
        the line because the metric-line format (``key=value`` space-
        separated) can't carry nested dicts inline; the operator's
        grep pipeline keys off these flat names.
        """
        from ._metrics import m as _m
        sf = self.state_fetch_lat_ms.summary()
        qd = self.queue_depth.summary()
        tiq = self.time_in_queue_ms.summary()
        # failure_class_counts is variable-cardinality; the line carries
        # the count but not the breakdown (operator queries
        # summary_dict() directly for the full mapping).
        _m(
            "daemon_obs_summary",
            events_handled_total=self.events_handled_total,
            state_fetch_n=sf["n"],
            state_fetch_p50_ms=sf["p50"],
            state_fetch_p95_ms=sf["p95"],
            state_fetch_p99_ms=sf["p99"],
            state_fetch_max_ms=sf["max"],
            queue_depth_n=qd["n"],
            queue_depth_p50=qd["p50"],
            queue_depth_p95=qd["p95"],
            queue_depth_p99=qd["p99"],
            queue_depth_max=qd["max"],
            time_in_queue_n=tiq["n"],
            time_in_queue_p50_ms=tiq["p50"],
            time_in_queue_p95_ms=tiq["p95"],
            time_in_queue_p99_ms=tiq["p99"],
            time_in_queue_max_ms=tiq["max"],
            n_failure_classes=len(self.failure_class_counts),
            n_failures_total=sum(self.failure_class_counts.values()),
        )
