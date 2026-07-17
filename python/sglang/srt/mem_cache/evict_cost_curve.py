"""Prefix-curve cache for the c^evict predictor.

The exact per-query victim walk (`predict_evict_cost_us`) prices eviction
perfectly but costs O(evictable nodes) on the scheduler thread, and the
Admitter runs it up to twice per arrival — measured ~0.9 ms/arrival at high
cache occupancy, against the design's near-zero arrival-path budget.

This cache trades bounded staleness for O(log n) lookups: one full victim
walk builds a cumulative (tokens_freed, cost_us) curve, queries
binary-search it, and the curve is rebuilt when older than
SGLANG_XPOOL_EVICT_CURVE_MAX_AGE_S (default 0.5 s — half the Budgeter
period, i.e. the same pricing granularity the planner already accepts) or
when eviction mutates the tree (the owner calls `invalidate()`).

Fidelity: victim order is deterministic (the same heap pops in the same
order wherever the walk stops), so the fresh curve matches the exact walk
at every prefix. Cascade-swept tombstones, whose sweep position inside the
walk is not exposed by the planner, are appended at the end of the curve —
a slight under-pricing of mid-range targets, bounded by the tombstone share.
Setting the max-age env to 0 disables the cache entirely (exact walk per
query).
"""

from __future__ import annotations

import time
from bisect import bisect_left
from typing import Callable, Dict, Iterable, Optional, Tuple


class EvictCostCurve:
    """Cumulative (tokens, cost) curve over the eviction-ordered victims."""

    __slots__ = ("built_at", "cum_tokens", "cum_cost_us")

    def __init__(
        self, entries: Iterable[Tuple[int, float]], built_at: Optional[float] = None
    ):
        """`entries` = (tokens_freed, cost_us) per victim, eviction order."""
        self.built_at = time.monotonic() if built_at is None else built_at
        self.cum_tokens = []
        self.cum_cost_us = []
        tok, cost = 0, 0.0
        for t, c in entries:
            tok += int(t)
            cost += float(c)
            self.cum_tokens.append(tok)
            self.cum_cost_us.append(cost)

    def lookup(self, num_tokens: int) -> float:
        """Cost (µs) to free ≥ num_tokens; +inf when supply is short
        (fail-closed, same contract as the exact walk)."""
        if num_tokens <= 0:
            return 0.0
        if not self.cum_tokens or num_tokens > self.cum_tokens[-1]:
            return float("inf")
        return self.cum_cost_us[bisect_left(self.cum_tokens, num_tokens)]


class EvictCostCurveCache:
    """Per-pool curve cache with age-based refresh.

    `max_age_s` is read per call so a runtime env flip takes effect
    immediately; `<= 0` disables the cache (caller falls back to the
    exact walk)."""

    def __init__(self):
        self._curves: Dict[str, EvictCostCurve] = {}

    def get_cost(
        self,
        pool: str,
        num_tokens: int,
        build_fn: Callable[[], EvictCostCurve],
        max_age_s: float,
    ) -> Optional[float]:
        """Curve-priced cost, or None when caching is disabled."""
        if max_age_s <= 0:
            return None
        curve = self._curves.get(pool)
        if curve is None or time.monotonic() - curve.built_at > max_age_s:
            curve = build_fn()
            self._curves[pool] = curve
        return curve.lookup(num_tokens)

    def invalidate(self) -> None:
        """Drop all curves (call after any eviction mutates the tree)."""
        self._curves.clear()
