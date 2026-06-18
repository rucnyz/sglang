"""T15 cross-rank hint-divergence detector (PLAN §2, DESIGN §6).

The eventual-consistent hint table (DESIGN §6) is justified by the
"eviction is the cross-rank sync point" argument: each rank's
scorer independently decides what to evict, and those decisions
must AGREE — otherwise stale hints could let two ranks evict
different units near-simultaneously and the hint-table can't
recover.

This detector takes a *time series* of per-rank state dumps and
flags any window where two ranks evicted DIFFERENT unit-hashes.
The detector is pure analysis — no sglang launch needed; it
operates on captured `/aginfer/state` JSON snapshots.

Method (window-based diff)
--------------------------
For consecutive state dumps S(t), S(t+1):
  * For each rank r, evicted_r = hashes_in(S(t), r) − hashes_in(S(t+1), r)
  * cross_rank_set_diff = symmetric_difference over all evicted_r
  * Divergence iff cross_rank_set_diff is non-empty AND |ranks| > 1

Output: list of ``DivergenceReport`` records — one per offending
window, with both per-rank evicted sets so an operator can audit
the actual hashes.

This module does NOT depend on the daemon — it's a standalone
analysis tool a CI job or a debug session can call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DivergenceReport:
    """One detected divergence window."""
    window_idx: int                     # which (t, t+1) pair (0-based)
    time_counter_prev: int              # state.time_counter at S(t)
    time_counter_curr: int              # state.time_counter at S(t+1)
    per_rank_evicted: Dict[int, FrozenSet[str]]
    # ``per_rank_evicted[r]`` = set of hashes rank r dropped this window.
    # Divergence ⇔ at least two ranks have NON-IDENTICAL eviction sets.

    def is_divergent(self) -> bool:
        sets = list(self.per_rank_evicted.values())
        if len(sets) < 2:
            return False
        first = sets[0]
        return any(s != first for s in sets[1:])


# ---------------------------------------------------------- parser


def _ranks_of(state_json: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Map ``rank_idx → units[]`` for both multi-rank and single-rank
    state dumps.

    Multi-rank: ``state["per_rank"] = [rank0, rank1, ...]`` →
        ``{0: rank0["units"], 1: rank1["units"], ...}``
    Single-rank: ``state["units"]`` present at top level →
        ``{0: state["units"]}``"""
    if "per_rank" in state_json:
        per = state_json["per_rank"]
        if not isinstance(per, list):
            raise ValueError(
                f"per_rank must be a list; got {type(per).__name__}"
            )
        return {i: r["units"] for i, r in enumerate(per)}
    return {0: state_json.get("units", [])}


def _hash_set(units: Sequence[Dict[str, Any]]) -> FrozenSet[str]:
    """Project a unit list to the set of unit hashes."""
    return frozenset(str(u["hash"]) for u in units if "hash" in u)


def _time_counter(state_json: Dict[str, Any]) -> int:
    """Time-counter of a state dump; per_rank dumps take MAX across
    ranks (matches ``daemon/kv_scheduler.py:_flatten_per_rank``)."""
    if "per_rank" in state_json:
        return max(
            int(r.get("time_counter", 0))
            for r in state_json["per_rank"]
        )
    return int(state_json.get("time_counter", 0))


# ---------------------------------------------------------- detector


def detect_divergence(
    state_dumps: Sequence[Dict[str, Any]],
) -> List[DivergenceReport]:
    """Scan a time-ordered list of state dumps; return one report per
    window where ranks evicted DIFFERENT hashes.

    Single-rank dumps yield no reports (no cross-rank comparison
    possible).  Identical-eviction windows yield no reports.

    The window-diff is strict ``hash ∈ prev − hash ∈ curr``; a
    hash that re-appears later (e.g. via re-prefill) does NOT
    invalidate this window's report — that's a separate hash-
    churn dynamic, not a divergence.
    """
    if len(state_dumps) < 2:
        return []
    reports: List[DivergenceReport] = []
    prev = state_dumps[0]
    prev_ranks = _ranks_of(prev)
    for window_idx, curr in enumerate(state_dumps[1:]):
        curr_ranks = _ranks_of(curr)
        # Rank set should be stable; we assert on disagreement.
        if set(curr_ranks.keys()) != set(prev_ranks.keys()):
            raise ValueError(
                f"rank set changed across window {window_idx}: "
                f"prev={sorted(prev_ranks.keys())} → "
                f"curr={sorted(curr_ranks.keys())}"
            )
        per_rank_evicted: Dict[int, FrozenSet[str]] = {}
        for r, prev_units in prev_ranks.items():
            curr_units = curr_ranks[r]
            evicted = _hash_set(prev_units) - _hash_set(curr_units)
            per_rank_evicted[r] = evicted
        report = DivergenceReport(
            window_idx=window_idx,
            time_counter_prev=_time_counter(prev),
            time_counter_curr=_time_counter(curr),
            per_rank_evicted=per_rank_evicted,
        )
        if report.is_divergent():
            reports.append(report)
        prev = curr
        prev_ranks = curr_ranks
    return reports


# ---------------------------------------------------------- summary


def summarise(reports: Sequence[DivergenceReport]) -> str:
    """Human-readable summary for a debug session / CI log line."""
    if not reports:
        return "T15: no cross-rank eviction divergence observed"
    lines = [
        f"T15: {len(reports)} divergence window(s) detected:"
    ]
    for r in reports:
        lines.append(
            f"  window {r.window_idx} (t={r.time_counter_prev}→"
            f"{r.time_counter_curr}):"
        )
        for rank, evicted in sorted(r.per_rank_evicted.items()):
            preview = ",".join(sorted(evicted)[:5])
            if len(evicted) > 5:
                preview += f",…(+{len(evicted) - 5})"
            lines.append(
                f"    rank {rank} evicted {len(evicted)} hashes: "
                f"{{{preview}}}"
            )
    return "\n".join(lines)


__all__ = [
    "DivergenceReport",
    "detect_divergence",
    "summarise",
]
