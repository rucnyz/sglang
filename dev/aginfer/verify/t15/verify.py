"""T15 verify — hint-table cross-rank divergence detector (PLAN §2).

T15 spec has two halves:

  (1) Build a divergence-detector tool over time-series of per-rank
      `/aginfer/state` dumps.  Pure analysis, no sglang needed.
  (2) Run sglang TP > 1 under high hint churn, capture state dumps,
      feed through the detector; assert no divergence over a real
      workload.

Half (2) is gated on:
  * TP > 1 sglang launch (real GPU time, infrastructure work)
  * A workload that demonstrates "high hint churn" — needs a
    benchmark that triggers repeated scorer turn-over

Defer (2) until a benchmark cycle is running for another reason
(e.g. a real T11 calibration run or T44 S1 verify); pipe the
captured states through the detector then.  Until then we ship
half (1) and prove the detector flags the right windows.

Stage list (10):

  A. Single-rank inputs (the detector must NEVER flag divergence)
     A0 single-rank, single-snapshot     → []
     A1 single-rank, multi-snapshot      → []
  B. Multi-rank, no divergence
     B0 multi-rank, no eviction at all   → []
     B1 multi-rank, identical eviction   → []
  C. Multi-rank, divergence present
     C0 rank-0 evicts {u1}, rank-1 evicts {u2}   → 1 report
     C1 rank-0 evicts {u1}, rank-1 evicts {u1,u2}→ 1 report (partial)
     C2 3 ranks, 2 agree + 1 diverges            → 1 report
     C3 sustained divergence across 4 windows    → 4 reports
  D. Wire-format / robustness
     D0 rank-set changes between windows         → ValueError
     D1 time_counter copied through              → asserted
     D2 summarise() produces non-empty string    → quick smoke

Usage:
    python dev/aginfer/verify/t15/verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from detector import (  # noqa: E402
    DivergenceReport,
    detect_divergence,
    summarise,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---- fixture helpers ----


def _unit(uhash: str) -> Dict[str, Any]:
    """Minimal unit shape — detector only reads ``hash``."""
    return {"hash": uhash}


def _rank(units: List[Dict[str, Any]], *, time_counter: int = 0) -> Dict[str, Any]:
    return {"units": list(units), "time_counter": time_counter}


def _multi(ranks: List[List[Dict[str, Any]]], *, time_counter: int = 0) -> Dict[str, Any]:
    """Build a per-rank state-dump JSON: ranks is a list of unit
    lists; each rank gets the same time_counter."""
    return {
        "per_rank": [
            _rank(units, time_counter=time_counter)
            for units in ranks
        ],
    }


def _single(units: List[Dict[str, Any]], *, time_counter: int = 0) -> Dict[str, Any]:
    """Single-rank state dump."""
    return {"units": list(units), "time_counter": time_counter}


# ============================================================ A. single-rank


def stage_a0_single_rank_single_snapshot() -> None:
    reports = detect_divergence([_single([_unit("u0")])])
    if reports:
        raise StageFail(f"single snapshot must yield no reports: {reports}")


def stage_a1_single_rank_multi_snapshot() -> None:
    """Single-rank state dumps, multiple windows.  No cross-rank
    comparison possible → no divergence ever."""
    seq = [
        _single([_unit("u0"), _unit("u1"), _unit("u2")]),
        _single([_unit("u0"), _unit("u2")]),                        # u1 evicted
        _single([_unit("u0")]),                                     # u2 also evicted
    ]
    reports = detect_divergence(seq)
    if reports:
        raise StageFail(f"single-rank must never diverge: {reports}")


# ============================================================ B. multi-rank, no divergence


def stage_b0_multi_rank_no_eviction() -> None:
    """2 ranks, no eviction in either rank → no report."""
    base = [[_unit("u0"), _unit("u1")], [_unit("u0"), _unit("u1")]]
    seq = [_multi(base), _multi(base)]
    reports = detect_divergence(seq)
    if reports:
        raise StageFail(f"no eviction → no reports: {reports}")


def stage_b1_multi_rank_identical_eviction() -> None:
    """2 ranks, both evict u1 in the same window → no divergence."""
    seq = [
        _multi([[_unit("u0"), _unit("u1")],
                [_unit("u0"), _unit("u1")]]),
        _multi([[_unit("u0")],
                [_unit("u0")]]),  # both evict u1
    ]
    reports = detect_divergence(seq)
    if reports:
        raise StageFail(
            f"identical eviction across ranks → no report: {reports}"
        )


# ============================================================ C. divergence


def stage_c0_divergence_distinct_evictions() -> None:
    """rank-0 evicts {u1}, rank-1 evicts {u2} → 1 report."""
    seq = [
        _multi([[_unit("u0"), _unit("u1"), _unit("u2")],
                [_unit("u0"), _unit("u1"), _unit("u2")]],
               time_counter=10),
        _multi([[_unit("u0"), _unit("u2")],            # rank-0 lost u1
                [_unit("u0"), _unit("u1")]],            # rank-1 lost u2
               time_counter=20),
    ]
    reports = detect_divergence(seq)
    if len(reports) != 1:
        raise StageFail(f"expected 1 report; got {len(reports)}: {reports}")
    r = reports[0]
    if r.per_rank_evicted[0] != frozenset({"u1"}):
        raise StageFail(f"rank-0 evicted: {r.per_rank_evicted[0]}")
    if r.per_rank_evicted[1] != frozenset({"u2"}):
        raise StageFail(f"rank-1 evicted: {r.per_rank_evicted[1]}")


def stage_c1_partial_divergence_overlap() -> None:
    """rank-0 evicts {u1}, rank-1 evicts {u1, u2} → divergence
    (rank-1 evicted an extra u2 that rank-0 still holds)."""
    seq = [
        _multi([[_unit("u0"), _unit("u1"), _unit("u2")],
                [_unit("u0"), _unit("u1"), _unit("u2")]]),
        _multi([[_unit("u0"), _unit("u2")],            # rank-0 lost u1
                [_unit("u0")]]),                       # rank-1 lost u1 + u2
    ]
    reports = detect_divergence(seq)
    if len(reports) != 1:
        raise StageFail(f"expected 1 report; got {len(reports)}: {reports}")
    if reports[0].per_rank_evicted[0] != frozenset({"u1"}):
        raise StageFail(f"rank-0: {reports[0].per_rank_evicted[0]}")
    if reports[0].per_rank_evicted[1] != frozenset({"u1", "u2"}):
        raise StageFail(f"rank-1: {reports[0].per_rank_evicted[1]}")


def stage_c2_three_ranks_2v1() -> None:
    """3 ranks: rank-0 + rank-1 agree, rank-2 diverges → 1 report."""
    seq = [
        _multi([[_unit("u0"), _unit("u1")]] * 3),
        _multi([
            [_unit("u0")],         # rank-0 evicted u1
            [_unit("u0")],         # rank-1 evicted u1
            [_unit("u0"), _unit("u1")],  # rank-2 kept u1
        ]),
    ]
    reports = detect_divergence(seq)
    if len(reports) != 1:
        raise StageFail(f"3-rank 2v1 → 1 report; got {len(reports)}")
    pre = reports[0].per_rank_evicted
    if pre[0] != frozenset({"u1"}) or pre[1] != frozenset({"u1"}):
        raise StageFail(f"ranks 0/1 should both have evicted u1: {pre}")
    if pre[2] != frozenset():
        raise StageFail(f"rank 2 should have evicted nothing: {pre[2]}")


def stage_c3_sustained_divergence_4_windows() -> None:
    """Sustained divergence — 4 consecutive windows each show
    different per-rank eviction → 4 reports."""
    snapshots = []
    # Start state: both ranks have u0..u9.
    base = [[_unit(f"u{i}") for i in range(10)]] * 2
    snapshots.append(_multi(base))
    # Each window: rank-0 evicts u(i*2), rank-1 evicts u(i*2+1).
    for i in range(4):
        next_rank_0 = [_unit(f"u{j}") for j in range(10) if j != i * 2]
        next_rank_1 = [_unit(f"u{j}") for j in range(10) if j != i * 2 + 1]
        # Carry the previous "kept" sets forward across windows.
        snapshots.append(_multi([
            [u for u in snapshots[-1]["per_rank"][0]["units"]
             if u["hash"] in {x["hash"] for x in next_rank_0}],
            [u for u in snapshots[-1]["per_rank"][1]["units"]
             if u["hash"] in {x["hash"] for x in next_rank_1}],
        ]))
    reports = detect_divergence(snapshots)
    if len(reports) != 4:
        raise StageFail(
            f"4 sustained-divergence windows → 4 reports; got {len(reports)}"
        )


# ============================================================ D. robustness


def stage_d0_rank_set_changes_raises() -> None:
    """A scale-up / scale-down between windows (different rank count)
    is a deployment-bug class signal — raise rather than silently
    proceed.  Cross-rank comparison is undefined."""
    seq = [
        _multi([[_unit("u0")], [_unit("u0")]]),   # 2 ranks
        _multi([[_unit("u0")]]),                  # 1 rank
    ]
    try:
        detect_divergence(seq)
    except ValueError:
        return
    raise StageFail(
        "rank-count change must raise ValueError; got no exception"
    )


def stage_d1_time_counter_propagated() -> None:
    """Reports carry both endpoints' time_counter for log correlation."""
    seq = [
        _multi([[_unit("u0"), _unit("u1")],
                [_unit("u0"), _unit("u1")]],
               time_counter=100),
        _multi([[_unit("u0")],                # rank-0 evicted u1
                [_unit("u1")]],                # rank-1 evicted u0
               time_counter=200),
    ]
    reports = detect_divergence(seq)
    if len(reports) != 1:
        raise StageFail(f"expected 1 report; got {len(reports)}")
    r = reports[0]
    if r.time_counter_prev != 100 or r.time_counter_curr != 200:
        raise StageFail(
            f"time_counter not propagated: prev={r.time_counter_prev}, "
            f"curr={r.time_counter_curr}"
        )


def stage_d2_summarise_smoke() -> None:
    """summarise() produces a non-empty string with both ranks'
    hashes listed.  Smoke test only — exact format is operator-
    facing, not contract."""
    seq = [
        _multi([[_unit("u0"), _unit("u1")],
                [_unit("u0"), _unit("u1")]]),
        _multi([[_unit("u0")], [_unit("u1")]]),
    ]
    reports = detect_divergence(seq)
    out = summarise(reports)
    if "divergence" not in out.lower():
        raise StageFail(f"summary missing 'divergence': {out!r}")
    if "u0" not in out or "u1" not in out:
        raise StageFail(f"summary missing hashes: {out!r}")
    empty_out = summarise([])
    if "no cross-rank" not in empty_out.lower():
        raise StageFail(f"no-divergence summary: {empty_out!r}")


# ============================================================ run


_STAGES = [
    ("A0 single-rank, single snapshot → no report",   stage_a0_single_rank_single_snapshot),
    ("A1 single-rank, multi-snapshot → no report",    stage_a1_single_rank_multi_snapshot),
    ("B0 multi-rank, no eviction → no report",        stage_b0_multi_rank_no_eviction),
    ("B1 multi-rank, identical eviction → no report", stage_b1_multi_rank_identical_eviction),
    ("C0 rank-0 evicts u1, rank-1 evicts u2 → divergence", stage_c0_divergence_distinct_evictions),
    ("C1 partial divergence (rank-1 evicts extra)",   stage_c1_partial_divergence_overlap),
    ("C2 3 ranks, 2 agree + 1 diverges",              stage_c2_three_ranks_2v1),
    ("C3 sustained divergence across 4 windows",      stage_c3_sustained_divergence_4_windows),
    ("D0 rank-set changes between windows → ValueError", stage_d0_rank_set_changes_raises),
    ("D1 time_counter propagated to reports",         stage_d1_time_counter_propagated),
    ("D2 summarise() smoke",                          stage_d2_summarise_smoke),
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
        print(_red(f"\nT15 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT15 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
