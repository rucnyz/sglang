"""T34 (#156, DESIGN §9) — multi-axis sparse 0/1 knapsack DP.

Verifies the two `joint_decide` primitives against an EXHAUSTIVE
brute-force oracle (enumerate all 2^K subsets, take the optimum) — the
gold standard for an exact DP.  At bucket_size=1 the DP works in raw
bytes, so the DP optimum must equal the brute-force optimum exactly.
Separate stages pin the per-axis bucket quantisation (relief/budget
round DOWN, destination consumption rounds UP) and the infeasibility
contract.

Stages:
  A. knapsack_min_cost_multi (pressure phase)
    A0 single relief axis: min-cost subset hits bytes_needed
    A1 multi-axis: 2 HBM relief axes + 1 DRAM cap axis, both targets hit
       without overflowing the destination cap
    A2 EXACTNESS vs brute force (random fixtures, bucket_size=1): the
       DP's chosen-subset cost == the brute-force min feasible cost
    A3 destination cap is a hard constraint: a cheap Migrate that would
       overflow DRAM is rejected in favour of a Pause (no acquired)
    A4 the returned subset is itself feasible + its cost == the DP min
  B. quantisation
    B0 relief rounds DOWN (sub-bucket relief doesn't count toward the
       target); destination consumption rounds UP (sub-bucket acquire
       costs a whole bucket)
  C. infeasibility
    C0 no subset can hit bytes_needed → KnapsackInfeasibleError w/ ctx
    C1 DROP / Pause are always feasible: even with destinations FULL, a
       Pause (relief, no acquired) satisfies the target
  D. knapsack_max_value_multi (headroom phase)
    D0 max-gain subset within a single budget axis
    D1 EXACTNESS vs brute force (random, bucket_size=1)
    D2 budget is a hard constraint; re_use rounds UP
    D3 the returned subset is feasible + its gain == the DP max
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.knapsack import (  # noqa: E402
    KnapsackInfeasibleError,
    Migrate,
    Pause,
    Resume,
    _acquire_at,
    _relief_at,
    knapsack_max_value_multi,
    knapsack_min_cost_multi,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ----------------------------------------------------------- brute-force oracle


def _brute_min_cost(items, bytes_needed, cap_left):
    """Exhaustive optimum (raw bytes — match the DP at bucket_size=1).
    Returns (best_cost, n_feasible) or (None, 0) if infeasible."""
    best = None
    n_feasible = 0
    for r in range(len(items) + 1):
        for subset in itertools.combinations(items, r):
            if all(sum(_relief_at(c, *a) for c in subset) >= bytes_needed[a]
                   for a in bytes_needed) and \
               all(sum(_acquire_at(c, *a) for c in subset) <= cap_left[a]
                   for a in cap_left):
                n_feasible += 1
                cost = sum(c.cost for c in subset)
                if best is None or cost < best:
                    best = cost
    return best, n_feasible


def _brute_max_value(items, budget):
    best = 0.0  # empty subset is always feasible, gain 0
    for r in range(len(items) + 1):
        for subset in itertools.combinations(items, r):
            if all(sum(c.re_use.get(a[0], {}).get(a[1], 0) for c in subset)
                   <= budget[a] for a in budget):
                gain = sum(c.gain for c in subset)
                if gain > best:
                    best = gain
    return best


def _feasible_min(subset, bytes_needed, cap_left) -> bool:
    return (all(sum(_relief_at(c, *a) for c in subset) >= bytes_needed[a]
                for a in bytes_needed)
            and all(sum(_acquire_at(c, *a) for c in subset) <= cap_left[a]
                    for a in cap_left))


def _unit_buckets(*axes) -> Dict[Any, int]:
    return {a: 1 for a in axes}


# ============================================================ A. min-cost


def stage_a0_single_axis() -> None:
    need = {("HBM", "kv"): 100}
    cap = {}
    items = [
        Migrate(cost=5.0, relief={"HBM": {"kv": 60}}, id="m60"),
        Migrate(cost=3.0, relief={"HBM": {"kv": 40}}, id="m40"),
        Migrate(cost=9.0, relief={"HBM": {"kv": 100}}, id="m100"),
    ]
    bs = _unit_buckets(("HBM", "kv"))
    chosen = knapsack_min_cost_multi(items, need, cap, bs)
    # cheapest way to free >=100 is m60+m40 (cost 8) vs m100 (cost 9).
    if abs(sum(c.cost for c in chosen) - 8.0) > 1e-9:
        raise StageFail(f"expected min cost 8.0 (m60+m40); got {sum(c.cost for c in chosen)} ({[c.id for c in chosen]})")


def stage_a1_multi_axis() -> None:
    """2 HBM relief axes (full, mamba) + 1 DRAM destination cap."""
    need = {("HBM", "full"): 50, ("HBM", "mamba"): 30}
    cap = {("DRAM", "full"): 1000, ("DRAM", "mamba"): 1000}
    items = [
        Migrate(cost=2.0, relief={"HBM": {"full": 50}}, acquired={"DRAM": {"full": 50}}, id="full"),
        Migrate(cost=2.0, relief={"HBM": {"mamba": 30}}, acquired={"DRAM": {"mamba": 30}}, id="mamba"),
        Migrate(cost=10.0, relief={"HBM": {"full": 50, "mamba": 30}}, acquired={"DRAM": {"full": 50, "mamba": 30}}, id="both"),
    ]
    bs = _unit_buckets(("HBM", "full"), ("HBM", "mamba"), ("DRAM", "full"), ("DRAM", "mamba"))
    chosen = knapsack_min_cost_multi(items, need, cap, bs)
    # full+mamba (cost 4) beats the single "both" (cost 10).
    if {c.id for c in chosen} != {"full", "mamba"}:
        raise StageFail(f"expected {{full, mamba}}; got {[c.id for c in chosen]}")


def stage_a2_exactness_vs_brute() -> None:
    rng = random.Random(34_156)
    bs = _unit_buckets(("HBM", "kv"), ("DRAM", "kv"))
    for trial in range(60):
        K = rng.randint(3, 9)
        items: List[Any] = []
        for i in range(K):
            relief = rng.randint(10, 80)
            # mix Migrate (with destination acquire) and Pause (no acquire)
            if rng.random() < 0.5:
                items.append(Migrate(cost=round(rng.uniform(1, 10), 2),
                                     relief={"HBM": {"kv": relief}},
                                     acquired={"DRAM": {"kv": rng.randint(0, 80)}},
                                     id=f"m{i}"))
            else:
                items.append(Pause(cost=round(rng.uniform(1, 10), 2),
                                   relief={"HBM": {"kv": relief}}, pid=f"p{i}"))
        total_relief = sum(_relief_at(c, "HBM", "kv") for c in items)
        need = {("HBM", "kv"): rng.randint(0, total_relief)}
        cap = {("DRAM", "kv"): rng.randint(0, 400)}
        brute, n_feas = _brute_min_cost(items, need, cap)
        if brute is None:
            continue  # infeasible fixtures handled in C0
        chosen = knapsack_min_cost_multi(items, need, cap, bs)
        dp_cost = sum(c.cost for c in chosen)
        if abs(dp_cost - brute) > 1e-9:
            raise StageFail(
                f"trial {trial}: DP cost {dp_cost} != brute-force min {brute} "
                f"(K={K}, need={need}, cap={cap}, feasible={n_feas})"
            )
        if not _feasible_min(chosen, need, cap):
            raise StageFail(f"trial {trial}: DP returned an INFEASIBLE subset")


def stage_a3_dest_cap_hard_constraint() -> None:
    """A cheap Migrate would overflow DRAM; the DP must pick the (more
    expensive) Pause instead, which consumes no destination capacity."""
    need = {("HBM", "kv"): 100}
    cap = {("DRAM", "kv"): 50}   # only 50 bytes of DRAM room
    items = [
        Migrate(cost=1.0, relief={"HBM": {"kv": 100}}, acquired={"DRAM": {"kv": 100}}, id="cheap-but-overflows"),
        Pause(cost=5.0, relief={"HBM": {"kv": 100}}, pid="pause"),
    ]
    bs = _unit_buckets(("HBM", "kv"), ("DRAM", "kv"))
    chosen = knapsack_min_cost_multi(items, need, cap, bs)
    ids = {getattr(c, "id", None) or getattr(c, "pid", None) for c in chosen}
    if ids != {"pause"}:
        raise StageFail(
            f"cheap Migrate overflows DRAM cap (100 > 50) and must be "
            f"rejected for the Pause; got {ids}"
        )


def stage_a4_returned_subset_optimal_and_feasible() -> None:
    need = {("HBM", "kv"): 70}
    cap = {("DRAM", "kv"): 90}
    items = [
        Migrate(cost=4.0, relief={"HBM": {"kv": 40}}, acquired={"DRAM": {"kv": 40}}, id="a"),
        Migrate(cost=4.0, relief={"HBM": {"kv": 40}}, acquired={"DRAM": {"kv": 40}}, id="b"),
        Migrate(cost=7.5, relief={"HBM": {"kv": 70}}, acquired={"DRAM": {"kv": 70}}, id="c"),
    ]
    bs = _unit_buckets(("HBM", "kv"), ("DRAM", "kv"))
    chosen = knapsack_min_cost_multi(items, need, cap, bs)
    brute, _ = _brute_min_cost(items, need, cap)
    if not _feasible_min(chosen, need, cap):
        raise StageFail("returned subset not feasible")
    if abs(sum(c.cost for c in chosen) - brute) > 1e-9:
        raise StageFail(f"returned cost {sum(c.cost for c in chosen)} != optimum {brute}")


# ============================================================ B. quantisation


def stage_b0_bucket_rounding() -> None:
    # bucket = 64.  relief rounds DOWN, destination acquire rounds UP.
    bs = {("HBM", "kv"): 64, ("DRAM", "kv"): 64}
    # relief 63 → 0 buckets (rounds down): a single such Migrate cannot
    # satisfy a 64-byte (1-bucket) need.
    need = {("HBM", "kv"): 64}
    cap = {("DRAM", "kv"): 64}
    one_small = [Migrate(cost=1.0, relief={"HBM": {"kv": 63}}, acquired={"DRAM": {"kv": 1}}, id="x")]
    try:
        knapsack_min_cost_multi(one_small, need, cap, bs)
    except KnapsackInfeasibleError:
        pass
    else:
        raise StageFail("relief 63 must round DOWN to 0 buckets → infeasible for a 64-byte need")
    # two of them: relief 63+63=126 bytes → but each rounds down to 0
    # buckets BEFORE summing? No — the DP sums bucketised deltas, so
    # 0+0 = 0 buckets → still infeasible.  Confirms per-item round-down.
    two_small = one_small + [Migrate(cost=1.0, relief={"HBM": {"kv": 63}}, acquired={"DRAM": {"kv": 1}}, id="y")]
    try:
        knapsack_min_cost_multi(two_small, need, cap, bs)
    except KnapsackInfeasibleError:
        pass
    else:
        raise StageFail("per-item round-down: 63+63 each → 0 buckets → still infeasible")
    # destination acquire 1 byte → rounds UP to 1 bucket (64); a Migrate
    # freeing 64 (1 bucket) but acquiring 1 byte needs 1 DRAM bucket.
    need2 = {("HBM", "kv"): 64}
    items = [Migrate(cost=1.0, relief={"HBM": {"kv": 64}}, acquired={"DRAM": {"kv": 1}}, id="z")]
    chosen = knapsack_min_cost_multi(items, need2, {("DRAM", "kv"): 64}, bs)   # 1 bucket room → fits
    if {c.id for c in chosen} != {"z"}:
        raise StageFail("acquire 1B → 1 bucket should fit a 1-bucket (64B) DRAM cap")
    try:
        knapsack_min_cost_multi(items, need2, {("DRAM", "kv"): 0}, bs)   # 0 buckets room
    except KnapsackInfeasibleError:
        pass
    else:
        raise StageFail("acquire 1B rounds UP to 1 bucket → must NOT fit 0-byte DRAM cap")


# ============================================================ C. infeasibility


def stage_c0_infeasible_raises() -> None:
    need = {("HBM", "kv"): 1000}
    cap = {("DRAM", "kv"): 1000}
    items = [Migrate(cost=1.0, relief={"HBM": {"kv": 10}}, acquired={"DRAM": {"kv": 10}}, id="tiny")]
    bs = _unit_buckets(("HBM", "kv"), ("DRAM", "kv"))
    try:
        knapsack_min_cost_multi(items, need, cap, bs, context={"event": "TEST"})
    except KnapsackInfeasibleError as e:
        # forensic context present
        for key in ("bytes_needed", "cap_left", "n_items", "dp_size"):
            if key not in e.context:
                raise StageFail(f"infeasible context missing {key!r}: {e.context}")
        if e.context.get("event") != "TEST":
            raise StageFail("caller context not threaded into the forensic dump")
        return
    raise StageFail("total relief (10) < need (1000) must raise KnapsackInfeasibleError")


def stage_c1_pause_always_feasible() -> None:
    """Destinations are FULL (0 cap) — but a Pause frees HBM without
    consuming any destination, so a plan always exists."""
    need = {("HBM", "kv"): 100}
    cap = {("DRAM", "kv"): 0, ("DISK", "kv"): 0}
    items = [
        Migrate(cost=1.0, relief={"HBM": {"kv": 100}}, acquired={"DRAM": {"kv": 100}}, id="m"),  # would overflow
        Pause(cost=50.0, relief={"HBM": {"kv": 100}}, pid="p"),                                   # no acquire
    ]
    bs = _unit_buckets(("HBM", "kv"), ("DRAM", "kv"), ("DISK", "kv"))
    chosen = knapsack_min_cost_multi(items, need, cap, bs)
    ids = {getattr(c, "pid", None) or getattr(c, "id", None) for c in chosen}
    if ids != {"p"}:
        raise StageFail(f"with destinations full, only the Pause is feasible; got {ids}")


# ============================================================ D. max-value


def stage_d0_single_axis() -> None:
    budget = {("HBM", "kv"): 100}
    items = [
        Resume(gain=5.0, re_use={"HBM": {"kv": 60}}, pid="a"),
        Resume(gain=3.0, re_use={"HBM": {"kv": 40}}, pid="b"),
        Resume(gain=4.0, re_use={"HBM": {"kv": 50}}, pid="c"),
    ]
    bs = _unit_buckets(("HBM", "kv"))
    chosen = knapsack_max_value_multi(items, budget, bs)
    # within 100 bytes: a(60)+b(40)=gain 8 (uses 100); a+c=110 over; b+c=90 gain 7.
    if abs(sum(c.gain for c in chosen) - 8.0) > 1e-9:
        raise StageFail(f"expected max gain 8.0 (a+b); got {sum(c.gain for c in chosen)} ({[c.pid for c in chosen]})")


def stage_d1_exactness_vs_brute() -> None:
    rng = random.Random(34_157)
    bs = _unit_buckets(("HBM", "full"), ("HBM", "mamba"))
    for trial in range(60):
        K = rng.randint(3, 9)
        items = []
        for i in range(K):
            re_use = {"HBM": {}}
            if rng.random() < 0.7:
                re_use["HBM"]["full"] = rng.randint(0, 60)
            if rng.random() < 0.5:
                re_use["HBM"]["mamba"] = rng.randint(0, 40)
            items.append(Resume(gain=round(rng.uniform(1, 10), 2), re_use=re_use, pid=f"r{i}"))
        budget = {("HBM", "full"): rng.randint(0, 200), ("HBM", "mamba"): rng.randint(0, 150)}
        brute = _brute_max_value(items, budget)
        chosen = knapsack_max_value_multi(items, budget, bs)
        dp_gain = sum(c.gain for c in chosen)
        if abs(dp_gain - brute) > 1e-9:
            raise StageFail(
                f"trial {trial}: DP gain {dp_gain} != brute max {brute} "
                f"(K={K}, budget={budget})"
            )
        # feasible?
        for a in budget:
            used = sum(c.re_use.get(a[0], {}).get(a[1], 0) for c in chosen)
            if used > budget[a]:
                raise StageFail(f"trial {trial}: chosen subset overspends {a}: {used} > {budget[a]}")


def stage_d2_budget_hard_and_roundup() -> None:
    bs = {("HBM", "kv"): 64}
    budget = {("HBM", "kv"): 64}   # 1 bucket
    # re_use 1 byte → rounds UP to 1 bucket; two of them → 2 buckets > 1.
    items = [
        Resume(gain=5.0, re_use={"HBM": {"kv": 1}}, pid="a"),
        Resume(gain=5.0, re_use={"HBM": {"kv": 1}}, pid="b"),
    ]
    chosen = knapsack_max_value_multi(items, budget, bs)
    # each costs 1 bucket; budget is 1 bucket → only ONE fits.
    if len(chosen) != 1:
        raise StageFail(f"re_use rounds UP (1B→1 bucket); only one fits a 1-bucket budget; got {len(chosen)}")


def stage_d3_returned_subset_optimal() -> None:
    budget = {("HBM", "kv"): 100}
    items = [
        Resume(gain=4.0, re_use={"HBM": {"kv": 50}}, pid="a"),
        Resume(gain=4.0, re_use={"HBM": {"kv": 50}}, pid="b"),
        Resume(gain=7.0, re_use={"HBM": {"kv": 100}}, pid="c"),
    ]
    bs = _unit_buckets(("HBM", "kv"))
    chosen = knapsack_max_value_multi(items, budget, bs)
    brute = _brute_max_value(items, budget)
    if abs(sum(c.gain for c in chosen) - brute) > 1e-9:
        raise StageFail(f"returned gain {sum(c.gain for c in chosen)} != optimum {brute}")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 min-cost single relief axis",              stage_a0_single_axis),
    ("A1 min-cost multi-axis (2 HBM relief + DRAM cap)", stage_a1_multi_axis),
    ("A2 min-cost EXACT vs brute force (60 random)", stage_a2_exactness_vs_brute),
    ("A3 destination cap is a hard constraint",     stage_a3_dest_cap_hard_constraint),
    ("A4 returned subset feasible + optimal",       stage_a4_returned_subset_optimal_and_feasible),
    ("B0 relief rounds down, acquire rounds up",    stage_b0_bucket_rounding),
    ("C0 infeasible → KnapsackInfeasibleError + ctx", stage_c0_infeasible_raises),
    ("C1 Pause always feasible (destinations full)", stage_c1_pause_always_feasible),
    ("D0 max-value single budget axis",             stage_d0_single_axis),
    ("D1 max-value EXACT vs brute force (60 random)", stage_d1_exactness_vs_brute),
    ("D2 budget hard constraint; re_use rounds up", stage_d2_budget_hard_and_roundup),
    ("D3 returned subset optimal",                  stage_d3_returned_subset_optimal),
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
            print(f"  {_red('FAIL')}  Stage {label}: "
                  f"unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT34 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT34 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
