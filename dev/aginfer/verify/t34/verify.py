"""T34 (#156, DESIGN §9) — multi-axis sparse 0/1 knapsack DP.

DESIGN §9 is **value-gated**: every phase runs the SAME value-maximising
primitive ``knapsack_max_value_multi`` and may pick the empty set.  Relief
feeds it Migrate value-items (gain = −cost, re_use = destination
``acquired``); resume feeds it Resume candidates (gain = V_u, re_use =
HBM re-entry).  This suite verifies that ONE primitive against an
EXHAUSTIVE brute-force oracle (enumerate all subset choices, take the
optimum) — the gold standard for an exact DP.  At bucket_size=1 the DP
works in raw bytes, so DP optimum == brute optimum exactly; separate
stages pin per-axis bucket quantisation (round UP, safe for ``<=``),
multiple-choice grouping (#194), and the DP-cell ceiling.

Stages:
  D0 single budget axis: max-gain subset within one budget
  D1 EXACTNESS vs brute force (random, bucket_size=1, 2 axes)
  D2 budget is a hard constraint; re_use rounds UP
  D3 the returned subset is feasible + its gain == the DP max
  G0 multiple-choice groups (#194): at most one member per group; EXACT
     vs a grouped brute oracle (this is what makes a unit's mutually-
     exclusive transitions safe to feed as relief value-items)
  E0 EXACT vs a quantising brute oracle at bucket_size>1, multi-axis
  E1 DP cell ceiling fails loud (KnapsackBudgetExceededError + ctx)
  E2 empty items / zero budget → empty plan (value-gated no-op)
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
    KnapsackBudgetExceededError,
    Resume,
    _bk,
    _bk_up,
    knapsack_max_value_multi,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ----------------------------------------------------------- brute-force oracle


def _use_at(c, a):
    return c.re_use.get(a[0], {}).get(a[1], 0)


def _brute_max_value(items, budget):
    """Exhaustive optimum at bucket_size=1 (raw bytes).  Empty subset is
    always feasible with gain 0, so a negative-only fixture → 0."""
    best = 0.0
    for r in range(len(items) + 1):
        for subset in itertools.combinations(items, r):
            if all(sum(_use_at(c, a) for c in subset) <= budget[a]
                   for a in budget):
                gain = sum(c.gain for c in subset)
                if gain > best:
                    best = gain
    return best


def _brute_max_value_q(items, budget, bucket_size):
    """Brute optimum at bucket_size>1 — quantises IDENTICALLY to the DP
    (round each item's re_use UP, sum, compare to the round-DOWN budget)."""
    Wb = {a: _bk(budget[a], bucket_size[a]) for a in budget}
    best = 0.0
    for r in range(len(items) + 1):
        for subset in itertools.combinations(items, r):
            use = {a: sum(_bk_up(_use_at(c, a), bucket_size[a]) for c in subset)
                   for a in budget}
            if all(use[a] <= Wb[a] for a in budget):
                gain = sum(c.gain for c in subset)
                if gain > best:
                    best = gain
    return best


def _brute_max_value_grouped(groups, budget):
    """Exhaustive optimum with at-most-one-per-group (multiple-choice)."""
    best = 0.0
    opts = [[None] + list(g) for g in groups]
    for combo in itertools.product(*opts):
        subset = [m for m in combo if m is not None]
        if all(sum(_use_at(c, a) for c in subset) <= budget[a]
               for a in budget):
            gain = sum(c.gain for c in subset)
            if gain > best:
                best = gain
    return best


def _unit_buckets(*axes) -> Dict[Any, int]:
    return {a: 1 for a in axes}


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
        raise StageFail(f"expected max gain 8.0 (a+b); got "
                        f"{sum(c.gain for c in chosen)} ({[c.pid for c in chosen]})")


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
            # gains may be negative — the empty set must dominate a
            # net-negative pick (value-gated no-op).
            items.append(Resume(gain=round(rng.uniform(-3, 10), 2),
                                 re_use=re_use, pid=f"r{i}"))
        budget = {("HBM", "full"): rng.randint(0, 200),
                  ("HBM", "mamba"): rng.randint(0, 150)}
        brute = _brute_max_value(items, budget)
        chosen = knapsack_max_value_multi(items, budget, bs)
        dp_gain = sum(c.gain for c in chosen)
        if abs(dp_gain - brute) > 1e-9:
            raise StageFail(
                f"trial {trial}: DP gain {dp_gain} != brute max {brute} "
                f"(K={K}, budget={budget})")
        for a in budget:
            used = sum(_use_at(c, a) for c in chosen)
            if used > budget[a]:
                raise StageFail(f"trial {trial}: chosen overspends {a}: "
                                f"{used} > {budget[a]}")


def stage_d2_budget_hard_and_roundup() -> None:
    bs = {("HBM", "kv"): 64}
    budget = {("HBM", "kv"): 64}   # 1 bucket
    # re_use 1 byte → rounds UP to 1 bucket; two of them → 2 buckets > 1.
    items = [
        Resume(gain=5.0, re_use={"HBM": {"kv": 1}}, pid="a"),
        Resume(gain=5.0, re_use={"HBM": {"kv": 1}}, pid="b"),
    ]
    chosen = knapsack_max_value_multi(items, budget, bs)
    if len(chosen) != 1:
        raise StageFail(f"re_use rounds UP (1B→1 bucket); only one fits a "
                        f"1-bucket budget; got {len(chosen)}")


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
        raise StageFail(f"returned gain {sum(c.gain for c in chosen)} "
                        f"!= optimum {brute}")


# ============================================================ G. grouping


def stage_g0_multiple_choice_groups() -> None:
    """#194 multiple-choice: candidates sharing a non-None ``group`` are
    at-most-one (a unit's evict/spill/DROP transitions are alternatives —
    taking two double-counts the unit's bytes).  EXACT vs a grouped brute
    oracle, including the same-budget case where a plain 0/1 knapsack would
    wrongly stack two members of one group."""
    # direct: two members of group "u" each fit; the DP takes the higher-
    # value ONE, never both (even though both would fit the budget).
    a = Resume(gain=10.0, re_use={"HBM": {"kv": 30}}, pid="u-a", group="u")
    b = Resume(gain=4.0, re_use={"HBM": {"kv": 30}}, pid="u-b", group="u")
    bs = _unit_buckets(("HBM", "kv"))
    chosen = knapsack_max_value_multi([a, b], {("HBM", "kv"): 1000}, bs)
    if [c.pid for c in chosen] != ["u-a"]:
        raise StageFail(f"group 'u': at most one member, the higher-value; "
                        f"got {[c.pid for c in chosen]}")

    # exact vs grouped brute over random grouped fixtures, 2 axes.
    rng = random.Random(34_222)
    bs2 = _unit_buckets(("HBM", "full"), ("DRAM", "kv"))
    for trial in range(60):
        n_groups = rng.randint(2, 5)
        groups = []
        for gi in range(n_groups):
            members = []
            for mi in range(rng.randint(1, 3)):
                re_use = {"HBM": {"full": rng.randint(0, 40)},
                          "DRAM": {"kv": rng.randint(0, 40)}}
                members.append(Resume(gain=round(rng.uniform(-2, 9), 2),
                                       re_use=re_use, pid=(gi, mi),
                                       group=f"g{gi}"))
            groups.append(members)
        items = [m for g in groups for m in g]
        budget = {("HBM", "full"): rng.randint(0, 120),
                  ("DRAM", "kv"): rng.randint(0, 120)}
        brute = _brute_max_value_grouped(groups, budget)
        chosen = knapsack_max_value_multi(items, budget, bs2)
        # at most one per group
        seen = [c.group for c in chosen]
        if len(seen) != len(set(seen)):
            raise StageFail(f"trial {trial}: 2+ members of one group chosen: "
                            f"{[c.pid for c in chosen]}")
        if abs(sum(c.gain for c in chosen) - brute) > 1e-9:
            raise StageFail(f"trial {trial}: grouped DP gain "
                            f"{sum(c.gain for c in chosen)} != brute {brute}")


# ============================================================ E. audit closure


def stage_e0_bs_gt1_multi_axis_exactness() -> None:
    """EXACT vs a quantising brute oracle at bucket_size>1, with MULTIPLE
    budget axes simultaneously (the genuinely multi-axis case bucket=1 did
    not cover).  re_use rounds UP per item, budget rounds DOWN."""
    rng = random.Random(34_999)
    BS = 64
    bs = {("DRAM", "full"): BS, ("DRAM", "mamba"): BS}
    for trial in range(60):
        K = rng.randint(3, 8)
        items = []
        for i in range(K):
            re_use = {"DRAM": {}}
            if rng.random() < 0.7:
                re_use["DRAM"]["full"] = rng.randint(0, 200)
            if rng.random() < 0.6:
                re_use["DRAM"]["mamba"] = rng.randint(0, 200)
            items.append(Resume(gain=round(rng.uniform(-2, 9), 2),
                                 re_use=re_use, pid=f"r{i}"))
        budget = {("DRAM", "full"): rng.randint(0, 600),
                  ("DRAM", "mamba"): rng.randint(0, 600)}
        brute = _brute_max_value_q(items, budget, bs)
        chosen = knapsack_max_value_multi(items, budget, bs)
        if abs(sum(c.gain for c in chosen) - brute) > 1e-9:
            raise StageFail(
                f"trial {trial} (bs={BS}, multi-axis): DP gain "
                f"{sum(c.gain for c in chosen)} != brute {brute} "
                f"(budget={budget})")


def stage_e1_dp_cell_ceiling() -> None:
    """#156 audit #9: a candidate set with large, distinct bucket-deltas
    blows up the sparse DP (state cross-product).  Past ``max_dp_cells``
    the primitive FAILS LOUD (KnapsackBudgetExceededError + forensic ctx)
    instead of stalling the event loop.  A normal small fixture stays well
    under the ceiling."""
    bs = _unit_buckets(("HBM", "kv"))
    # powers-of-two re_use → 2^K distinct partial sums; large budget so
    # nothing rejects → |dp| grows to ~2^K.
    blow = [Resume(gain=1.0, re_use={"HBM": {"kv": 2 ** i}}, pid=i)
            for i in range(12)]
    try:
        knapsack_max_value_multi(blow, {("HBM", "kv"): 10 ** 9}, bs,
                                 max_dp_cells=50, context={"event": "BLOW"})
    except KnapsackBudgetExceededError as e:
        for key in ("dp_size", "max_dp_cells", "item_index", "n_items",
                    "items", "axes"):
            if key not in e.context:
                raise StageFail(f"blowup ctx missing {key!r}: {list(e.context)}")
        if e.context["dp_size"] <= 50:
            raise StageFail(f"ceiling should trip ABOVE max_dp_cells; "
                            f"dp_size={e.context['dp_size']}")
    else:
        raise StageFail("a 2^12-state fixture must trip max_dp_cells=50")
    # a normal small fixture stays well under the (default) ceiling
    ok = knapsack_max_value_multi(
        [Resume(gain=1.0, re_use={"HBM": {"kv": 100}}, pid="x")],
        {("HBM", "kv"): 1000}, bs)
    if [c.pid for c in ok] != ["x"]:
        raise StageFail("a normal fixture must NOT trip the ceiling")


def stage_e2_empty_and_zero() -> None:
    """#156 audit #12: empty items / zero budget / all-negative gains →
    the empty plan (value-gated no-op)."""
    bs = _unit_buckets(("HBM", "kv"))
    if knapsack_max_value_multi([], {("HBM", "kv"): 100}, bs) != []:
        raise StageFail("empty items → []")
    # zero budget excludes a positive-re_use item
    if knapsack_max_value_multi(
            [Resume(gain=5.0, re_use={"HBM": {"kv": 5}}, pid="a")],
            {("HBM", "kv"): 0}, bs) != []:
        raise StageFail("zero budget must exclude a positive-re_use item")
    # all-negative gains → empty (no item pays for itself)
    neg = knapsack_max_value_multi(
        [Resume(gain=-1.0, re_use={"HBM": {"kv": 1}}, pid="a"),
         Resume(gain=-3.0, re_use={"HBM": {"kv": 1}}, pid="b")],
        {("HBM", "kv"): 100}, bs)
    if neg != []:
        raise StageFail(f"all-negative gains must yield [] (no-op), got "
                        f"{[c.pid for c in neg]}")


def stage_e3_nonpositive_bucket_guard() -> None:
    """#218: a non-positive ``bucket_size`` (page granularity) divides in
    _bk/_bk_up — must raise a CLEAR, contextful ValueError up front, not a
    bare ZeroDivisionError deep in the DP (which the daemon would surface as
    an unhelpful crash instead of fatal('nonpositive_page_bytes'))."""
    item = [Resume(gain=5.0, re_use={"HBM": {"kv": 100}}, pid="a")]
    for bad_bs in (0, -64):
        try:
            knapsack_max_value_multi(item, {("HBM", "kv"): 1000},
                                     {("HBM", "kv"): bad_bs})
        except ValueError as e:
            if "bucket_size" not in str(e):
                raise StageFail(f"guard message must name bucket_size: {e}")
        except ZeroDivisionError:
            raise StageFail(f"bucket_size={bad_bs} raised a bare "
                            "ZeroDivisionError — the #218 positivity guard "
                            "must catch it first with a clear message")
        else:
            raise StageFail(f"bucket_size={bad_bs} must raise ValueError")


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("D0 max-value single budget axis",             stage_d0_single_axis),
    ("D1 max-value EXACT vs brute force (60 random, +/- gains)",
     stage_d1_exactness_vs_brute),
    ("D2 budget hard constraint; re_use rounds up", stage_d2_budget_hard_and_roundup),
    ("D3 returned subset optimal",                  stage_d3_returned_subset_optimal),
    ("G0 multiple-choice groups EXACT vs grouped brute (#194)",
     stage_g0_multiple_choice_groups),
    ("E0 EXACT vs brute at bucket>1, multi-axis (#10/#11)",
     stage_e0_bs_gt1_multi_axis_exactness),
    ("E1 DP cell ceiling fails loud (#9 blow-up guard)",
     stage_e1_dp_cell_ceiling),
    ("E2 empty items / zero budget / negative gains (#12)",
     stage_e2_empty_and_zero),
    ("E3 non-positive bucket_size → clear ValueError, not ZeroDiv (#218)",
     stage_e3_nonpositive_bucket_guard),
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
