"""T34 (#156, DESIGN §9 "Joint decide") — multi-axis sparse 0/1
knapsack DP primitive + the candidate contract it consumes.

`joint_decide` (DESIGN §9) is **value-gated**: every phase runs the SAME
value-MAXIMISING knapsack and may pick the empty set (no-op).  There is
no min-cost-cover phase and no forced relief.

  * ``knapsack_max_value_multi``: choose the maximum-total-net-value
    subset of candidates whose re-entering bytes stay within
    ``budget[(tier, sp)]`` per axis (<= axes).  Relief uses it with
    value = −cost and the destination ``(DRAM|DISK, sp)`` caps as the
    budget; Resume uses it with value = V_u gain and the per-HBM-subpool
    free room as the budget.  An item enters the plan only if it is part
    of a net-positive subset — a non-relievable pegged subpool yields no
    positive item, so the knapsack returns ``[]`` (do-no-harm).

Multi-axis because items consume bytes from *different* (tier, subpool)
budgets.  Each axis quantises at its OWN ``page_bytes`` granularity
(``bucket_size`` is per-axis — no global LCM/min collapse).  The DP is
**sparse**: only cells reachable from some subset-take materialise in
the ``dp`` dict; a dense table would be exponential in axis count.

Quantisation safety: budget rounds **down** (`_bk`, never over-states
available room); consumption rounds **up** (`_bk_up`, never under-counts
what an item consumes).

Candidate contract (the items the DP consumes — string tier keys
"HBM"/"DRAM"/"DISK", matching the §5 pool_usage / §6 wire):

  * ``Migrate(cost, relief, acquired)`` — ``relief``/``acquired`` are
    sparse ``{tier: {subpool: bytes}}``; absent key = contributes 0 to
    that axis (a meaningful encoding, not a guard).  ``joint_decide``
    wraps a net-positive Migrate as a value-item (value = −cost,
    consumption = ``acquired``) before calling the primitive.
  * ``Resume(gain, re_use)`` — ``re_use`` is ``{tier: {sp: bytes}}``.
  * ``Pause(cost, relief)`` — retained for the candidate generators, but
    the **Pause lever is DORMANT**: ``joint_decide`` does not generate
    pauses (their cost misses the paused agent's forgone progress and
    their OOM-benefit is unmodelled, so a Pause cannot yet be valued —
    DESIGN §8/§9).

`joint_decide` normalises Resume's flat ``{sp: bytes}`` (HBM-only) into
the nested ``{HBM: {...}}`` shape before calling; the primitive operates
purely on the nested contract.

Mutual exclusion (#194): a unit emits SEVERAL migrate transitions
(evict / spill / DROP) that are alternatives — applying two is
physically incoherent (their relief double-counts the unit's bytes;
their costs aren't additive).  Candidates sharing a non-None ``group``
attribute are treated as **at-most-one** (multiple-choice knapsack);
``group=None`` is an independent 0/1 item.  ``migrate_candidates``
sets ``group = unit hash``; Resume leaves it None (one per program).
This is a DESIGN §9 "exact 0/1 knapsack" correction — plain 0/1 would
double-count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ------------------------------------------------------------- candidates


@dataclass
class Migrate:
    """A residence-set transition candidate.  ``relief`` frees bytes on
    the source (HBM) tier; ``acquired`` consumes bytes on the
    destination (DRAM/DISK) tier.  A pure-evict (DROP) has
    ``acquired={}``.  ``id`` is opaque (unit hash) for traceability.

    ``group`` is the mutual-exclusion key (#194): two transitions of the
    SAME unit are alternatives — applying both is physically incoherent
    (their relief would double-count shared bytes and their costs are
    not additive, since each candidate is scored as a marginal change
    from the ORIGINAL residence).  The DP treats items sharing a
    non-None ``group`` as at-most-one (multiple-choice knapsack).
    ``group=None`` (the default) ⇒ an independent 0/1 item — preserving
    the original single-item behaviour for callers that don't set it."""
    cost: float
    relief: Dict[str, Dict[str, int]] = field(default_factory=dict)
    acquired: Dict[str, Dict[str, int]] = field(default_factory=dict)
    id: Any = None
    group: Any = None


@dataclass
class Pause:
    """A program-pause candidate.  Frees HBM bytes (``relief``), consumes
    no destination capacity (no ``acquired``).  ``group`` (mutual
    exclusion) defaults to None: a program emits a single Pause.

    Retained for the candidate generators / baselines, but **DORMANT** in
    ``joint_decide`` (not generated — see the module docstring)."""
    cost: float
    relief: Dict[str, Dict[str, int]] = field(default_factory=dict)
    pid: Any = None
    group: Any = None


@dataclass
class Resume:
    """A program-resume candidate (the resume step).  ``re_use`` is the
    per-(HBM, subpool) bytes that re-enter HBM on resume; ``gain`` is
    the V_u recovered.  ``group`` defaults to None (one Resume per
    program — independent 0/1 items)."""
    gain: float
    re_use: Dict[str, Dict[str, int]] = field(default_factory=dict)
    pid: Any = None
    group: Any = None


# Sparse-DP cell ceiling (#156 audit #9).  DESIGN §9 estimates ≤10^5
# reachable cells on real T9/T11 workloads and ~10^6 worst case; Python
# materialises ~50k cells/sec, so 10^6 cells ≈ 20 s — a real event-loop
# stall, NOT "microseconds".  The state space is the cross-product of
# per-axis reachable bucket counts; it blows up when candidates carry
# large, DISTINCT relief/acquire bucket-deltas across many axes (e.g.
# large units against small page_bytes, or wide destination caps).  We
# FAIL LOUD past this ceiling rather than stall silently: the primitive
# raises ``KnapsackBudgetExceededError`` and the daemon's ``joint_decide``
# maps it to ``fatal()`` (crash-only, DESIGN §10) — a blow-up means the
# candidate set / quantisation is misconfigured (top-k undersized at the
# wrong granularity), an operator bug, not a workload reality.  Generous
# default (10× DESIGN's real-case estimate); override per call.
_MAX_DP_CELLS = 1_000_000


class KnapsackBudgetExceededError(Exception):
    """Raised when the sparse DP's reachable-cell count exceeds
    ``max_dp_cells`` (#156 audit #9).  Carries a forensic ``context``;
    ``joint_decide`` maps it to ``fatal("joint_decide_dp_blowup", …)`` —
    a pathological candidate set (excess relief/acquire variance at the
    chosen quantisation), not a workload reality."""
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        super().__init__(
            f"joint_decide DP blew up: {context.get('dp_size')} reachable "
            f"cells > max_dp_cells={context.get('max_dp_cells')} at item "
            f"{context.get('item_index')}/{context.get('n_items')} "
            f"(axes={context.get('axes')}) — candidate set / quantisation "
            f"misconfigured"
        )


# ------------------------------------------------------------- quantisation


def _bk(n: int, bucket_size: int) -> int:
    """Round-DOWN quantisation (safe for a relief ">= target" axis: never
    over-claims freed bytes; and for a budget "<= budget" axis: never
    over-states available room)."""
    return int(n) // int(bucket_size)


def _bk_up(n: int, bucket_size: int) -> int:
    """Round-UP quantisation (safe for a destination "<= cap" axis: never
    under-counts consumed bytes)."""
    bs = int(bucket_size)
    return (int(n) + bs - 1) // bs


def _grouped(items: List[Any]) -> List[List[Any]]:
    """Partition ``items`` into mutual-exclusion groups (#194).

    Items sharing a non-None ``group`` attribute land in ONE group (the
    DP takes at most one member — multiple-choice knapsack); an item
    with ``group=None`` (or no such attribute) is its own singleton
    group (independent 0/1 item, the original behaviour).  Insertion
    order is preserved so reconstruction is deterministic.
    """
    groups: List[List[Any]] = []
    index: Dict[Any, int] = {}
    for it in items:
        g = getattr(it, "group", None)
        if g is None:
            groups.append([it])
        elif g in index:
            groups[index[g]].append(it)
        else:
            index[g] = len(groups)
            groups.append([it])
    return groups


# ------------------------------------------------------------- DP primitives


def knapsack_max_value_multi(
    items: List[Any],
    budget: Dict[Any, int],
    bucket_size: Dict[Any, int],
    *,
    context: Dict[str, Any] = None,
    max_dp_cells: int = _MAX_DP_CELLS,
) -> List[Any]:
    """0/1 knapsack: subset S maximising Σ gain(s∈S) s.t. every
    (tier, sp) axis:  Σ re_use <= budget.  ``re_use`` rounds up (safe for
    <=).  Multi-axis sparse DP: only reachable ``dp`` cells materialise.

    ``items`` must expose ``.gain`` + ``.re_use`` — ``joint_decide`` feeds
    both the relief value-items (Migrate wrapped as gain = −cost,
    ``re_use`` = destination ``acquired``) and the Resume candidates
    through this one primitive (DESIGN §9 value-gated).  ``bucket_size``
    values MUST be > 0.  Raises ``KnapsackBudgetExceededError`` if the
    reachable-cell count exceeds ``max_dp_cells``.  Returns the chosen
    candidate list (empty when no positive-value subset fits)."""
    axes = list(budget.keys())
    # Positivity guard (#218): bucket_size divides in _bk/_bk_up; a 0 (or
    # negative) page granularity would raise a bare ZeroDivisionError deep
    # in the DP.  Fail with a clear, contextful error instead — the daemon
    # maps it to fatal (a non-positive page_bytes is a deployment bug).
    bad = [a for a in axes if int(bucket_size.get(a, 0)) <= 0]
    if bad:
        raise ValueError(
            f"knapsack_max_value_multi: bucket_size must be > 0 for every "
            f"axis; non-positive at {bad} (bucket_size={bucket_size})")
    W = {a: _bk(budget[a], bucket_size[a]) for a in axes}
    K = len(items)
    NEG = float("-inf")

    # Parent-pointer reconstruction + multiple-choice grouping (#194).
    # Singleton groups (group=None) reproduce plain 0/1 behaviour.  The
    # parent pointer (not a subtract-the-delta traceback) keeps the
    # chosen-subset readout exact even when a transition is rejected for
    # cap, so reconstruction never returns an over-budget subset.
    grouped = _grouped(items)
    dp: Dict[tuple, float] = {tuple(0 for _ in axes): 0.0}
    parent: Dict[tuple, tuple] = {}
    for gi, group in enumerate(grouped):
        base = dp
        new_dp = dict(dp)                            # option: take none
        for member in group:
            d = tuple(_bk_up(member.re_use.get(t, {}).get(sp, 0),
                             bucket_size[(t, sp)])
                      for (t, sp) in axes)
            for s, gain in base.items():
                s_new = tuple(s[i] + d[i] for i in range(len(axes)))
                if any(s_new[i] > W[axes[i]] for i in range(len(axes))):
                    continue
                new_gain = gain + member.gain
                if new_gain > new_dp.get(s_new, NEG):
                    new_dp[s_new] = new_gain
                    parent[(gi, s_new)] = (s, member)
        dp = new_dp
        if len(dp) > max_dp_cells:
            ctx = dict(context or {})
            ctx.update({
                "dp_size": len(dp), "max_dp_cells": max_dp_cells,
                "item_index": gi, "n_items": K, "axes": axes,
                "items": list(items),
            })
            raise KnapsackBudgetExceededError(ctx)

    s_pick = max(dp, key=dp.get)
    chosen: List[Any] = []
    gi = len(grouped)
    while gi > 0:
        gi -= 1
        if (gi, s_pick) in parent:
            s_pred, member = parent[(gi, s_pick)]
            chosen.append(member)
            s_pick = s_pred
    return chosen
