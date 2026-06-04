"""T34 (#156, DESIGN §9 "Joint decide") — multi-axis sparse 0/1
knapsack DP primitives + the candidate contract they consume.

`joint_decide` (DESIGN §9) runs one of two exact 0/1 knapsacks per
event:

  * **Pressure phase** — ``knapsack_min_cost_multi``: choose the
    minimum-total-V_u-cost subset of {Migrate-HBM-out, Pause} that
    frees at least ``bytes_needed[(HBM, sp)]`` from EVERY pressured
    HBM subpool (>= axes) WITHOUT overflowing any destination
    ``cap_left[(DRAM|DISK, sp)]`` (<= axes).
  * **Headroom phase** — ``knapsack_max_value_multi``: choose the
    maximum-total-V_u-gain subset of {Resume} whose re-entering HBM
    bytes stay within ``budget[(HBM, sp)]`` per HBM subpool (<= axes).

Multi-axis because items consume bytes from *different* (tier, subpool)
budgets.  Each axis quantises at its OWN ``page_bytes`` granularity
(``bucket_size`` is per-axis — no global LCM/min collapse).  The DP is
**sparse**: only cells reachable from some subset-take materialise in
the ``dp`` dict; a dense table would be exponential in axis count.

Quantisation safety: relief / budget round **down** (`_bk`, safe for
">= target" and "<= budget" respectively — never over-claims relief,
never over-spends budget); destination consumption rounds **up**
(`_bk_up`, safe for "<= cap" — never under-counts what we consume).

Candidate contract (the items the DP consumes — string tier keys
"HBM"/"DRAM"/"DISK", matching the §5 pool_usage / §6 wire):

  * ``Migrate(cost, relief, acquired)`` — ``relief``/``acquired`` are
    sparse ``{tier: {subpool: bytes}}``; absent key = contributes 0 to
    that axis (a meaningful encoding, not a guard).
  * ``Pause(cost, relief)`` — ``relief`` is ``{tier: {sp: bytes}}``;
    pausing consumes no destination capacity (no ``acquired``).
  * ``Resume(gain, re_use)`` — ``re_use`` is ``{tier: {sp: bytes}}``.

`joint_decide` normalises Pause's flat ``{sp: bytes}`` and Resume's
flat ``{sp: bytes}`` (HBM-only) into the nested ``{HBM: {...}}`` shape
before calling these primitives; the primitives operate purely on the
nested contract.

Mutual exclusion (#194): a unit emits SEVERAL migrate transitions
(evict / spill / DROP) that are alternatives — applying two is
physically incoherent (their relief double-counts the unit's bytes;
their costs aren't additive).  Candidates sharing a non-None ``group``
attribute are treated as **at-most-one** (multiple-choice knapsack);
``group=None`` is an independent 0/1 item.  ``migrate_candidates``
sets ``group = unit hash``; Pause / Resume leave it None (one per
program).  This is a DESIGN §9 "exact 0/1 knapsack" correction — plain
0/1 would double-count.

Infeasibility (pressure phase): no subset hits every relief target
under the destination caps.  The DESIGN claim that this is "always an
algorithm bug" is FALSE in the common case (#194): in-flight-dominated
pressure migration can't touch, with no Pause candidate available, is
a *workload reality*.  Callers pass ``best_effort=True`` to free the
max-relief subset and re-evaluate next event instead of crashing
(``joint_decide`` does this); ``best_effort=False`` (the default) keeps
raising ``KnapsackInfeasibleError`` so the infeasible path stays unit-
testable.  ``cap_left`` is clamped to ≥ 0 by the caller (a negative
budget = over-subscribed destination = 0 room).
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
    no destination capacity (no ``acquired`` — see ``_acquire_at``).
    ``group`` (mutual exclusion) defaults to None: a program emits a
    single Pause, so pauses are independent 0/1 items."""
    cost: float
    relief: Dict[str, Dict[str, int]] = field(default_factory=dict)
    pid: Any = None
    group: Any = None


@dataclass
class Resume:
    """A program-resume candidate (headroom phase).  ``re_use`` is the
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


class KnapsackInfeasibleError(Exception):
    """Raised by ``knapsack_min_cost_multi`` when no subset satisfies
    every relief target under the destination caps.  Carries a forensic
    ``context`` dict (including the candidate ``items`` per DESIGN §9's
    ``fatal(candidates=…)``, so ops can see WHICH candidates were
    available — diagnosing top-k undersizing vs a filter dropping a
    needed candidate); the daemon's ``joint_decide`` re-raises it as
    ``fatal("joint_decide_infeasible", **context)`` (DESIGN §9/§10)."""
    def __init__(self, context: Dict[str, Any]):
        self.context = context
        super().__init__(
            f"joint_decide infeasible: no subset hits every relief "
            f"target under destination caps "
            f"(bytes_needed={context.get('bytes_needed')}, "
            f"cap_left={context.get('cap_left')}, "
            f"items={context.get('n_items')}, dp_size={context.get('dp_size')})"
        )


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


def _relief_at(c: Any, tier: str, sp: str) -> int:
    """Bytes candidate ``c`` frees on axis (tier, sp).  Sparse: absent
    key = 0 (legitimate, not a guard)."""
    return c.relief.get(tier, {}).get(sp, 0)


def _acquire_at(c: Any, tier: str, sp: str) -> int:
    """Bytes candidate ``c`` adds to destination axis (tier, sp).  Only
    Migrate has ``acquired``; Pause never consumes destination
    capacity, so return 0 without indexing."""
    if isinstance(c, Migrate):
        return c.acquired.get(tier, {}).get(sp, 0)
    return 0


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


def knapsack_min_cost_multi(
    items: List[Any],
    bytes_needed: Dict[Any, int],
    cap_left: Dict[Any, int],
    bucket_size: Dict[Any, int],
    *,
    context: Dict[str, Any] = None,
    max_dp_cells: int = _MAX_DP_CELLS,
    best_effort: bool = False,
) -> List[Any]:
    """0/1 knapsack: subset S minimising Σ cost(s∈S) s.t.
      (a) every (HBM, sp) relief axis:  Σ relief >= bytes_needed
      (b) every (DRAM|DISK, sp) cap axis: Σ acquired <= cap_left

    ``items`` are pressure-phase candidates (``Migrate`` / ``Pause``) —
    each must expose ``.cost`` + ``.relief`` (and ``Migrate.acquired``).
    Passing a ``Resume`` is a caller bug (``joint_decide`` enforces the
    phase split); it raises ``AttributeError`` rather than silently
    mis-scoring.

    ``bucket_size`` keyed by axis (tier, sp) — each axis at its own page
    granularity; values MUST be > 0 (sourced from
    ``pool_usage[τ].subpools[sp].page_bytes``).  Relief rounds DOWN
    (never over-claims), destination consumption rounds UP (never
    under-counts).  NOTE (#156 audit B): the round-UP target +
    round-DOWN relief can report a marginally raw-feasible instance
    (sub-page residual need) as INFEASIBLE; with ``best_effort`` the DP
    frees as much as it can in that case (safe direction: under-free,
    re-evaluate next event).

    ``best_effort`` (#194): when no subset reaches the full relief target
    (workload reality — e.g. in-flight-dominated pressure no migrate can
    touch and no Pause candidate is available), return the reachable
    cell with the MAXIMUM total relief, tie-broken by minimum cost,
    instead of raising.  When the target IS reachable this is identical
    to the strict optimum (full-relief cells have max total relief, and
    min cost among them is the strict answer).  ``best_effort=False``
    (the default, used by the t34 primitives test) raises
    ``KnapsackInfeasibleError`` so the infeasible path stays unit-
    testable.  ``cap_left`` should be pre-clamped to ≥ 0 by the caller
    (a negative budget means the destination is over-subscribed = no
    room = 0, never "less than zero room").

    Raises ``KnapsackBudgetExceededError`` if the reachable-cell count
    exceeds ``max_dp_cells``.  Returns the chosen candidate list
    (subset of ``items``)."""
    relief_axes = list(bytes_needed.keys())          # [(HBM, sp), ...]
    cap_axes = list(cap_left.keys())                 # [(DRAM|DISK, sp), ...]
    W = {a: _bk_up(bytes_needed[a], bucket_size[a]) for a in relief_axes}
    Wcap = {a: _bk(cap_left[a], bucket_size[a]) for a in cap_axes}
    K = len(items)
    INF = float("inf")
    n = len(relief_axes)

    def zero_state():
        return (*(0 for _ in relief_axes), *(0 for _ in cap_axes))

    # PARENT-POINTER reconstruction (not the DESIGN's subtract-the-delta
    # traceback): the relief axis is CAPPED via min(W, …) on the forward
    # step, so the state is NON-invertible — subtracting an item's relief
    # delta cannot recover the true predecessor once a transition
    # saturated the cap, and the reconstructed subset would be wrong
    # (the DP COST stays correct; only the chosen-subset readout breaks).
    # Recording the predecessor state per IMPROVING transition makes the
    # readout exact regardless of capping.  (T34 verify A2 caught the
    # subtract version returning a too-cheap, infeasible subset.)
    # Multiple-choice over mutual-exclusion groups (#194): each group
    # contributes AT MOST ONE member.  Singleton groups (group=None)
    # reproduce the original 0/1 behaviour exactly.  Each member is
    # scored from the PRE-GROUP states (``base = dp``), never from a
    # sibling-take, so two members of one group can't both be chosen.
    grouped = _grouped(items)
    dp: Dict[tuple, float] = {zero_state(): 0.0}
    # parent[(gi, s_new)] = (s_pred, member) — which member of group gi
    # produced s_new; absent ⇒ group gi took none on the path to s_new.
    parent: Dict[tuple, tuple] = {}
    for gi, group in enumerate(grouped):
        base = dp
        new_dp = dict(dp)                            # option: take none
        for member in group:
            d_relief = tuple(_bk(_relief_at(member, t, sp), bucket_size[(t, sp)])
                             for (t, sp) in relief_axes)
            d_cap = tuple(_bk_up(_acquire_at(member, t, sp), bucket_size[(t, sp)])
                          for (t, sp) in cap_axes)
            for s, cost in base.items():
                r_buckets, cap_buckets = s[:n], s[n:]
                r_new = tuple(min(W[relief_axes[i]], r_buckets[i] + d_relief[i])
                              for i in range(n))
                cap_new = tuple(cap_buckets[i] + d_cap[i]
                                for i in range(len(cap_axes)))
                if any(cap_new[i] > Wcap[cap_axes[i]]
                       for i in range(len(cap_axes))):
                    continue
                s_new = r_new + cap_new
                new_cost = cost + member.cost
                if new_cost < new_dp.get(s_new, INF):
                    new_dp[s_new] = new_cost
                    parent[(gi, s_new)] = (s, member)
        dp = new_dp
        if len(dp) > max_dp_cells:
            ctx = dict(context or {})
            ctx.update({
                "dp_size": len(dp), "max_dp_cells": max_dp_cells,
                "item_index": gi, "n_items": K,
                "axes": relief_axes + cap_axes, "items": list(items),
            })
            raise KnapsackBudgetExceededError(ctx)

    full_r = tuple(W[a] for a in relief_axes)
    feasible = [(c, s) for s, c in dp.items() if s[:n] == full_r]
    if feasible:
        _, s_pick = min(feasible)                    # min cost among full-relief
    elif best_effort:
        # No subset reaches the full target — free as much as possible:
        # max total relief (sum across capped axes), tie-broken by min
        # cost.  The next event re-evaluates; sglang eviction backstops.
        s_pick = max(dp, key=lambda s: (sum(s[:n]), -dp[s]))
    else:
        ctx = dict(context or {})
        ctx.update({
            "bytes_needed": dict(bytes_needed),
            "cap_left": dict(cap_left),
            "bucket_size": dict(bucket_size),
            "n_items": K,
            "items": list(items),          # #156 audit #8: which candidates
            "dp_size": len(dp),
        })
        raise KnapsackInfeasibleError(ctx)

    chosen: List[Any] = []
    gi = len(grouped)
    while gi > 0:
        gi -= 1
        if (gi, s_pick) in parent:
            s_pred, member = parent[(gi, s_pick)]
            chosen.append(member)
            s_pick = s_pred
    return chosen


def knapsack_max_value_multi(
    items: List[Any],
    budget: Dict[Any, int],
    bucket_size: Dict[Any, int],
    *,
    context: Dict[str, Any] = None,
    max_dp_cells: int = _MAX_DP_CELLS,
) -> List[Any]:
    """0/1 knapsack: subset S maximising Σ gain(s∈S) s.t. every
    (HBM, sp) axis:  Σ re_use <= budget.  ``re_use`` rounds up (safe for
    <=).  Same sparse multi-axis DP shape as ``knapsack_min_cost_multi``.

    ``items`` are headroom-phase candidates (``Resume``) — each must
    expose ``.gain`` + ``.re_use``; passing a ``Migrate``/``Pause`` is a
    caller bug (raises ``AttributeError``).  ``bucket_size`` values MUST
    be > 0.  Raises ``KnapsackBudgetExceededError`` if the reachable-cell
    count exceeds ``max_dp_cells``.  Returns the chosen candidate list."""
    axes = list(budget.keys())
    W = {a: _bk(budget[a], bucket_size[a]) for a in axes}
    K = len(items)
    NEG = float("-inf")

    # Parent-pointer reconstruction + multiple-choice grouping (#194),
    # uniform with knapsack_min_cost_multi.  Singleton groups reproduce
    # the original 0/1 behaviour.
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
