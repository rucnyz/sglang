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

Infeasibility (pressure phase): no subset hits every relief target
under the destination caps.  DROP (``acquired={}``) and Pause
(``acquired={}``) never consume destination capacity, so a feasible
plan ALWAYS exists unless top-k under-sized the candidate set or a
filter dropped a candidate it shouldn't have — i.e. an algorithm bug,
not a workload reality.  The primitive raises
``KnapsackInfeasibleError`` carrying a forensic context; the daemon's
``joint_decide`` maps that to ``fatal("joint_decide_infeasible", ...)``
(DESIGN §10) — kept as an exception here so the primitive stays pure
and the infeasible path is unit-testable.
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
    ``acquired={}``.  ``id`` is opaque (unit hash) for traceability."""
    cost: float
    relief: Dict[str, Dict[str, int]] = field(default_factory=dict)
    acquired: Dict[str, Dict[str, int]] = field(default_factory=dict)
    id: Any = None


@dataclass
class Pause:
    """A program-pause candidate.  Frees HBM bytes (``relief``), consumes
    no destination capacity (no ``acquired`` — see ``_acquire_at``)."""
    cost: float
    relief: Dict[str, Dict[str, int]] = field(default_factory=dict)
    pid: Any = None


@dataclass
class Resume:
    """A program-resume candidate (headroom phase).  ``re_use`` is the
    per-(HBM, subpool) bytes that re-enter HBM on resume; ``gain`` is
    the V_u recovered."""
    gain: float
    re_use: Dict[str, Dict[str, int]] = field(default_factory=dict)
    pid: Any = None


class KnapsackInfeasibleError(Exception):
    """Raised by ``knapsack_min_cost_multi`` when no subset satisfies
    every relief target under the destination caps.  Carries a forensic
    ``context`` dict; the daemon's ``joint_decide`` re-raises it as
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


# ------------------------------------------------------------- DP primitives


def knapsack_min_cost_multi(
    items: List[Any],
    bytes_needed: Dict[Any, int],
    cap_left: Dict[Any, int],
    bucket_size: Dict[Any, int],
    *,
    context: Dict[str, Any] = None,
) -> List[Any]:
    """0/1 knapsack: subset S minimising Σ cost(s∈S) s.t.
      (a) every (HBM, sp) relief axis:  Σ relief >= bytes_needed
      (b) every (DRAM|DISK, sp) cap axis: Σ acquired <= cap_left

    ``bucket_size`` keyed by axis (tier, sp) — each axis at its own page
    granularity.  Relief rounds down; destination consumption rounds up.
    Raises ``KnapsackInfeasibleError`` if no subset satisfies (a) under
    (b).  Returns the chosen candidate list (subset of ``items``)."""
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
    dp: Dict[tuple, float] = {zero_state(): 0.0}
    parent: Dict[tuple, tuple] = {}                  # (k, s_new) -> s_pred
    for k, c in enumerate(items, start=1):
        d_relief = tuple(_bk(_relief_at(c, t, sp), bucket_size[(t, sp)])
                         for (t, sp) in relief_axes)
        d_cap = tuple(_bk_up(_acquire_at(c, t, sp), bucket_size[(t, sp)])
                      for (t, sp) in cap_axes)
        new_dp = dict(dp)
        for s, cost in dp.items():
            r_buckets, cap_buckets = s[:n], s[n:]
            r_new = tuple(min(W[relief_axes[i]], r_buckets[i] + d_relief[i])
                          for i in range(n))
            cap_new = tuple(cap_buckets[i] + d_cap[i]
                            for i in range(len(cap_axes)))
            if any(cap_new[i] > Wcap[cap_axes[i]] for i in range(len(cap_axes))):
                continue
            s_new = r_new + cap_new
            new_cost = cost + c.cost
            if new_cost < new_dp.get(s_new, INF):
                new_dp[s_new] = new_cost
                parent[(k, s_new)] = s
        dp = new_dp

    full_r = tuple(W[a] for a in relief_axes)
    feasible = [(c, s) for s, c in dp.items() if s[:n] == full_r]
    if not feasible:
        ctx = dict(context or {})
        ctx.update({
            "bytes_needed": dict(bytes_needed),
            "cap_left": dict(cap_left),
            "bucket_size": dict(bucket_size),
            "n_items": K,
            "dp_size": len(dp),
        })
        raise KnapsackInfeasibleError(ctx)

    _, s_pick = min(feasible)                        # min over cost
    chosen: List[Any] = []
    k = K
    while k > 0:
        if (k, s_pick) in parent:
            chosen.append(items[k - 1])
            s_pick = parent[(k, s_pick)]
        k -= 1
    return chosen


def knapsack_max_value_multi(
    items: List[Any],
    budget: Dict[Any, int],
    bucket_size: Dict[Any, int],
) -> List[Any]:
    """0/1 knapsack: subset S maximising Σ gain(s∈S) s.t. every
    (HBM, sp) axis:  Σ re_use <= budget.  ``re_use`` rounds up (safe for
    <=).  Same sparse multi-axis DP shape as ``knapsack_min_cost_multi``.
    Returns the chosen candidate list."""
    axes = list(budget.keys())
    W = {a: _bk(budget[a], bucket_size[a]) for a in axes}
    K = len(items)
    NEG = float("-inf")

    # Parent-pointer reconstruction (uniform with knapsack_min_cost_multi;
    # robust even though the headroom state has no cap-clamp).
    dp: Dict[tuple, float] = {tuple(0 for _ in axes): 0.0}
    parent: Dict[tuple, tuple] = {}
    for k, c in enumerate(items, start=1):
        d = tuple(_bk_up(c.re_use.get(t, {}).get(sp, 0), bucket_size[(t, sp)])
                  for (t, sp) in axes)
        new_dp = dict(dp)
        for s, gain in dp.items():
            s_new = tuple(s[i] + d[i] for i in range(len(axes)))
            if any(s_new[i] > W[axes[i]] for i in range(len(axes))):
                continue
            new_gain = gain + c.gain
            if new_gain > new_dp.get(s_new, NEG):
                new_dp[s_new] = new_gain
                parent[(k, s_new)] = s
        dp = new_dp

    s_pick = max(dp, key=dp.get)
    chosen: List[Any] = []
    k = K
    while k > 0:
        if (k, s_pick) in parent:
            chosen.append(items[k - 1])
            s_pick = parent[(k, s_pick)]
        k -= 1
    return chosen
