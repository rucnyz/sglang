"""
Ours: closed-form greedy policy (paper §7, post-T17 / residence-set form).

For each unit u in the decision set, enumerate the meaningful per-unit
residence transitions (DESIGN §7 transfer-window semantics, 6 cases)
and pick the argmax V_u over the candidates that fit current capacity:

    V_u(next_residence)
        = p_hat_u · [R(u, DROP) - R(u, authoritative_tier(next))]
        − h_(τ, sp)(occ) · b_u · 1/λ_u
        − M_eff(u, current → next, t)

T34 will replace the per-unit greedy with the multi-axis sparse DP over
the union action space; T12 will replace the soft-quadratic placeholder
holding-cost shape.  T33 (this commit) covers the data-flow + 6-
transition enumeration only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base import Action, ReuseUnit, SchedulerState, Tier
from .knapsack import Migrate

# String tier labels matching the §5 pool_usage / §6 wire (the
# knapsack candidate contract keys relief/acquired by these strings).
_TIER_LABEL = {Tier.HBM: "HBM", Tier.DRAM: "DRAM", Tier.DISK: "DISK"}


@dataclass
class TierCosts:
    """All per-tier scalars. Calibrate from trace warm-up."""
    # rho_τ: reload cost per token (sec/tok). DROP tier uses prefill cost pi_u.
    rho: Dict[Tier, float]
    # h_τ baseline; multiplied by (1 + occ^2) so cost rises near cap.
    # Per-(tier, subpool) refinement is T12 calibration future-work; for
    # now we treat all subpools in a tier as having the same h baseline.
    h_base: Dict[Tier, float]
    # BW (bytes/sec) per directional link (src, dst).
    bw: Dict[Tuple[Tier, Tier], float]


# Per DESIGN §7 transfer-window semantics, 6 meaningful per-unit
# transitions (each is a (add_tiers, remove_tiers) edit to residence).
# Keyed by the "current residence as frozenset" to keep the table
# index-stable.
_TRANSITIONS: Dict[frozenset, List[Tuple[List[Tier], List[Tier]]]] = {
    # HBM only → HBM+DRAM (write_through) | HBM only → DRAM (host-only)
    # | HBM only → DROP (full eviction)
    frozenset({Tier.HBM}): [
        ([Tier.DRAM], []),               # write_through; keep HBM
        ([Tier.DRAM], [Tier.HBM]),       # evict HBM, host backup created
        ([], [Tier.HBM]),                # DROP entirely
    ],
    # HBM+DRAM → DRAM (HBM evict) | HBM+DRAM → HBM (DRAM drop)
    # | HBM+DRAM → DROP (full)
    frozenset({Tier.HBM, Tier.DRAM}): [
        ([], [Tier.HBM]),                # HBM eviction
        ([], [Tier.DRAM]),               # DRAM drop, device retained
        ([], [Tier.HBM, Tier.DRAM]),     # DROP entirely
    ],
    # DRAM only → HBM+DRAM (load_back) | DRAM only → DRAM+DISK (spill)
    # | DRAM only → DROP
    frozenset({Tier.DRAM}): [
        ([Tier.HBM], []),                # load_back to device
        ([Tier.DISK], []),               # Mooncake spill
        ([], [Tier.DRAM]),               # DROP
    ],
    # DRAM+DISK → DRAM (drop disk) | DRAM+DISK → HBM+DRAM (promote+drop_disk)
    # | DRAM+DISK → DISK
    frozenset({Tier.DRAM, Tier.DISK}): [
        ([], [Tier.DISK]),               # disk spill rolled back
        ([Tier.HBM], [Tier.DISK]),       # promote to device, lose disk
        ([], [Tier.DRAM]),               # forget device-side, keep disk
    ],
    # DISK only → DRAM+DISK (Mooncake load) | DISK only → DROP
    frozenset({Tier.DISK}): [
        ([Tier.DRAM], []),               # Mooncake load to DRAM
        ([], [Tier.DISK]),               # DROP
    ],
}


def reload_cost(u: ReuseUnit, tier: Tier, costs: TierCosts,
                pi_u: float) -> float:
    """R(u, τ) — paper §2.2."""
    if tier == Tier.DROP:
        return pi_u * u.n_tokens
    return costs.rho[tier] * u.n_tokens


def holding_unit_cost(tier: Tier, occ: float, costs: TierCosts) -> float:
    """h_τ(occ) — non-decreasing in occupancy; soft-quadratic placeholder.

    T12 calibration replaces this shape (linear vs power vs hyperbolic)
    per subpool.  For T33 we use max-over-subpools occupancy passed in
    by the caller.
    """
    if tier == Tier.DROP:
        return 0.0
    return costs.h_base[tier] * (1.0 + occ * occ)


def migration_cost_effective(
    u: ReuseUnit,
    src: Tier,
    dst: Tier,
    bw_free: Dict[Tuple[Tier, Tier], float],
    costs: TierCosts,
) -> float:
    """M_eff(u, src → dst; t) = (b_u / BW) · g(bw_used).
    g(0)=1; g → ∞ as bw_used/total → 1."""
    if src == dst or dst == Tier.DROP:
        return 0.0
    bw_total = costs.bw.get((src, dst), 0.0)
    if bw_total <= 0:
        return float("inf")
    base = u.n_bytes / bw_total
    free = bw_free.get((src, dst), bw_total)
    bw_used_frac = max(0.0, 1.0 - free / bw_total)
    if bw_used_frac >= 0.999:
        return float("inf")
    g = 1.0 / max(1e-6, 1.0 - bw_used_frac)
    return base * g


def _authoritative_of(residence: List[Tier]) -> Tier:
    """Highest-compute-readiness tier in ``residence`` (HBM > DRAM >
    DISK); empty residence ≡ DROP (DESIGN §7 ``authoritative_tier``)."""
    for t in (Tier.HBM, Tier.DRAM, Tier.DISK):
        if t in residence:
            return t
    return Tier.DROP


def value_residence(u: ReuseUnit, next_residence: List[Tier],
                    state: SchedulerState, costs: TierCosts,
                    pi_u: float) -> float:
    """V_u over a candidate residence — paper §7 / DESIGN §7 ``_value``.

    Module-level so the §9 ``migrate_candidates`` generator and the
    admission program-value aggregation can share ONE V_u definition
    with ``OursGreedyPolicy`` (DESIGN §9: one decision pipeline, one
    value function).  The authoritative tier of ``next_residence``
    drives both the reload cost (the tier that services the next read)
    and the holding tax.
    """
    tier = _authoritative_of(next_residence)
    save_prefill = u.p_hat * (
        reload_cost(u, Tier.DROP, costs, pi_u)
        - reload_cost(u, tier, costs, pi_u)
    )
    occ = state.tier_usage.occupancy_ratio(tier) if tier != Tier.DROP else 0.0
    h = holding_unit_cost(tier, occ, costs)
    hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6
    return save_prefill - h * u.n_bytes * hold_time


def migrate_candidates(
    state: SchedulerState,
    decision_set: List[str],
    costs: TierCosts,
    pi_u: float = 1.0e-4,
) -> List[Migrate]:
    """DESIGN §7 / §9 unit-level candidate generator.

    For each unit in ``decision_set``, enumerate the meaningful
    residence-set transitions (the ``_TRANSITIONS`` table, DESIGN §7
    transfer-window semantics) and emit one ``knapsack.Migrate`` per
    pressure-relieving transition:

        cost     = V_u(current) − V_u(next)            # value forgone
                 + M_eff(current_auth → next_auth)     # link bandwidth
                 + unavailability_cost (= 0 under write-through HiCache)
        relief   = {tier_str: {sp: bytes}} freed on each removed tier
        acquired = {tier_str: {sp: bytes}} consumed on each added tier
        id       = (unit_id, add_tiers, remove_tiers)  # for dispatch

    ``id`` carries the residence-set edit verbatim so §9's chosen
    subset maps straight onto ``assignments_to_wire`` without a second
    lookup.  Transitions whose ``relief`` is empty across every
    (tier, subpool) are dropped (DESIGN §7: a meaningful candidate
    frees bytes on at least one axis) — e.g. a pure write-through
    ``add {DRAM}`` keeps HBM and relieves nothing.

    Same-unit transitions are emitted as INDEPENDENT 0/1 items; the §9
    DP must not pick two transitions of one unit (their relief would
    double-count physically-shared bytes).  ``joint_decide`` enforces
    this via per-unit grouping before the knapsack — see
    ``daemon/joint_decide.py``.
    """
    out: List[Migrate] = []
    for uid in decision_set:
        u = state.units.get(uid)
        if u is None:
            continue
        current = list(u.residence)
        current_key = frozenset(current)
        transitions = _TRANSITIONS.get(current_key)
        if transitions is None:
            # Residence the §7 6-transition table doesn't enumerate
            # (e.g. {HBM, DISK}); skip rather than invent a transition
            # — same contract as OursGreedyPolicy.decide.
            continue
        src = _authoritative_of(current)
        for add_tiers, remove_tiers in transitions:
            new_residence = [t for t in current if t not in remove_tiers] \
                + list(add_tiers)
            if frozenset(new_residence) == current_key:
                continue  # no-op edit

            cost = value_residence(u, current, state, costs, pi_u) \
                - value_residence(u, new_residence, state, costs, pi_u)
            # Migration (link) cost: each ADDED tier not already resident
            # copies u's bytes from the source over the relevant link.
            for t in add_tiers:
                if t in current or t == Tier.DROP:
                    continue
                cost += migration_cost_effective(
                    u, src, t, state.tier_usage.bw_free, costs)
            # unavailability_cost == 0 under write-through HiCache
            # (DESIGN §7); kept implicit (the +0 term).

            relief: Dict[str, Dict[str, int]] = {}
            for t in remove_tiers:
                if t not in current or t == Tier.DROP:
                    continue
                sp_bytes = u.n_bytes_by_tier.get(t, {})
                if sp_bytes:
                    relief[_TIER_LABEL[t]] = {sp: int(b)
                                              for sp, b in sp_bytes.items()}
            if not any(b > 0 for sub in relief.values()
                       for b in sub.values()):
                continue  # no pressure relieved (DESIGN §7 filter)

            acquired: Dict[str, Dict[str, int]] = {}
            src_bytes = u.n_bytes_by_tier.get(src, {})
            for t in add_tiers:
                if t in current or t == Tier.DROP:
                    continue
                # Same physical bytes land on the destination tier
                # (write-through copies bit-for-bit; subpool layout is
                # architecture-fixed) — size from the source tier.
                acquired[_TIER_LABEL[t]] = {sp: int(b)
                                            for sp, b in src_bytes.items()}

            out.append(Migrate(
                cost=cost,
                relief=relief,
                acquired=acquired,
                id=(u.id, list(add_tiers), list(remove_tiers)),
                group=u.id,   # #194: a unit's transitions are alternatives
            ))
    return out


class OursGreedyPolicy:
    name = "ours_greedy"

    def __init__(self, costs: TierCosts, prefill_cost_per_token: float = 1.0e-4):
        self.costs = costs
        self.pi_u = prefill_cost_per_token

    def _value(self, u: ReuseUnit, next_residence: List[Tier],
               state: SchedulerState) -> float:
        """V_u over a candidate residence — delegates to the module-level
        :func:`value_residence` so the greedy policy, ``migrate_candidates``
        (§9), and admission share one value definition."""
        return value_residence(u, next_residence, state, self.costs, self.pi_u)

    def _score_transition(self, u: ReuseUnit, next_residence: List[Tier],
                          state: SchedulerState) -> float:
        """V_u(next) − M_eff(current_auth → next_auth)."""
        v = self._value(u, next_residence, state)
        # Migration cost: src is current authoritative_tier, dst is next.
        src = u.authoritative_tier
        if next_residence:
            for t in (Tier.HBM, Tier.DRAM, Tier.DISK):
                if t in next_residence:
                    dst = t
                    break
            else:
                dst = Tier.DROP
        else:
            dst = Tier.DROP
        mig = migration_cost_effective(
            u, src, dst, state.tier_usage.bw_free, self.costs)
        return v - mig

    def decide(self, state: SchedulerState) -> Action:
        plan: List[Tuple[str, List[Tier], List[Tier]]] = []

        # Per-tier capacity_left: cap_total − used_total (sum over
        # subpools).  T34's multi-axis DP replaces this single-axis
        # check with per-(tier, subpool) constraints; for T33 the
        # tier-level check is sufficient and matches the paper §7
        # closed-form greedy.
        capacity_left = {
            t: state.tier_usage.cap_bytes_total(t)
               - state.tier_usage.used_bytes_total(t)
            for t in (Tier.HBM, Tier.DRAM, Tier.DISK)
        }

        for uid in state.decision_set:
            u = state.units[uid]
            current_residence = list(u.residence)
            current_key = frozenset(current_residence)
            candidates = _TRANSITIONS.get(current_key)
            if candidates is None:
                # Unrecognised residence set (e.g. {HBM, DISK} which
                # the §7 6-transition table doesn't enumerate); skip
                # rather than try to invent a transition.
                continue

            # Baseline: no migrate.
            best_next = current_residence
            best_score = self._score_transition(u, current_residence, state)

            for add_tiers, remove_tiers in candidates:
                next_residence = (
                    [t for t in current_residence if t not in remove_tiers]
                    + list(add_tiers)
                )
                # Capacity check: every tier we're ADDING to must have
                # room for u's bytes.
                fits = True
                for t in add_tiers:
                    if t == Tier.DROP:
                        continue
                    if capacity_left.get(t, 0) < u.n_bytes:
                        fits = False
                        break
                if not fits:
                    continue

                score = self._score_transition(u, next_residence, state)
                if score > best_score:
                    best_score = score
                    best_next = next_residence

            best_set = frozenset(best_next)
            if best_set == current_key:
                continue  # no migrate this unit
            add = [t for t in best_next if t not in current_residence]
            remove = [t for t in current_residence if t not in best_next]
            plan.append((u.id, add, remove))
            # Update capacity bookkeeping.
            for t in add:
                if t != Tier.DROP:
                    capacity_left[t] = capacity_left.get(t, 0) - u.n_bytes
            for t in remove:
                if t != Tier.DROP:
                    capacity_left[t] = capacity_left.get(t, 0) + u.n_bytes

        return Action(assignments=plan)
