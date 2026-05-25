"""
Ours: closed-form greedy policy from paper Section 7.

For each unit u and each tier candidate tau in {0,1,2,3}:
    V_u(tau)  = p_hat_u * [R(u,0) - R(u,tau)]  -  h_tau(used_tau) * b_u / lambda_u
    Vt_u(old->new) = V_u(new) - M_eff(u, old->new, t)

Decision = arg max over D_t under capacity (per-event multi-knapsack).
Without the top/bottom-k reduction (Section 7.1) -- that lives in
ours_topk_rl.py once we move to the learned variant.

Cost knobs are wired through TierCosts so calibration is one file change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .base import Action, ReuseUnit, SchedulerState, Tier


@dataclass
class TierCosts:
    """All per-tier scalars. Calibrate from trace warm-up."""
    # rho_tau: reload cost per token (sec/tok). tier 0 = re-prefill rate pi_u (per-unit).
    rho: Dict[Tier, float]
    # h_tau(used_tau) -> holding cost per byte per second. Function form is open in
    # the paper; we use a soft-quadratic on occupancy: h_tau * (1 + (used/cap)^2).
    h_base: Dict[Tier, float]
    # BW (bytes/sec) for each (src, dst) tier pair.
    bw: Dict[tuple, float]


def reload_cost(u: ReuseUnit, tier: Tier, costs: TierCosts, pi_u: float) -> float:
    """R(u, tau) -- paper Section 2.2."""
    if tier == Tier.DROP:
        return pi_u * u.n_tokens
    return costs.rho[tier] * u.n_tokens


def holding_unit_cost(tier: Tier, used: int, cap: int, costs: TierCosts) -> float:
    """h_tau(used_tau) -- non-decreasing in occupancy; soft-quadratic."""
    if tier == Tier.DROP or cap == 0:
        return 0.0
    occ = used / cap
    return costs.h_base[tier] * (1.0 + occ * occ)


def migration_cost_effective(
    u: ReuseUnit,
    src: Tier,
    dst: Tier,
    bw_free: Dict[tuple, float],
    costs: TierCosts,
) -> float:
    """M_eff(u, src->dst; t) = (b_u / BW) * g(bw_used).
    g(0)=1; g -> inf as bw_used/total -> 1. We use 1/(1 - bw_used/total)."""
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


class OursGreedyPolicy:
    name = "ours_greedy"

    def __init__(self, costs: TierCosts, prefill_cost_per_token: float = 1.0e-4):
        self.costs = costs
        self.pi_u = prefill_cost_per_token   # per-token prefill cost; pi_u

    def _value(self, u: ReuseUnit, tier: Tier, state: SchedulerState) -> float:
        """V_u(tau) from Section 7."""
        save_prefill = u.p_hat * (
            reload_cost(u, Tier.DROP, self.costs, self.pi_u)
            - reload_cost(u, tier, self.costs, self.pi_u)
        )
        used = state.tier_usage.used_bytes.get(tier, 0)
        cap = state.tier_usage.capacity_bytes.get(tier, 0)
        h = holding_unit_cost(tier, used, cap, self.costs)
        # Holding for expected reuse interval 1/lambda; if lambda~0 use large constant.
        hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6
        return save_prefill - h * u.n_bytes * hold_time

    def _net_value(self, u: ReuseUnit, target: Tier, state: SchedulerState) -> float:
        v = self._value(u, target, state)
        mig = migration_cost_effective(u, u.tier, target, state.tier_usage.bw_free, self.costs)
        return v - mig

    def decide(self, state: SchedulerState) -> Action:
        plan = []
        # Greedy per-unit: pick the argmax tier. Capacity is checked post-hoc;
        # in a real implementation we'd run a true multi-knapsack solver here,
        # but for V1 the greedy is the paper's Section 7 closed-form rule.
        capacity_left = {
            t: state.tier_usage.capacity_bytes.get(t, 0)
               - state.tier_usage.used_bytes.get(t, 0)
            for t in (Tier.HBM, Tier.DRAM, Tier.DISK)
        }
        for uid in state.decision_set:
            u = state.units[uid]
            best_tier = u.tier
            best_score = self._net_value(u, u.tier, state)
            for tau in (Tier.HBM, Tier.DRAM, Tier.DISK, Tier.DROP):
                if tau == u.tier:
                    continue
                if tau != Tier.DROP and capacity_left.get(tau, 0) < u.n_bytes:
                    continue
                score = self._net_value(u, tau, state)
                if score > best_score:
                    best_score = score
                    best_tier = tau
            if best_tier != u.tier:
                plan.append((u.id, best_tier))
                if best_tier != Tier.DROP:
                    capacity_left[best_tier] -= u.n_bytes
                capacity_left[u.tier] = capacity_left.get(u.tier, 0) + u.n_bytes
        return Action(assignments=plan)
