"""
KVFlow baseline (paper Section 8).

Specialization:
    p_hat = steps-to-execution    -- proxy for reuse probability is how many
                                     DAG steps until this unit is consumed
                                     next. Closer == higher reuse prob.
    only tau_old <= tau_target     -- monotone: KVFlow only promotes units
                                     toward HBM, never demotes (no DRAM->disk,
                                     no HBM->DRAM). Demotion is implicit via
                                     the engine's default eviction.

Reads `state.units[uid].p_hat` as the (precomputed) inverse-steps-to-execution
score. The workload driver fills this in (steps_until_next_use(unit) -> p_hat).
"""
from __future__ import annotations
from typing import List

from .base import Action, ReuseUnit, SchedulerState, Tier


_PROMOTE_ORDER = {Tier.DROP: 0, Tier.DISK: 1, Tier.DRAM: 2, Tier.HBM: 3}


class KVFlowPolicy:
    name = "kvflow"

    def __init__(self, hbm_threshold: float = 0.6, dram_threshold: float = 0.3):
        # p_hat thresholds for *promoting* into each tier
        self.t_hbm = hbm_threshold
        self.t_dram = dram_threshold

    def _target_tier(self, p_hat: float) -> Tier:
        if p_hat >= self.t_hbm:
            return Tier.HBM
        if p_hat >= self.t_dram:
            return Tier.DRAM
        return Tier.DISK

    def decide(self, state: SchedulerState) -> Action:
        plan: List[tuple] = []
        usage = state.tier_usage
        cap_left = {
            t: usage.capacity_bytes.get(t, 0) - usage.used_bytes.get(t, 0)
            for t in (Tier.HBM, Tier.DRAM, Tier.DISK)
        }

        # Sort the decision set by p_hat descending; highest-priority units
        # get a shot at HBM first.
        ranked = sorted(
            (state.units[uid] for uid in state.decision_set),
            key=lambda u: -u.p_hat,
        )

        for u in ranked:
            target = self._target_tier(u.p_hat)
            if _PROMOTE_ORDER[target] <= _PROMOTE_ORDER[u.tier]:
                # Monotone constraint: never demote.
                continue
            if cap_left.get(target, 0) < u.n_bytes:
                # No room at the target tier; KVFlow refuses to evict to make
                # room (that's the engine's job).
                continue
            plan.append((u.id, target))
            cap_left[target] -= u.n_bytes
            cap_left[u.tier] = cap_left.get(u.tier, 0) + u.n_bytes

        return Action(assignments=plan)
