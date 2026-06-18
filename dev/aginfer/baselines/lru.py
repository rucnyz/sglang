"""
LRU baseline.

Paper Section 8 specialization:
    p_hat_u = f(age),  tau in {3, 0}    (binary: keep on HBM or drop)

Concretely: when the engine asks the policy to act on D_t (typically only on
memory_pressure events), demote the oldest units in D_t down to tier 0 until
HBM occupancy is back under threshold. Never use DRAM/Disk — pure LRU is
HBM-or-drop.
"""
from __future__ import annotations
from typing import List

from .base import Action, Policy, ReuseUnit, SchedulerState, Tier


class LRUPolicy:
    name = "lru"

    def __init__(self, high_watermark: float = 0.85, low_watermark: float = 0.70):
        self.hi = high_watermark
        self.lo = low_watermark

    def decide(self, state: SchedulerState) -> Action:
        # Only act on memory pressure; ignore all other event kinds.
        if state.event_kind != "memory_pressure":
            return Action()

        hbm = Tier.HBM
        usage = state.tier_usage
        if usage.occupancy_ratio(hbm) < self.hi:
            return Action()

        target_bytes = int(usage.capacity_bytes[hbm] * self.lo)
        bytes_to_free = max(0, usage.used_bytes[hbm] - target_bytes)

        # Eldest first among the decision set (units the engine offered up).
        candidates: List[ReuseUnit] = sorted(
            (state.units[uid] for uid in state.decision_set if state.units[uid].tier == hbm),
            key=lambda u: -u.age_seconds,
        )

        plan = []
        freed = 0
        for u in candidates:
            if freed >= bytes_to_free:
                break
            plan.append((u.id, Tier.DROP))
            freed += u.n_bytes
        return Action(assignments=plan)
