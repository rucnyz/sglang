"""
ThunderAgent baseline (paper Section 8).

Specialization: action space restricted to {keep, invalidate}. No tiering, no
DRAM, no disk -- a unit either lives on HBM (tau=3) or is dropped (tau=0).
The unit-of-action is also coarser than ours: ThunderAgent operates at
session granularity, not per-reuse-unit, so a single decision invalidates
every unit a session owns. We approximate that here by grouping `state.units`
by `holders[0]` (=== owning session) and letting the policy invalidate the
oldest *session* whole when HBM is over the high watermark.

Decision rule:
    * on memory_pressure, find the session whose units have the largest
      max(age_seconds). Invalidate every unit of that session.
    * repeat until HBM occupancy < low_watermark.
    * on every other event, no-op.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

from .base import Action, ReuseUnit, SchedulerState, Tier


class ThunderAgentPolicy:
    name = "thunder_agent"

    def __init__(self, high_watermark: float = 0.85, low_watermark: float = 0.70):
        self.hi = high_watermark
        self.lo = low_watermark

    def _sessions(self, state: SchedulerState) -> Dict[str, List[ReuseUnit]]:
        groups: Dict[str, List[ReuseUnit]] = defaultdict(list)
        for uid in state.decision_set:
            u = state.units[uid]
            owner = u.holders[0] if u.holders else u.id
            groups[owner].append(u)
        return groups

    def decide(self, state: SchedulerState) -> Action:
        if state.event_kind != "memory_pressure":
            return Action()
        usage = state.tier_usage
        if usage.occupancy_ratio(Tier.HBM) < self.hi:
            return Action()

        target_bytes = int(usage.capacity_bytes[Tier.HBM] * self.lo)
        bytes_to_free = max(0, usage.used_bytes[Tier.HBM] - target_bytes)

        groups = self._sessions(state)
        ranked = sorted(
            groups.items(),
            key=lambda kv: -max((u.age_seconds for u in kv[1]), default=0.0),
        )

        plan: List[tuple] = []
        freed = 0
        for _sess, units in ranked:
            if freed >= bytes_to_free:
                break
            for u in units:
                if u.tier != Tier.HBM:
                    continue
                plan.append((u.id, Tier.DROP))
                freed += u.n_bytes
        return Action(assignments=plan)
