"""
Continuum baseline (paper Section 8).

Specialization:
    Pin / unpin, tau in {2, 3}            -- only DRAM and HBM, never disk, never drop.
    p_hat via TTL                          -- units have a finite lifetime; freshness
                                              is (TTL - age) / TTL clipped to [0, 1].

So units are either "pinned" on HBM (fresh, p_hat above pin_threshold) or
"unpinned" to DRAM (stale, p_hat below pin_threshold). Once a unit's TTL
expires, p_hat goes to 0 and it is dropped from the active pool by the
engine (Continuum does not explicitly invalidate; expiry is a side-effect).

For the simulator we expose:
    * On memory_pressure: demote DRAM-pinning of units whose p_hat < pin_threshold.
    * On session_arrival / llm_prefill: promote DRAM->HBM if p_hat >= pin_threshold.
"""
from __future__ import annotations
from typing import List

from .base import Action, ReuseUnit, SchedulerState, Tier


class ContinuumPolicy:
    name = "continuum"

    def __init__(self, ttl_seconds: float = 60.0, pin_threshold: float = 0.5):
        self.ttl = ttl_seconds
        self.thr = pin_threshold

    def _ttl_freshness(self, u: ReuseUnit) -> float:
        if self.ttl <= 0:
            return 0.0
        return max(0.0, 1.0 - u.age_seconds / self.ttl)

    def decide(self, state: SchedulerState) -> Action:
        plan: List[tuple] = []
        usage = state.tier_usage

        if state.event_kind == "memory_pressure":
            for uid in state.decision_set:
                u = state.units[uid]
                if u.tier != Tier.HBM:
                    continue
                if self._ttl_freshness(u) < self.thr:
                    plan.append((u.id, Tier.DRAM))

        elif state.event_kind in ("session_arrival", "llm_prefill", "tool_call_end"):
            cap_left = (
                usage.capacity_bytes.get(Tier.HBM, 0)
                - usage.used_bytes.get(Tier.HBM, 0)
            )
            promote: List[tuple] = []
            for uid in state.decision_set:
                u = state.units[uid]
                if u.tier != Tier.DRAM:
                    continue
                if self._ttl_freshness(u) < self.thr:
                    continue
                if cap_left < u.n_bytes:
                    break
                promote.append((u.id, Tier.HBM))
                cap_left -= u.n_bytes
            plan.extend(promote)

        return Action(assignments=plan)
