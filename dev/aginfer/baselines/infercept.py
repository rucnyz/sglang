"""
InferCept baseline (paper Section 8).

Specialization:
    D_t only on interception   ->  policy only acts on tool_call_start /
                                   tool_call_end events.
    actions in {0, 2, 3}       ->  drop, DRAM, HBM (no disk tier).
    p_hat                       ->  binary, derived from whether the session is
                                   currently inside a tool call.

Rationale: the original InferCept paper is about "intercepting" pause points
in the agent (tool calls being the canonical one) and parking the KV cache
elsewhere so the HBM slot can be reused by another session. So:
    * on tool_call_start: demote that session's units HBM -> DRAM.
    * on tool_call_end:    promote that session's units DRAM -> HBM (if room).
    * never use disk; never drop.

On every other event, no-op (action set is empty by spec).
"""
from __future__ import annotations
from typing import List

from .base import Action, ReuseUnit, SchedulerState, Tier


class InferCeptPolicy:
    name = "infercept"

    def __init__(self):
        pass

    def _units_for_session(
        self, state: SchedulerState, session_id: str, on_tier: Tier
    ) -> List[ReuseUnit]:
        out = []
        for uid in state.decision_set:
            u = state.units[uid]
            if u.tier != on_tier:
                continue
            if u.holders and session_id in u.holders:
                out.append(u)
            elif not u.holders and u.id.startswith(session_id):
                out.append(u)
        return out

    def decide(self, state: SchedulerState) -> Action:
        plan: List[tuple] = []
        sess = state.event_session_id

        if state.event_kind == "tool_call_start" and sess:
            for u in self._units_for_session(state, sess, Tier.HBM):
                plan.append((u.id, Tier.DRAM))

        elif state.event_kind == "tool_call_end" and sess:
            cap_left = (
                state.tier_usage.capacity_bytes.get(Tier.HBM, 0)
                - state.tier_usage.used_bytes.get(Tier.HBM, 0)
            )
            for u in self._units_for_session(state, sess, Tier.DRAM):
                if cap_left < u.n_bytes:
                    break
                plan.append((u.id, Tier.HBM))
                cap_left -= u.n_bytes

        return Action(assignments=plan)
