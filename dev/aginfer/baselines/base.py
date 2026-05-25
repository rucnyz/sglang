"""
Common types and Policy interface.

Mapping to paper symbols:
    ReuseUnit       <-> u = (id, type, scope, hash, n_u, b_u, chunks)        [Section 2.1]
    SchedulerState  <-> s_t = ({(tier(u), age(u), p_hat_u)}, {g_i}, used_tau, bw_free)  [Section 3]
    Action          <-> a_t = {(u, tau_target) : u in D_t}                   [Section 4]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, List, Optional, Protocol


class Tier(IntEnum):
    DROP = 0
    DISK = 1
    DRAM = 2
    HBM = 3


class UnitType(str, Enum):
    PLATFORM = "platform"
    TOOL_DEF = "tool_def"
    SUBAGENT_CTX = "subagent_ctx"
    SESSION = "session"


class Scope(str, Enum):
    GLOBAL = "global"
    TENANT = "tenant"
    SESSION = "session"


@dataclass
class ReuseUnit:
    """Paper Section 2.1: u = (id, type, scope, hash, n_u, b_u, chunks_u)."""
    id: str
    type: UnitType
    scope: Scope
    n_tokens: int                  # n_u
    n_bytes: int                   # b_u
    chunk_ids: List[str] = field(default_factory=list)  # Mooncake/LMCache interface

    # Runtime-tracked fields (paper Section 3 state)
    tier: Tier = Tier.HBM
    age_seconds: float = 0.0
    p_hat: float = 0.0             # predicted reuse prob over Delta t
    lambda_rate: float = 0.0       # Poisson reuse rate (Section 7)

    # Per-session holders (used by 'platform' / 'tool_def' typed estimator)
    holders: List[str] = field(default_factory=list)


@dataclass
class TierUsage:
    """used_tau, BW_free for paper Section 3."""
    used_bytes: Dict[Tier, int] = field(default_factory=lambda: {Tier.HBM: 0, Tier.DRAM: 0, Tier.DISK: 0})
    capacity_bytes: Dict[Tier, int] = field(default_factory=dict)
    bw_free: Dict[tuple, float] = field(default_factory=dict)  # (src_tier, dst_tier) -> free BW

    def occupancy_ratio(self, tier: Tier) -> float:
        cap = self.capacity_bytes.get(tier, 0)
        return self.used_bytes[tier] / cap if cap else 0.0


@dataclass
class SchedulerState:
    """Paper Section 3 reduced state for one decision event."""
    t: float
    units: Dict[str, ReuseUnit]
    tier_usage: TierUsage
    # Event metadata: what triggered this decision (Section 4 table)
    event_kind: str           # e.g. 'session_arrival', 'tool_call_start', ...
    event_session_id: Optional[str] = None
    # The decision subset D_t the engine is asking the policy to act on.
    decision_set: List[str] = field(default_factory=list)


@dataclass
class Action:
    """a_t = {(u, tau_target)}. Empty list = no migration."""
    assignments: List[tuple] = field(default_factory=list)  # List[(unit_id, Tier)]


class Policy(Protocol):
    """Stateless interface: given a state + event, return an action.

    Concrete implementations may carry internal counters (LRU age table, etc.)."""

    name: str

    def decide(self, state: SchedulerState) -> Action: ...

    def update_after_step(self, state: SchedulerState, action: Action, hits: List[str]) -> None:
        """Optional hook for policies that need to update internal state (e.g. LRU
        recency table, learned model). Default no-op."""
        return None
