"""
Common types and Policy interface (post-T17 / DESIGN §5 + §6).

Mapping to paper symbols (post-round-9):
    ReuseUnit       <-> u = (id, type, scope, residence, n_bytes_by_tier, ...)  [§2.1 + §5]
    SchedulerState  <-> s_t = ({u}, {g_i}, pool_usage, bw_free, ...)            [§3]
    Action          <-> a_t = {(u, add_tiers, remove_tiers) : u in D_t}         [§6 migrate]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple


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
    """Paper §2.1 reuse unit, lifted to DESIGN §5 residence-set semantics.

    A unit can simultaneously occupy multiple tiers (e.g. post-write-through
    a node is in HBM AND DRAM until HBM eviction).  `residence` is the SET
    of tiers the unit currently has bytes in; `n_bytes_by_tier` is the
    per-(tier, subpool) byte breakdown that §9's multi-axis DP indexes.
    """
    id: str
    type: UnitType
    scope: Scope
    n_tokens: int
    # Per-(tier, subpool) bytes.  Mirrors /aginfer/state.units[*].n_bytes
    # shape exactly.  Empty dict ⇒ the unit has been dropped (in which
    # case it shouldn't appear in units[] at all per DESIGN §5).
    n_bytes_by_tier: Dict[Tier, Dict[str, int]] = field(default_factory=dict)
    chunk_ids: List[str] = field(default_factory=list)

    # Residence as a SET (list for JSON-stability).  Equals
    # `list(n_bytes_by_tier.keys())` by construction; the field exists
    # explicitly so consumers don't have to derive it on every read.
    residence: List[Tier] = field(default_factory=list)

    # Runtime metadata (paper §3)
    age_seconds: float = 0.0
    p_hat: float = 0.0
    lambda_rate: float = 0.0

    # Per-session holders (admission §8 uses this for 1/holders weighting)
    holders: List[str] = field(default_factory=list)

    # DESIGN §7/§9 (#210): the three structural leaf predicates sglang
    # checks before applying a remove (unified_radix_cache apply site
    # 2673/2684/2687).  migrate_candidates mirrors them so it never
    # proposes a reject-guaranteed migrate (which is pure waste — under
    # A3 saturation the unfiltered policy produced ~86k apply_failed/cycle,
    # zero relief, daemon thrash).
    #   is_device_leaf — no child holds device Full-KV, unlocked, not root
    #                    ⇒ removable from HBM.
    #   is_host_leaf   — device-evicted, backuped, no children, unlocked
    #                    ⇒ removable from DRAM.  (Implies is_tree_leaf.)
    #   is_tree_leaf   — len(children) == 0 ⇒ a full-drop (post-residence
    #                    empty) is allowed.  is_device_leaf does NOT imply
    #                    this (disk-only children), so it is carried apart.
    # Default True keeps pre-#210 fixtures valid; the live dump fills them.
    is_device_leaf: bool = True
    is_host_leaf: bool = True
    is_tree_leaf: bool = True

    # DESIGN §7/§3 (S1 demote↔predictive-promote coupling): set True on the
    # caller's exclusive tail at TOOL_CALL_START, when the scheduler is ALSO
    # placing a predictive promote-back (T_start+ETA−load_back) for this unit.
    # The pair means the next reuse is served from HBM (the promote pre-stages
    # it), so the demote's value must NOT charge the reuse a DRAM/DISK load_back
    # — value_residence reads this to value the reuse at the HBM reload (0).
    # Worst case if the promote fails to land == baseline (reuse pays load_back),
    # so crediting it is the correct proactive bet, not an unsafe optimism.
    promote_pending: bool = False

    @property
    def n_bytes(self) -> int:
        """Total bytes across all (tier, subpool).  Convenience for
        callers that just need the memory footprint."""
        return sum(b for sp_dict in self.n_bytes_by_tier.values()
                   for b in sp_dict.values())

    @property
    def authoritative_tier(self) -> Tier:
        """DESIGN §7 `authoritative_tier(residence)` — the tier whose
        h_(τ, sp) is the denominator of V_u.  HBM if in residence,
        else DRAM, else DISK.

        Empty residence is a contract violation (the unit should not
        appear in `units[]` at all per DESIGN §5); raises rather than
        silently picking a tier.
        """
        for t in (Tier.HBM, Tier.DRAM, Tier.DISK):
            if t in self.residence:
                return t
        raise ValueError(
            f"ReuseUnit {self.id!r}: empty residence — "
            f"unit should have been dropped from units[]")

    def bytes_in_tier(self, tier: Tier) -> int:
        """Sum of n_bytes across subpools for a given tier."""
        return sum(self.n_bytes_by_tier.get(tier, {}).values())


@dataclass
class TierUsage:
    """DESIGN §5 `pool_usage` allocator-truth view, per-(tier, subpool).

    `pool_used` etc. are nested dicts keyed by (tier, subpool_name).
    Mirrors `/aginfer/state.pool_usage` shape exactly.  The legacy
    flat-by-tier view is derivable via the `*_total` helpers.

    `bw_free` is per-(σ, τ) directional link free bandwidth (bytes/s)
    derived from DESIGN §5 `link_stats` (peak_bw_bps - recent_throughput
    on cold links, peak on idle).
    """
    pool_used: Dict[Tier, Dict[str, int]] = field(default_factory=dict)
    pool_cap: Dict[Tier, Dict[str, int]] = field(default_factory=dict)
    pool_available: Dict[Tier, Dict[str, int]] = field(default_factory=dict)
    pool_evictable: Dict[Tier, Dict[str, int]] = field(default_factory=dict)
    page_bytes: Dict[Tier, Dict[str, int]] = field(default_factory=dict)
    # DESIGN §8 bytes_per_token_in_subpool (#199): per-token HBM byte
    # growth during decode, by subpool (attention = bpt, mamba = 0).
    # Sourced from sglang's pool_usage[*].decode_bytes_per_token; the
    # daemon's forecast_inflight_demand multiplies decode-token growth by
    # this.  Defaults to {} (≡ 0) when an older sglang omits the field.
    decode_bytes_per_token: Dict[Tier, Dict[str, int]] = field(
        default_factory=dict)
    bw_free: Dict[Tuple[Tier, Tier], float] = field(default_factory=dict)

    def occupancy_ratio(self, tier: Tier) -> float:
        """Max occupancy across subpools — DESIGN §5 'admission acts
        when ANY subpool crosses theta_hi, not when the aggregate does'.
        A Mamba snapshot pool at 95 % with attention at 60 % is the
        failure mode the aggregate view hides.
        """
        used_by_sp = self.pool_used.get(tier, {})
        cap_by_sp = self.pool_cap.get(tier, {})
        ratios = [
            used / cap_by_sp[sp]
            for sp, used in used_by_sp.items()
            if cap_by_sp.get(sp, 0) > 0
        ]
        return max(ratios, default=0.0)

    def subpool_occupancy(self, tier: Tier, subpool: str) -> float:
        cap = self.pool_cap[tier][subpool]
        return self.pool_used[tier][subpool] / cap if cap > 0 else 0.0

    def used_bytes_total(self, tier: Tier) -> int:
        return sum(self.pool_used.get(tier, {}).values())

    def cap_bytes_total(self, tier: Tier) -> int:
        return sum(self.pool_cap.get(tier, {}).values())


@dataclass
class SchedulerState:
    """Paper §3 reduced state for one decision event.

    Per DESIGN §5 admission gates per-(tier, subpool) — `pool_pressure`
    is nested accordingly.  The aggregate-by-tier view is derivable
    via `tier_usage.occupancy_ratio(tier)`.
    """
    t: float
    units: Dict[str, ReuseUnit]
    tier_usage: TierUsage
    event_kind: str
    event_session_id: Optional[str] = None
    decision_set: List[str] = field(default_factory=list)
    # Per-(tier, subpool) allocator-truth occupancy ratio.
    # admission_controller.gate keys per-subpool: ANY subpool crossing
    # theta_hi triggers a pause, not just the aggregate.
    pool_pressure: Dict[Tier, Dict[str, float]] = field(default_factory=dict)
    # DESIGN §8 program-level inputs (the admission candidate generator
    # iterates these): {pid: {"state", "pre_pause_state", "unit_hashes",
    # "hbm": {"committed": {sp: bytes}, "inflight": {sp: bytes}},
    # "dram": {"committed": {sp: bytes}}}}.  Authored by sglang's
    # /aginfer/state per_program_usage (DESIGN §5/§8); the daemon reads
    # it the same way it reads pool_usage — no tracker join.
    per_program_usage: Dict[str, Any] = field(default_factory=dict)
    # DESIGN §8 forecast inputs: {"prefill_bps": float,
    # "decode_per_program": {pid: bytes_per_sec}}.  prefill_bps and
    # decode rates are T26 measurement (ship as 0.0 placeholders
    # pre-T26); forecast degrades to the snapshot term until then.
    throughput_ema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """DESIGN §6 migrate action.

    Each assignment is `(unit_id, add_tiers, remove_tiers)` per the
    residence-set transition spec.  Empty list = no migration.

    The 6 meaningful per-unit transitions (DESIGN §7):
      {HBM}        → {HBM, DRAM}    (write_through)
      {HBM, DRAM}  → {DRAM}         (HBM eviction, host backup kept)
      {HBM, DRAM}  → {HBM}          (DRAM drop, device retained)
      {DRAM}       → {HBM, DRAM}    (load_back to device)
      {DRAM}       → {}             (DROP)
      {DISK}       → {DRAM, DISK}   (Mooncake load)
    """
    assignments: List[Tuple[str, List[Tier], List[Tier]]] = field(
        default_factory=list)


class Policy(Protocol):
    """Stateless interface: given a state + event, return an action.

    Concrete implementations may carry internal counters (LRU age table,
    learned model, etc.).
    """

    name: str

    def decide(self, state: SchedulerState) -> Action: ...

    def update_after_step(self, state: SchedulerState, action: Action,
                          hits: List[str]) -> None:
        """Optional hook for policies that need to update internal state
        (e.g. LRU recency table). Default no-op."""
        return None
