"""DESIGN §9 — joint decision over the union action space (#194).

Each event triggers ONE decision function over
``A = {unit migrate} ∪ {program pause/resume}``.  ``joint_decide``
replaces the old sequential decompose (``OursGreedyPolicy.decide``
greedy migrate, THEN admission ``_on_pressure`` / ``_on_resolved``):
the two lever types draw on the same byte budget, so the choice
between them is made by ONE value criterion (DESIGN §9).

**Value-gated, not cover.**  The daemon only ever takes NET-POSITIVE
actions (value = −cost > 0); every phase is a value-MAXIMISING knapsack
that may pick the empty set.  There is no ``bytes_needed`` to cover and
no forced relief: ``theta_hi`` gates only WHETHER relief candidates are
generated; it is not a wall occupancy must be pushed back under.  If a
subpool is pegged but no net-positive action relieves it, the daemon
NO-OPS and lets the bottleneck be a bottleneck (do-no-harm).  The two
phases are NOT mutually exclusive — resume runs every event alongside
relief; resuming a starved-out program is itself a net-positive action.

  * **Relief** (any HBM subpool's ``forecast`` > ``theta_hi × cap``):
    take net-positive Migrate AND Pause actions that RELIEVE A PRESSURED
    subpool, maximising total value within each destination subpool's free
    room — ``knapsack_max_value_multi``.  A hot unit (cost > 0) or a migrate
    that would relieve a healthy subpool is excluded.  The **Pause lever is
    LIVE** (#260): a Pause's HBM relief is valued at the pressured subpool's
    shadow price (the eviction it averts), net of its cost — now including the
    forgone-progress term (§8) that makes pausing an actively-decoding agent
    net-negative, so only a parked low-value agent is ever paused (do-no-harm).
  * **Resume** (runs every event): take net-positive Resume actions
    that fit per-subpool at ``theta_lo`` headroom — same
    ``knapsack_max_value_multi``.  A zero-re_use resume (un-starve of a
    program whose units were dropped while gated) fits even under
    pressure.

``LLM_PREFILL`` runs ``joint_decide`` like every other event — its
``D_t`` is ∅ so ``migrate_candidates`` returns ``[]``; the entry point
is uniform (DESIGN §9) and typically the plan is empty.

The DP-blowup path maps to ``fatal`` (crash-only, DESIGN §9/§10): the
candidate set is bounded by the live unit/program count, so a knapsack
budget overflow means the generators produced more axes than the DP can
hold — an algorithm bug, not a workload reality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List

from .base import SchedulerState, Tier
from .knapsack import (
    KnapsackBudgetExceededError,
    knapsack_max_value_multi,
)
from .ours_greedy import migrate_candidates

from . import admission_controller as adm
from .events import EventKind
from ._fatal import fatal

logger = logging.getLogger(__name__)

_TIER = {"HBM": Tier.HBM, "DRAM": Tier.DRAM, "DISK": Tier.DISK}


@dataclass
class _ValueItem:
    """Adapter that lets a relief candidate (Migrate / Pause) feed the SAME
    value-maximising knapsack the Resume path uses (§9 value-gated rewrite).

      gain   = the action's NET VALUE = −cost.  A Migrate that improves
               total V_u (moving a COLD unit out of the pressured tier) has
               cost < 0 → gain > 0; a HOT unit's move and every Pause
               (cost > 0, benefit unmodelled = 0) are net-negative.
      re_use = the action's destination-tier ``acquired`` bytes — the
               (DRAM|DISK, subpool) budget it consumes; the knapsack reads
               it exactly like a Resume's HBM re_use.
      group  = the per-unit MCKP group (a unit's transitions are mutually
               exclusive alternatives); src = the original candidate, put
               back into the plan when chosen."""
    gain: float
    re_use: Dict[str, Dict[str, int]]
    group: Any
    src: Any


def _page_bytes(state: SchedulerState, tier_label: str, sp: str) -> int:
    """``page_bytes`` for axis (tier_label, sp).  A missing key is a
    schema-contract violation (DESIGN §10 subpool-key consistency) — let
    the KeyError surface rather than defaulting to a wrong granularity.

    A configured subpool reporting ``page_bytes <= 0`` is a deployment bug
    (#218): it would divide-by-zero in the DP quantisation.  Fail loud with
    a forensic dump, in the same "required positivity" family as
    ``peak_bw_bps`` / ``h_max`` / ``prefill_bps`` (DESIGN §10)."""
    pb = int(state.tier_usage.page_bytes[_TIER[tier_label]][sp])
    if pb <= 0:
        fatal("nonpositive_page_bytes", tier=tier_label, subpool=sp,
              page_bytes=pb)
    return pb


def _budget_and_buckets(state, base_budget, items):
    """Budget + bucket-size dicts for ``knapsack_max_value_multi``, covering
    EVERY axis any item actually consumes — not just the configured ones.

    The DP reads consumption only on axes present in ``budget``; an axis a
    candidate consumes but that is ABSENT from ``budget`` would be silently
    treated as 0-bytes-consumed (free), letting the DP over-subscribe a
    destination subpool that isn't mirrored across tiers.  That contradicts
    the module's fail-loud stance (cf. ``_page_bytes``).  So: start from the
    configured destination axes (``base_budget``), then add every
    ``(tier, subpool)`` appearing in any item's ``re_use`` with room =
    ``max(0, cap − used)`` — defaulting an UNCONFIGURED subpool to **0 room**
    (a unit cannot be written to a subpool that does not exist, so any
    consumption there rounds up to ≥1 bucket > 0 → the item is rejected).
    ``bucket_size`` is the per-axis ``page_bytes``; an unconfigured axis has
    no ``page_bytes``, but its room is 0 so the bucket only needs to be
    positive (1) to make round-up reject it."""
    tu = state.tier_usage
    budget = dict(base_budget)
    for it in items:
        for tier_label, d in (it.re_use or {}).items():
            tier = _TIER[tier_label]
            caps = tu.pool_cap.get(tier, {})
            used = tu.pool_used.get(tier, {})
            for sp in d:
                ax = (tier_label, sp)
                if ax not in budget:
                    budget[ax] = max(0, int(caps.get(sp, 0)) - int(used.get(sp, 0)))
    bucket = {}
    for ax in budget:
        try:
            bucket[ax] = _page_bytes(state, ax[0], ax[1])
        except KeyError:
            bucket[ax] = 1   # unconfigured ⇒ 0-room axis; bucket only needs
                             # to be positive so consumption rounds up > 0.
    return budget, bucket


def _shadow_prices(state, pressured_sps, costs, pi_u) -> Dict[str, float]:
    """#260 — marginal value (inference-time per byte) of freeing one HBM byte
    in each pressured subpool: the V_u-DENSITY of the lowest-V_u HBM-resident
    unit in that subpool — the next eviction victim that freed room spares.
    Clamped >= 0 (a marginal resident with net-negative V_u should be evicted
    anyway, so making room to keep it has no relief value).

    This is the ``OOM averted`` benefit (DESIGN §8) that lets a Pause's HBM
    relief be VALUED against Migrate in the SAME value-knapsack — the term whose
    absence (relief measured in bytes, never converted to time) was half of why
    Pause stayed dormant."""
    best: Dict[str, Any] = {}   # sp -> (lowest V_u, that unit's bytes in sp)
    for u in state.units.values():
        if Tier.HBM not in u.residence:
            continue
        hb = u.n_bytes_by_tier.get(Tier.HBM, {})
        v = None
        for sp, b in hb.items():
            if sp not in pressured_sps or int(b) <= 0:
                continue
            if v is None:
                v = adm._value_at_current_tier(u, state, costs, pi_u)
            cur = best.get(sp)
            if cur is None or v < cur[0]:
                best[sp] = (v, int(b))
    return {sp: max(0.0, vb[0] / vb[1]) for sp, vb in best.items()}


def joint_decide(
    state: SchedulerState,
    event: Any,
    *,
    costs: Any,
    pi_u: float,
    theta_hi: float,
    theta_lo: float,
    heartbeat_s: float,
    admission_enabled: bool = True,
) -> List[Any]:
    """Value-gated joint decision over {migrate, pause, resume} (DESIGN §9).

    Returns the chosen plan; an EMPTY list = no-op — every available action
    is net-negative, or there is no pressure and nothing worth resuming.
    BOTH pieces below are value-MAXIMISING knapsacks that may pick the empty
    set: there is no cover and no forced relief.  ``theta_hi`` only gates
    whether relief candidates are generated; it is not a wall the daemon
    must push occupancy back under.

    Relief (runs when any HBM subpool is pressured): take net-positive
    Migrate actions that RELIEVE A PRESSURED subpool.  value = −cost, so a
    Migrate with cost < 0 (moving a COLD unit out improves total V_u) is
    taken; a HOT unit (cost > 0) is dropped; and a Migrate that relieves a
    HEALTHY subpool is dropped too (relief targets the bottleneck, never
    churns a subpool that has room).  So a non-migratable, all-in-flight
    pressured subpool (attention ``swa``) has NO migrate that relieves it →
    relief no-ops (do-no-harm); a cold-cache-pressured subpool yields many →
    the daemon relieves it.  No per-subpool threshold tuning — V_u carries
    the per-unit decision, and the pressured-subpool filter carries the
    targeting.

    The **Pause lever is LIVE** (#260).  A Pause's HBM relief is valued at the
    pressured subpool's shadow price (``_shadow_prices`` — the V_u-density of
    the unit eviction would next claim, the ``OOM averted`` benefit), net of
    its cost = ``V_u_program + marginal_pause_cost + forgone_progress``.  The
    forgone-progress term (``admission_controller``, §8) costs the wall-time the
    held agent makes no progress — a full horizon for a REASONING agent — so a
    Pause that would stall an actively-decoding agent is net-negative and never
    chosen (the A3 regression is closed by the cost MODEL, not a state filter).
    Only net-positive pauses enter the knapsack; with no pressure or no
    low-value parked agent, none do (do-no-harm).  Gated on ``admission_enabled``.

    Resume (runs EVERY event, independent of relief — not mutually
    exclusive): take net-positive Resume actions that fit per-subpool at
    theta_lo headroom; a zero-re_use resume fits even under pressure (the
    free un-starve of a program whose units were dropped while gated).

    ``admission_enabled=False`` (Run-K kv-only ablation) drops resume:
    relief is migrate-only either way."""
    tu = state.tier_usage
    hbm_cap = tu.pool_cap.get(Tier.HBM, {})
    fc = adm.forecast(state, heartbeat_s)            # {sp: forecast bytes}
    plan: List[Any] = []

    # ---- Relief (value-gated, migrate-only, pressured-subpool-targeted):
    #      net-positive migrates that relieve a pegged subpool; may no-op.
    pressured_sps = {sp for sp, cap in hbm_cap.items()
                     if fc.get(sp, 0.0) > theta_hi * float(cap)}
    if pressured_sps:
        cands = migrate_candidates(state, state.decision_set, costs, pi_u)
        # #223: at TOOL_CALL_END the decision set is the caller's session
        # TAIL, which is REUSE-IMMINENT — the session resumes and extends it
        # next turn (DESIGN §7 decision_set table marks it a *promote*
        # candidate, "about to reuse").  Evicting it is futile: the frontier
        # leaf gains a device-KV child by apply time → sglang rejects
        # remove_hbm_not_device_leaf (a dump→apply TOCTOU), and it is against
        # the event's promote intent.  So suppress remove-HBM relief for a
        # reuse-imminent tail; genuine HBM pressure is relieved by the
        # MEMORY_PRESSURE events' cold-unit top-k, not by evicting the hot
        # frontier.  (The per-hash cooldown #223 stays as the backstop for
        # the residual same-tick race on the MEMORY_PRESSURE path.)
        # TOOL_CALL_END is the only IMPLEMENTED reuse-imminent (promote)
        # event; SUB_RETURN (§7 — parent tail becomes a promote candidate when
        # a sub-agent returns) is in DESIGN/the paper but not yet in EventKind.
        # When it lands, add it here — its decision set is likewise a
        # about-to-reuse tail that must not be evicted.
        reuse_imminent = getattr(event, "kind", None) == EventKind.TOOL_CALL_END
        # Keep a Migrate iff (a) net-positive (cost < 0 ⇒ moving improves
        # total V_u — a COLD unit) AND (b) it relieves a subpool that is
        # ACTUALLY pressured AND (c) it is not a remove-HBM on a reuse-imminent
        # tail.  (a) protects hot units, (b) keeps relief on the bottleneck
        # (never churns a healthy subpool).  Together: an all-in-flight pegged
        # subpool (swa) has no relieving migrate → the relief no-ops.  Each
        # item carries its destination ``acquired`` bytes as the budget it
        # consumes.  No Pause items (dormant lever).  group = the unit hash
        # (migrate_candidates always sets it) — the multiple-choice exclusion
        # of a unit's mutually-exclusive transitions RELIES on this; a
        # per-transition id would split them into singletons and let the DP
        # double-count the unit's bytes.
        items = [
            _ValueItem(gain=-float(c.cost),
                       re_use=(getattr(c, "acquired", None) or {}),
                       group=c.group,
                       src=c)
            for c in cands
            if float(c.cost) < 0.0
            and any(sp in pressured_sps for sp in c.relief.get("HBM", {}))
            # guard the c.id[2] unpack (consistency with scheduler_driver's id guards):
            # a live migrate_candidate's id is always (uid, add, remove); behaviour-identical
            # for valid 3-tuples, and a malformed/None id can never crash this filter.
            and not (reuse_imminent and isinstance(c.id, tuple) and len(c.id) >= 3
                     and Tier.HBM in c.id[2])
        ]
        # Pause lever — now LIVE (#260; was DORMANT).  Value each Pause's HBM
        # relief at the pressured-subpool shadow price (the eviction it averts),
        # NET of the pause cost (V_u_program + marginal re-prefill + forgone
        # progress).  do-no-harm: only NET-POSITIVE pauses enter; a REASONING
        # agent's forgone-progress term makes its pause net-negative, so it is
        # never chosen — the A3 regression is closed by the cost MODEL, not a
        # state filter.  re_use={} — a pause consumes no destination capacity
        # (it frees HBM), so it always fits; group=("pause", pid) keeps a
        # program's pause one item, independent of every unit's migrate group
        # (the radix-vs-inflight byte overlap is already removed in
        # pause_relief via ``_committed_in_dt``).  Gated on admission_enabled
        # (the Run-K kv-only ablation suppresses it).
        if admission_enabled:
            shadow = _shadow_prices(state, pressured_sps, costs, pi_u)
            for pc in adm.pause_candidates(state, heartbeat_s=heartbeat_s):
                # The relief from pausing p PERSISTS only while p stays paused —
                # the every-event Resume lever pulls it back the moment p is
                # ready.  A REASONING agent resumes at once (τ_idle=0) → its
                # relief does not persist → hold_frac 0 → gain = −cost ≤ 0
                # ROBUSTLY, independent of the relief magnitude: the benefit-side
                # half of the A3 do-no-harm guarantee (the cost-side half is
                # forgone_progress).  A parked ACTING agent persists for its tool
                # gap → hold_frac 1 (cold; a per-program ETA tightens it to
                # min(1, τ_idle / W)).
                pstate = (state.per_program_usage.get(pc.pid, {}) or {}).get("state")
                hold_frac = 0.0 if pstate == "REASONING" else 1.0
                relief_value = hold_frac * sum(
                    float(pc.relief.get(sp, 0)) * shadow.get(sp, 0.0)
                    for sp in pressured_sps)
                gain = relief_value - float(pc.cost)
                if gain > 0.0:
                    items.append(_ValueItem(
                        gain=gain, re_use={}, group=("pause", pc.pid), src=pc))
        if items:
            # Only constraint: do not overflow any destination (DRAM|DISK,
            # subpool) cap.  Clamp to >= 0 (an over-subscribed destination
            # is 0 room, not negative).  ``_budget_and_buckets`` then extends
            # this with every axis the items actually consume (0 room for an
            # unconfigured subpool) so consumption is never silently free.
            cap_left: Dict[Any, int] = {}
            for tier_label, tier in (("DRAM", Tier.DRAM), ("DISK", Tier.DISK)):
                used = tu.pool_used.get(tier, {})
                for sp, cap in tu.pool_cap.get(tier, {}).items():
                    cap_left[(tier_label, sp)] = max(
                        0, int(cap) - int(used.get(sp, 0)))
            cap_left, bucket_size = _budget_and_buckets(state, cap_left, items)
            ctx = {"event": getattr(event, "kind", event), "phase": "relief",
                   "forecast": fc, "theta_hi": theta_hi}
            try:
                chosen = knapsack_max_value_multi(
                    items, cap_left, bucket_size, context=ctx)
            except KnapsackBudgetExceededError as exc:
                fatal("joint_decide_dp_blowup", **exc.context)
                chosen = []  # unreachable (fatal never returns)
            plan += [it.src for it in chosen]

    # ---- Resume (value-gated): runs every event; net-positive fits; no-op.
    if admission_enabled and hbm_cap:
        free_room = {("HBM", sp): max(0.0, theta_lo * float(cap) - fc.get(sp, 0.0))
                     for sp, cap in hbm_cap.items()}
        rcands = adm.resume_candidates(state, heartbeat_s, theta_hi)
        rcands = [replace(c, re_use={"HBM": c.re_use}) for c in rcands]
        if rcands:
            # Cover every HBM subpool a resume re-enters, not just the
            # configured ones (an off-budget subpool = 0 room → rejected).
            free_room, rbucket = _budget_and_buckets(state, free_room, rcands)
            rctx = {"event": getattr(event, "kind", event), "phase": "resume",
                    "forecast": fc, "theta_lo": theta_lo}
            try:
                plan += knapsack_max_value_multi(
                    rcands, free_room, rbucket, context=rctx)
            except KnapsackBudgetExceededError as exc:
                fatal("joint_decide_dp_blowup", **exc.context)

    return plan
