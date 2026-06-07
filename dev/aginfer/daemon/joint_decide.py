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
    take net-positive Migrate actions that RELIEVE A PRESSURED subpool,
    maximising total −cost within each destination subpool's free room
    — ``knapsack_max_value_multi``.  A hot unit (cost > 0) or a migrate
    that would relieve a healthy subpool is excluded.  The **Pause
    lever is DORMANT** (not generated): its cost misses the paused
    agent's forgone progress and its OOM-benefit is unmodelled, so it
    cannot yet be valued (§8).
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

from baselines.base import SchedulerState, Tier
from baselines.knapsack import (
    KnapsackBudgetExceededError,
    knapsack_max_value_multi,
)
from baselines.ours_greedy import migrate_candidates

from . import admission_controller as adm
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

    The **Pause lever is NOT generated** here — it is DORMANT.  A Pause's
    cost (``V_u_program + marginal_pause_cost``) misses the paused agent's
    forgone PROGRESS, and its benefit (the OOM it averts) is unmodelled, so
    a Pause cannot yet be correctly valued — and an under-costed Pause whose
    ``V_u_program`` goes negative under the holding tax would wrongly fire
    and stall an active agent (the A3 regression).  Until both the
    progress-cost and the OOM-benefit are modelled (§8), the daemon never
    pauses.  Migrate is the working relief lever.

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
        # Keep a Migrate iff (a) net-positive (cost < 0 ⇒ moving improves
        # total V_u — a COLD unit) AND (b) it relieves a subpool that is
        # ACTUALLY pressured.  (a) protects hot units, (b) keeps relief on
        # the bottleneck (never churns a healthy subpool).  Together: an
        # all-in-flight pegged subpool (swa) has no relieving migrate → the
        # relief no-ops.  Each item carries its destination ``acquired``
        # bytes as the budget it consumes.  No Pause items (dormant lever).
        # group = the unit hash (migrate_candidates always sets it) — the
        # multiple-choice exclusion of a unit's mutually-exclusive
        # transitions RELIES on this; a per-transition id would split them
        # into singletons and let the DP double-count the unit's bytes.
        items = [
            _ValueItem(gain=-float(c.cost),
                       re_use=(getattr(c, "acquired", None) or {}),
                       group=c.group,
                       src=c)
            for c in cands
            if float(c.cost) < 0.0
            and any(sp in pressured_sps for sp in c.relief.get("HBM", {}))
        ]
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
