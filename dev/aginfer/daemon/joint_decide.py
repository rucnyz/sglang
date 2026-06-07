"""DESIGN §9 — joint decision over the union action space (#194).

Each event triggers ONE decision function over
``A = {unit migrate} ∪ {program pause/resume}``.  ``joint_decide``
replaces the old sequential decompose (``OursGreedyPolicy.decide``
greedy migrate, THEN admission ``_on_pressure`` / ``_on_resolved``):
the two lever types attack different parts of HBM (radix vs in-flight)
but draw on the same byte budget, so the choice between them must be
JOINT (DESIGN §9 "Why not sequential migrate-then-pause").

Two phases, mutually exclusive per event, each an exact DP:

  * **Pressure** (any HBM subpool's ``forecast`` ≥ ``theta_hi × cap``):
    minimise total V_u cost of a {Migrate, Pause} subset that frees
    ``bytes_needed[sp]`` from every pressured HBM subpool without
    overflowing any destination subpool — ``knapsack_min_cost_multi``.
  * **Headroom** (EVERY HBM subpool's ``forecast`` < ``theta_lo ×
    cap``): maximise total V_u gain of a {Resume} subset within each
    HBM subpool's free room — ``knapsack_max_value_multi``.

In between is the hysteresis dead-zone (``return []``).  Pressure
suppresses headroom across ALL subpools (DESIGN §9): if any subpool is
pressured the headroom phase does not run, because resuming a program
grows inflight bytes in every subpool it decodes into.

``LLM_PREFILL`` runs ``joint_decide`` like every other event — its
``D_t`` is ∅ so ``migrate_candidates`` returns ``[]``, but the
admission generators still produce Pause/Resume candidates from live
state; typically the plan is empty, but the entry point is uniform
(DESIGN §9).

The infeasible / DP-blowup paths map to ``fatal`` (crash-only,
DESIGN §9/§10): a feasible pressure plan ALWAYS exists (DROP + Pause
consume no destination capacity), so infeasibility means the candidate
set was undersized — an algorithm bug, not a workload reality.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List

from baselines.base import SchedulerState, Tier
from baselines.knapsack import (
    KnapsackBudgetExceededError,
    Migrate,
    Pause,
    Resume,
    knapsack_max_value_multi,
    knapsack_min_cost_multi,
)
from baselines.ours_greedy import migrate_candidates

from . import admission_controller as adm
from ._fatal import fatal

logger = logging.getLogger(__name__)

_TIER = {"HBM": Tier.HBM, "DRAM": Tier.DRAM, "DISK": Tier.DISK}


def _has_relief(c: Any) -> bool:
    return any(b > 0 for sub in c.relief.values() for b in sub.values())


def _drop_relief_axis_acquirers(cands: List[Any], relief_axes: List[Any]) -> List[Any]:
    """Drop pressure candidates that ACQUIRE bytes into a relief (HBM)
    subpool.

    The pressure DP models only DRAM/DISK as destination-cap axes
    (``cap_left``); HBM is a relief axis, so a candidate's HBM ``acquired``
    is invisible to the cap constraint and treated as FREE.  The one
    relief-bearing migrate that acquires HBM is ``promote+drop_disk``
    ({DRAM,DISK} → {HBM,DRAM}): it relieves DISK (irrelevant to HBM
    pressure) while pulling the unit's bytes BACK into HBM.  With a negative
    cost the min-cost DP would greedily take it as free cost reduction —
    silently GROWING the very tier the pressure phase is relieving.
    Applying such a transition during HBM pressure is incoherent; exclude
    it before the knapsack runs.  (``promote+drop_disk``'s cost is provably
    ≥ 0 under the default h_HBM ≥ h_DRAM calibration, so this is normally
    inert — but the DP must not depend on a calibration invariant to stay
    correct.)"""
    pressured_subpools: Dict[str, set] = {}
    for (tier, sp) in relief_axes:
        pressured_subpools.setdefault(tier, set()).add(sp)
    out: List[Any] = []
    for c in cands:
        acq = getattr(c, "acquired", None) or {}
        if any(sp in pressured_subpools.get(tier, ())
               for tier, sub in acq.items()
               for sp, b in sub.items() if b > 0):
            continue
        out.append(c)
    return out


def _page_bytes(state: SchedulerState, tier_label: str, sp: str) -> int:
    """``page_bytes`` for axis (tier_label, sp).  A missing key is a
    schema-contract violation (DESIGN §10 subpool-key consistency) — let
    the KeyError surface rather than defaulting to a wrong granularity."""
    return int(state.tier_usage.page_bytes[_TIER[tier_label]][sp])


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
    """Return the chosen mixed plan (a list of ``Migrate`` / ``Pause`` /
    ``Resume`` candidates) for the live handler to dispatch.  Empty list
    = nothing to do this event (declined or hysteresis dead-zone).

    ``admission_enabled=False`` drops the program-level Pause/Resume
    levers: the pressure phase becomes migrate-only and the headroom
    phase is skipped (nothing to resume).  This is the kv-only ablation
    arm (Run K) — the union action space collapses to {migrate}."""
    tu = state.tier_usage
    hbm_cap = tu.pool_cap.get(Tier.HBM, {})
    fc = adm.forecast(state, heartbeat_s)            # {sp: forecast bytes}

    def _resume_plan() -> List[Any]:
        """#213: a paused program resumes as soon as IT fits, per-subpool —
        decoupled from pressure on OTHER subpools.  ``free_room`` is the
        theta_lo headroom (hysteresis margin) clamped to >=0; a resume with
        zero re_use on a tight/pegged subpool has zero weight THERE and so
        fits even under pressure (the un-starve path for dropped-units
        programs).  Run EVERY event — not just when all subpools have cooled.

        Why: the pressure and resume phases used to be mutually exclusive
        (pressure ran whenever ANY subpool crossed theta_hi, resume only in
        the else).  Under a permanently-pegged subpool (A3 swa ~0.99) the
        pressure phase is always active, so resume never fired → the daemon
        paused monotonically and agents starved.  Running resume alongside
        pressure makes admission a feedback controller: pause the heavy
        programs, run what fits, and resume paused programs as running ones
        finish and free capacity.  capacity_fits (theta_hi, inside
        resume_candidates) already keeps a resume from re-pressuring a tight
        subpool, so a resumed program never worsens the pegged axis."""
        if not (admission_enabled and hbm_cap):
            return []
        free_room = {
            ("HBM", sp): max(0.0, theta_lo * float(cap) - fc.get(sp, 0.0))
            for sp, cap in hbm_cap.items()
        }
        rcands = adm.resume_candidates(state, heartbeat_s, theta_hi)
        rcands = [replace(c, re_use={"HBM": c.re_use}) for c in rcands]
        if not rcands:
            return []
        rbucket = {axis: _page_bytes(state, axis[0], axis[1])
                   for axis in free_room}
        rctx = {"event": getattr(event, "kind", event), "phase": "headroom",
                "forecast": fc, "theta_lo": theta_lo}
        try:
            return knapsack_max_value_multi(
                rcands, free_room, rbucket, context=rctx)
        except KnapsackBudgetExceededError as exc:
            fatal("joint_decide_dp_blowup", **exc.context)
            return []

    # ---- Pressure phase ----
    bytes_needed = {
        ("HBM", sp): max(0.0, fc.get(sp, 0.0) - theta_hi * float(cap))
        for sp, cap in hbm_cap.items()
    }
    if any(b > 0 for b in bytes_needed.values()):
        # Destination capacity: every DRAM + DISK subpool (≤ axes).
        # Clamp to ≥ 0 — a negative cap_left means the destination tier
        # is already over-subscribed (no room), which is 0 room, never
        # "less than zero".  Without the clamp a negative budget rejects
        # even zero-acquire DROP candidates (the DP's cap accumulator
        # starts at 0 > a negative Wcap) → spurious infeasibility (#194,
        # caught by integration_stress: DRAM at -32 GB cap_left).
        cap_left: Dict[Any, int] = {}
        for tier_label, tier in (("DRAM", Tier.DRAM), ("DISK", Tier.DISK)):
            used = tu.pool_used.get(tier, {})
            for sp, cap in tu.pool_cap.get(tier, {}).items():
                cap_left[(tier_label, sp)] = max(
                    0, int(cap) - int(used.get(sp, 0)))

        cands: List[Any] = migrate_candidates(state, state.decision_set,
                                              costs, pi_u)
        if admission_enabled:
            cands += adm.pause_candidates(state, heartbeat_s=heartbeat_s)
        # Normalise Pause's flat {sp: bytes} relief into the nested
        # {"HBM": {...}} shape Migrate already uses (DESIGN §9).
        cands = [replace(c, relief={"HBM": c.relief}) if isinstance(c, Pause)
                 else c for c in cands]
        cands = [c for c in cands if _has_relief(c)]
        # A candidate that ACQUIRES into a relief (HBM) subpool would GROW
        # the pressured tier — and the DP's cap model (DRAM/DISK only) can't
        # see it.  Exclude such candidates (e.g. promote+drop_disk) before
        # the knapsack so they can never be selected as "free" relief.
        cands = _drop_relief_axis_acquirers(cands, list(bytes_needed.keys()))

        bucket_size = {
            axis: _page_bytes(state, axis[0], axis[1])
            for axis in (list(bytes_needed) + list(cap_left))
        }
        ctx = {
            "event": getattr(event, "kind", event),
            "phase": "pressure",
            "forecast": fc,
            "theta_hi": theta_hi,
        }
        # best_effort: free as much as the available candidates allow.
        # Pressure the migrate+pause levers can't fully relieve (in-flight
        # -dominated occupancy with no Pause candidate available, or a D_t
        # too small to cover bytes_needed) is a WORKLOAD REALITY (#194,
        # contra DESIGN §9's "infeasible = always a bug"): under-freeing +
        # re-evaluating next event (and sglang's own eviction backstop) is
        # correct — crashing the daemon on transient over-pressure is not.
        # A genuine misconfiguration still fails loud via the DP-blowup
        # ceiling below.
        try:
            plan = knapsack_min_cost_multi(
                cands, bytes_needed, cap_left, bucket_size,
                context=ctx, best_effort=True)
        except KnapsackBudgetExceededError as exc:
            fatal("joint_decide_dp_blowup", **exc.context)
            return []  # unreachable (fatal never returns)
        # Observe the shortfall when best-effort couldn't fully relieve.
        freed = {}
        for c in plan:
            for sp, b in c.relief.get("HBM", {}).items():
                freed[sp] = freed.get(sp, 0) + b
        for (_, sp), need in bytes_needed.items():
            if need > 0 and freed.get(sp, 0) < need:
                logger.warning(
                    "joint_decide pressure under-relieved HBM/%s: freed "
                    "%d of %d needed bytes (%d candidates, admission=%s); "
                    "re-evaluating next event",
                    sp, freed.get(sp, 0), int(need), len(cands),
                    admission_enabled,
                )
        # #213: resume runs ALONGSIDE pressure now (no longer suppressed).
        # The pressure plan pauses heavy programs / migrates KV; the resume
        # plan admits paused programs that fit per-subpool (zero weight on the
        # pegged subpool).  Different programs, independent levers — together
        # they make admission a feedback controller instead of monotonic
        # pausing.  (#211 zero-re_use Resumes are kept — a free un-starve.)
        return plan + _resume_plan()

    # ---- No pressure on any subpool: resume what fits (cool-down). ----
    return _resume_plan()
