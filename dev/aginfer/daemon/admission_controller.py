"""Aginfer admission — the program-level candidate generator (DESIGN §8).

Admission does NOT run its own decision loop.  Per DESIGN §9 (#194)
the daemon makes ONE joint decision per event over the union action
space ``{unit migrate} ∪ {program pause/resume}``; admission's job is
to produce the program-level half of the candidate set that
``joint_decide`` (``daemon/joint_decide.py``) consumes:

  * ``pause_candidates(state)``  — one ``Pause(cost, relief)`` per
    REASONING / ACTING program (work-loss cost + HBM relief).
  * ``resume_candidates(state)`` — one ``Resume(gain, re_use)`` per
    PAUSED program that ``capacity_fits``.
  * ``forecast(state)`` — per-HBM-subpool predicted bytes at the next
    event (the §9 pressure / headroom trigger input).

The old event-driven ``_on_pressure`` / ``_on_resolved`` pause loops
(the "Gauss-Seidel decompose" that ran admission as a separate handler
composed on top of kv_scheduler) are superseded by ``joint_decide``
and were removed in #194.  Thresholds (theta_hi / theta_lo) live on the
EventRouter (the single source of truth, T22 / §10); the paused set is
read from sglang's ``per_program_usage`` state each event — no daemon
FIFO, so a restart loses no admission bookkeeping.

All inputs come from sglang's ``/aginfer/state`` (``per_program_usage``
+ ``pool_usage`` + ``throughput_ema``), read the same way kv_scheduler
reads ``pool_usage`` — no tracker join.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from baselines.base import ReuseUnit, SchedulerState, Tier
from baselines.costs import default_costs
from baselines.ours_greedy import (
    holding_unit_cost,
    reload_cost,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- scoring


def _value_at_current_tier(
    u: ReuseUnit, state: SchedulerState, costs, pi_u: float
) -> float:
    """Paper §7 ``V_u(tau)`` evaluated at the unit's CURRENT tier.

    The steady-state "keep value" — saved-prefill minus the holding
    tax over the expected reuse interval.  Matches
    ``OursGreedyPolicy._value`` line for line (B1 audit fix: round-1
    passed ``used=0, cap=0`` to ``holding_unit_cost`` which short-
    circuited to 0, silently dropping the holding-tax term and
    making admission's score = ``p_hat * saved_prefill`` only).
    """
    tier = u.authoritative_tier
    save_prefill = u.p_hat * (
        reload_cost(u, Tier.DROP, costs, pi_u)
        - reload_cost(u, tier, costs, pi_u)
    )
    # Max-over-subpools occupancy at the authoritative tier; matches
    # OursGreedyPolicy._value (post-T33 phase 2) and DESIGN §5
    # 'admission acts when ANY subpool crosses theta_hi'.
    occ = state.tier_usage.occupancy_ratio(tier)
    h = holding_unit_cost(tier, occ, costs)
    hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6
    return save_prefill - h * u.n_bytes * hold_time


def shared_aware_prog_scores(
    state: SchedulerState,
    pi_u: float = 1.0e-4,
) -> Dict[str, float]:
    """Compute the per-program aggregate value with shared-aware
    division: each unit's V_u is split across its holders.

    Returns ``{program_id: aggregate_score}`` for every program that
    holds at least one unit.  A program scoring LOW is a good pause
    candidate (its KV footprint contributes little value relative to
    its byte cost).
    """
    costs = default_costs()
    scores: Dict[str, float] = {}
    for u in state.units.values():
        if not u.holders:
            # An "unowned" unit (e.g. system prefix with no session
            # tags yet) doesn't contribute to any program's score.
            continue
        v = _value_at_current_tier(u, state, costs, pi_u)
        share = v / len(u.holders)
        for sid in u.holders:
            scores[sid] = scores.get(sid, 0.0) + share
    return scores


# ----------------------------------------------------------- §8 forecast


# Active program states that emit a Pause candidate (DESIGN §8:
# "one Pause for every REASONING or ACTING program"; ENDED + PAUSED
# skipped).
_ACTIVE_STATES = ("REASONING", "ACTING")


def forecast_horizon(state: SchedulerState, heartbeat_s: float) -> float:
    """DESIGN §8 ``forecast_horizon`` — expected seconds to the next
    event: ``min(heartbeat_s, 1 / recent_event_rate)``.

    ``recent_event_rate`` is not yet tracked by the daemon (it needs an
    event-timestamp EMA — a T26-adjacent measurement), so this returns
    the cold-start value ``heartbeat_s`` (the next *guaranteed* event
    arrival is sglang's heartbeat).  That is the correct upper bound;
    once the event-rate EMA lands the horizon shrinks under load.
    """
    return float(heartbeat_s)


def forecast_inflight_demand(
    state: SchedulerState, horizon_s: float
) -> Dict[str, float]:
    """DESIGN §8 ``forecast_inflight_demand`` — per-HBM-subpool expected
    byte growth before the next event, summed over programs actively
    decoding in that subpool.

    The formula is
      ``Σ_p min(E[remaining_tokens(p)], horizon × decode_throughput(p))
            × bytes_per_token_in_subpool(p, sp)``
    over ``p`` with ``inflight[sp] > 0``.  All THREE inputs are
    currently unwired:
      * ``decode_throughput(p)`` — T26 measurement (``decode_per_program``
        ships empty pre-T26),
      * ``E[remaining_tokens(p)]`` — the T11 estimator (#126),
      * ``bytes_per_token_in_subpool(p, sp)`` — a model-architecture
        constant not exposed in ``/aginfer/state``.
    With any of them absent the term is 0, so ``forecast`` degrades to
    the snapshot ``used_bytes`` (see :func:`forecast`).  This is the
    honest cold-start state — the trajectory term activates when T26 +
    T11 + the architecture constant are wired (tracked as a #194
    follow-on).  Returns an empty dict (≡ 0 per subpool).
    """
    decode = state.throughput_ema.get("decode_per_program", {}) or {}
    if not decode:
        return {}
    # decode_per_program is populated but bytes_per_token_in_subpool /
    # E[remaining_tokens] are still unavailable — we cannot complete the
    # product, so the term remains 0.  Kept as an explicit branch so the
    # wiring point is visible when those inputs land.
    return {}


def forecast(state: SchedulerState, heartbeat_s: float) -> Dict[str, float]:
    """DESIGN §8 ``forecast(state)`` — per-HBM-subpool predicted bytes at
    the next event if no action is taken:
    ``pool_usage.HBM.subpools[sp].used_bytes + forecast_inflight_demand[sp]``.

    The inflight term is 0 under the current schema (see
    :func:`forecast_inflight_demand`), so today ``forecast[sp] ==
    used_bytes[sp]`` and §9's pressure/headroom triggers reduce exactly
    to the allocator-truth HBM occupancy the admission loop used before
    the joint rewrite — behaviour-preserving, and trajectory-aware once
    the inflight inputs are wired.
    """
    horizon = forecast_horizon(state, heartbeat_s)
    demand = forecast_inflight_demand(state, horizon)
    used = state.tier_usage.pool_used.get(Tier.HBM, {})
    return {sp: float(used_bytes) + float(demand.get(sp, 0.0))
            for sp, used_bytes in used.items()}


# ----------------------------------------------- §8 program candidates


def _program_inflight(pu: Dict[str, Any]) -> Dict[str, int]:
    return {sp: int(b) for sp, b in pu.get("hbm", {}).get("inflight", {}).items()}


def _program_committed(pu: Dict[str, Any]) -> Dict[str, int]:
    return {sp: int(b) for sp, b in pu.get("hbm", {}).get("committed", {}).items()}


def marginal_pause_cost(pu: Dict[str, Any], prefill_bps: float) -> float:
    """DESIGN §8 ``marginal_pause_cost`` — work lost pausing p NOW vs at
    its next natural off-GPU boundary: p's in-flight decoded-so-far
    bytes (summed across HBM subpools) re-prefilled on resume, divided
    by the prefill throughput.

    ``prefill_bps`` is T26 measurement (ships 0.0 pre-T26); a
    non-positive rate means "no measurement yet", so the term is 0 and
    Pause.cost reduces to the snapshot ``V_u_program`` — the same
    interim degradation as :func:`forecast`.
    """
    if prefill_bps <= 0.0:
        return 0.0
    inflight_bytes = sum(_program_inflight(pu).values())
    return inflight_bytes / prefill_bps


def pause_relief(pu: Dict[str, Any]) -> Dict[str, int]:
    """DESIGN §8 ``pause_relief`` — per-HBM-subpool bytes pausing p frees:
    ``snapshot_relief[sp] + future_inflight_savings[sp]`` where
    ``snapshot_relief = inflight[sp] + committed[sp]`` (p's in-flight
    decode bytes + its exclusive radix share) and
    ``future_inflight_savings`` mirrors :func:`forecast_inflight_demand`
    (0 under the current schema)."""
    inflight = _program_inflight(pu)
    committed = _program_committed(pu)
    relief: Dict[str, int] = {}
    for sp in set(inflight) | set(committed):
        v = inflight.get(sp, 0) + committed.get(sp, 0)
        if v > 0:
            relief[sp] = v
    return relief


def pause_candidates(
    state: SchedulerState,
    prefill_bps: Optional[float] = None,
) -> List[Any]:
    """DESIGN §8 ``pause_candidates`` — one ``Pause`` per REASONING /
    ACTING program (ENDED + PAUSED skipped).

      cost   = V_u_program(p) + marginal_pause_cost(p)     # work-loss (s)
      relief = pause_relief(p)                             # HBM bytes (flat sp dict)

    ``relief`` is the flat ``{sp: bytes}`` shape; ``joint_decide``
    normalises it to ``{"HBM": {...}}`` before the DP (DESIGN §9).
    ``V_u_program`` uses the shared-aware aggregate (each unit's V_u
    split across holders) — the interim attribution under §7's binary
    p_hat; DESIGN's no-weight form is correct once T11's holder-product
    conditional p_hat lands (#126).
    """
    from baselines.knapsack import Pause
    if prefill_bps is None:
        prefill_bps = float(state.throughput_ema.get("prefill_bps", 0.0))
    vprog = shared_aware_prog_scores(state)
    out: List[Any] = []
    for pid, pu in state.per_program_usage.items():
        if pu.get("state") not in _ACTIVE_STATES:
            continue
        relief = pause_relief(pu)
        cost = vprog.get(pid, 0.0) + marginal_pause_cost(pu, prefill_bps)
        out.append(Pause(cost=cost, relief=relief, pid=pid))
    return out


def capacity_fits(
    forecast_dict: Dict[str, float],
    re_use: Dict[str, int],
    hbm_subpools: Dict[str, Dict[str, int]],
    theta_hi: float,
) -> bool:
    """DESIGN §8 ``capacity_fits`` — true iff for EVERY HBM subpool:
    ``forecast[sp] + re_use[sp] ≤ theta_hi × cap[sp]``."""
    for sp, fields in hbm_subpools.items():
        cap = float(fields["cap_bytes"])
        proj = forecast_dict.get(sp, 0.0) + float(re_use.get(sp, 0))
        if proj > theta_hi * cap:
            return False
    return True


def resume_candidates(
    state: SchedulerState,
    heartbeat_s: float,
    theta_hi: float,
) -> List[Any]:
    """DESIGN §8 ``resume_candidates`` — one ``Resume`` per PAUSED program
    that passes :func:`capacity_fits`.

      gain   = V_u_program_if_active(p, pre_pause_state)   # counterfactual (s)
      re_use = expected_peak_hbm_after_resume(p)           # HBM bytes (flat sp dict)

    The counterfactual ``gain`` overrides p's state to its
    ``pre_pause_state``; under §7's binary p_hat a PAUSED holder already
    counts as alive (only ENDED zeroes p_hat in ``build_paper_state``),
    so the override is a no-op today and ``gain`` is the same shared-aware
    aggregate as the Pause cost.  It becomes a true counterfactual once
    T11's conditional p_hat zeroes paused holders' contribution (#126).
    ``re_use`` is the flat ``{sp: bytes}`` shape; ``joint_decide``
    normalises it to ``{"HBM": {...}}``.
    """
    from baselines.knapsack import Resume
    from ._admission_math import expected_peak_hbm_after_resume
    hbm_subpools = state.tier_usage.pool_cap.get(Tier.HBM, {})
    # cap dict in the {sp: {"cap_bytes": ...}} shape capacity_fits wants.
    cap_view = {sp: {"cap_bytes": cap} for sp, cap in hbm_subpools.items()}
    fc = forecast(state, heartbeat_s)
    vprog = shared_aware_prog_scores(state)
    out: List[Any] = []
    for pid, pu in state.per_program_usage.items():
        if pu.get("state") != "PAUSED":
            continue
        re_use = expected_peak_hbm_after_resume(
            pu.get("unit_hashes", []), state.units)
        if not capacity_fits(fc, re_use, cap_view, theta_hi):
            continue
        out.append(Resume(gain=vprog.get(pid, 0.0), re_use=re_use, pid=pid))
    return out

