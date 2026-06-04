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
import math
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


def expected_remaining_tokens(
    pid: str, state: SchedulerState
) -> Optional[float]:
    """DESIGN §8 ``E[remaining_tokens(p)]`` — expected residual decode
    length for program ``pid``, conditional on its observable state.

    Estimator priority (§8): (1) event-payload hint, (2) per-turn decode
    history, (3) workload-prior fit, (4) bootstrap = ``max_completion_
    tokens`` (cold-start only).  Levels 1-3 are the T11 estimator (#126);
    none are wired yet, and the dump carries no ``max_completion_tokens``
    for the bootstrap either.  We read an OPTIONAL per-program field
    ``expected_remaining_tokens`` (which T11 / a future sglang dump will
    populate) and return ``None`` when it is absent.

    Returning ``None`` — rather than falling back to ``max_completion_
    tokens`` — is deliberate: §8 warns that using the bootstrap ``max`` as
    the steady-state rule over-forecasts by ~5× (``max/mean`` on agent
    workloads) and makes admission over-pause.  So with no real estimate,
    :func:`_program_inflight_growth` contributes 0 for that program rather
    than projecting a worst-case decode.  The trajectory term thus stays
    off until a genuine ``E[remaining]`` source exists.
    """
    pu = state.per_program_usage.get(pid, {})
    v = pu.get("expected_remaining_tokens")
    if v is None:
        return None
    # Defensive coercion (#199 audit): a malformed value (NaN / inf /
    # negative / non-numeric) must NOT slip past the downstream guards
    # (``nan <= 0`` is False) and poison the forecast or crash
    # ``int(nan)`` in pause_relief.  Treat any non-finite-or-negative as
    # "no estimate" → None (the program is skipped, never bootstrapped).
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 0.0:
        return None
    return f


def _program_inflight_growth(
    pid: str,
    pu: Dict[str, Any],
    state: SchedulerState,
    decode_per_program: Dict[str, Any],
    dbpt: Dict[str, int],
    horizon_s: float,
) -> Dict[str, float]:
    """DESIGN §8 per-program near-term HBM growth (the term
    :func:`forecast_inflight_demand` sums and ``pause_relief``'s
    ``future_inflight_savings`` reuses):

      growth[sp] = min(E[remaining], horizon × decode_throughput)
                 × bytes_per_token_in_subpool[sp]    for sp with inflight[sp] > 0

    Gated per input — the term is 0 (sp omitted) unless ALL hold:
      * ``decode_throughput(p) > 0`` (T26 measurement),
      * ``E[remaining_tokens(p))`` is a real estimate, not None (T11),
      * ``inflight[sp] > 0`` (p is actively decoding in sp), and
      * ``decode_bytes_per_token[sp] > 0`` (attention subpool; Mamba = 0,
        so its snapshot state is never projected as per-token growth).
    Under the current placeholders (decode empty, no E[remaining]) every
    program returns ``{}`` and the trajectory term vanishes — forecast
    degrades to ``used_bytes``.
    """
    # Defensive coercion (#199 audit): a malformed decode rate (NaN / inf
    # / negative / non-numeric) must not pass the ``<= 0`` guard
    # (``nan <= 0`` is False) and produce a NaN demand.
    try:
        dt = float(decode_per_program.get(pid, 0.0))
    except (TypeError, ValueError):
        return {}
    if not math.isfinite(dt) or dt <= 0.0:
        return {}
    e_rem = expected_remaining_tokens(pid, state)  # already finite or None
    if e_rem is None:
        return {}
    growth_tokens = min(e_rem, horizon_s * dt)
    if not math.isfinite(growth_tokens) or growth_tokens <= 0.0:
        return {}
    inflight = pu.get("hbm", {}).get("inflight", {})
    out: Dict[str, float] = {}
    for sp, b in inflight.items():
        if int(b) <= 0:
            continue
        bpt = int(dbpt.get(sp, 0))
        if bpt <= 0:
            continue
        out[sp] = growth_tokens * bpt
    return out


def forecast_inflight_demand(
    state: SchedulerState, horizon_s: float
) -> Dict[str, float]:
    """DESIGN §8 ``forecast_inflight_demand`` — per-HBM-subpool expected
    byte growth before the next event, summed over programs actively
    decoding in that subpool:

      ``Σ_p min(E[remaining_tokens(p)], horizon × decode_throughput(p))
            × bytes_per_token_in_subpool(p, sp)``    for p with inflight[sp]>0

    Fully assembled from `_program_inflight_growth`; see its docstring for
    the per-input gating.  Returns ``{}`` (≡ 0 per subpool) under the
    current placeholders, so :func:`forecast` degrades to ``used_bytes``
    — behaviour-preserving until T26 (decode throughput + per-program
    inflight) and T11 (E[remaining]) wire the live inputs (#199 / #126).
    """
    decode = state.throughput_ema.get("decode_per_program", {}) or {}
    if not decode:
        return {}
    dbpt = state.tier_usage.decode_bytes_per_token.get(Tier.HBM, {})
    demand: Dict[str, float] = {}
    # No program-STATE filter — DESIGN §8 is explicit that the physical
    # signal "is subpool sp growing for p RIGHT NOW" is answered by
    # ``inflight[sp] > 0`` (inside `_program_inflight_growth`), NOT by
    # ``state == REASONING`` (a just-resumed REASONING program with no
    # in-flight request must not be forecast).  A PAUSED/ENDED program is
    # off-GPU, so sglang reports ``inflight[sp] == 0`` for it and it
    # contributes nothing here — the inflight signal is authoritative.
    for pid, pu in state.per_program_usage.items():
        for sp, g in _program_inflight_growth(
            pid, pu, state, decode, dbpt, horizon_s
        ).items():
            demand[sp] = demand.get(sp, 0.0) + g
    return demand


def forecast(state: SchedulerState, heartbeat_s: float) -> Dict[str, float]:
    """DESIGN §8 ``forecast(state)`` — per-HBM-subpool predicted bytes at
    the next event if no action is taken:
    ``pool_usage.HBM.subpools[sp].used_bytes + forecast_inflight_demand[sp]``.

    The inflight (trajectory) term is 0 while its inputs are sglang
    placeholders (decode throughput / per-program inflight unmeasured),
    so today ``forecast[sp] == used_bytes[sp]`` and §9's pressure /
    headroom triggers reduce exactly to allocator-truth HBM occupancy —
    behaviour-preserving, and trajectory-aware once T26/T11 wire the
    inputs (#199 / #200 / #126).
    """
    horizon = forecast_horizon(state, heartbeat_s)
    demand = forecast_inflight_demand(state, horizon)
    used = state.tier_usage.pool_used.get(Tier.HBM, {})
    # Iterate the UNION (#199 audit): a demand subpool absent from
    # pool_used would otherwise be silently dropped before §9 ever sees
    # it (used_bytes defaults to 0 for such a subpool).
    return {sp: float(used.get(sp, 0)) + float(demand.get(sp, 0.0))
            for sp in (set(used) | set(demand))}


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


def _committed_in_dt(
    pu: Dict[str, Any], state: SchedulerState
) -> Dict[str, int]:
    """Per-HBM-subpool committed (radix) bytes of p's units that are ALSO
    in this event's decision_set (D_t) — the bytes the MIGRATE lever will
    free (holistic-review #2).

    Sized to match sglang's ``committed`` construction (``unit_hbm_bytes
    // n_holders``) so subtracting it from ``pause_relief`` leaves the two
    levers freeing physically-disjoint HBM (DESIGN §9 'radix vs
    in-flight'): without this, a ``Migrate(u)`` and a ``Pause(p∋u)`` in
    the same pressure plan both count u's bytes, the DP over-estimates
    relief and under-frees.  A D_t unit that the DP does NOT migrate is
    simply not credited to the pause either (conservative under-free,
    recovered next event) — never double-counted."""
    dt = set(state.decision_set)
    out: Dict[str, int] = {}
    for h in pu.get("unit_hashes", []):
        if h not in dt:
            continue
        u = state.units.get(h)
        if u is None:
            continue
        n_holders = max(1, len(u.holders))
        for sp, b in u.n_bytes_by_tier.get(Tier.HBM, {}).items():
            out[sp] = out.get(sp, 0) + int(b) // n_holders
    return out


def pause_relief(
    pu: Dict[str, Any],
    future_inflight_savings: Optional[Dict[str, float]] = None,
    exclude_committed: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """DESIGN §8 ``pause_relief`` — per-HBM-subpool bytes pausing p frees:
    ``snapshot_relief[sp] + future_inflight_savings[sp]`` where
    ``snapshot_relief = inflight[sp] + committed[sp]`` (p's in-flight
    decode bytes + its exclusive radix share).

    ``future_inflight_savings`` is the near-term decode growth pausing p
    averts — the same per-program term :func:`forecast_inflight_demand`
    sums (computed by :func:`_program_inflight_growth`; 0 under the
    current placeholders).  ``pause_candidates`` computes it and passes
    it in; absent ⇒ snapshot relief only.

    ``exclude_committed`` (holistic-review #2) is the committed (radix)
    bytes of p's units that are in this event's D_t — subtracted so the
    pause and the migrate lever free disjoint HBM (no double-count).
    See :func:`_committed_in_dt`."""
    inflight = _program_inflight(pu)
    committed = _program_committed(pu)
    fut = future_inflight_savings or {}
    excl = exclude_committed or {}
    relief: Dict[str, int] = {}
    for sp in set(inflight) | set(committed) | set(fut):
        c = max(0, committed.get(sp, 0) - int(excl.get(sp, 0)))
        v = inflight.get(sp, 0) + c + int(fut.get(sp, 0.0))
        if v > 0:
            relief[sp] = v
    return relief


def pause_candidates(
    state: SchedulerState,
    prefill_bps: Optional[float] = None,
    heartbeat_s: float = 5.0,
) -> List[Any]:
    """DESIGN §8 ``pause_candidates`` — one ``Pause`` per REASONING /
    ACTING program (ENDED + PAUSED skipped).

      cost   = V_u_program(p) + marginal_pause_cost(p)     # work-loss (s)
      relief = snapshot_relief(p) + future_inflight_savings(p)   # HBM bytes

    ``relief`` is the flat ``{sp: bytes}`` shape; ``joint_decide``
    normalises it to ``{"HBM": {...}}`` before the DP (DESIGN §9).
    ``future_inflight_savings`` (the near-term decode growth pausing p
    averts) is 0 under the current placeholders, so ``relief`` is the
    snapshot ``inflight + committed`` until T26/T11 land (#199).
    ``V_u_program`` uses the shared-aware aggregate (each unit's V_u
    split across holders) — the interim attribution under §7's binary
    p_hat; DESIGN's no-weight form is correct once T11's holder-product
    conditional p_hat lands (#126).
    """
    from baselines.knapsack import Pause
    if prefill_bps is None:
        prefill_bps = float(state.throughput_ema.get("prefill_bps", 0.0))
    horizon = forecast_horizon(state, heartbeat_s)
    decode = state.throughput_ema.get("decode_per_program", {}) or {}
    dbpt = state.tier_usage.decode_bytes_per_token.get(Tier.HBM, {})
    vprog = shared_aware_prog_scores(state)
    out: List[Any] = []
    for pid, pu in state.per_program_usage.items():
        if pu.get("state") not in _ACTIVE_STATES:
            continue
        fut = _program_inflight_growth(
            pid, pu, state, decode, dbpt, horizon)
        # #2: drop the committed bytes of p's units that the migrate lever
        # owns this event (units in D_t), so Migrate∩Pause don't double-
        # count the same radix bytes.
        excl = _committed_in_dt(pu, state)
        relief = pause_relief(pu, fut, excl)
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

