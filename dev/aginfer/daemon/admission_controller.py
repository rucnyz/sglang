"""Aginfer daemon admission_controller (T8).

Event-driven program-level pause/resume for back-pressure.  Reacts
to sglang's webhook events (``memory_pressure`` / ``pressure_resolved``);
**no periodic timer**.

For each ``memory_pressure`` event:

1. Fetch a fresh ``/aginfer/state``.
2. If ``HBM_occ < theta_hi``: do nothing (the watermark is gone).
3. Score every program holding HBM-resident units by **shared-aware
   aggregate V_u** (paper §7 unit value divided by holder count, so
   the platform / tool_def prefix doesn't double-count across
   programs that share it).
4. Pause the program with the LOWEST aggregate score via
   ``program_tracker.pause(pid)``.
5. Re-poll ``/aginfer/state`` to see if the migrate side (T7 +
   sglang's eviction) already cleared the pressure.  If not, loop;
   re-score and pause again.  Bound the iteration at ``max_pauses``
   per event so a single tick can't pause the whole world.

For each ``pressure_resolved`` event:

1. Resume programs in FIFO order (oldest pause first) ONE at a time.
2. After each resume, re-poll state; if HBM occ would cross
   ``theta_hi`` again, stop (hysteresis).
3. No timer / no sleep — the next ``pressure_resolved`` event will
   resume the next program.

Composition with kv_scheduler (T7): the EventRouter's MEMORY_PRESSURE
and PRESSURE_RESOLVED handlers are WRAPPED.  T7's kv_scheduler.handle
runs first (issues the migrate POST), then T8's admission.handle
runs (re-checks occ after the migrate landed and pauses if needed).

Design constraints (verify/t8/README.md):

* No ``asyncio.sleep`` / ``time.sleep`` / ``loop.call_later`` in this
  module.  Pure reactive.
* Per-event handler wall time: < 10 ms at 32 programs.  The
  ``prog_score`` aggregation builds a single program→units index in
  O(N) and iterates programs in O(P log P) (sort).
* FIFO state lives on the controller instance, not module-global, so
  a daemon restart resets cleanly (same pattern as
  ``KvScheduler._unknown_tier_log``).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from baselines.base import ReuseUnit, SchedulerState, Tier
from baselines.costs import default_costs
from baselines.ours_greedy import (
    OursGreedyPolicy,
    holding_unit_cost,
    reload_cost,
)

from .events import Event, EventBus, EventKind
from .kv_scheduler import _env_float, build_paper_state
from .program_tracker import ProgramTracker, State

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- calibration

# Hysteresis watermarks.  theta_hi is the pause trigger; theta_lo is
# the resume trigger so we don't ping-pong on a single occupancy
# fluctuation.  Default 0.85 / 0.70 (16 pp gap = ~1.3 GiB on a B300
# HBM tier of ~8 GiB usable cache).
_DEFAULT_THETA_HI = _env_float("AGINFER_ADMISSION_THETA_HI", "0.85")
_DEFAULT_THETA_LO = _env_float("AGINFER_ADMISSION_THETA_LO", "0.70")

# Max pauses per single memory_pressure event.  Without a cap, a
# single oscillation could pause every program in sight.  16 is
# enough to react to a real burst (e.g., 32 programs and the lowest
# half are clearly idle) without runaway.
_DEFAULT_MAX_PAUSES_PER_EVENT = int(
    os.environ.get("AGINFER_ADMISSION_MAX_PAUSES_PER_EVENT", "16")
)


# ----------------------------------------------------------------- scoring


def _value_at_current_tier(u: ReuseUnit, costs, pi_u: float) -> float:
    """Paper §7 ``V_u(tau)`` evaluated at the unit's CURRENT tier.

    This is the steady-state "keep value" — saved-prefill minus the
    holding tax over the expected reuse interval.  Used in the
    program score as the per-unit contribution before shared-aware
    division.
    """
    save_prefill = u.p_hat * (
        reload_cost(u, Tier.DROP, costs, pi_u)
        - reload_cost(u, u.tier, costs, pi_u)
    )
    h = holding_unit_cost(u.tier, 0, 0, costs)  # tier_usage unused at unit scope
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
        v = _value_at_current_tier(u, costs, pi_u)
        share = v / len(u.holders)
        for sid in u.holders:
            scores[sid] = scores.get(sid, 0.0) + share
    return scores


# ----------------------------------------------------------------- controller


class AdmissionController:
    """Per-instance admission controller.

    State (telemetry + FIFO) lives on the instance so a daemon restart
    resets cleanly.  No module-level mutable state.
    """

    def __init__(
        self,
        *,
        tracker: ProgramTracker,
        theta_hi: float = _DEFAULT_THETA_HI,
        theta_lo: float = _DEFAULT_THETA_LO,
        max_pauses_per_event: int = _DEFAULT_MAX_PAUSES_PER_EVENT,
    ) -> None:
        if not 0.0 < theta_lo < theta_hi < 1.0:
            raise ValueError(
                f"admission watermarks must satisfy "
                f"0 < theta_lo < theta_hi < 1; got "
                f"theta_hi={theta_hi}, theta_lo={theta_lo}"
            )
        self.tracker = tracker
        self.theta_hi = theta_hi
        self.theta_lo = theta_lo
        self.max_pauses_per_event = max_pauses_per_event
        # FIFO of paused programs (oldest at index 0).
        self._paused_fifo: List[str] = []
        # Telemetry for tests.
        self.pause_decisions: int = 0
        self.resume_decisions: int = 0
        # Per-instance log set for any one-shot warnings (mirrors
        # KvScheduler's pattern).
        self._unknown_tier_log: set = set()

    # ---- inspectors (for tests / observability) ----

    def paused(self) -> List[str]:
        """Return the FIFO of currently-paused programs (oldest first)."""
        return list(self._paused_fifo)

    # ---- handler ----

    async def handle(self, event: Event, router) -> None:  # noqa: ANN001
        """Single entry point.  Routed to MEMORY_PRESSURE +
        PRESSURE_RESOLVED via :func:`attach_admission_controller`."""
        if event.kind == EventKind.MEMORY_PRESSURE:
            await self._on_pressure(event, router)
        elif event.kind == EventKind.PRESSURE_RESOLVED:
            await self._on_resolved(event, router)
        # All other kinds are routed elsewhere (T7); ignore here.

    async def _on_pressure(self, event: Event, router) -> None:  # noqa: ANN001
        """Pause the lowest-scoring program(s) until HBM occ < theta_hi."""
        for _ in range(self.max_pauses_per_event):
            try:
                state_json = await router.fetch_state()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "admission: state fetch failed for %s: %s",
                    event.kind.value, exc,
                )
                return
            sched_state = build_paper_state(
                state_json,
                event=event,
                tracker=self.tracker,
                unknown_tier_log=self._unknown_tier_log,
            )
            occ = self._hbm_occ(sched_state)
            if occ < self.theta_hi:
                # Pressure cleared by the migrate (T7) or natural eviction.
                return
            scores = shared_aware_prog_scores(sched_state)
            # Filter out already-paused programs.
            eligible = {
                pid: s for pid, s in scores.items()
                if self.tracker.state(pid) != State.PAUSED
            }
            if not eligible:
                logger.info(
                    "admission: occ=%.3f >= theta_hi=%.3f but no "
                    "eligible victim (all programs already paused)",
                    occ, self.theta_hi,
                )
                return
            # Pick the LOWEST scoring program.  Tie-break by pid for
            # determinism (= FIFO of program_id string order, which
            # the test fixtures exploit).
            victim = min(eligible.items(), key=lambda kv: (kv[1], kv[0]))[0]
            self.tracker.pause(victim)
            self._paused_fifo.append(victim)
            self.pause_decisions += 1
            logger.info(
                "admission: paused %s (score=%.4g; HBM occ=%.3f >= %.3f)",
                victim, eligible[victim], occ, self.theta_hi,
            )

    async def _on_resolved(self, event: Event, router) -> None:  # noqa: ANN001
        """Resume programs in FIFO order one at a time; stop when the
        next resume would cross theta_hi (hysteresis)."""
        # First, prune any pids from the FIFO that aren't actually
        # PAUSED any more (e.g., an external resume() call).
        self._paused_fifo = [
            pid for pid in self._paused_fifo
            if self.tracker.state(pid) == State.PAUSED
        ]
        if not self._paused_fifo:
            logger.info("admission: pressure_resolved but nothing to resume")
            return

        try:
            state_json = await router.fetch_state()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "admission: state fetch failed for %s: %s",
                event.kind.value, exc,
            )
            return
        sched_state = build_paper_state(
            state_json,
            event=event,
            tracker=self.tracker,
            unknown_tier_log=self._unknown_tier_log,
        )
        occ = self._hbm_occ(sched_state)
        if occ >= self.theta_lo:
            # Watermark hasn't truly dropped below the resume floor —
            # don't release pressure prematurely.  Sglang's webhook
            # will fire another pressure_resolved when occ drops further.
            logger.info(
                "admission: pressure_resolved but occ=%.3f >= theta_lo=%.3f; "
                "hold FIFO (no resume this event)",
                occ, self.theta_lo,
            )
            return
        # Resume one (the oldest paused).
        victim = self._paused_fifo.pop(0)
        self.tracker.resume(victim)
        self.resume_decisions += 1
        logger.info(
            "admission: resumed %s (oldest in FIFO; occ=%.3f < theta_lo=%.3f)",
            victim, occ, self.theta_lo,
        )

    # ---- helpers ----

    @staticmethod
    def _hbm_occ(state: SchedulerState) -> float:
        cap = state.tier_usage.capacity_bytes.get(Tier.HBM, 0)
        if cap == 0:
            return 0.0
        used = state.tier_usage.used_bytes.get(Tier.HBM, 0)
        return used / cap


# ----------------------------------------------------------------- attach


def attach_admission_controller(
    router, admission: AdmissionController  # noqa: ANN001
) -> None:
    """Wrap T7's MEMORY_PRESSURE / PRESSURE_RESOLVED handlers so both
    kv_scheduler.handle AND admission.handle fire on each pressure
    event.  kv_scheduler runs FIRST (migrate POST may relieve the
    pressure); admission re-checks state and pauses only if needed.

    Composition order matters: if admission ran first, it would over-
    pause based on stale (pre-migrate) state.
    """
    for kind in (EventKind.MEMORY_PRESSURE, EventKind.PRESSURE_RESOLVED):
        prior = router._handlers.get(kind.value)

        async def _composite(evt, r, _prior=prior, _adm=admission):
            if _prior is not None:
                await _prior(evt, r)
            await _adm.handle(evt, r)

        router.set_handler(kind, _composite)
