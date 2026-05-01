"""
Phase 2e.5.6.3.c — cross-pool capacity planner driven by real per-pool
pressure signals.

Reads engine snapshot fields (`kv_used_tokens`, `mamba_usage`,
`num_running_reqs`, `num_queue_reqs`, `cache_hit_rate`) and decides
whether to fire a cross-pool transfer in either direction. Replaces
the 2e.5.6.2 / 2e.5.6.3.a oscillator with a workload-aware policy.

Policy form (paper §4.3 reduced to threshold-with-hysteresis):
  Marginal value of holding capacity in pool σ ≈ usage_σ. The Lagrange
  optimum equalizes marginal values by transferring chunks toward the
  higher-pressure pool. Concretely:

    if usage_kv > kv_high_water and usage_mamba < mamba_low_water:
        transfer mamba → kv (KV donates is wrong — mamba donates to KV)
    if usage_mamba > mamba_high_water and usage_kv < kv_low_water:
        transfer kv → mamba

  Hysteresis (high vs low watermark) prevents thrashing during edge
  fluctuations. When neither side is stressed (or both are stressed),
  no transfer fires.

This is intentionally not a full Lagrange bisection: the per-tick
planning surface is two-pool, and threshold-with-hysteresis on the
marginal_value=usage approximation produces the same decision as
formal Lagrange equalization for the two-pool case.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CrossPoolPolicyConfig:
    kv_high_water: float = 0.85    # KV usage above this → KV-stressed
    kv_low_water: float = 0.50     # KV usage below this → KV-relaxed
    mamba_high_water: float = 0.80
    mamba_low_water: float = 0.40
    cooldown_ticks: int = 16       # Don't fire two transfers in a row.
                                   # 16 ticks × 2 s = 32 s — enough for an
                                   # lcm-aware transfer (~3 s GPU time) to
                                   # settle, request queue to drain, and a
                                   # post-fire effect to manifest before
                                   # the next decision.
    dst_chunks_per_action: int = 1  # In balanced units (lcm-aware).
    # Setting 4 follow-up: V_σ' ≈ usage_σ is saturation-blind. When
    # qdepth_trigger > 0, the planner ALSO fires a transfer when one
    # pool is saturated (above its high watermark) and the queue depth
    # exceeds qdepth_trigger — even if the other pool is above its low
    # watermark. This recovers gradient information at saturation: a
    # rising queue is the signal that pool capacity is the bottleneck.
    qdepth_trigger: int = 0        # 0 = disabled; e.g. 4 enables the new rule
    # Setting 1 v9 follow-up: level-triggered planning fires repeatedly
    # while usage stays above the high watermark (cooldown-bounded),
    # accumulating actuator overhead even on steady-state workloads.
    # Edge-triggered mode fires ONLY at state transitions (BELOW_LOW ↔
    # IN_BAND ↔ ABOVE_HIGH). Steady state above high → 0 transfers
    # after the first crossing → 0 actuator overhead.
    edge_trigger: bool = True      # paper §design-l2 default; SGLANG_XPOOL_EDGE_TRIGGER=0 to disable
    # Net-benefit gate (paper §design-l2). Engine-agnostic abstract:
    #   B (benefit) ≥ C (cost) × margin
    # B is the sum of admission-pressure signals translated through the
    # `EnginePressureAdapter` (see budgeter/pressure_adapter.py).
    # Different engines provide different adapters that map their native
    # pool-pressure mechanisms into a uniform "us of GPU time saved"
    # space. SGLang's adapter dominates the eviction term (its primary
    # pressure-relief mechanism), vLLM's would dominate swap/preempt,
    # etc. C is the empirically measured per-fire wall time
    # (`nb_chunk_cost_us`).
    net_benefit_enabled: bool = True   # paper §design-l2 default; SGLANG_XPOOL_NET_BENEFIT=0 to disable
    # nb_chunk_cost_us: empirically measured per-fire cuMemUnmap+cuMemMap
    # wall time. ~5,000 us on H200 with 256 MB chunks per the Stage 1
    # calibration in sglang `87360b2c7`. Adapter-specific signal
    # coefficients live in the adapter; the planner only needs the cost.
    nb_chunk_cost_us: float = 5000.0
    nb_margin: float = 1.5
    # Period (in stable ticks) at which the planner re-evaluates the
    # net-benefit gate while a pool is sustainedly ABOVE_HIGH. Without
    # this, edge-trigger mode would only check at state transitions and
    # miss steady-state pressure that built up after the initial cross.
    nb_persist_eval_period: int = 10


def _policy_from_env() -> CrossPoolPolicyConfig:
    return CrossPoolPolicyConfig(
        kv_high_water=float(os.environ.get("SGLANG_XPOOL_KV_HIGH", "0.85")),
        kv_low_water=float(os.environ.get("SGLANG_XPOOL_KV_LOW", "0.50")),
        mamba_high_water=float(os.environ.get("SGLANG_XPOOL_MAMBA_HIGH", "0.80")),
        mamba_low_water=float(os.environ.get("SGLANG_XPOOL_MAMBA_LOW", "0.40")),
        cooldown_ticks=int(os.environ.get("SGLANG_XPOOL_COOLDOWN", "16")),
        dst_chunks_per_action=int(os.environ.get("SGLANG_XPOOL_UNIT", "1")),
        qdepth_trigger=int(os.environ.get("SGLANG_XPOOL_QDEPTH_TRIGGER", "0")),
        edge_trigger=bool(int(os.environ.get("SGLANG_XPOOL_EDGE_TRIGGER", "1"))),
        net_benefit_enabled=bool(int(os.environ.get("SGLANG_XPOOL_NET_BENEFIT", "1"))),
        # Per-fire cuMemUnmap+cuMemMap wall time, empirically measured on
        # Qwen3.5-35B-A3B / H200 / 256 MB chunks (sglang `87360b2c7` Stage 1
        # calibration: 2 fires, 120 chunks, 9.5 ms total = ~80 us/chunk,
        # ~4.7 ms/fire). 5,000 us is a conservative round-up; operators
        # should re-measure on their hardware via the actuator's
        # `xpool_fire_total_us` log field.
        nb_chunk_cost_us=float(os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_US", "5000")),
        nb_margin=float(os.environ.get("SGLANG_XPOOL_NB_MARGIN", "1.5")),
        nb_persist_eval_period=int(os.environ.get("SGLANG_XPOOL_NB_PERSIST_EVAL_PERIOD", "10")),
    )


@dataclass
class PlanDecision:
    direction: Optional[str]    # "kv_to_mamba", "mamba_to_kv", or None
    reason: str
    usage_kv: float
    usage_mamba: float
    queue_depth: int = 0        # admission-pressure signal at decision time


class CrossPoolPlanner:
    """Threshold-with-hysteresis cross-pool transfer planner.

    Read-only with respect to scheduler state. Returns a `PlanDecision`
    each tick; the caller (BudgetAgent) decides whether to apply it
    (e.g., gating on `num_running_reqs == 0` for safety in this milestone).
    """

    # Per-pool state machine constants for edge-triggered mode.
    BELOW_LOW = "below_low"
    IN_BAND = "in_band"
    ABOVE_HIGH = "above_high"

    def __init__(
        self,
        config: Optional[CrossPoolPolicyConfig] = None,
        adapter=None,
    ) -> None:
        self.config = config if config is not None else _policy_from_env()
        if adapter is None:
            from sglang.srt.budgeter.pressure_adapter import get_default_adapter
            adapter = get_default_adapter()
        self._adapter = adapter
        self._cooldown_remaining: int = 0
        self._tick_count: int = 0
        # Edge-triggered state per pool. Initialized to IN_BAND so the
        # very first tick's reading establishes the baseline state without
        # forcing a transfer at startup.
        self._kv_state: str = self.IN_BAND
        self._mamba_state: str = self.IN_BAND
        # Persist counters: number of consecutive ticks each pool has been
        # in ABOVE_HIGH. Used by net-benefit gate's B_persist term to score
        # sustained pool saturation as a real (if soft) cost, even when the
        # scheduler doesn't formally retract or pause requests.
        self._kv_above_high_consec: int = 0
        self._mamba_above_high_consec: int = 0
        logger.info(
            "CrossPoolPlanner: kv_high=%.2f kv_low=%.2f mamba_high=%.2f "
            "mamba_low=%.2f cooldown=%d unit=%d edge_trigger=%s",
            self.config.kv_high_water, self.config.kv_low_water,
            self.config.mamba_high_water, self.config.mamba_low_water,
            self.config.cooldown_ticks, self.config.dst_chunks_per_action,
            self.config.edge_trigger,
        )

    def _classify(self, usage: float, low: float, high: float) -> str:
        """Map a usage value to a discrete state for edge-triggered mode."""
        if usage >= high:
            return self.ABOVE_HIGH
        if usage <= low:
            return self.BELOW_LOW
        return self.IN_BAND

    def _net_benefit_ok(
        self,
        snapshot: dict | None,
        kv_above_consec: int = 0,
        mamba_above_consec: int = 0,
    ) -> tuple[bool, str]:
        """Return (allow_fire, why) per the engine-agnostic cost/benefit
        estimator.

        Paper §design-l2: `B (benefit) ≥ C (cost) × margin`. Benefit is
        the sum of admission-pressure signals translated through the
        engine-specific `EnginePressureAdapter` (sglang
        `pressure_adapter.py`). Cost is `nb_chunk_cost_us` (empirically
        measured wall time per fire, e.g., ~5 ms on H200 with 256 MB
        chunks per the Stage 1 calibration in `87360b2c7`).

        For SGLang the dominant signal is tree-cache eviction (the
        engine's primary pressure-relief mechanism); paused/retracted
        rarely surface because eviction satisfies pressure first.
        Persist provides a saturation prior. The adapter normalizes
        all of these to "us of GPU time" so the inequality is unit-clean.

        Disabled (returns True unconditionally) when `net_benefit_enabled`
        is False — useful as a kill switch but not the paper-faithful
        default.
        """
        c = self.config
        if not c.net_benefit_enabled:
            return True, "nb=off"
        signals = self._adapter.signals_from_snapshot(
            snapshot or {}, kv_above_consec, mamba_above_consec,
        )
        benefit_us = signals.total_benefit_us
        cost_us = c.dst_chunks_per_action * c.nb_chunk_cost_us
        if benefit_us <= 0:
            return False, f"nb: no pressure ({signals.reason_str()})"
        if benefit_us < cost_us * c.nb_margin:
            return False, (
                f"nb: B={benefit_us:.0f}us < C={cost_us:.0f}us × "
                f"margin={c.nb_margin} ({signals.reason_str()})"
            )
        return True, (
            f"nb: B={benefit_us:.0f}us >= C={cost_us:.0f}us × "
            f"margin={c.nb_margin} ({signals.reason_str()})"
        )

    def decide(
        self,
        usage_kv: float,
        usage_mamba: float,
        queue_depth: int = 0,
        snapshot: dict | None = None,
    ) -> PlanDecision:
        """Decide whether to fire a cross-pool transfer this tick.

        `snapshot` is the budgeter snapshot dict (passed through from
        agent.py). The pressure adapter reads engine-native fields from
        it (num_evicted_tokens_recent / num_retracted_reqs /
        num_paused_reqs / num_queue_reqs) when net-benefit gate is on.
        """
        self._tick_count += 1
        c = self.config

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return PlanDecision(
                direction=None,
                reason=f"cooldown ({self._cooldown_remaining} left)",
                usage_kv=usage_kv,
                usage_mamba=usage_mamba,
                queue_depth=queue_depth,
            )

        # Edge-triggered mode: only fire on state transitions. Steady state
        # (no transition) → 0 transfers → 0 actuator overhead. This is the
        # property that lets Layer 2 not regress on common-case workloads.
        if c.edge_trigger:
            new_kv = self._classify(usage_kv, c.kv_low_water, c.kv_high_water)
            new_mamba = self._classify(usage_mamba, c.mamba_low_water, c.mamba_high_water)
            kv_changed = new_kv != self._kv_state
            mamba_changed = new_mamba != self._mamba_state
            old_kv, old_mamba = self._kv_state, self._mamba_state
            # Update state BEFORE returning so next tick sees the new state.
            self._kv_state, self._mamba_state = new_kv, new_mamba
            # Update consec counters for B_persist accumulation.
            self._kv_above_high_consec = (
                self._kv_above_high_consec + 1 if new_kv == self.ABOVE_HIGH else 0
            )
            self._mamba_above_high_consec = (
                self._mamba_above_high_consec + 1 if new_mamba == self.ABOVE_HIGH else 0
            )
            kv_consec = self._kv_above_high_consec
            mamba_consec = self._mamba_above_high_consec
            if not (kv_changed or mamba_changed):
                # Stable. Persist re-evaluation: even without a fresh edge,
                # if a pool has been ABOVE_HIGH long enough that B_persist
                # has built up, give the gate another chance to fire. This
                # closes the v9-auto v2 failure mode where stock cache hides
                # the true cost of sustained mamba=1.0 saturation behind
                # zero paused/retracted counters.
                period = c.nb_persist_eval_period
                if c.net_benefit_enabled and period > 0:
                    # Re-eval at every period-th tick of sustained ABOVE_HIGH.
                    if (mamba_consec >= period and (mamba_consec % period) == 0
                            and new_mamba == self.ABOVE_HIGH
                            and new_kv != self.ABOVE_HIGH):
                        ok, why = self._net_benefit_ok(
                            snapshot, kv_consec, mamba_consec,
                        )
                        if ok:
                            self._cooldown_remaining = c.cooldown_ticks
                            return PlanDecision(
                                direction="kv_to_mamba",
                                reason=f"persist: mamba ABOVE_HIGH×{mamba_consec} "
                                       f"({usage_mamba:.2f}) kv={new_kv} "
                                       f"({usage_kv:.2f}) [{why}]",
                                usage_kv=usage_kv,
                                usage_mamba=usage_mamba,
                                queue_depth=queue_depth,
                            )
                    if (kv_consec >= period and (kv_consec % period) == 0
                            and new_kv == self.ABOVE_HIGH
                            and new_mamba != self.ABOVE_HIGH):
                        ok, why = self._net_benefit_ok(
                            snapshot, kv_consec, mamba_consec,
                        )
                        if ok:
                            self._cooldown_remaining = c.cooldown_ticks
                            return PlanDecision(
                                direction="mamba_to_kv",
                                reason=f"persist: kv ABOVE_HIGH×{kv_consec} "
                                       f"({usage_kv:.2f}) mamba={new_mamba} "
                                       f"({usage_mamba:.2f}) [{why}]",
                                usage_kv=usage_kv,
                                usage_mamba=usage_mamba,
                                queue_depth=queue_depth,
                            )
                return PlanDecision(
                    direction=None,
                    reason=f"edge: stable kv={new_kv} mamba={new_mamba} "
                           f"(kv={usage_kv:.2f} mamba={usage_mamba:.2f}) "
                           f"persist=({kv_consec},{mamba_consec})",
                    usage_kv=usage_kv,
                    usage_mamba=usage_mamba,
                    queue_depth=queue_depth,
                )
            # Helper: try to fire `direction`, but suppress if the
            # net-benefit gate says actuator overhead would exceed the
            # admission-pressure cost we'd relieve.
            def _try_fire(direction: str, edge_reason: str) -> PlanDecision:
                ok, why = self._net_benefit_ok(
                    snapshot, kv_consec, mamba_consec,
                )
                if not ok:
                    return PlanDecision(
                        direction=None,
                        reason=f"edge: would fire {direction} ({edge_reason}) but {why}",
                        usage_kv=usage_kv,
                        usage_mamba=usage_mamba,
                        queue_depth=queue_depth,
                    )
                self._cooldown_remaining = c.cooldown_ticks
                return PlanDecision(
                    direction=direction,
                    reason=f"edge: {edge_reason} [{why}]",
                    usage_kv=usage_kv,
                    usage_mamba=usage_mamba,
                    queue_depth=queue_depth,
                )

            # Mamba just crossed into ABOVE_HIGH — fire kv→mamba once.
            if mamba_changed and new_mamba == self.ABOVE_HIGH and \
               new_kv != self.ABOVE_HIGH:
                return _try_fire(
                    "kv_to_mamba",
                    f"mamba {old_mamba}→ABOVE_HIGH ({usage_mamba:.2f}); "
                    f"kv={new_kv} ({usage_kv:.2f})",
                )
            # KV just crossed into ABOVE_HIGH — fire mamba→kv once.
            if kv_changed and new_kv == self.ABOVE_HIGH and \
               new_mamba != self.ABOVE_HIGH:
                return _try_fire(
                    "mamba_to_kv",
                    f"kv {old_kv}→ABOVE_HIGH ({usage_kv:.2f}); "
                    f"mamba={new_mamba} ({usage_mamba:.2f})",
                )
            # Reverse trigger: a pool just dropped to BELOW_LOW while the
            # other pool is ABOVE_HIGH — opportunity to undo a prior transfer.
            if mamba_changed and new_mamba == self.BELOW_LOW and \
               new_kv == self.ABOVE_HIGH:
                return _try_fire(
                    "mamba_to_kv",
                    f"mamba {old_mamba}→BELOW_LOW ({usage_mamba:.2f}); "
                    f"kv ABOVE_HIGH ({usage_kv:.2f})",
                )
            if kv_changed and new_kv == self.BELOW_LOW and \
               new_mamba == self.ABOVE_HIGH:
                return _try_fire(
                    "kv_to_mamba",
                    f"kv {old_kv}→BELOW_LOW ({usage_kv:.2f}); "
                    f"mamba ABOVE_HIGH ({usage_mamba:.2f})",
                )
            # State transition exists but doesn't match any actionable
            # pattern (e.g., IN_BAND→BELOW_LOW with the other pool also
            # IN_BAND or BELOW_LOW). No-op.
            return PlanDecision(
                direction=None,
                reason=f"edge: transition {old_kv}→{new_kv} mamba "
                       f"{old_mamba}→{new_mamba} (no actionable pattern)",
                usage_kv=usage_kv,
                usage_mamba=usage_mamba,
                queue_depth=queue_depth,
            )

        # ---- LEGACY level-triggered path (preserved when edge_trigger=False) ----
        # KV-stressed, mamba-relaxed → take from mamba.
        if usage_kv >= c.kv_high_water and usage_mamba <= c.mamba_low_water:
            self._cooldown_remaining = c.cooldown_ticks
            return PlanDecision(
                direction="mamba_to_kv",
                reason=f"kv={usage_kv:.2f}>={c.kv_high_water:.2f} & "
                       f"mamba={usage_mamba:.2f}<={c.mamba_low_water:.2f}",
                usage_kv=usage_kv,
                usage_mamba=usage_mamba,
                queue_depth=queue_depth,
            )
        # Mamba-stressed, KV-relaxed → take from KV.
        if usage_mamba >= c.mamba_high_water and usage_kv <= c.kv_low_water:
            self._cooldown_remaining = c.cooldown_ticks
            return PlanDecision(
                direction="kv_to_mamba",
                reason=f"mamba={usage_mamba:.2f}>={c.mamba_high_water:.2f} & "
                       f"kv={usage_kv:.2f}<={c.kv_low_water:.2f}",
                usage_kv=usage_kv,
                usage_mamba=usage_mamba,
                queue_depth=queue_depth,
            )
        # Setting 4 follow-up: at saturation, V_σ' ≈ usage_σ is blind to
        # which pool is the bottleneck. Add a queue-depth-driven rule:
        # when one pool is saturated and queue is non-trivial, fire a
        # transfer toward the saturated pool. This rule is gated on
        # SGLANG_XPOOL_QDEPTH_TRIGGER > 0 to preserve the legacy default.
        if c.qdepth_trigger > 0 and queue_depth >= c.qdepth_trigger:
            if usage_mamba >= c.mamba_high_water and usage_kv < c.kv_high_water:
                self._cooldown_remaining = c.cooldown_ticks
                return PlanDecision(
                    direction="kv_to_mamba",
                    reason=f"saturation+queue: mamba={usage_mamba:.2f}>="
                           f"{c.mamba_high_water:.2f} & qdepth={queue_depth}>="
                           f"{c.qdepth_trigger}",
                    usage_kv=usage_kv,
                    usage_mamba=usage_mamba,
                    queue_depth=queue_depth,
                )
            if usage_kv >= c.kv_high_water and usage_mamba < c.mamba_high_water:
                self._cooldown_remaining = c.cooldown_ticks
                return PlanDecision(
                    direction="mamba_to_kv",
                    reason=f"saturation+queue: kv={usage_kv:.2f}>="
                           f"{c.kv_high_water:.2f} & qdepth={queue_depth}>="
                           f"{c.qdepth_trigger}",
                    usage_kv=usage_kv,
                    usage_mamba=usage_mamba,
                    queue_depth=queue_depth,
                )

        return PlanDecision(
            direction=None,
            reason=f"both within band: kv={usage_kv:.2f} mamba={usage_mamba:.2f}",
            usage_kv=usage_kv,
            usage_mamba=usage_mamba,
            queue_depth=queue_depth,
        )
