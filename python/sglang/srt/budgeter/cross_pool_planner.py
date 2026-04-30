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
    cooldown_ticks: int = 3        # Don't fire two transfers in a row;
                                   # let the system settle.
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
    edge_trigger: bool = False     # gate via SGLANG_XPOOL_EDGE_TRIGGER=1
    # Net-benefit gate (Setting 1 v9-auto follow-up). v9-auto's L1+L2 cell
    # showed L2 firing 15 transfers under L1's mamba_usage signal even when
    # L1's HPB-LRU snapshot retention had already absorbed the binding shift,
    # making the cuMemUnmap+cuMemMap cycle pure overhead and turning L1+L2
    # into a +42% Phase C P99 regression vs L1-only. The gate computes
    #   expected_benefit_us = retracted * avg_input * 1e6 / prefill_tps
    #                       + paused * pause_penalty_us
    #   expected_cost_us    = n_chunks * chunk_cost_us
    # and refuses to fire unless benefit >= cost * margin. With no retracts
    # and no paused reqs, benefit is 0 and the gate always blocks — which
    # is exactly the v9-auto failure mode we need to suppress.
    net_benefit_enabled: bool = False  # gate via SGLANG_XPOOL_NET_BENEFIT=1
    nb_avg_prefill_tokens: int = 4096  # avg input length used for benefit est
    nb_prefill_tps: float = 50000.0    # prefill throughput est (H200 default)
    nb_pause_penalty_us: float = 1000.0  # cost of one paused-req tick
    nb_chunk_cost_us: float = 50000.0  # cuMemUnmap+cuMemMap per chunk (~50ms)
    nb_margin: float = 1.5             # benefit must exceed cost * margin


def _policy_from_env() -> CrossPoolPolicyConfig:
    return CrossPoolPolicyConfig(
        kv_high_water=float(os.environ.get("SGLANG_XPOOL_KV_HIGH", "0.85")),
        kv_low_water=float(os.environ.get("SGLANG_XPOOL_KV_LOW", "0.50")),
        mamba_high_water=float(os.environ.get("SGLANG_XPOOL_MAMBA_HIGH", "0.80")),
        mamba_low_water=float(os.environ.get("SGLANG_XPOOL_MAMBA_LOW", "0.40")),
        cooldown_ticks=int(os.environ.get("SGLANG_XPOOL_COOLDOWN", "3")),
        dst_chunks_per_action=int(os.environ.get("SGLANG_XPOOL_UNIT", "1")),
        qdepth_trigger=int(os.environ.get("SGLANG_XPOOL_QDEPTH_TRIGGER", "0")),
        edge_trigger=bool(int(os.environ.get("SGLANG_XPOOL_EDGE_TRIGGER", "0"))),
        net_benefit_enabled=bool(int(os.environ.get("SGLANG_XPOOL_NET_BENEFIT", "0"))),
        nb_avg_prefill_tokens=int(os.environ.get("SGLANG_XPOOL_NB_AVG_PREFILL_TOKENS", "4096")),
        nb_prefill_tps=float(os.environ.get("SGLANG_XPOOL_NB_PREFILL_TPS", "50000")),
        nb_pause_penalty_us=float(os.environ.get("SGLANG_XPOOL_NB_PAUSE_PENALTY_US", "1000")),
        nb_chunk_cost_us=float(os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_US", "50000")),
        nb_margin=float(os.environ.get("SGLANG_XPOOL_NB_MARGIN", "1.5")),
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
    ) -> None:
        self.config = config if config is not None else _policy_from_env()
        self._cooldown_remaining: int = 0
        self._tick_count: int = 0
        # Edge-triggered state per pool. Initialized to IN_BAND so the
        # very first tick's reading establishes the baseline state without
        # forcing a transfer at startup.
        self._kv_state: str = self.IN_BAND
        self._mamba_state: str = self.IN_BAND
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
        self, num_paused: int, num_retracted: int
    ) -> tuple[bool, str]:
        """Return (allow_fire, why) per the cost/benefit estimator.

        Disabled (returns True) unless `net_benefit_enabled`. When enabled,
        treats `num_retracted` as full re-prefill avoided and `num_paused`
        as one tick of waiting avoided. Refuses to fire when benefit < cost
        × margin — the case where actuator overhead would exceed the
        admission-pressure cost the transfer is meant to relieve.
        """
        c = self.config
        if not c.net_benefit_enabled:
            return True, "nb=off"
        if num_paused == 0 and num_retracted == 0:
            return False, "nb: no admission pressure (paused=0 retracted=0)"
        benefit_us = (
            num_retracted * c.nb_avg_prefill_tokens * 1e6 / c.nb_prefill_tps
            + num_paused * c.nb_pause_penalty_us
        )
        cost_us = c.dst_chunks_per_action * c.nb_chunk_cost_us
        if benefit_us < cost_us * c.nb_margin:
            return False, (
                f"nb: benefit {benefit_us:.0f}us < cost {cost_us:.0f}us × "
                f"margin {c.nb_margin} (paused={num_paused} retracted={num_retracted})"
            )
        return True, (
            f"nb: benefit {benefit_us:.0f}us >= cost {cost_us:.0f}us × "
            f"margin {c.nb_margin} (paused={num_paused} retracted={num_retracted})"
        )

    def decide(
        self,
        usage_kv: float,
        usage_mamba: float,
        queue_depth: int = 0,
        num_paused_reqs: int = 0,
        num_retracted_reqs: int = 0,
    ) -> PlanDecision:
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
            if not (kv_changed or mamba_changed):
                return PlanDecision(
                    direction=None,
                    reason=f"edge: stable kv={new_kv} mamba={new_mamba} "
                           f"(kv={usage_kv:.2f} mamba={usage_mamba:.2f})",
                    usage_kv=usage_kv,
                    usage_mamba=usage_mamba,
                    queue_depth=queue_depth,
                )
            # Helper: try to fire `direction`, but suppress if the
            # net-benefit gate says actuator overhead would exceed the
            # admission-pressure cost we'd relieve.
            def _try_fire(direction: str, edge_reason: str) -> PlanDecision:
                ok, why = self._net_benefit_ok(num_paused_reqs, num_retracted_reqs)
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
