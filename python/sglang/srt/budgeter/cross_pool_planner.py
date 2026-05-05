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

import json
import logging
import os
import time
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
    edge_trigger: bool = True      # paper §sec:design-l2 default; SGLANG_XPOOL_EDGE_TRIGGER=0 to disable
    # Net-benefit gate (paper §sec:design-l2). Engine-agnostic abstract:
    #   B (benefit) ≥ C (cost) × margin
    # B is the sum of admission-pressure signals translated through the
    # `EnginePressureAdapter` (see budgeter/pressure_adapter.py).
    # Different engines provide different adapters that map their native
    # pool-pressure mechanisms into a uniform "us of GPU time saved"
    # space. SGLang's adapter dominates the eviction term (its primary
    # pressure-relief mechanism), vLLM's would dominate swap/preempt,
    # etc. C is the empirically measured per-fire wall time
    # (`nb_chunk_cost_us`).
    net_benefit_enabled: bool = True   # paper §sec:design-l2 default; SGLANG_XPOOL_NET_BENEFIT=0 to disable
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
    # Hysteresis Δ on the both-pools-ABOVE_HIGH branch: when both are
    # saturated, fire toward the less-saturated pool only if the
    # |usage_kv - usage_mamba| gap exceeds this margin. Prevents
    # oscillation when both pools sit at near-equal pressure.
    hysteresis_delta: float = 0.05
    # Both-above branch tight activation gate (fix#37 v2 — prevents
    # oscillation on workloads where both pools are merely "high" but
    # neither is truly binding). Must satisfy:
    #   max(usage_kv, usage_mamba) >= both_above_max_threshold AND
    #   |usage_kv - usage_mamba|       >= both_above_min_gap.
    # Without these, fire toggles direction every persist period as the
    # gap reverses sign post-fire (each fire shifts capacity by ~10-20%
    # in usage units, exceeding hysteresis_delta=0.05).
    both_above_max_threshold: float = 0.95
    both_above_min_gap: float = 0.20
    # Paper §sec:design-l2-firegate (Eq nb-direction-gate): direction-aware
    # net benefit. For each ordered pair (σ_src, σ_dst) compute
    #   NB(σ_src→σ_dst) = c_dst(L̄_dst)·P_save(σ_dst) - c_src(L̄_src)·P_loss(σ_src)
    # with P_save = P_loss = max(0, (u_σ - u_low) / (1 - u_low)) keyed on
    # the pool's own low-water; arg-max over directions and fire iff
    # max-NB ≥ α·C_act. When enabled, this REPLACES the saturation-driven
    # direction selection — the legacy edge-trigger / persist re-eval paths
    # are not used. SGLANG_XPOOL_NB_DIRECTION_AWARE=0 to fall back to the
    # legacy gate. Default ON to match paper.
    nb_direction_aware: bool = True


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
        hysteresis_delta=float(os.environ.get("SGLANG_XPOOL_HYSTERESIS_DELTA", "0.05")),
        both_above_max_threshold=float(os.environ.get("SGLANG_XPOOL_BOTH_ABOVE_MAX", "0.95")),
        both_above_min_gap=float(os.environ.get("SGLANG_XPOOL_BOTH_ABOVE_GAP", "0.20")),
        nb_direction_aware=bool(int(os.environ.get("SGLANG_XPOOL_NB_DIRECTION_AWARE", "1"))),
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
        # Cost-curve handle (paper §sec:design-l2-firegate). The adapter
        # carries the calibrated curves; we expose them on the planner so
        # decision logging can quote c_σ(\bar{L}) and the L<L* / L>L* regime
        # without re-loading.
        self._cost_curves = getattr(adapter, "cost_curves", None)
        # Optional JSONL log of every cost-aware decision (one record per
        # decide() call regardless of fire/no-fire). Set SGLANG_XPOOL_COST_LOG
        # to a writable path to enable.
        self._cost_log_path: Optional[str] = (
            os.environ.get("SGLANG_XPOOL_COST_LOG") or None
        )
        self._cost_log_fp = None
        if self._cost_log_path:
            try:
                self._cost_log_fp = open(self._cost_log_path, "a", buffering=1)
                logger.info(
                    "[xpool-cost] decisions logged to %s", self._cost_log_path
                )
            except OSError as e:
                logger.warning(
                    "[xpool-cost] cannot open SGLANG_XPOOL_COST_LOG=%s: %s",
                    self._cost_log_path, e,
                )
                self._cost_log_fp = None
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

    @staticmethod
    def _p_func(usage: float, low_water: float) -> float:
        """P_save / P_loss as defined in paper Eq p-loss-save:
            P(σ) = max(0, (u_σ - u_low) / (1 - u_low))
        Hits 0 below low-water (slack regime), rises smoothly to 1 at
        full saturation. Same shape for both save and loss.
        """
        if low_water >= 1.0:
            return 1.0 if usage >= 1.0 else 0.0
        return max(0.0, min(1.0, (usage - low_water) / (1.0 - low_water)))

    def _pick_direction_by_nb(
        self,
        usage_kv: float,
        usage_mamba: float,
        snapshot: dict | None,
    ) -> tuple[Optional[str], float, str]:
        """Paper §sec:design-l2-firegate Eq nb-direction-gate.

        Computes NB for each candidate direction and returns the arg-max if
        it clears the gate; otherwise returns (None, ...).

        Returns:
            (best_direction, best_nb_us, reason_str)
            best_direction is "kv_to_mamba" / "mamba_to_kv" or None.
        """
        if self._cost_curves is None:
            return None, 0.0, "nb_direction: no cost curves"
        c = self.config
        # Per-pool L̄_i (paper §sec:design-formalism-offline). Each pool's
        # EWMA is fed by its own evict events: KV evict for L_kv, mamba
        # snapshot evict for L_m. mean_recovery_len_retract (req kicked
        # out) is a separate retract-pressure signal and not used here.
        L_kv = float((snapshot or {}).get("mean_recovery_len_kv", 0) or 0)
        L_m = float((snapshot or {}).get("mean_recovery_len_rec", L_kv) or L_kv)
        if L_kv <= 0 and L_m <= 0:
            # No L̄ observed yet — fall back to legacy path.
            return None, 0.0, "nb_direction: no recovery_len observed"
        c_kv = self._cost_curves.c_kv_us(L_kv if L_kv > 0 else L_m)
        c_m = self._cost_curves.c_m_us(L_m if L_m > 0 else L_kv)

        p_save_kv = self._p_func(usage_kv, c.kv_low_water)
        p_loss_kv = p_save_kv  # same functional form per Eq p-loss-save
        p_save_m = self._p_func(usage_mamba, c.mamba_low_water)
        p_loss_m = p_save_m

        # NB amortization: a fired chunk persists at least `cooldown_ticks`
        # ticks before the gate re-evaluates. The per-chunk gain/loss
        # accumulates over that interval, so we scale by lifetime to get a
        # meaningful comparison against the (one-shot) actuator cost.
        # Without this scaling the gate under-counts benefit by ~cooldown_ticks×
        # and misses fires that are net-positive across their amortization
        # window. We use cooldown_ticks as a conservative lower bound on
        # chunk lifetime — actual chunks usually persist longer.
        lifetime = max(1, c.cooldown_ticks)
        # NB(kv_to_mamba): grow mamba (gain c_m × P_save_m), shrink kv
        # (lose c_kv × P_loss_kv), amortized over lifetime ticks.
        nb_k2m = lifetime * (c_m * p_save_m - c_kv * p_loss_kv)
        # NB(mamba_to_kv): grow kv (gain c_kv × P_save_kv), shrink mamba
        # (lose c_m × P_loss_m), amortized over lifetime ticks.
        nb_m2k = lifetime * (c_kv * p_save_kv - c_m * p_loss_m)

        # Saturation guard: a source pool above its high-water cannot
        # afford to give up bytes, regardless of how cheap its per-byte
        # recovery looks. The linear P_loss in Eq p-loss-save captures
        # the per-chunk *re-prefill* cost but does not capture the
        # super-linear queueing-breakdown cost when the source is already
        # at admission-saturation: shrinking a 95%-full pool by even one
        # chunk drives many subsequent requests to stall in admission,
        # an effect the per-byte cost model under-counts. We reject any
        # direction whose source's *admission-side* usage is at or above
        # its high-water mark.
        #
        # Critically, the mamba pool's "usage" reported by the allocator
        # mixes two semantically different occupancies: active slots
        # (1 per running req — these are the admission gate; can't be
        # shrunk without queueing breakdown) and radix-tree-cached
        # snapshots (cache fill — can be cheaply LRU-evicted). When the
        # snapshot decorates the saturation, m2k fires would be wrongly
        # blocked even though shrinking the snapshot side is exactly
        # what the c_M(L) × P_loss term is already pricing in. We
        # therefore consult the active-only signal when the agent
        # provides it (via snapshot["usage_mamba_active"]); fall back
        # to total usage when that field is missing for back-compat.
        # (The KV pool has no equivalent "snapshot-only" occupancy
        # distinction, so usage_kv is used directly.)
        kv_active_for_guard = usage_kv
        m_active_for_guard = float(
            (snapshot or {}).get("usage_mamba_active", usage_mamba) or usage_mamba
        )
        if kv_active_for_guard >= c.kv_high_water:
            nb_k2m = float("-inf")  # KV is saturated, can't shrink it
        if m_active_for_guard >= c.mamba_high_water:
            nb_m2k = float("-inf")  # mamba ACTIVE-slots saturated, can't shrink it

        # Actuator cost: prefer the runtime EWMA over the static config.
        # Stage-0 / config nb_chunk_cost_us is an idle-time lower bound;
        # under live traffic the cuMemUnmap+cuMemMap pair pays additional
        # CUDA-graph deferral and allocator-contention cost. The runtime
        # EWMA tracks fire wall-times observed via cudaSynchronize-bracketed
        # measurement (paper §sec:design-l2-firegate runtime self-calibration).
        from sglang.srt.budgeter.cost_model import get_runtime_actuator_cost
        runtime_cost = get_runtime_actuator_cost()
        c_actuator_us = (
            runtime_cost.current_us if runtime_cost.is_calibrated
            else max(runtime_cost.current_us, c.nb_chunk_cost_us)
        )
        threshold = c.nb_margin * c.dst_chunks_per_action * c_actuator_us

        if nb_k2m >= nb_m2k and nb_k2m >= threshold:
            best_dir = "kv_to_mamba"
            best_nb = nb_k2m
        elif nb_m2k > nb_k2m and nb_m2k >= threshold:
            best_dir = "mamba_to_kv"
            best_nb = nb_m2k
        else:
            best_dir = None
            best_nb = max(nb_k2m, nb_m2k)

        reason = (
            f"NB[k2m]={nb_k2m:.0f}us NB[m2k]={nb_m2k:.0f}us "
            f"threshold={threshold:.0f}us "
            f"(c_kv={c_kv:.0f}us@L={L_kv:.0f}, c_m={c_m:.0f}us@L={L_m:.0f}, "
            f"P_save: kv={p_save_kv:.2f} m={p_save_m:.2f}, "
            f"P_loss: kv={p_loss_kv:.2f} m={p_loss_m:.2f})"
        )
        return best_dir, best_nb, reason

    def _net_benefit_ok(
        self,
        snapshot: dict | None,
        kv_above_consec: int = 0,
        mamba_above_consec: int = 0,
        edge_active: bool = False,
    ) -> tuple[bool, str]:
        """Return (allow_fire, why) per the engine-agnostic cost/benefit
        estimator.

        Paper §sec:design-l2: `B (benefit) ≥ C (cost) × margin`. Benefit is
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
            edge_active=edge_active,
        )
        benefit_us = signals.total_benefit_us
        cost_us = c.dst_chunks_per_action * c.nb_chunk_cost_us

        # Quote the c_σ(\bar{L}) regime so the asymmetry direction is
        # visible in the log without re-deriving from raw fields.
        regime_str = ""
        if self._cost_curves is not None and snapshot:
            kv_L = float(snapshot.get("mean_recovery_len_kv", 0) or 0)
            if kv_L > 0:
                kv_ms = self._cost_curves.c_kv_ms(kv_L)
                m_ms = self._cost_curves.c_m_ms(kv_L)
                ratio = m_ms / kv_ms if kv_ms > 0 else float("inf")
                side = "M-expensive" if ratio > 1 else "KV-expensive"
                regime_str = (
                    f" [csigma@L={kv_L:.0f}: c_KV={kv_ms:.2f}ms "
                    f"c_M={m_ms:.2f}ms ratio={ratio:.2f}x "
                    f"L*={self._cost_curves.L_star:.0f} → {side}]"
                )

        if benefit_us <= 0:
            return False, f"nb: no pressure ({signals.reason_str()}){regime_str}"
        if benefit_us < cost_us * c.nb_margin:
            return False, (
                f"nb: B={benefit_us:.0f}us < C={cost_us:.0f}us × "
                f"margin={c.nb_margin} ({signals.reason_str()}){regime_str}"
            )
        return True, (
            f"nb: B={benefit_us:.0f}us >= C={cost_us:.0f}us × "
            f"margin={c.nb_margin} ({signals.reason_str()}){regime_str}"
        )

    def close(self) -> None:
        """Flush + close the cost JSONL handle (if any)."""
        fp = self._cost_log_fp
        if fp is not None:
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass
            self._cost_log_fp = None

    def _emit_cost_log(
        self,
        decision: "PlanDecision",
        snapshot: dict | None,
        edge_active: bool,
    ) -> None:
        """Append one JSONL record per decide() call when SGLANG_XPOOL_COST_LOG
        is set. Carries enough state to plot the gate's behavior offline:
        usage levels, recovery lengths, cost-curve evaluations, signals
        breakdown from the adapter, fire direction.
        """
        if self._cost_log_fp is None:
            return
        rec: dict = dict(
            ts=round(time.time(), 3),
            tick=self._tick_count,
            usage_kv=decision.usage_kv,
            usage_mamba=decision.usage_mamba,
            queue_depth=decision.queue_depth,
            kv_state=self._kv_state,
            mamba_state=self._mamba_state,
            kv_consec=self._kv_above_high_consec,
            mamba_consec=self._mamba_above_high_consec,
            edge_active=edge_active,
            cooldown=self._cooldown_remaining,
            direction=decision.direction,
            reason=decision.reason,
        )
        if snapshot:
            for k in (
                "mean_recovery_len_kv",
                "mean_recovery_len_rec",
                "mean_recovery_len_retract",
                "num_evicted_tokens_recent",
                "num_retracted_reqs",
                "num_paused_reqs",
                "num_queue_reqs",
                "kv_used_tokens",
                "mamba_usage",
                # Active-slot mamba usage (saturation-guard input distinct
                # from total mamba usage; see paper §sec:design-l2-firegate
                # "Active-slot vs cache-fill saturation").
                "usage_mamba_active",
            ):
                if k in snapshot:
                    v = snapshot[k]
                    t = getattr(v, "total", None)
                    rec[k] = t if isinstance(t, (int, float)) else v
        # Only attach the adapter's breakdown if it was produced THIS tick
        # (otherwise we'd ship the previous decide()'s numbers, which is
        # confusing on cooldown / no-pressure ticks that skip the gate).
        breakdown = getattr(self._adapter, "last_breakdown", None)
        breakdown_serial = breakdown.get("_serial") if breakdown else None
        prev_serial = getattr(self, "_last_emitted_breakdown_serial", None)
        if breakdown and breakdown_serial != prev_serial:
            rec["benefit_breakdown"] = breakdown
            self._last_emitted_breakdown_serial = breakdown_serial
        if self._cost_curves is not None and snapshot:
            kv_L = float(snapshot.get("mean_recovery_len_kv", 0) or 0)
            if kv_L > 0:
                rec["c_kv_ms_at_L"] = self._cost_curves.c_kv_ms(kv_L)
                rec["c_m_ms_at_L"] = self._cost_curves.c_m_ms(kv_L)
                rec["L_star"] = self._cost_curves.L_star
                rec["regime"] = (
                    "M-expensive" if kv_L < self._cost_curves.L_star
                    else "KV-expensive"
                )
        try:
            self._cost_log_fp.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def decide(
        self,
        usage_kv: float,
        usage_mamba: float,
        queue_depth: int = 0,
        snapshot: dict | None = None,
        edge_active: bool = False,
    ) -> PlanDecision:
        """Wrapper around `_decide_inner` that emits one JSONL record per
        call (SGLANG_XPOOL_COST_LOG) and INFO-logs every fire decision."""
        decision = self._decide_inner(
            usage_kv, usage_mamba, queue_depth, snapshot, edge_active
        )
        if decision.direction is not None:
            # Fire decisions always at INFO; quote regime + benefit so post-
            # hoc analysis can correlate fires with the asymmetry direction.
            extra = ""
            if self._cost_curves is not None and snapshot:
                kv_L = float(snapshot.get("mean_recovery_len_kv", 0) or 0)
                if kv_L > 0:
                    kv_ms = self._cost_curves.c_kv_ms(kv_L)
                    m_ms = self._cost_curves.c_m_ms(kv_L)
                    side = (
                        "M>KV" if kv_L < self._cost_curves.L_star else "KV>M"
                    )
                    extra = (
                        f" csigma@L={kv_L:.0f}: c_KV={kv_ms:.2f}ms "
                        f"c_M={m_ms:.2f}ms ({side})"
                    )
            logger.info(
                "[xpool-cost] FIRE tick=%d direction=%s usage_kv=%.2f "
                "usage_mamba=%.2f reason=%s%s",
                self._tick_count, decision.direction,
                decision.usage_kv, decision.usage_mamba,
                decision.reason, extra,
            )
        self._emit_cost_log(decision, snapshot, edge_active)
        return decision

    def _decide_inner(
        self,
        usage_kv: float,
        usage_mamba: float,
        queue_depth: int = 0,
        snapshot: dict | None = None,
        edge_active: bool = False,
    ) -> PlanDecision:
        """Original decision logic. Same return/side-effect behavior as
        before; logging is layered in `decide()` above.
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

        # Paper §sec:design-l2-firegate Eq nb-direction-gate: direction-aware
        # net benefit over both candidate transfer pairs. When enabled,
        # this REPLACES the legacy saturation-driven direction selection
        # (and its persist re-eval / edge-trigger machinery) — fires are
        # arg-max NB across {kv_to_mamba, mamba_to_kv}, gated by α·C_act.
        # Always also update the consec counters before returning so any
        # diagnostic logging downstream still sees the right values.
        if c.nb_direction_aware:
            new_kv = self._classify(
                usage_kv, c.kv_low_water, c.kv_high_water
            )
            new_mamba = self._classify(
                usage_mamba, c.mamba_low_water, c.mamba_high_water
            )
            self._kv_state, self._mamba_state = new_kv, new_mamba
            self._kv_above_high_consec = (
                self._kv_above_high_consec + 1
                if new_kv == self.ABOVE_HIGH else 0
            )
            self._mamba_above_high_consec = (
                self._mamba_above_high_consec + 1
                if new_mamba == self.ABOVE_HIGH else 0
            )
            best_dir, best_nb, why = self._pick_direction_by_nb(
                usage_kv, usage_mamba, snapshot
            )
            if best_dir is not None:
                self._cooldown_remaining = c.cooldown_ticks
                return PlanDecision(
                    direction=best_dir,
                    reason=f"nb_direction: best={best_dir} NB={best_nb:.0f}us "
                           f"[{why}]",
                    usage_kv=usage_kv,
                    usage_mamba=usage_mamba,
                    queue_depth=queue_depth,
                )
            return PlanDecision(
                direction=None,
                reason=f"nb_direction: no candidate cleared gate [{why}]",
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
                # Stable. Two re-evaluation paths from inside a stable
                # state, both pointing the gate at signals it would
                # otherwise miss:
                #   (1) Edge re-eval (paper §sec:design-l2 S_edge): when
                #       the agent observes |Δu| > θ_edge this tick, a
                #       phase transition is in progress even if the
                #       discrete state classifier hasn't crossed yet
                #       (or just crossed and we're now in the
                #       post-crossing stable run). The S_edge benefit
                #       term is one-tick-bounded, so we must re-eval
                #       the gate this tick or lose it.
                #   (2) Persist re-eval: a pool that's been ABOVE_HIGH
                #       long enough has accumulated B_persist; give the
                #       gate another chance even without an edge.
                if c.net_benefit_enabled and edge_active:
                    if (new_mamba == self.ABOVE_HIGH
                            and new_kv != self.ABOVE_HIGH):
                        ok, why = self._net_benefit_ok(
                            snapshot, kv_consec, mamba_consec,
                            edge_active=True,
                        )
                        if ok:
                            self._cooldown_remaining = c.cooldown_ticks
                            return PlanDecision(
                                direction="kv_to_mamba",
                                reason=f"edge_signal: mamba ABOVE_HIGH "
                                       f"({usage_mamba:.2f}) kv={new_kv} "
                                       f"({usage_kv:.2f}) [{why}]",
                                usage_kv=usage_kv,
                                usage_mamba=usage_mamba,
                                queue_depth=queue_depth,
                            )
                    if (new_kv == self.ABOVE_HIGH
                            and new_mamba != self.ABOVE_HIGH):
                        ok, why = self._net_benefit_ok(
                            snapshot, kv_consec, mamba_consec,
                            edge_active=True,
                        )
                        if ok:
                            self._cooldown_remaining = c.cooldown_ticks
                            return PlanDecision(
                                direction="mamba_to_kv",
                                reason=f"edge_signal: kv ABOVE_HIGH "
                                       f"({usage_kv:.2f}) mamba={new_mamba} "
                                       f"({usage_mamba:.2f}) [{why}]",
                                usage_kv=usage_kv,
                                usage_mamba=usage_mamba,
                                queue_depth=queue_depth,
                            )
                period = c.nb_persist_eval_period
                if c.net_benefit_enabled and period > 0:
                    # Re-eval at every period-th tick of sustained ABOVE_HIGH.
                    if (mamba_consec >= period and (mamba_consec % period) == 0
                            and new_mamba == self.ABOVE_HIGH
                            and new_kv != self.ABOVE_HIGH):
                        ok, why = self._net_benefit_ok(
                            snapshot, kv_consec, mamba_consec,
                            edge_active=edge_active,
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
                            edge_active=edge_active,
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

                    # Both pools ABOVE_HIGH simultaneously: the binding
                    # pool is the one with HIGHER usage. The persist re-
                    # eval branches above only fire when exactly one is
                    # ABOVE_HIGH; without this both-pools branch, a
                    # post-fire over-shrink that lifts the destination
                    # pool above HIGH alongside an already-saturating
                    # source pool leaves both saturated and the planner
                    # stuck (paper §sec:design-l2-firegate Lagrange
                    # equalization: arbitrate toward the larger marginal
                    # value).
                    #
                    # Tight activation gate: only fire when one pool is
                    # actually saturated (max usage ≥ both_above_max_threshold,
                    # default 0.95) AND the gap exceeds both_above_min_gap
                    # (default 0.20). The proxy V'_σ ≈ usage_σ saturates
                    # at high usage; firing on small post-fire gap with
                    # both pools merely "high" (e.g., 0.6/0.8) leads to
                    # ping-pong because each fire shifts capacity by
                    # ~10-20% which immediately reverses the gap. Tight
                    # gate restricts this branch to the "one pool truly
                    # binding" regime where Lagrange equalization is
                    # well-defined under the proxy.
                    both_above = (new_kv == self.ABOVE_HIGH
                                  and new_mamba == self.ABOVE_HIGH)
                    long_persist = (kv_consec >= period
                                    and (kv_consec % period) == 0
                                    and mamba_consec >= period)
                    max_u = max(usage_kv, usage_mamba)
                    truly_binding = max_u >= c.both_above_max_threshold
                    if both_above and long_persist and truly_binding:
                        gap = usage_kv - usage_mamba
                        if abs(gap) >= c.both_above_min_gap:
                            ok, why = self._net_benefit_ok(
                                snapshot, kv_consec, mamba_consec,
                                edge_active=edge_active,
                            )
                            if ok:
                                self._cooldown_remaining = c.cooldown_ticks
                                if gap > 0:
                                    direction = "mamba_to_kv"
                                    reason_head = (
                                        f"persist (both above): kv "
                                        f"({usage_kv:.2f}) > mamba "
                                        f"({usage_mamba:.2f}) by "
                                        f"{gap:+.2f} > Δ="
                                        f"{c.hysteresis_delta}; relieve KV"
                                    )
                                else:
                                    direction = "kv_to_mamba"
                                    reason_head = (
                                        f"persist (both above): mamba "
                                        f"({usage_mamba:.2f}) > kv "
                                        f"({usage_kv:.2f}) by "
                                        f"{-gap:+.2f} > Δ="
                                        f"{c.hysteresis_delta}; relieve mamba"
                                    )
                                return PlanDecision(
                                    direction=direction,
                                    reason=f"{reason_head} [{why}]",
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
