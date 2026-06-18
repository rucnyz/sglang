"""
Cross-pool capacity planner (paper §sec:appendix-trigger) driven by real
per-pool pressure signals.

Reads engine snapshot fields (`kv_used_tokens`, `mamba_usage`,
`num_running_reqs`, `num_queue_reqs`, `cache_hit_rate`) and decides
whether to fire a cross-pool transfer in either direction.

Policy: τ-invariant net-benefit (NB) arg-max direction planner
(`_pick_direction_by_nb`). For each ordered pair (σ_src → σ_dst) it scores a
net benefit NB(σ_src→σ_dst) from the per-second pressure signals of the
admission-pressure adapter (paper `eq:nb-lb`) plus the reuse-aware grow/drain
cost, takes the arg-max over the two directions, and fires iff max-NB clears
an actuator-cost threshold α·C_act (the dead-zone gate of paper
`eq:nb-direction-gate`). Per-second rates and seconds-horizons make the
decision invariant to the polling interval τ.
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
class XPoolPolicyConfig:
    kv_high_water: float = 0.85    # KV usage above this → KV-stressed
    kv_low_water: float = 0.50     # KV usage below this → KV-relaxed
    mamba_high_water: float = 0.80
    mamba_low_water: float = 0.40
    # Wall-clock control timescale (τ-invariant — design.md §"Three separated
    # concerns"). `cooldown_min_s` is THE customer-facing knob: the min
    # SECONDS between fires (= how often, at most, to rebalance). Default 32 s
    # reproduces the historical 16-tick × 2 s operating point. In seconds so
    # behaviour is invariant to the polling interval τ
    # (`SGLANG_HIMA_TICK_S`), which is a pure sampling rate, not a knob.
    cooldown_min_s: float = 32.0
    # Payback window the NB benefit is integrated over. design.md keeps this
    # SEPARATE from cooldown (continuous-time/BOCPD needs both), but the
    # customer should reason about ONE timescale: when left None it DERIVES to
    # `cooldown_min_s` (a fire pays back over exactly its lockout — the natural
    # discrete-controller setting; EWMA smoothing damps the noise the
    # design's optional cooldown>amortize buffer would otherwise guard). Set
    # explicitly (< cooldown_min_s) only for extra oscillation margin in noisy
    # regimes; __post_init__ enforces cooldown_min_s >= amortize_horizon_s.
    amortize_horizon_s: float | None = None
    dst_chunks_per_action: int = 1  # In balanced units (lcm-aware).
    # Net-benefit gate (paper §sec:appendix-trigger). Engine-agnostic abstract:
    #   B (benefit) ≥ C (cost) × margin
    # B is the sum of admission-pressure signals translated through the
    # `EnginePressureAdapter` (see budgeter/pressure_adapter.py).
    # Different engines provide different adapters that map their native
    # pool-pressure mechanisms into a uniform "us of GPU time saved"
    # space. SGLang's adapter dominates the eviction term (its primary
    # pressure-relief mechanism), vLLM's would dominate swap/preempt,
    # etc. C is the empirically measured per-fire wall time
    # (`nb_chunk_cost_us`).
    # nb_chunk_cost_us: empirically measured per-fire cuMemUnmap+cuMemMap
    # wall time. ~5,000 us on H200 with 256 MB chunks (Stage-0 calibration).
    # Adapter-specific signal coefficients live in the adapter; the planner
    # only needs the cost.
    nb_chunk_cost_us: float = 5000.0
    nb_margin: float = 1.5
    # Both-full (no-slack) guard toggle. The guard suppresses cross-fire when
    # BOTH pools are occupancy-saturated. On an arena-backed deployment KV is
    # genuinely growable, so growing the bound pool from the peer's cold-cache
    # slack is beneficial (static sweep: shrinking mamba 256→160 at conc22
    # lifts cache_hit 0.32→0.52). SGLANG_XPOOL_BOTH_FULL_GUARD=0 disables it so
    # the m2k grow-KV path can engage when both pools read full but one holds
    # donatable cold cache.
    both_full_guard: bool = True
    # Coupling threshold for the both-full guard. The guard protects COUPLED
    # prefix-cache: it suppresses cross-fire only when cache_hit_rate is at least
    # this (KV tokens and their paired mamba snapshot are co-needed, so draining
    # one orphans the peer and craters cache_hit). Below it the cache is cold /
    # uncoupled (distinct-context workloads) and a cache-full pool is reclaimable
    # slack the transfer should lend — so the guard does not fire.
    both_full_coupling_min: float = 0.05

    def __post_init__(self):
        # ONE customer knob: amortize_horizon_s derives to cooldown_min_s when
        # not set explicitly (a fire pays back over exactly its lockout).
        if self.amortize_horizon_s is None:
            self.amortize_horizon_s = self.cooldown_min_s
        # Fail-loud (no silent clamp): a fire must pay back within the
        # cooldown, else the next fire can re-evaluate and reverse it before
        # payback completes (design.md §"Convergence + oscillation guard").
        if self.cooldown_min_s < self.amortize_horizon_s:
            raise ValueError(
                f"cooldown_min_s ({self.cooldown_min_s}) must be >= "
                f"amortize_horizon_s ({self.amortize_horizon_s}): a fire "
                f"committed for amortize_horizon_s seconds cannot be allowed "
                f"to re-fire (and possibly reverse) before payback. Set "
                f"SGLANG_XPOOL_COOLDOWN_S >= SGLANG_XPOOL_AMORTIZE_S."
            )


def _policy_from_env() -> XPoolPolicyConfig:
    return XPoolPolicyConfig(
        kv_high_water=float(os.environ.get("SGLANG_XPOOL_KV_HIGH", "0.85")),
        kv_low_water=float(os.environ.get("SGLANG_XPOOL_KV_LOW", "0.50")),
        mamba_high_water=float(os.environ.get("SGLANG_XPOOL_MAMBA_HIGH", "0.80")),
        mamba_low_water=float(os.environ.get("SGLANG_XPOOL_MAMBA_LOW", "0.40")),
        cooldown_min_s=float(os.environ.get("SGLANG_XPOOL_COOLDOWN_S", "32")),
        # Advanced override only; derives to cooldown_min_s when unset so the
        # customer reasons about a single timescale.
        amortize_horizon_s=(
            float(os.environ["SGLANG_XPOOL_AMORTIZE_S"])
            if "SGLANG_XPOOL_AMORTIZE_S" in os.environ
            else None
        ),
        dst_chunks_per_action=int(os.environ.get("SGLANG_XPOOL_UNIT", "1")),
        # Per-fire cuMemUnmap+cuMemMap wall time, empirically measured on
        # H200 / 256 MB chunks (~80 us/chunk, ~4.7 ms/fire). 5,000 us is a
        # conservative round-up; operators should re-measure on their hardware
        # via the actuator's `xpool_fire_total_us` log field.
        nb_chunk_cost_us=float(os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_US", "5000")),
        nb_margin=float(os.environ.get("SGLANG_XPOOL_NB_MARGIN", "1.5")),
        both_full_guard=bool(int(os.environ.get("SGLANG_XPOOL_BOTH_FULL_GUARD", "1"))),
        both_full_coupling_min=float(
            os.environ.get("SGLANG_XPOOL_BOTH_FULL_COUPLING_MIN", "0.05")),
    )


@dataclass
class PlanDecision:
    direction: Optional[str]    # "kv_to_mamba", "mamba_to_kv", or None
    reason: str
    usage_kv: float
    usage_mamba: float
    queue_depth: int = 0        # admission-pressure signal at decision time


class XPoolPlanner:
    """τ-invariant net-benefit (NB) arg-max direction cross-pool transfer
    planner, gated by an actuator-cost threshold.

    `_pick_direction_by_nb` scores both transfer directions from the
    admission-pressure adapter's per-second signals plus the reuse-aware
    grow/drain cost and fires the arg-max direction iff it clears α·C_act.

    Read-only with respect to scheduler state. Returns a `PlanDecision`
    each tick; the caller (BudgetAgent) decides whether to apply it
    (e.g., gating on `num_running_reqs == 0` for safety).
    """

    # Per-pool saturation states. `_classify` maps active usage onto these to
    # drive the persist (dwell-above-high) prior in the NB calculation.
    BELOW_LOW = "below_low"
    IN_BAND = "in_band"
    ABOVE_HIGH = "above_high"

    def __init__(
        self,
        config: Optional[XPoolPolicyConfig] = None,
        adapter=None,
    ) -> None:
        self.config = config if config is not None else _policy_from_env()
        if adapter is None:
            from sglang.srt.budgeter.pressure_adapter import get_default_adapter
            adapter = get_default_adapter()
        self._adapter = adapter
        # Cost-curve handle (paper §sec:appendix-trigger). The adapter
        # carries the calibrated curves; we expose them on the planner so
        # decision logging can quote c_σ(\bar{L}) and the L<L* / L>L* regime
        # without re-loading.
        self._cost_curves = adapter.cost_curves
        # Optional JSONL log of every cost-aware decision (one record per
        # decide() call regardless of fire/no-fire). Set SGLANG_XPOOL_COST_LOG
        # to a writable path to enable.
        self._cost_log_path: Optional[str] = (
            os.environ.get("SGLANG_XPOOL_COST_LOG") or None
        )
        self.cost_log_enabled = False
        self._cost_log_fp = None
        if self._cost_log_path:
            try:
                self._cost_log_fp = open(self._cost_log_path, "a", buffering=1)
                self.cost_log_enabled = True
                logger.info(
                    "[xpool-cost] decisions logged to %s", self._cost_log_path
                )
            except OSError as e:
                logger.warning(
                    "[xpool-cost] cannot open SGLANG_XPOOL_COST_LOG=%s: %s",
                    self._cost_log_path, e,
                )
        # Wall clock accumulated from per-tick `dt` (seconds). Using the dt
        # stream (not time.time()) keeps the cooldown τ-invariant and
        # replayable. `_last_fire_clock_s = -inf` so the first fire is never
        # blocked by cooldown.
        self._clock_s: float = 0.0
        self._last_fire_clock_s: float = float("-inf")
        self._tick_count: int = 0
        # Per-pool saturation state. Initialized to IN_BAND so the very first
        # tick's reading establishes the baseline without forcing a transfer
        # at startup.
        self._kv_state: str = self.IN_BAND
        self._mamba_state: str = self.IN_BAND
        # Persist counters: consecutive ABOVE_HIGH ticks (carried to the
        # adapter so its cold-start fallback can read saturation length) and
        # dwell SECONDS above high-water (the τ-invariant persist signal: how
        # long the pool has been saturated, independent of the polling
        # interval).
        self._kv_above_high_consec: int = 0
        self._mamba_above_high_consec: int = 0
        self._kv_dwell_s: float = 0.0
        self._mamba_dwell_s: float = 0.0
        # Last `last_breakdown._serial` we shipped in a JSONL record.
        # Lets `decide()` ship the adapter's breakdown only when it was
        # produced THIS tick, not the previous one (matters on cooldown /
        # no-pressure ticks that skip the gate).
        self._last_emitted_breakdown_serial = None
        logger.info(
            "XPoolPlanner: kv_high=%.2f kv_low=%.2f mamba_high=%.2f "
            "mamba_low=%.2f cooldown_min_s=%.1f amortize_horizon_s=%.1f "
            "unit=%d",
            self.config.kv_high_water, self.config.kv_low_water,
            self.config.mamba_high_water, self.config.mamba_low_water,
            self.config.cooldown_min_s, self.config.amortize_horizon_s,
            self.config.dst_chunks_per_action,
        )

    def _classify(self, usage: float, low: float, high: float) -> str:
        """Map active usage onto a saturation state for the persist prior."""
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
        """Paper §sec:appendix-trigger admission-pressure adapter
        (which design.md §"Empirical pressure signal" elevates to PRIMARY
        signal — "always uses the paper's admission-pressure adapter" — since
        paper's closed-form $\\hat\\pi_i$ assumes IRM/Poisson which agent
        traffic violates, so we use the multi-source adapter always).

        NB for each direction sums:
          - eviction-cost term:  c_σ(L_σ) × P_save_σ  (when L observed)
          - admission-pressure:  queue_us, paused_us, retract_us
                                 attributed to the saturated pool
          - persist-cost:        dwell SECONDS above high-water
                                 (_{kv,mamba}_dwell_s) × persist_tick_us,
                                 per direction (τ-invariant saturation prior)

        Cold-start (L=0): the L-cost terms drop out but pressure terms
        carry signal — paper §sec:appendix-trigger (the `eq:nb-lb` adapter
        surrogate) explicitly engages this way on the saturated face of the
        dual.

        Returns:
            (best_direction, best_nb_us, reason_str)
            best_direction is "kv_to_mamba" / "mamba_to_kv" or None.
        """
        if self._cost_curves is None:
            return None, 0.0, "nb_direction: no cost curves"
        c = self.config
        snap = snapshot or {}
        # `dt` (wall seconds since previous tick) is validated present + >0 by
        # `_decide_inner`; the adapter prices flow signals as per-second rates
        # (count ÷ dt) and the per-tick grow flow is converted below.
        dt = float(snap["dt"])

        # ---- (1) Eviction-cost terms (c_σ(L)) — when L observed ----
        L_kv = float(snap.get("slow_recovery_len_kv", 0) or 0)
        L_m = float(snap.get("slow_recovery_len_rec", L_kv) or L_kv)
        if L_kv > 0 or L_m > 0:
            c_kv = self._cost_curves.c_kv_us(L_kv if L_kv > 0 else L_m)
            c_m = self._cost_curves.c_m_us(L_m if L_m > 0 else L_kv)
        else:
            c_kv = c_m = 0.0  # cold start — eviction-cost contributes 0

        p_save_kv = self._p_func(usage_kv, c.kv_low_water)
        p_loss_kv = p_save_kv  # same functional form per Eq p-loss-save
        p_save_m = self._p_func(usage_mamba, c.mamba_low_water)
        p_loss_m = p_save_m

        # ---- (2) Multi-source admission-pressure signals (per-second rates) ----
        # paper §sec:appendix-trigger Adapter(SGLang) flow/stock signals =
        #   { S_evict, S_retract, S_pause, S_queue } (S_persist added below)
        # The adapter returns RATES (µs/s): flow signals (evict/retract) are
        # divided by dt; stock signals (queue/paused depth) are priced per
        # second. EWMA-smoothing of the rate stream lives in the adapter.
        signals = self._adapter.signals_from_snapshot(
            snap,
            kv_consec=self._kv_above_high_consec,
            mamba_consec=self._mamba_above_high_consec,
            dt=dt,
        )
        # S_evict is direction-aware via c_σ(L) × P_save above; the
        # adapter's evict_us is a single scalar (KV-side observation
        # only), used here only as a fallback for cold-start when L
        # is 0 but evict has just begun.
        evict_us_cold = signals.evict_us if (L_kv <= 0 and L_m <= 0) else 0.0

        # Saturation-weighted attribution. Queue / pause / retract /
        # cold-evict pressure is attributed to relieving pool σ in
        # proportion to σ's saturation index `P_save_σ` (see `_p_func`).
        # This mirrors the eviction-cost `c_σ(L) × P_save_σ` form: pool σ
        # is queue-responsible to the extent it is near saturation. Below
        # low_water (P_save_σ = 0) the signal does not count as memory-bound
        # — queue is then compute / batch / SLO bound and a memory-fire
        # cannot relieve it.
        #
        # Properties of saturation-weighted attribution:
        #   - pressure_to_σ ≤ admit_pressure_aggregate (P_save_σ ≤ 1)
        #   - pressure_to_kv + pressure_to_m can exceed admit_pressure
        #     when both pools are near saturation (this is correct: the
        #     same queue stall can be relieved by either fire direction,
        #     and the NB calculation picks max — counted once via argmax,
        #     not summed)
        #   - At marginal saturation (P_save ≈ 0.03) the attribution is
        #     marginal (~3% of queue), restoring proportionality.
        admit_pressure_aggregate = (
            signals.queue_us + signals.paused_us + signals.retract_us
            + evict_us_cold
        )
        # Memory-bound credibility gate. admit_pressure_aggregate signals
        # SOMETHING is stalling admission, but doesn't say WHAT. P_save_σ
        # already weights "value of relieving σ", but that assumes the stall
        # IS memory-bound. If neither pool is credibly near saturation, the
        # queue is compute / batch / SLO-bound and a memory fire delivers
        # zero — without this gate the cost model would fire at marginal
        # P_save × big queue (the queue contribution alone clearing the
        # threshold while max P_save ≈ 0.03).
        #
        # Bayesian factor: P(stall is memory-bound) ≈ max(P_save_σ).
        # Discount the WHOLE admit_pressure by this credibility. Same
        # P_save ramp the eviction-cost term (c_σ × P_save_σ) and the
        # per-pool attribution (admit × P_save_σ) already use — this
        # extends the ramp one level up to gate the signal's claim to
        # being memory-bound at all.
        mem_bound_credibility = max(p_save_kv, p_save_m)
        admit_pressure_credible = admit_pressure_aggregate * mem_bound_credibility
        pressure_to_kv = admit_pressure_credible * p_save_kv
        pressure_to_m = admit_pressure_credible * p_save_m
        # Marginal-fire cap. NB prices the net benefit of ONE fire, but
        # `admit_pressure` prices the WHOLE admission backlog; one fire moves
        # only the available source free pages and so may admit only a few of the
        # queued requests. Crediting the full backlog to a single fire makes
        # the budgeter fire when the MARGINAL relief is near-zero — e.g. it
        # drains a hot recur mamba cache to grow a saturated KV pool that the
        # fire can only nudge (a zero-downside violation: dynamic worse than
        # static). Scale the per-direction pressure benefit by the fraction
        # of the queue ONE fire actually clears: `min(1, fire_admit_σ /
        # num_queue)`. The agent supplies `fire_admit_σ` = #queued reqs one
        # σ-grow fire admits (grant ÷ avg req size). Absent (back-compat) ⇒ 1.0
        # (whole-queue). The eviction-cost term (`c_σ × P_save`) and the
        # reuse-aware grow term are already per-unit bounded, so only the
        # queue/pressure term needs this; this is the single over-counting
        # benefit that breaks zero-downside on phase-shifting (Case-3) loads.
        snap_q = snap.get("num_queue_reqs", 0)
        queued = getattr(snap_q, "total", snap_q)
        queued = int(queued) if isinstance(queued, (int, float)) else 0
        marg_kv = 1.0
        marg_m = 1.0
        if queued > 0:
            marg_kv = min(1.0, float(snap.get("fire_admit_kv", queued)) / queued)
            marg_m = min(1.0, float(snap.get("fire_admit_mamba", queued)) / queued)
            pressure_to_kv *= marg_kv
            pressure_to_m *= marg_m
        # Persist (saturation prior), per-direction: dwell SECONDS above
        # high-water × a per-second-of-dwell weight (`persist_tick_us` is now
        # interpreted as µs of accumulated evidence per second saturated).
        # τ-invariant (dwell is wall-time, not a tick count). Added to NB
        # directly as a one-shot benefit (µs), NOT integrated over the
        # amortization horizon.
        # The marginal-fire cap applies to ALL saturation-relief benefit
        # terms — persist (dwell prior) and the c_σ×P_save eviction-cost term
        # (below) — not just `pressure_to`: each prices the value of relieving
        # σ's saturation, which ONE fire only partially delivers. Without it
        # the residual c_σ×P_save + persist tip NB[m2k] positive even after
        # the coupled grow_σ and the paired mamba drain cancel, so the
        # budgeter churns the coupled recur cache for ~0 net.
        persist_kv = self._kv_dwell_s * self._adapter.persist_tick_us * marg_kv
        persist_m = self._mamba_dwell_s * self._adapter.persist_tick_us * marg_m

        # ---- (3) NB over the amortization horizon (τ-invariant) ----
        # Benefit RATES (µs/s) are integrated over `amortize_horizon_s`
        # seconds. `pressure_to_*` is already a per-second rate (the adapter
        # divides flow signals by dt and prices stock signals per second);
        # the c_σ(L)×P_save eviction-cost rate joins it, as does the
        # reuse-aware grow signal (a per-tick flow → ÷dt). One-time costs
        # (the reuse-aware drain) and the persist prior are added/subtracted
        # directly, not scaled by the horizon.
        horizon_s = c.amortize_horizon_s
        # Mamba drain cost for m2k. Prefer the reuse-aware (hit-weighted)
        # eviction cost of the snapshots an m2k drain would force out,
        # supplied by the agent as snapshot["mamba_drain_cost_us"]
        # (MambaRadixCache.predict_evict_cost_us summed over the LRU+LPB
        # eviction victims with their hit counts). This is the REALIZED
        # loss and is paid once per fire — it is NOT scaled by p_loss_m.
        # The legacy estimate c_m × p_loss_m prices the drain off ACTIVE
        # utilization, which is blind to cache reuse: on a hot-but-active-
        # low mamba cache both c_m (re-prefill curve, 0 at L=0) and
        # p_loss_m (active-based P_save) collapse to ~0, so the drain
        # reads as free and m2k drains a HOT cache. Fall back to the
        # active estimate when the agent does not supply the reuse-aware
        # cost (keeps NB byte-identical when the field is absent).
        mamba_drain_cost_reuse = snap.get("mamba_drain_cost_us")
        kv_drain_cost_reuse = snap.get("kv_drain_cost_us")
        # NB(kv_to_mamba) = grow mamba → relieve mamba-side pressure
        # + avoid future mamba evict;  lose kv-side eviction-cost.
        #
        # Grow-side eviction benefit (symmetric to the drain cost): a pool
        # actively shedding (hot) cache should be GROWN. The per-tick benefit
        # of growing pool σ is the reuse-aware cost of the evictions σ is
        # currently forced into — supplied by the agent as
        # snapshot["{mamba,kv}_evict_grow_us"] (predict_evict_cost_us over
        # σ's recent eviction count). This is NOT gated by the active-
        # utilization P_save_σ: a pool 97%-occupied and shedding hot snapshots
        # reads as "not pressured" because its ACTIVE slot use is moderate, so
        # the active-gated `c_σ × P_save_σ` term collapses to ~0 and the
        # planner would never grow it. The grow term is naturally reuse-aware
        # (cold-only shedding → ~0) and 0 when σ is not evicting (preserves
        # test_G).
        grow_mamba = float(snap.get("mamba_evict_grow_us", 0.0) or 0.0)
        grow_kv = float(snap.get("kv_evict_grow_us", 0.0) or 0.0)
        nb_k2m = horizon_s * (
            c_m * p_save_m * marg_m       # marginal-fire cap: one fire avoids
            + pressure_to_m               # only grant/L of σ's evictions
            + grow_mamba / dt             # grow_* is reuse-aware (already bounded)
        ) + persist_m
        # KV drain cost for k2m (symmetric to the mamba drain cost
        # below): prefer the reuse-aware (hit-weighted) KV eviction cost the
        # agent supplies as snapshot["kv_drain_cost_us"] — a walk of the KV
        # RadixCache victims a k2m fire would force out — subtracted ONCE per
        # fire (the eviction is a one-time event), so a hot KV cache resists
        # the drain. Fall back to the legacy active `amortize_horizon_s ×
        # c_kv × P_loss_kv` estimate when the agent does not supply it
        # (preserves prior behavior for non-reuse-aware callers).
        if kv_drain_cost_reuse is not None:
            nb_k2m -= float(kv_drain_cost_reuse)  # reuse-aware, paid once
        else:
            nb_k2m -= horizon_s * c_kv * p_loss_kv  # legacy active estimate
        # NB(mamba_to_kv) = symmetric. Grow KV (relieve KV-side pressure,
        # avoid future KV evict + the KV eviction it is currently paying)
        # minus the cost of draining mamba.
        nb_m2k = horizon_s * (
            c_kv * p_save_kv * marg_kv    # marginal-fire cap
            + pressure_to_kv
            + grow_kv / dt
        ) + persist_kv
        if mamba_drain_cost_reuse is not None:
            nb_m2k -= float(mamba_drain_cost_reuse)  # reuse-aware, paid once
        else:
            nb_m2k -= horizon_s * c_m * p_loss_m  # legacy active estimate

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

        # Both-full (no-slack) guard, gated on cache COUPLING. Cross-pool
        # transfer moves capacity from a SLACK pool to a BOUND one. When BOTH
        # pools are cache-inclusive-saturated, neither has free slack, so a fire
        # can only evict cached entries to grow the peer. That craters cache_hit
        # ONLY when the caches are COUPLED: under prefix-cache demand a hit needs
        # both the KV tokens AND the paired mamba snapshot, so draining one
        # orphans the other's still-hot paired entries (measured: cc conc-22 LPB,
        # m2k 27x, cache_hit 0.50->0.10). The NB misses this — the per-pool drain
        # cost reads a victim cold on its OWN reuse even when its peer half is
        # hot. The coupling signal is `cache_hit_rate`: with hits present the
        # entries are co-needed (guard); with cache_hit below
        # `both_full_coupling_min` (distinct-context workloads — the cc KV-bound
        # and dynamic-shift cases) a cache-full pool is COLD reclaimable slack the
        # transfer SHOULD lend, so the guard stands down. Uses cache-inclusive
        # occupancy (the harm is cache eviction), gated by coupling.
        snap_occ_kv = float((snapshot or {}).get("pool_occupancy_kv", usage_kv))
        snap_occ_m = float((snapshot or {}).get("pool_occupancy_mamba", usage_mamba))
        cache_hit_rate = float((snapshot or {}).get("cache_hit_rate", 0.0) or 0.0)
        if (
            c.both_full_guard
            and cache_hit_rate >= c.both_full_coupling_min
            and snap_occ_kv >= c.kv_high_water
            and snap_occ_m >= c.mamba_high_water
        ):
            nb_k2m = float("-inf")
            nb_m2k = float("-inf")

        # Actuator cost: prefer the runtime EWMA over the static config.
        # Static config nb_chunk_cost_us is an idle-time lower bound;
        # under live traffic the cuMemUnmap+cuMemMap pair pays additional
        # CUDA-graph deferral and allocator-contention cost. The runtime
        # EWMA tracks fire wall-times observed via cudaSynchronize-bracketed
        # measurement (paper §sec:appendix-trigger "Runtime actuator-cost
        # EWMA").
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
            f"pressure_to: kv={pressure_to_kv:.0f}us m={pressure_to_m:.0f}us, "
            f"persist: kv={persist_kv:.0f}us m={persist_m:.0f}us)"
        )
        return best_dir, best_nb, reason

    def close(self) -> None:
        """Flush + close the cost JSONL handle (if any)."""
        if self.cost_log_enabled:
            try:
                self._cost_log_fp.flush()
                self._cost_log_fp.close()
            except Exception:
                pass
            self.cost_log_enabled = False
            self._cost_log_fp = None

    def _emit_cost_log(
        self,
        decision: "PlanDecision",
        snapshot: dict | None,
    ) -> None:
        """Append one JSONL record per decide() call when SGLANG_XPOOL_COST_LOG
        is set. Carries enough state to plot the gate's behavior offline:
        usage levels, recovery lengths, signals breakdown from the adapter,
        fire direction.
        """
        if not self.cost_log_enabled:
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
            cooldown_s_left=max(
                0.0,
                self.config.cooldown_min_s
                - (self._clock_s - self._last_fire_clock_s),
            ),
            direction=decision.direction,
            reason=decision.reason,
        )
        if snapshot:
            for k in (
                "slow_recovery_len_kv",
                "slow_recovery_len_rec",
                "slow_recovery_len_retract",
                "num_evicted_tokens_recent",
                "num_retracted_reqs",
                "num_paused_reqs",
                "num_queue_reqs",
                "kv_used_tokens",
                "mamba_usage",
                # Active-slot mamba usage (saturation-guard input distinct
                # from total mamba usage; the saturation index P_save_j keys
                # on active utilization u_j per paper §sec:appendix-trigger
                # `eq:nb-lb`, not cache fill).
                "usage_mamba_active",
            ):
                if k in snapshot:
                    v = snapshot[k]
                    t = getattr(v, "total", None)
                    rec[k] = t if isinstance(t, (int, float)) else v
        # Only attach the adapter's breakdown if it was produced THIS tick
        # (otherwise we'd ship the previous decide()'s numbers, which is
        # confusing on cooldown / no-pressure ticks that skip the gate).
        breakdown = self._adapter.last_breakdown
        breakdown_serial = breakdown.get("_serial") if breakdown else None
        prev_serial = self._last_emitted_breakdown_serial
        if breakdown and breakdown_serial != prev_serial:
            rec["benefit_breakdown"] = breakdown
            self._last_emitted_breakdown_serial = breakdown_serial
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
    ) -> PlanDecision:
        """Wrapper around `_decide_inner` that emits one JSONL record per
        call (SGLANG_XPOOL_COST_LOG) and INFO-logs every fire decision."""
        decision = self._decide_inner(
            usage_kv, usage_mamba, queue_depth, snapshot
        )
        if decision.direction is not None:
            logger.info(
                "[xpool-cost] FIRE tick=%d direction=%s usage_kv=%.2f "
                "usage_mamba=%.2f reason=%s",
                self._tick_count, decision.direction,
                decision.usage_kv, decision.usage_mamba,
                decision.reason,
            )
        self._emit_cost_log(decision, snapshot)
        return decision

    def _decide_inner(
        self,
        usage_kv: float,
        usage_mamba: float,
        queue_depth: int = 0,
        snapshot: dict | None = None,
    ) -> PlanDecision:
        """Arg-max NB decision logic. Same return/side-effect behavior as the
        wrapper sees; logging is layered in `decide()` above.
        """
        self._tick_count += 1
        c = self.config

        # Advance the τ-invariant wall clock by this tick's elapsed seconds.
        # `dt` is REQUIRED (no fallback): a caller that does not supply the
        # elapsed time cannot be priced in rates, and a silent default would
        # reintroduce the tick-coupling this design removes.
        snap_for_dt = snapshot or {}
        _dt_raw = snap_for_dt.get("dt")
        if _dt_raw is None or float(_dt_raw) <= 0.0:
            raise ValueError(
                "XPoolPlanner.decide requires snapshot['dt'] > 0 (wall "
                "seconds since the previous tick); got "
                f"{_dt_raw!r}. The planner prices signals as per-second "
                "rates and gates the cooldown in wall-clock seconds."
            )
        dt = float(_dt_raw)
        self._clock_s += dt

        cooldown_elapsed_s = self._clock_s - self._last_fire_clock_s
        if cooldown_elapsed_s < c.cooldown_min_s:
            return PlanDecision(
                direction=None,
                reason=(
                    f"cooldown ({c.cooldown_min_s - cooldown_elapsed_s:.1f}s "
                    f"left of {c.cooldown_min_s:.1f}s)"
                ),
                usage_kv=usage_kv,
                usage_mamba=usage_mamba,
                queue_depth=queue_depth,
            )

        # Paper §sec:appendix-trigger: direction-aware net benefit over both
        # candidate transfer pairs, built on the `eq:nb-lb` adapter surrogate
        # and gated by the `eq:nb-direction-gate` dead-zone (α·C_act). Fires
        # are arg-max NB across {kv_to_mamba, mamba_to_kv}.
        #
        # Persist consec counter uses ACTIVE-only usage (design.md
        # §"Empirical pressure signal", the active-utilization u_j of the
        # adapter — only live state pins capacity). Cached snapshots can be
        # LRU-evicted on next admission and never stall it; counting them
        # inflates persist signal and fires on phantom saturation (idle
        # workloads where radix cache fills to 99% but admission has lots of
        # headroom). Falls back to total usage if the snapshot doesn't carry
        # the active fields (back-compat with callers that don't populate them).
        snap_ = snapshot or {}
        # Use explicit `in` check + `is None` guard. Don't use
        # `snap_.get(k, fallback) or fallback` — that pattern treats the legit
        # value 0.0 (zero live state) as falsy and falls back to the total,
        # defeating the whole point of the active signal on idle workloads
        # where usage_*_active is literally 0 by design.
        if "usage_kv_active" in snap_ and snap_["usage_kv_active"] is not None:
            usage_kv_active = float(snap_["usage_kv_active"])
        else:
            usage_kv_active = usage_kv
        if "usage_mamba_active" in snap_ and snap_["usage_mamba_active"] is not None:
            usage_mamba_active = float(snap_["usage_mamba_active"])
        else:
            usage_mamba_active = usage_mamba
        new_kv = self._classify(
            usage_kv_active, c.kv_low_water, c.kv_high_water
        )
        new_mamba = self._classify(
            usage_mamba_active, c.mamba_low_water, c.mamba_high_water
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
        # Dwell SECONDS above high-water (τ-invariant persist signal):
        # accumulate this tick's `dt` while saturated, reset on exit.
        self._kv_dwell_s = (
            self._kv_dwell_s + dt if new_kv == self.ABOVE_HIGH else 0.0
        )
        self._mamba_dwell_s = (
            self._mamba_dwell_s + dt if new_mamba == self.ABOVE_HIGH else 0.0
        )
        # Direction NB must read the active-only signal: mamba's raw
        # usage mixes admission slots (real bottleneck) with radix-tree
        # snapshots (cache fill — cheaply LRU-evictable, NOT pressure).
        # On CC traces with a hot mamba cache (raw ≈ 0.95, active ≈ 0.20),
        # feeding raw usage would make P_save_m read the cache as pressure
        # and fire k2m, shrinking the *real* bottleneck KV to grow a pool
        # with 75%+ slack. The source-saturation guard in
        # `_pick_direction_by_nb` already consults `_active`; carry that
        # distinction through to the NB calculation itself.
        best_dir, best_nb, why = self._pick_direction_by_nb(
            usage_kv_active, usage_mamba_active, snapshot
        )
        if best_dir is not None:
            self._last_fire_clock_s = self._clock_s
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
