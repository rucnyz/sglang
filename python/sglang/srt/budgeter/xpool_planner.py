"""Cross-pool capacity planner: payback-based pool sizing.

Three signals capture "this pool is too small" (all in µs/s):
  1. R_evict:     eviction rate × recovery cost (re-computation waste)
  2. R_admission: queue depth × opportunity cost (concurrency waste from max_running cap)
  3. urgency:     time-to-fill multiplier (proactive: act before pool fills)

Unified formula:
  R(pool) = urgency(pool) × [R_evict(pool) + R_admission(pool)]
  fire iff: (R(dst) - R(src)) × cooldown > fire_cost

Direction: grow the pool with higher total harm rate.
Self-converges: pool grows → harm rate drops → fires stop.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PaybackConfig:
    cooldown_s: float = 10.0
    ewma_tau_s: float = 5.0
    # Fire iff payback > fire_cost × margin. 1.0 = break-even threshold;
    # higher demands the transfer recoup a multiple of its cost before firing.
    payback_margin: float = 1.0
    # Convergence backoff: the planner assumes a fire (grow the pool) reduces
    # that pool's harm — self-convergence. When the harm is NOT capacity-elastic
    # (the working set exceeds total memory, so growing the pool never drops the
    # eviction rate), that assumption fails and the planner would fire every
    # tick forever for zero gain, paying only cuMem cost. So each direction's
    # cooldown widens by 2× per consecutive ineffective fire (harm did not drop
    # by ≥ converge_eps since the last fire in that direction), capped at
    # 2**converge_backoff_cap; a single effective fire resets it. The widened
    # cooldown still lets a periodic probe fire re-measure elasticity, so the
    # planner resumes immediately if the workload becomes elastic again.
    converge_backoff_cap: int = 4
    converge_eps: float = 0.05
    # Active-usage direction guard. r_evict is a CACHE-reuse harm signal and is
    # blind to the fact that SHRINKING a pool also displaces that pool's LIVE
    # working set. When the harm points at growing the pool with LESS live
    # demand (e.g. 35B swarm: the smaller mamba pool sheds more re-prefill cost
    # by tombstoning the hot shared-prefix trunk, yet KV carries more live
    # work), the transfer steals memory from the genuinely-bottlenecked pool.
    # This guard completes the demand accounting: never grow a pool by shrinking
    # one with strictly higher active usage.
    active_usage_guard: bool = True

    def __post_init__(self):
        if self.cooldown_s <= 0:
            raise ValueError(f"cooldown_s must be positive, got {self.cooldown_s}")
        if self.payback_margin <= 0:
            raise ValueError(
                f"payback_margin must be positive, got {self.payback_margin}"
            )
        if self.converge_backoff_cap < 0:
            raise ValueError(
                f"converge_backoff_cap must be >= 0, got {self.converge_backoff_cap}"
            )


def _config_from_env() -> PaybackConfig:
    from sglang.srt.environ import envs

    return PaybackConfig(
        cooldown_s=float(os.environ.get("SGLANG_XPOOL_COOLDOWN_S", "10.0")),
        payback_margin=envs.SGLANG_XPOOL_PAYBACK_MARGIN.get(),
    )


@dataclass
class PlanDecision:
    direction: Optional[str]
    reason: str


class PaybackPlanner:
    """Three-signal payback planner for cross-pool transfers.

    Every tick, computes the total harm rate R(pool) = R_evict + R_admission
    for each pool, applies an urgency multiplier, and fires toward the pool
    with higher harm rate if the payback condition is met.
    """

    def __init__(self, cost_curves=None, config: Optional[PaybackConfig] = None,
                 fire_cost_us: float = 5000.0):
        self.config = config or _config_from_env()
        self._cost_curves = cost_curves
        self._fire_cost_us = fire_cost_us
        self._kv_evict_ewma: float = 0.0
        self._m_evict_ewma: float = 0.0
        self._kv_occ_prev: float = 0.0
        self._m_occ_prev: float = 0.0
        self._last_fire_clock: float = 0.0
        self._fire_count: int = 0
        # Convergence-backoff state (per direction). _dir_last_harm: r_dst at the
        # last committed fire; _dir_ineffective: consecutive ineffective-fire
        # count (drives the 2** backoff); _dir_last_fire_clock: last committed
        # fire time in that direction.
        self._dir_last_harm: dict = {}
        self._dir_ineffective: dict = {}
        self._dir_last_fire_clock: dict = {}
        logger.info(
            "PaybackPlanner: cooldown=%.1fs fire_cost=%.0fus",
            self.config.cooldown_s, self._fire_cost_us,
        )

    def update_fire_cost(self, us: float) -> None:
        self._fire_cost_us = us

    def decide(self, snapshot: dict, clock_s: float, dt: float) -> PlanDecision:
        # Signal 1: R_evict — the reuse-weighted LPB LOSS shed per pool (us),
        # NOT the raw evicted-token/slot count. A block's loss is
        # `n_b · c_recompute(s_b)`: never-reused cache (n_b=0) costs ~0 to
        # evict, so a low-reuse pool's churn no longer inflates its harm rate
        # (the swarm k2m/m2k oscillation root). The value is already the exact
        # recompute cost, so r_evict IS the EWMA'd loss rate directly.
        kv_evict = float(snapshot.get("kv_evicted_lpb_loss_recent", 0) or 0)
        m_evict = float(snapshot.get("mamba_evicted_lpb_loss_recent", 0) or 0)

        alpha = 1.0 - math.exp(-dt / self.config.ewma_tau_s) if dt > 0 else 0.1
        safe_dt = max(dt, 1e-6)
        self._kv_evict_ewma = alpha * (kv_evict / safe_dt) + (1 - alpha) * self._kv_evict_ewma
        self._m_evict_ewma = alpha * (m_evict / safe_dt) + (1 - alpha) * self._m_evict_ewma

        n_running = float(snapshot.get("num_running_reqs", 0) or 0)

        r_evict_kv = self._kv_evict_ewma
        r_evict_m = self._m_evict_ewma

        # Signal 2: R_admission (concurrency waste from queue depth)
        # Attribution: in sglang, max_running is ALWAYS determined by mamba
        # pool size (mamba_live / ratio). So if requests are queuing because
        # num_running == max_running_mamba, the queue is caused by mamba.
        W = float(snapshot.get("num_queue_reqs", 0) or 0)
        max_running_m = float(snapshot.get("max_running_mamba", 0) or 0)
        N = max(n_running, 1.0)
        r_admit_kv = 0.0
        r_admit_m = 0.0
        if W > 0 and max_running_m > 0 and n_running >= max_running_m * 0.9:
            r_admit_m = W * 1e6 / N
        kv_occ = float(snapshot.get("pool_occupancy_kv", 0) or 0)
        m_occ = float(snapshot.get("mamba_usage", 0) or 0)

        # Signal 3: urgency (proactive: time-to-fill multiplier)
        tick_s = max(dt, 1.0)
        urgency_kv = self._urgency(kv_occ, self._kv_occ_prev, dt, tick_s)
        urgency_m = self._urgency(m_occ, self._m_occ_prev, dt, tick_s)
        self._kv_occ_prev = kv_occ
        self._m_occ_prev = m_occ

        # Total harm rate per pool
        r_kv = urgency_kv * (r_evict_kv + r_admit_kv)
        r_m = urgency_m * (r_evict_m + r_admit_m)

        if r_kv >= r_m:
            direction = "mamba_to_kv"
            net_benefit_rate = r_kv - r_m
        else:
            direction = "kv_to_mamba"
            net_benefit_rate = r_m - r_kv

        # Active-usage direction guard (see PaybackConfig.active_usage_guard):
        # never grow a pool by shrinking one that carries strictly more live
        # work. r_evict measures only cache-reuse harm, so on its own it will
        # feed an inelastic cache signal (the source's re-prefill churn) at the
        # expense of the destination's genuinely-bottlenecked live working set.
        if self.config.active_usage_guard:
            u_kv_active = float(snapshot.get("usage_kv_active", 0.0) or 0.0)
            u_m_active = float(snapshot.get("usage_mamba_active", 0.0) or 0.0)
            if direction == "kv_to_mamba" and u_kv_active > u_m_active:
                return PlanDecision(
                    direction=None,
                    reason=f"active-usage guard: KV more live-active "
                           f"({u_kv_active:.2f} > {u_m_active:.2f}); won't shrink it to grow mamba",
                )
            if direction == "mamba_to_kv" and u_m_active > u_kv_active:
                return PlanDecision(
                    direction=None,
                    reason=f"active-usage guard: mamba more live-active "
                           f"({u_m_active:.2f} > {u_kv_active:.2f}); won't shrink it to grow KV",
                )

        if clock_s - self._last_fire_clock < self.config.cooldown_s:
            return PlanDecision(
                direction=None,
                reason=f"cooldown ({clock_s - self._last_fire_clock:.1f}s "
                       f"< {self.config.cooldown_s:.1f}s, "
                       f"R_kv={r_kv:.0f} R_m={r_m:.0f})",
            )

        payback = net_benefit_rate * self.config.cooldown_s
        threshold_us = self._fire_cost_us * self.config.payback_margin
        if payback > threshold_us:
            # Convergence backoff. r_dst is the harm of the pool we would grow.
            # If the last fire in this direction did NOT drop r_dst by at least
            # converge_eps, that fire was ineffective (harm not capacity-elastic)
            # — widen this direction's cooldown 2× per consecutive miss so we
            # stop paying cuMem cost for a transfer that cannot help. An
            # effective fire (harm dropped) resets the streak.
            r_dst = r_m if direction == "kv_to_mamba" else r_kv
            prev_harm = self._dir_last_harm.get(direction)
            effective = (
                prev_harm is None
                or r_dst < prev_harm * (1.0 - self.config.converge_eps)
            )
            streak = 0 if effective else self._dir_ineffective.get(direction, 0)
            backoff = 2 ** min(streak, self.config.converge_backoff_cap)
            eff_cooldown = self.config.cooldown_s * backoff
            since = clock_s - self._dir_last_fire_clock.get(direction, -1e18)
            if since < eff_cooldown:
                return PlanDecision(
                    direction=None,
                    reason=f"convergence backoff ({direction}: harm not responding, "
                           f"streak={streak}, eff_cd={eff_cooldown:.0f}s, "
                           f"since={since:.0f}s, R_dst={r_dst:.0f})",
                )
            self._dir_ineffective[direction] = 0 if effective else streak + 1
            self._dir_last_harm[direction] = r_dst
            self._dir_last_fire_clock[direction] = clock_s
            self._last_fire_clock = clock_s
            self._fire_count += 1
            return PlanDecision(
                direction=direction,
                reason=f"payback: net={net_benefit_rate:.0f}us/s × {self.config.cooldown_s:.0f}s "
                       f"= {payback:.0f}us > {threshold_us:.0f}us "
                       f"(margin {self.config.payback_margin:.1f}, backoff×{backoff}, "
                       f"R_kv={r_kv:.0f}[evict={r_evict_kv:.0f}+admit={r_admit_kv:.0f}] "
                       f"R_m={r_m:.0f}[evict={r_evict_m:.0f}+admit={r_admit_m:.0f}])",
            )

        return PlanDecision(
            direction=None,
            reason=f"no payback: net={net_benefit_rate:.0f}us/s "
                   f"(R_kv={r_kv:.0f} R_m={r_m:.0f})",
        )

    @staticmethod
    def _urgency(occ: float, occ_prev: float, dt: float, tick_s: float) -> float:
        """Time-pressure multiplier: > 1 when pool is filling fast."""
        if dt <= 0 or occ <= 0:
            return 1.0
        d_occ = occ - occ_prev
        if d_occ <= 0:
            return 1.0
        rate = d_occ / dt
        free = max(1.0 - occ, 1e-6)
        time_to_fill = free / rate
        return max(1.0, tick_s / time_to_fill)
