"""Phase 2b: pressure-driven eviction policy.

Computes per-pool eviction targets each tick. Phase 2b's policy is
deliberately simple: it pre-evicts prefix-cached blocks when queue depth
indicates imminent KV pressure, and pre-evicts mamba state when admission
is bottlenecked by mamba slots. A more sophisticated Lagrange-equalized
policy across all four pools is deferred until 2c+.

The policy is **pure** (no I/O, no side effects). The agent calls it with a
snapshot dict and gets back a target spec to feed into
`tree_cache.evict(EvictParams(...))`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvictTarget:
    """How many tokens to release from each pool this tick (zero = don't touch)."""

    num_tokens: int = 0       # full-attn KV prefix-cached tokens
    swa_num_tokens: int = 0   # sliding-window KV prefix-cached tokens
    mamba_num: int = 0        # mamba/DeltaNet state slots
    reason: str = "noop"

    def is_noop(self) -> bool:
        return (
            self.num_tokens == 0
            and self.swa_num_tokens == 0
            and self.mamba_num == 0
        )


@dataclass
class PolicyConfig:
    """Tunables for `PressurePolicy`."""

    # Trigger thresholds. The policy is intentionally CONSERVATIVE: it only fires
    # when SGLang is already showing failure-mode signals (queue building, requests
    # paused/retracted). On healthy workloads the trigger conditions are never met
    # so the budgeter is a no-op — that's the no-regression guarantee.
    queue_depth_to_trigger: int = 1     # any queued request triggers consideration
    kv_evictable_floor_frac: float = 0.05   # only evict if evictable >= 5% of pool
    mamba_usage_pressure: float = 0.85       # evict mamba when usage >= 0.85

    # Per-tick eviction caps (fraction of evictable region)
    kv_evict_step_frac: float = 0.20    # release at most 20% of evictable per tick
    kv_evict_max_frac: float = 0.10     # ... and at most 10% of total pool

    # Hysteresis: don't evict twice within this many ticks of last evict
    evict_cooldown_ticks: int = 3


class PressurePolicy:
    """Reactive eviction policy.

    The policy looks only at the most recent snapshot and a small amount of
    state (last-evict tick, last-evict counts). It does NOT fit utility curves
    yet — that's 2c.
    """

    def __init__(self, cfg: Optional[PolicyConfig] = None) -> None:
        self.cfg = cfg or PolicyConfig()
        self._last_evict_tick: int = -10**9
        self._cumulative_evicted_kv: int = 0
        self._cumulative_evicted_mamba: int = 0
        self._cumulative_decisions: int = 0

    @property
    def stats(self) -> dict:
        return {
            "decisions": self._cumulative_decisions,
            "evicted_kv": self._cumulative_evicted_kv,
            "evicted_mamba": self._cumulative_evicted_mamba,
            "last_evict_tick": self._last_evict_tick,
        }

    # ----- main entry -----

    def decide(self, snap: dict, tick: int) -> EvictTarget:
        """Return what to evict this tick (may be a noop)."""
        cfg = self.cfg
        if tick - self._last_evict_tick < cfg.evict_cooldown_ticks:
            return EvictTarget(reason="cooldown")

        def _as_int(v) -> int:
            """Extract a count from int / float / QueueCount-like / None."""
            if v is None:
                return 0
            t = getattr(v, "total", None)
            if isinstance(t, (int, float)):
                return int(t)
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        def _as_float(v) -> float:
            t = getattr(v, "total", None)
            if isinstance(t, (int, float)):
                return float(t)
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        kv_max = _as_float(snap.get("max_total_num_tokens"))
        kv_evictable = _as_float(snap.get("kv_evictable_tokens"))
        num_queue = _as_int(snap.get("num_queue_reqs"))
        num_paused = _as_int(snap.get("num_paused_reqs"))
        num_retracted = _as_int(snap.get("num_retracted_reqs"))
        mamba_usage = _as_float(snap.get("mamba_usage"))

        # Default: no action.
        target = EvictTarget()

        # --- 1. KV pressure trigger ---
        # If there are queued requests AND prefix-cache holds non-trivial bytes,
        # release some to lower TTFT. Also fires when retract/paused gauges blink.
        kv_floor = cfg.kv_evictable_floor_frac * kv_max
        # Pressure = SGLang is already in a failure mode:
        #   - queue is building up (admission can't keep up), OR
        #   - admission is paused / requests retracted (KV ran out)
        # We deliberately do NOT trigger on raw token_usage because high usage
        # with a calm queue means the system is performing fine — eviction would
        # only thin a cache that was about to be useful.
        kv_under_pressure = (
            num_queue >= cfg.queue_depth_to_trigger
            or num_paused > 0
            or num_retracted > 0
        )
        if kv_under_pressure and kv_evictable >= kv_floor and kv_max > 0:
            step = int(min(
                cfg.kv_evict_step_frac * kv_evictable,
                cfg.kv_evict_max_frac * kv_max,
            ))
            if step > 0:
                target.num_tokens = step
                target.reason = (
                    f"kv_pressure(q={num_queue},p={num_paused},r={num_retracted})"
                )

        # --- 2. Mamba pressure trigger ---
        # If mamba pool nearly full and there's a queue, force a slot release.
        if mamba_usage >= cfg.mamba_usage_pressure and num_queue > 0:
            target.mamba_num = 1     # one slot is enough — admission needs one
            target.reason = target.reason if not target.is_noop() else "mamba_pressure"

        if not target.is_noop():
            self._last_evict_tick = tick
            self._cumulative_decisions += 1
            self._cumulative_evicted_kv += target.num_tokens
            self._cumulative_evicted_mamba += target.mamba_num

        return target
