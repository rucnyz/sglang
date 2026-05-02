"""Engine-native pressure adapter for L2's net-benefit gate.

Paper §design-l2 states the gate as `B (benefit) ≥ C (cost) × margin`.
The benefit term `B` is the sum of admission-pressure signals translated
to a uniform "us of GPU time saved" space. Different engines surface
admission pressure through different mechanisms:

  - SGLang: tree-cache (radix prefix) eviction is the primary
    pressure-relief mechanism. When KV is tight, the engine evicts
    completed-but-cached prefix entries from the tree cache; those
    entries' future re-prefill cost is the deferred admission-pressure
    cost. Retract is a rare fallback that fires only when eviction
    can't keep up.
  - vLLM: CPU swap-out is the primary mechanism — KV blocks evicted
    to host memory; preemption is the fallback.
  - Generic / no-eviction engine: paused + retracted + queue_depth
    are the only signals.

This module defines an abstract `EnginePressureAdapter` interface and a
concrete `SGLangPressureAdapter` that translates SGLang's snapshot
fields into the uniform `PressureSignals` namedtuple. New engines
register their own adapter; the gate code in `cross_pool_planner.py` is
engine-agnostic.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PressureSignals:
    """Engine-native admission-pressure signals translated to a uniform
    benefit-microsecond space.

    Each field is "expected GPU time (us) saved by averting one tick's
    worth of accumulated pressure of that type". The L2 fire decision
    sums all fields; the gate fires when sum ≥ chunk_cost_us × margin.

    Why microseconds: matches `chunk_cost_us` units (the empirically
    measured cuMemUnmap+cuMemMap wall time, e.g. ~5,000 us/fire on
    H200 with 256 MB chunks per sglang `87360b2c7` calibration).
    """
    evict_us: float = 0.0    # tree-cache eviction backlog (re-prefill cost deferral)
    retract_us: float = 0.0  # currently/recently retracted reqs
    paused_us: float = 0.0   # admission-paused reqs
    queue_us: float = 0.0    # queued reqs waiting for admission
    persist_us: float = 0.0  # pool above-high dwell time (saturation prior)
    edge_us: float = 0.0     # phase-transition signal: |du/dt| above threshold

    @property
    def total_benefit_us(self) -> float:
        return (self.evict_us + self.retract_us + self.paused_us
                + self.queue_us + self.persist_us + self.edge_us)

    def reason_str(self) -> str:
        return (f"evict={self.evict_us:.0f} retract={self.retract_us:.0f} "
                f"paused={self.paused_us:.0f} queue={self.queue_us:.0f} "
                f"persist={self.persist_us:.0f} edge={self.edge_us:.0f}")


class EnginePressureAdapter(ABC):
    """Translates engine-native pool-pressure signals to uniform
    benefit-microseconds usable by L2's net-benefit gate.

    Subclasses must implement `signals_from_snapshot`. The budgeter calls
    this on every gate evaluation; it should be cheap (no I/O, just
    arithmetic on snapshot fields the engine already populates).
    """

    @abstractmethod
    def signals_from_snapshot(
        self,
        snapshot: dict,
        kv_consec: int,
        mamba_consec: int,
        edge_active: bool = False,
    ) -> PressureSignals:
        ...


class SGLangPressureAdapter(EnginePressureAdapter):
    """SGLang's evict-first scheduler. Tree-cache eviction is the
    primary signal — when KV is tight, `check_decode_mem` calls
    `evict_from_tree_cache` to free completed-but-cached prefix tokens;
    those evicted tokens carry deferred re-prefill cost. Retract path
    rarely triggers because eviction usually satisfies pressure.

    Coefficients:
      prefill_save_us_per_token  - GPU cost per token of prefill on the
        target hardware/model. Default 12.5 us/token corresponds to
        ~80 K tok/s prefill throughput (Qwen3.5-35B-A3B / H200 / TP=1).
        Override via SGLANG_XPOOL_PREFILL_SAVE_US_PER_TOKEN or the
        per-deployment auto-calibration in agent.py (Stage 3).
      full_prefill_us  - "Full" re-prefill cost for one retracted req,
        approximated as avg_input_tokens × prefill_save_us_per_token.
        For ~6K-token avg input on H200: 6000 × 12.5 = 75,000 us = 75 ms.
      pause_penalty_us  - Cost of one paused tick. Conservative 1 ms.
      queue_wait_us  - Penalty per queued req. Conservative 100 us
        (assumes queueing displaces some idle GPU time at the margin).
      persist_tick_us  - Per-tick value of sustained pool above-high.
        Acts as a "I should fire even if no explicit signal yet"
        accumulator.

    All coefficients can be overridden via env (`SGLANG_XPOOL_*_US`)
    for empirical re-calibration.
    """

    def __init__(
        self,
        prefill_save_us_per_token: float | None = None,
        full_prefill_us: float | None = None,
        pause_penalty_us: float | None = None,
        queue_wait_us: float | None = None,
        persist_tick_us: float | None = None,
        edge_us: float | None = None,
    ):
        self.prefill_save_us_per_token = (
            prefill_save_us_per_token
            if prefill_save_us_per_token is not None
            else float(os.environ.get(
                "SGLANG_XPOOL_PREFILL_SAVE_US_PER_TOKEN", "12.5"))
        )
        self.full_prefill_us = (
            full_prefill_us
            if full_prefill_us is not None
            else float(os.environ.get(
                "SGLANG_XPOOL_FULL_PREFILL_US", "75000"))
        )
        self.pause_penalty_us = (
            pause_penalty_us
            if pause_penalty_us is not None
            else float(os.environ.get(
                "SGLANG_XPOOL_PAUSE_PENALTY_US", "1000"))
        )
        self.queue_wait_us = (
            queue_wait_us
            if queue_wait_us is not None
            else float(os.environ.get(
                "SGLANG_XPOOL_QUEUE_WAIT_US", "100"))
        )
        self.persist_tick_us = (
            persist_tick_us
            if persist_tick_us is not None
            else float(os.environ.get(
                "SGLANG_XPOOL_PERSIST_TICK_US", "5000"))
        )
        # S_edge: paper §design-l2-actuator. Bounded one-tick benefit
        # added when the planner observes |du_σ/dt| > θ_edge — the
        # gradient signature of a phase transition. Calibrated as one
        # control interval's worth of avoided re-prefill at the engine's
        # current prefill throughput. Default 100 ms ≈ τ × prefill_tps
        # × prefill_save_us_per_token at τ=2 s (i.e., ~25K tokens worth
        # of deferred re-prefill avoided by re-allocating now rather
        # than waiting one more tick).
        self.edge_us = (
            edge_us
            if edge_us is not None
            else float(os.environ.get("SGLANG_XPOOL_EDGE_US", "100000"))
        )
        logger.info(
            "SGLangPressureAdapter: prefill_save=%.1f us/tok "
            "full_prefill=%.0f us pause_penalty=%.0f us "
            "queue_wait=%.0f us persist_tick=%.0f us edge=%.0f us",
            self.prefill_save_us_per_token, self.full_prefill_us,
            self.pause_penalty_us, self.queue_wait_us, self.persist_tick_us,
            self.edge_us,
        )

    @staticmethod
    def _to_int(v) -> int:
        """Snapshot fields may be int, float, or composite metric objects
        with .total field; normalize to int."""
        if v is None:
            return 0
        t = getattr(v, "total", None)
        if isinstance(t, (int, float)):
            return int(t)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    def signals_from_snapshot(
        self,
        snapshot: dict,
        kv_consec: int,
        mamba_consec: int,
        edge_active: bool = False,
    ) -> PressureSignals:
        evicted_tokens = self._to_int(snapshot.get("num_evicted_tokens_recent", 0))
        retracted = self._to_int(snapshot.get("num_retracted_reqs", 0))
        paused = self._to_int(snapshot.get("num_paused_reqs", 0))
        queued = self._to_int(snapshot.get("num_queue_reqs", 0))
        # Persist uses the max of (kv, mamba) consec since either pool's
        # sustained saturation is evidence the workload could benefit
        # from re-allocation.
        max_consec = max(kv_consec, mamba_consec)

        return PressureSignals(
            evict_us=evicted_tokens * self.prefill_save_us_per_token,
            retract_us=retracted * self.full_prefill_us,
            paused_us=paused * self.pause_penalty_us,
            queue_us=queued * self.queue_wait_us,
            persist_us=max_consec * self.persist_tick_us,
            edge_us=self.edge_us if edge_active else 0.0,
        )


def get_default_adapter() -> EnginePressureAdapter:
    """Factory: returns the SGLang adapter (the only one implemented
    today). Future: dispatch on engine identity."""
    return SGLangPressureAdapter()
