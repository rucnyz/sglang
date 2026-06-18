"""Engine-native pressure adapter for the admission gate's net-benefit check.

Paper §sec:design-l2 states the gate as `B (benefit) ≥ C (cost) × margin`.
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
register their own adapter; the gate code in `xpool_planner.py` is
engine-agnostic.
"""

from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PressureSignals:
    """Engine-native admission-pressure signals translated to a uniform
    benefit-microsecond space.

    Each field is "expected GPU time (us) saved by averting one tick's
    worth of accumulated pressure of that type". The admission gate
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

    @property
    def total_benefit_us(self) -> float:
        return (self.evict_us + self.retract_us + self.paused_us
                + self.queue_us + self.persist_us)


class EnginePressureAdapter(ABC):
    """Translates engine-native pool-pressure signals to uniform
    benefit-microseconds usable by the admission gate's net-benefit check.

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
        dt: float,
    ) -> PressureSignals:
        """Return per-second pressure RATES (µs/s). `dt` (wall seconds since
        the previous tick) is REQUIRED: flow signals are priced as rates
        (count ÷ dt) so the gate is invariant to the polling interval τ."""
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
        per-deployment auto-calibration in agent.py (auto-calibration).
      full_prefill_us  - "Full" re-prefill cost for one retracted req,
        approximated as avg_input_tokens × prefill_save_us_per_token.
        For ~6K-token avg input on H200: 6000 × 12.5 = 75,000 us = 75 ms.
      pause_penalty_us  - µs/s of pressure per PAUSED req (a stock: standing
        backlog, not divided by dt). Conservative 1 ms/s.
      queue_wait_us  - µs/s of pressure per QUEUED req (a stock, not divided
        by dt). Conservative 100 us/s (queueing displaces idle GPU time).
      persist_tick_us  - µs of saturation-prior evidence per SECOND of dwell
        above high-water (used by the planner as dwell_s × persist_tick_us).
        Acts as a "I should fire even if no explicit signal yet" accumulator.

    All coefficients can be overridden via env (`SGLANG_XPOOL_*_US`)
    for empirical re-calibration.
    """

    # Default L for "evict" benefit term derivation when calibrated curves
    # are present but the snapshot doesn't carry an observed mean recovery
    # length. 2048 is the upper end of the short-recovery regime where
    # c_KV is launch-bound; the per-token cost there matches the
    # uncalibrated 12.5 us/tok scalar within a factor of 2 on
    # H200/Qwen3.5-35B-A3B.
    _DEFAULT_KV_RECOVER_L: float = 2048.0
    # Default L for "retract" cost: a full-input re-prefill. 6K matches
    # the uncalibrated 75 ms scalar.
    _DEFAULT_RETRACT_L: float = 6144.0

    def __init__(
        self,
        prefill_save_us_per_token: float | None = None,
        full_prefill_us: float | None = None,
        pause_penalty_us: float | None = None,
        queue_wait_us: float | None = None,
        persist_tick_us: float | None = None,
    ):
        # Cost-curve-aware defaults (paper §sec:appendix-trigger /
        # calibration). When calibrated c_σ(L) curves are available, derive
        # the scalar coefficients from c_KV(L) at a default L; explicit env
        # / constructor overrides still win. The curves are also held on
        # self for the planner to query at runtime with live \bar{L}.
        from sglang.srt.budgeter.cost_model import get_cost_curves
        self.cost_curves = get_cost_curves()
        kv_default_us_per_tok = self.cost_curves.c_kv_per_token_us(
            self._DEFAULT_KV_RECOVER_L
        )
        retract_default_us = self.cost_curves.c_kv_us(self._DEFAULT_RETRACT_L)

        self.prefill_save_us_per_token = (
            prefill_save_us_per_token
            if prefill_save_us_per_token is not None
            else float(os.environ.get(
                "SGLANG_XPOOL_PREFILL_SAVE_US_PER_TOKEN",
                f"{kv_default_us_per_tok:.4f}"))
        )
        self.full_prefill_us = (
            full_prefill_us
            if full_prefill_us is not None
            else float(os.environ.get(
                "SGLANG_XPOOL_FULL_PREFILL_US",
                f"{retract_default_us:.0f}"))
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
        logger.info(
            "SGLangPressureAdapter: prefill_save=%.1f us/tok "
            "full_prefill=%.0f us pause_penalty=%.0f us "
            "queue_wait=%.0f us persist_tick=%.0f us "
            "[cost_curves source=%s, L*=%.0f tok]",
            self.prefill_save_us_per_token, self.full_prefill_us,
            self.pause_penalty_us, self.queue_wait_us, self.persist_tick_us,
            self.cost_curves.source, self.cost_curves.L_star,
        )
        # EWMA smoothing of the per-second RATE stream (design.md §"Empirical
        # pressure signal": "the rate stream is then smoothed ... EWMA").
        # The time-constant is in SECONDS and the decay adapts to dt
        # (η_eff = 1 − exp(−dt / ewma_tau_s)), so smoothing is invariant to
        # the polling interval τ. The first observation seeds the EWMA
        # undiluted (so a single-shot call returns the raw rate).
        self.ewma_tau_s = float(os.environ.get("SGLANG_XPOOL_EWMA_TAU_S", "5.0"))
        self._ewma: dict[str, float] = {}

        # Tick counter bumped on each `signals_from_snapshot` call so
        # downstream consumers can tell whether `last_breakdown` is from
        # the current tick. Initialized to 0; first call yields serial=1.
        self._breakdown_serial = 0
        self.last_breakdown = None

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

    @staticmethod
    def _to_float(v) -> float:
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _ewma_step(self, key: str, rate: float, eta: float) -> float:
        """EWMA one signal's per-second rate. First observation seeds the
        accumulator undiluted, so a single-shot call returns the raw rate."""
        prev = self._ewma.get(key)
        if prev is None:
            self._ewma[key] = rate
        else:
            self._ewma[key] = prev + eta * (rate - prev)
        return self._ewma[key]

    def signals_from_snapshot(
        self,
        snapshot: dict,
        kv_consec: int,
        mamba_consec: int,
        dt: float,
    ) -> PressureSignals:
        # `dt` (wall seconds since the previous tick) is REQUIRED: flow
        # signals are priced as per-second rates (count ÷ dt). A non-positive
        # dt is a caller bug — fail loud (no fallback).
        if dt is None or float(dt) <= 0.0:
            raise ValueError(
                f"signals_from_snapshot requires dt > 0 (wall seconds since "
                f"the previous tick); got {dt!r}."
            )
        dt = float(dt)
        evicted_tokens = self._to_int(snapshot.get("num_evicted_tokens_recent", 0))
        retracted = self._to_int(snapshot.get("num_retracted_reqs", 0))
        paused = self._to_int(snapshot.get("num_paused_reqs", 0))
        queued = self._to_int(snapshot.get("num_queue_reqs", 0))

        # L-aware recovery cost (paper §sec:appendix-trigger, Eq nb-lb /
        # Eq p-loss-save): ∑_b n_b · c_i(\\bar L_i). Per tick we observe a total
        # token-count `evicted_tokens` (sum of per-leaf L's); the number of
        # misses is approx evicted_tokens / \\bar L_KV, each paying
        # c_KV(\\bar L_KV). No-calibration path keeps the per-token scalar.
        kv_L = self._to_float(snapshot.get("slow_recovery_len_kv", 0.0))
        retract_L = self._to_float(snapshot.get("slow_recovery_len_retract", 0.0))
        use_curves = self.cost_curves.source != "builtin_default" or \
                     snapshot.get("slow_recovery_len_kv") is not None
        # S_evict is a FLOW (tokens evicted since the last tick) → per-second
        # rate (÷ dt).
        if use_curves and evicted_tokens > 0 and kv_L > 0:
            num_misses = evicted_tokens / kv_L
            evict_rate = num_misses * self.cost_curves.c_kv_us(kv_L) / dt
            evict_path = f"calibrated_n_misses={num_misses:.2f}@L={kv_L:.0f}"
        else:
            evict_rate = evicted_tokens * self.prefill_save_us_per_token / dt
            evict_path = "scalar_per_tok"
        if retract_L > 0:
            retract_us_per = self.cost_curves.c_kv_us(retract_L)
            retract_path = f"calibrated_L={retract_L:.0f}"
        else:
            retract_us_per = self.full_prefill_us
            retract_path = "scalar_total"
        # S_retract is a FLOW (reqs retracted since last tick) → per-second.
        retract_rate = retracted * retract_us_per / dt
        # S_pause / S_queue are STOCKS (instantaneous backlog depth): a
        # standing backlog of N reqs is an ongoing N × penalty µs/s of
        # pressure regardless of polling rate — NO ÷ dt. `pause_penalty_us` /
        # `queue_wait_us` are interpreted as µs/s per backlogged req.
        paused_rate = paused * self.pause_penalty_us
        queue_rate = queued * self.queue_wait_us

        # EWMA-smooth the per-second rate stream (τ-invariant: decay adapts
        # to dt). persist is no longer an adapter signal — it is the
        # planner's dwell-seconds prior.
        eta = 1.0 - math.exp(-dt / self.ewma_tau_s) if self.ewma_tau_s > 0 else 1.0
        evict_us = self._ewma_step("evict", evict_rate, eta)
        retract_us = self._ewma_step("retract", retract_rate, eta)
        paused_us = self._ewma_step("paused", paused_rate, eta)
        queue_us = self._ewma_step("queue", queue_rate, eta)

        signals = PressureSignals(
            evict_us=evict_us,
            retract_us=retract_us,
            paused_us=paused_us,
            queue_us=queue_us,
            persist_us=0.0,
        )

        # Cache breakdown for downstream JSONL logging. Cheap (just stashing
        # numbers), avoids re-evaluating curves in the planner. The
        # `_serial` field is bumped each call so a downstream consumer can
        # tell whether the snapshot it's reading is fresh-this-tick.
        self._breakdown_serial += 1
        self.last_breakdown = dict(
            _serial=self._breakdown_serial,
            evicted_tokens=evicted_tokens,
            retracted=retracted,
            paused=paused,
            queued=queued,
            dt=dt,
            kv_consec=kv_consec,
            mamba_consec=mamba_consec,
            kv_L=kv_L,
            retract_L=retract_L,
            evict_path=evict_path,
            retract_path=retract_path,
            evict_us=signals.evict_us,
            retract_us=signals.retract_us,
            paused_us=signals.paused_us,
            queue_us=signals.queue_us,
            persist_us=signals.persist_us,
            total_benefit_us=signals.total_benefit_us,
        )

        # Per-tick breakdown at DEBUG (low-volume but high-signal); operator
        # can flip with logging.getLogger("sglang.srt.budgeter").setLevel(DEBUG).
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[xpool-cost] signals: evict=%dtok→%.0fus(%s) retract=%dreq→%.0fus(%s) "
                "paused=%d queue=%d kv_consec=%d mamba_consec=%d "
                "→ total=%.0fus",
                evicted_tokens, signals.evict_us, evict_path,
                retracted, signals.retract_us, retract_path,
                paused, queued, kv_consec, mamba_consec,
                signals.total_benefit_us,
            )
        return signals


def get_default_adapter() -> EnginePressureAdapter:
    """Factory: returns the SGLang adapter (the only one implemented
    today). Future: dispatch on engine identity."""
    return SGLangPressureAdapter()
