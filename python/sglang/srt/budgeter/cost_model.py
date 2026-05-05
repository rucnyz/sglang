"""
Stage-0 deployment-time calibration consumer (paper §sec:design-l2-firegate).

Loads parametric c_σ(L) recovery-cost curves emitted by
`dev/eval/cost_model/stage0_calibrate.py`. Curves are loaded from
SGLANG_CSIGMA_* env vars (preferred, eval-able) or from the JSON path
SGLANG_CSIGMA_JSON. Falls back to a Qwen3.5-35B-A3B / H200 reference
default with a one-time warning when no calibration is present.

The admission gate consumes these curves to make recovery-cost-aware
fire decisions: at observed mean recovery length \bar{L}_σ, the cost of
a miss is c_σ(\bar{L}_σ) milliseconds; the gate compares net benefit
across pools using these L-aware costs rather than fixed scalars.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostCurves:
    """Parametric recovery-cost curves for KV and mamba pools.

    Stack-level wall-clock per L-token recovery (ms):
        c_KV(L) = kv_alpha · L² + kv_beta · L + kv_gamma
                  (parallel-batched attention re-prefill: L² compute term
                   plus per-layer kernel-launch overhead)
        c_M(L)  = m_alpha · L + m_beta
                  (chunked-scan recurrent re-derivation: linear-in-L
                   asymptote plus per-chunk setup cost)

    Crossover L_star: c_KV(L*) = c_M(L*). Below L*, mamba misses are
    more expensive (prefer evicting KV); above L*, the inequality
    inverts (prefer evicting mamba).
    """
    kv_alpha: float       # ms / token²
    kv_beta: float        # ms / token
    kv_gamma: float       # ms (kernel launch / fixed)
    m_alpha: float        # ms / token
    m_beta: float         # ms (chunk setup overhead)
    L_star: float         # tokens; 0 if no real crossover
    source: str           # "env" / "json:..." / "legacy_default"

    def c_kv_ms(self, L: float) -> float:
        """KV stack-level recovery time (ms) for L-token re-prefill."""
        if L <= 0:
            return self.kv_gamma
        return max(0.0, self.kv_alpha * L * L + self.kv_beta * L + self.kv_gamma)

    def c_m_ms(self, L: float) -> float:
        """Mamba stack-level recovery time (ms) for L-token re-derivation."""
        if L <= 0:
            return self.m_beta
        return max(0.0, self.m_alpha * L + self.m_beta)

    def c_kv_per_token_us(self, L: float) -> float:
        """KV per-token recovery cost (us) at observed length L."""
        if L <= 0:
            return 0.0
        return self.c_kv_ms(L) * 1000.0 / L

    def c_m_per_token_us(self, L: float) -> float:
        """Mamba per-token recovery cost (us) at observed length L."""
        if L <= 0:
            return 0.0
        return self.c_m_ms(L) * 1000.0 / L

    def c_kv_us(self, L: float) -> float:
        """Total KV recovery cost (us) at length L."""
        return self.c_kv_ms(L) * 1000.0

    def c_m_us(self, L: float) -> float:
        """Total mamba recovery cost (us) at length L."""
        return self.c_m_ms(L) * 1000.0


# Legacy default: Qwen3.5-35B-A3B / H200 BF16 reference run.
# Used when no calibration is present, accompanied by a one-time warning.
LEGACY_DEFAULT = CostCurves(
    kv_alpha=1.19e-7,
    kv_beta=0.0,
    kv_gamma=0.44,
    m_alpha=2.17e-3,
    m_beta=6.99,
    L_star=21780.0,
    source="legacy_default",
)


def _try_load_env() -> Optional[CostCurves]:
    """Path 1: SGLANG_CSIGMA_* env vars (set by stage0_calibrate.sh)."""
    if os.environ.get("SGLANG_CSIGMA_KV_ALPHA") is None:
        return None
    try:
        curves = CostCurves(
            kv_alpha=float(os.environ["SGLANG_CSIGMA_KV_ALPHA"]),
            kv_beta=float(os.environ.get("SGLANG_CSIGMA_KV_BETA", "0")),
            kv_gamma=float(os.environ["SGLANG_CSIGMA_KV_GAMMA"]),
            m_alpha=float(os.environ["SGLANG_CSIGMA_M_ALPHA"]),
            m_beta=float(os.environ["SGLANG_CSIGMA_M_BETA"]),
            L_star=float(os.environ.get("SGLANG_CSIGMA_LSTAR", "0")),
            source="env",
        )
        logger.info(
            "CostCurves[env]: c_KV=%.3eL²%+.3eL%+.3e ms, "
            "c_M=%.3eL%+.3e ms, L*=%.0f tok (model=%s, dev=%s)",
            curves.kv_alpha, curves.kv_beta, curves.kv_gamma,
            curves.m_alpha, curves.m_beta, curves.L_star,
            os.environ.get("SGLANG_CSIGMA_MODEL", "?"),
            os.environ.get("SGLANG_CSIGMA_DEVICE", "?"),
        )
        return curves
    except (ValueError, KeyError) as e:
        logger.warning("CostCurves: failed to parse SGLANG_CSIGMA_* env: %s", e)
        return None


def _try_load_json() -> Optional[CostCurves]:
    """Path 2: SGLANG_CSIGMA_JSON pointing at calibration record file."""
    path = os.environ.get("SGLANG_CSIGMA_JSON")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
        fit = rec["fit"]
        curves = CostCurves(
            kv_alpha=float(fit["c_kv"]["alpha_ms_per_tok2"]),
            kv_beta=float(fit["c_kv"].get("beta_ms_per_tok", 0.0)),
            kv_gamma=float(fit["c_kv"]["gamma_ms"]),
            m_alpha=float(fit["c_m"]["alpha_ms_per_tok"]),
            m_beta=float(fit["c_m"]["beta_ms"]),
            L_star=float(fit.get("crossover_L_star", 0.0)),
            source=f"json:{path}",
        )
        logger.info(
            "CostCurves[%s]: c_KV=%.3eL²%+.3eL%+.3e ms, "
            "c_M=%.3eL%+.3e ms, L*=%.0f tok",
            curves.source, curves.kv_alpha, curves.kv_beta, curves.kv_gamma,
            curves.m_alpha, curves.m_beta, curves.L_star,
        )
        return curves
    except Exception as e:
        logger.warning("CostCurves: failed to load SGLANG_CSIGMA_JSON=%s: %s", path, e)
        return None


_singleton: Optional[CostCurves] = None
_warned_legacy: bool = False


def get_cost_curves() -> CostCurves:
    """Return the process-wide CostCurves singleton.

    First call resolves the curves from env / JSON / legacy default and
    logs the source plus a few sample evaluations so a downstream reviewer
    can verify the asymmetry direction without re-deriving it. Subsequent
    calls return the cached instance.
    """
    global _singleton, _warned_legacy
    if _singleton is not None:
        return _singleton
    curves = _try_load_env() or _try_load_json()
    if curves is None:
        curves = LEGACY_DEFAULT
        if not _warned_legacy:
            logger.warning(
                "[xpool-cost] No SGLANG_CSIGMA_* calibration present. Using "
                "legacy default (Qwen3.5-35B-A3B / H200 reference). For a "
                "deployment-specific calibration: "
                "`eval \"$(bash dev/eval/cost_model/stage0_calibrate.sh)\"`."
            )
            _warned_legacy = True
    _singleton = curves
    # Sample evaluations at canonical recovery lengths so the asymmetry
    # direction is greppable at boot without running the bench again.
    samples = [256, 1024, 2048, 4096, 8192, 16384]
    rows = []
    for L in samples:
        kv = curves.c_kv_ms(L)
        m = curves.c_m_ms(L)
        ratio = m / kv if kv > 0 else float("inf")
        rows.append(f"L={L}: c_KV={kv:.2f}ms c_M={m:.2f}ms ratio={ratio:.2f}x")
    logger.info(
        "[xpool-cost] CostCurves ready (source=%s, L*=%.0f tok). Sample evals: %s",
        curves.source, curves.L_star, " | ".join(rows),
    )
    return curves


def reset_cost_curves() -> None:
    """Test hook: clear the singleton so the next get reloads from env."""
    global _singleton, _warned_legacy
    _singleton = None
    _warned_legacy = False


# ---------- Runtime actuator-cost EWMA (paper §sec:design-l2-firegate) ----------
#
# Stage-0 calibration measures recovery-cost curves c_σ(L) — these are
# kernel-level (model+hardware) properties stable under any traffic. The
# actuator unit cost c_actuator is different: it is the wall-clock of one
# cuMemUnmap+cuMemMap pair, but under live traffic it pays additional
# costs (CUDA-graph deferral while in-flight kernels finish, allocator
# lock contention with the scheduler, per-spec drain wait) that the
# offline idle-time probe does not see. We therefore measure c_actuator
# at runtime: every committed fire reports `fire_total_us` (shrink+grow,
# torch.cuda.synchronize-bracketed) which we EWMA into a process-wide
# estimate. The planner reads the current estimate when computing the
# fire-decision gate's α·C_act threshold each tick.

class RuntimeActuatorCost:
    """Process-wide EWMA of observed per-chunk actuator wall-time.

    Initialized to a conservative value (10 ms/chunk by default) so the
    cold-start gate suppresses fires until enough live observations have
    arrived. After each observed fire of `n_chunks` consuming
    `total_us` microseconds, the per-chunk estimate is updated as
        new = α · (total_us / n_chunks) + (1 - α) · old
    where α defaults to 0.3 (5-fire half-life ≈ 1.6 fires).
    """

    def __init__(self, initial_us: float = 10000.0, alpha: float = 0.3):
        self._initial = float(initial_us)
        self._current = float(initial_us)
        self._alpha = float(alpha)
        self._n_observations = 0
        self._last_observed_us: Optional[float] = None

    @property
    def current_us(self) -> float:
        return self._current

    @property
    def n_observations(self) -> int:
        return self._n_observations

    @property
    def is_calibrated(self) -> bool:
        """True once enough observations have arrived to trust the EWMA
        over the conservative initial value. We use 3 observations (with
        α=0.3 the EWMA is ~70% determined by observed data after 3 fires)."""
        return self._n_observations >= 3

    def update(self, total_us: float, n_chunks: int) -> None:
        if n_chunks <= 0 or total_us <= 0:
            return
        per_chunk = float(total_us) / float(n_chunks)
        if self._n_observations == 0:
            # First observation seeds the EWMA without dilution by the
            # conservative initial — that initial only governed cold-start.
            self._current = per_chunk
        else:
            self._current = self._alpha * per_chunk + (1.0 - self._alpha) * self._current
        self._last_observed_us = per_chunk
        self._n_observations += 1
        logger.info(
            "[xpool-cost] actuator EWMA update: observed=%.0fus/chunk "
            "(total=%.0fus over %d chunks), EWMA=%.0fus/chunk after %d obs",
            per_chunk, total_us, n_chunks, self._current, self._n_observations,
        )

    def reset(self) -> None:
        self._current = self._initial
        self._n_observations = 0
        self._last_observed_us = None


_runtime_actuator: Optional[RuntimeActuatorCost] = None


def get_runtime_actuator_cost() -> RuntimeActuatorCost:
    """Process-wide singleton. Initial value can be overridden via
    SGLANG_XPOOL_NB_CHUNK_COST_INIT_US (default 10000 = 10 ms; the
    legacy SGLANG_XPOOL_NB_CHUNK_COST_US is still honored as a fallback
    for older deployments)."""
    global _runtime_actuator
    if _runtime_actuator is None:
        initial = float(os.environ.get(
            "SGLANG_XPOOL_NB_CHUNK_COST_INIT_US",
            os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_US", "10000")
        ))
        alpha = float(os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_EWMA_ALPHA", "0.3"))
        _runtime_actuator = RuntimeActuatorCost(initial_us=initial, alpha=alpha)
        logger.info(
            "[xpool-cost] RuntimeActuatorCost initialized: initial=%.0fus/chunk "
            "(conservative, suppresses fires until 3+ observations) alpha=%.2f",
            initial, alpha,
        )
    return _runtime_actuator


def reset_runtime_actuator_cost() -> None:
    global _runtime_actuator
    _runtime_actuator = None
