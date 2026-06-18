"""
Deployment-time calibration consumer (paper §sec:design-l2-firegate).

Loads parametric c_σ(L) recovery-cost curves emitted by
`dev/eval/cost_model/calibrate.py`. Curves are loaded from
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
import math
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
    source: str           # "env" / "json:..." / "builtin_default"

    def __post_init__(self):
        """Reject a poisoned curve at construction — the single validation
        point for EVERY source (env / JSON / fit / builtin). κ_i reaches
        production only through `SGLANG_CSIGMA_*` (the in-engine probe is a
        no-op), and a thin/noisy offline `calibrate_kappa.py` fit — or a
        stale/edited profile `.sh` — can carry a NaN/inf/negative leading
        coeff: NaN slips past the `<= 0` guards downstream, and a negative
        `α` makes the curve go negative at large L. The env/JSON loaders
        construct under try/except, so a raise here falls back to the
        builtin curve + warns rather than silently mis-pricing."""
        for name in ("kv_alpha", "kv_beta", "kv_gamma",
                     "m_alpha", "m_beta", "L_star"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"CostCurves.{name} must be finite, got {v!r}")
        if self.kv_alpha < 0 or self.m_alpha < 0:
            raise ValueError(
                f"CostCurves leading coeffs must be >= 0 (the curve must not "
                f"go negative at large L): kv_alpha={self.kv_alpha}, "
                f"m_alpha={self.m_alpha}"
            )

    def c_kv_ms(self, L: float) -> float:
        """KV stack-level recovery time (ms) for L-token re-prefill."""
        if not (L > 0):  # NaN-safe: not (nan > 0) is True
            return self.kv_gamma
        return max(0.0, self.kv_alpha * L * L + self.kv_beta * L + self.kv_gamma)

    def c_m_ms(self, L: float) -> float:
        """Mamba stack-level recovery time (ms) for L-token re-derivation."""
        if not (L > 0):  # NaN-safe: not (nan > 0) is True
            return self.m_beta
        return max(0.0, self.m_alpha * L + self.m_beta)

    def c_kv_per_token_us(self, L: float) -> float:
        """KV per-token recovery cost (us) at observed length L."""
        if not (L > 0):  # NaN-safe: not (nan > 0) is True
            return 0.0
        return self.c_kv_ms(L) * 1000.0 / L

    def c_m_per_token_us(self, L: float) -> float:
        """Mamba per-token recovery cost (us) at observed length L."""
        if not (L > 0):  # NaN-safe: not (nan > 0) is True
            return 0.0
        return self.c_m_ms(L) * 1000.0 / L

    def c_kv_us(self, L: float) -> float:
        """Total KV recovery cost (us) at length L."""
        return self.c_kv_ms(L) * 1000.0

    def c_m_us(self, L: float) -> float:
        """Total mamba recovery cost (us) at length L."""
        return self.c_m_ms(L) * 1000.0


# Built-in default: Qwen3.5-35B-A3B / H200 BF16 reference run.
# Used when no calibration is present, accompanied by a one-time warning.
# Single-curve by design: a miss on either pool re-prefills the whole bound
# prefix = one forward at cost c(s), so the total is folded into c_KV and
# c_M = 0 (matching every real calibration, e.g. kappa_fit.json). A non-zero
# c_M here was a stale two-curve artifact that made the planner price a
# mamba miss off a phantom curve and fire the wrong cross-pool direction.
BUILTIN_DEFAULT = CostCurves(
    kv_alpha=1.19e-7,
    kv_beta=0.0,
    kv_gamma=0.44,
    m_alpha=0.0,
    m_beta=0.0,
    L_star=0.0,
    source="builtin_default",
)


def _try_load_env() -> Optional[CostCurves]:
    """Path 1: SGLANG_CSIGMA_* env vars (set by calibrate.sh)."""
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
_warned_builtin: bool = False
_model_mismatch_warned: bool = False


def get_cost_curves() -> CostCurves:
    """Return the process-wide CostCurves singleton.

    First call resolves the curves from env / JSON / built-in default and
    logs the source plus a few sample evaluations so a downstream reviewer
    can verify the asymmetry direction without re-deriving it. Subsequent
    calls return the cached instance.
    """
    global _singleton, _warned_builtin
    if _singleton is not None:
        return _singleton
    curves = _try_load_env() or _try_load_json()
    if curves is None:
        raise RuntimeError(
            "HiMA requires a calibrated cost model. Set SGLANG_CSIGMA_* env "
            "vars or SGLANG_CSIGMA_JSON=<path>. Calibrate with: "
            "`eval \"$(bash dev/eval/cost_model/calibrate.sh)\"`."
        )
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
    global _singleton, _warned_builtin, _model_mismatch_warned
    _singleton = None
    _warned_builtin = False
    _model_mismatch_warned = False


def _model_basename(m: Optional[str]) -> Optional[str]:
    return m.rstrip("/").split("/")[-1] if m else m


def check_model_mismatch(running_model: Optional[str]) -> None:
    """Fail-fast if the calibration profile's model doesn't match the
    running model. κ_i / c^xfer / c_m are per-(model, GPU) constants;
    a stale profile silently mis-prices every decision."""
    global _model_mismatch_warned
    if _model_mismatch_warned:
        return
    _model_mismatch_warned = True
    profile_model = os.environ.get("SGLANG_CSIGMA_MODEL")
    if not profile_model or not running_model:
        return
    if _model_basename(profile_model) != _model_basename(running_model):
        raise RuntimeError(
            f"Calibration profile model (SGLANG_CSIGMA_MODEL={profile_model}) "
            f"!= running model ({running_model}). The cost curves are calibrated "
            f"for a different model. Regenerate with "
            f"`calibrate_profile.sh {running_model} <device>`."
        )


def _crossover_L_star(kv_alpha: float, kv_beta: float, kv_gamma: float,
                      m_alpha: float, m_beta: float) -> float:
    """Smallest positive L where c_KV(L) == c_M(L), else 0.

    Solve `kv_alpha·L² + (kv_beta - m_alpha)·L + (kv_gamma - m_beta) = 0`.
    Returns 0 when there is no positive real crossover (curves don't
    cross in the physical L>0 domain — `L_star=0` is the "no real
    crossover" sentinel used by BUILTIN_DEFAULT and the JSON loader).
    """
    a = kv_alpha
    b = kv_beta - m_alpha
    c = kv_gamma - m_beta
    if a == 0:
        if b == 0:
            return 0.0
        L = -c / b
        return L if L > 0 else 0.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.0
    import math
    sq = math.sqrt(disc)
    roots = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
    pos = sorted(r for r in roots if r > 0)
    return pos[0] if pos else 0.0


def fit_cost_curves(
    kv_samples: list,
    m_samples: list,
    source: str = "boot_probe",
) -> CostCurves:
    """Fit `CostCurves` from boot-time prefill-probe timings (
    design.md §"Boot-time probe": `c_i(s)`'s `κ` coefficients).

    `kv_samples` / `m_samples` are lists of `(L_tokens, wall_ms)` from
    re-prefilling sequences of length `L` through the KV / mamba stack.
    Per design.md §"Shared cost model":
      * KV is quadratic — least-squares fit `c_KV = κ_α·L² + κ_β·L +
        κ_γ` over the basis `[L², L, 1]` (needs ≥3 distinct L).
      * Mamba is linear — least-squares fit `c_M = κ_α·L + κ_β` over
        `[L, 1]` (needs ≥2 distinct L).
    `κ_α < 0` (unphysical: cost decreasing in L) raises — a bad probe
    must fail loudly, not silently seed a degenerate curve.
    """
    import numpy as np

    kv = sorted(kv_samples)
    m = sorted(m_samples)
    if len({L for L, _ in kv}) < 3:
        raise ValueError(
            f"fit_cost_curves: KV quadratic fit needs ≥3 distinct L; "
            f"got {sorted({L for L, _ in kv})}"
        )
    if len({L for L, _ in m}) < 2:
        raise ValueError(
            f"fit_cost_curves: mamba linear fit needs ≥2 distinct L; "
            f"got {sorted({L for L, _ in m})}"
        )

    Lk = np.array([s[0] for s in kv], dtype=float)
    yk = np.array([s[1] for s in kv], dtype=float)
    Ak = np.vstack([Lk * Lk, Lk, np.ones_like(Lk)]).T
    kv_alpha, kv_beta, kv_gamma = (
        float(v) for v in np.linalg.lstsq(Ak, yk, rcond=None)[0]
    )

    Lm = np.array([s[0] for s in m], dtype=float)
    ym = np.array([s[1] for s in m], dtype=float)
    Am = np.vstack([Lm, np.ones_like(Lm)]).T
    m_alpha, m_beta = (
        float(v) for v in np.linalg.lstsq(Am, ym, rcond=None)[0]
    )

    if kv_alpha < 0 or m_alpha < 0:
        raise ValueError(
            f"fit_cost_curves: unphysical negative leading coefficient "
            f"(kv_alpha={kv_alpha:.3e}, m_alpha={m_alpha:.3e}); recompute "
            f"cost must be non-decreasing in L. Bad probe samples?"
        )

    return CostCurves(
        kv_alpha=kv_alpha,
        kv_beta=kv_beta,
        kv_gamma=kv_gamma,
        m_alpha=m_alpha,
        m_beta=m_beta,
        L_star=_crossover_L_star(kv_alpha, kv_beta, kv_gamma, m_alpha, m_beta),
        source=source,
    )


def set_cost_curves(curves: CostCurves) -> bool:
    """Install boot-probe-fitted curves as the process-wide singleton
    Returns True if installed, False if skipped.

    **Env precedence**: an explicit operator calibration
    (`SGLANG_CSIGMA_*` / `SGLANG_CSIGMA_JSON`) is the source of truth
    and is NEVER overwritten by the boot probe — the operator measured
    those deliberately. The boot probe only replaces the built-in
    default (the case where no calibration was supplied). This keeps
    the deterministic-prediction guarantee while letting an
    uncalibrated deployment self-calibrate at startup.
    """
    global _singleton
    if _try_load_env() is not None or _try_load_json() is not None:
        logger.info(
            "[xpool-cost] boot probe fitted curves (source=%s) but "
            "SGLANG_CSIGMA_* calibration is present — keeping the "
            "operator calibration (env precedence).", curves.source,
        )
        return False
    _singleton = curves
    logger.info(
        "[xpool-cost] boot-probe curves installed: kv_alpha=%.3e "
        "kv_beta=%.3e kv_gamma=%.3f m_alpha=%.3e m_beta=%.3f L*=%.0f",
        curves.kv_alpha, curves.kv_beta, curves.kv_gamma,
        curves.m_alpha, curves.m_beta, curves.L_star,
    )
    return True


# ---------- Runtime actuator-cost EWMA (paper §sec:design-l2-firegate) ----------
#
# Calibration measures recovery-cost curves c_σ(L) — these are
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
    """Process-wide EWMA of observed per-page actuator wall-time.

    The value is consumed per PAGE: `CostModel.c_xfer_us` computes
    `n_pages * current_us`, matching design.md which prices `c^xfer`
    per page. Initialized to a conservative value (3 ms/page by default)
    so the cold-start gate suppresses fires until enough live observations
    have arrived. After each observed fire of `n_chunks` pages consuming
    `total_us` microseconds, the per-page estimate is updated as
        new = α · (total_us / n_chunks) + (1 - α) · old
    where α defaults to 0.3 (5-fire half-life ≈ 1.6 fires).
    """

    # A single invalid measurement (non-finite / <=0 / n_chunks<=0) is a
    # transient and is skipped; this many CONSECUTIVE invalid samples means the
    # actuator wall-time path is broken (the EWMA can never update and the
    # Budgeter is stuck on the conservative default), so update() raises.
    # Normal c^xfer drift is expected and tracked by the EWMA — it does not
    # trip this; only impossible values in a row do.
    _INVALID_STREAK_LIMIT = 3

    def __init__(self, initial_us: float = 3000.0, alpha: float = 0.3):
        self._initial = float(initial_us)
        self._current = float(initial_us)
        self._alpha = float(alpha)
        self._n_observations = 0
        # Consecutive invalid measurements; reset by any valid update. A
        # sustained run trips the raise in update().
        self._consecutive_invalid = 0
        self._last_observed_us: Optional[float] = None
        # Set True once the boot-time c^xfer probe successfully seeds a
        # measured wall. Distinct from `is_calibrated` (≥3 LIVE fires,
        # always False at boot): this records that the seed is a real
        # measurement, not the conservative cold-start default — so a
        # calibration dump can tell a successful probe from a failed one.
        self._boot_seeded = False
        # Phase 6 introduced a second writer: `Admitter.execute_decision`
        # now calls `update()` from the scheduler thread on top of the
        # existing Budgeter worker thread caller. `_n_observations += 1`
        # decomposes into LOAD/INC/STORE in CPython; the GIL can switch
        # between them, so without this lock we'd lose observations
        # (or torn-read `_current` mid-update). See audit_phase6_meta.md.
        import threading
        self._lock = threading.Lock()

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

    @property
    def is_boot_seeded(self) -> bool:
        """True once `seed_from_boot_probe` installed a measured c^xfer
        wall. Unlike `is_calibrated` (≥3 live fires, always False at boot),
        this distinguishes a successful boot probe from one that failed and
        left the conservative default — the signal a calibration dump needs
        to decide whether to emit the c^xfer seed."""
        return self._boot_seeded

    def update(self, total_us: float, n_chunks: int) -> None:
        if n_chunks <= 0 or not math.isfinite(total_us) or total_us <= 0:
            with self._lock:
                self._consecutive_invalid += 1
                streak = self._consecutive_invalid
            if streak >= self._INVALID_STREAK_LIMIT:
                raise ValueError(
                    f"actuator c^xfer measurement invalid {streak} times in a "
                    f"row (latest total_us={total_us!r}, n_chunks={n_chunks!r}): "
                    f"the fire wall-time path is broken, the EWMA cannot update, "
                    f"and the Budgeter is stuck on the conservative default. "
                    f"Investigate the cudaSynchronize-bracketed fire timing "
                    f"Normal c^xfer drift does not trip this — only "
                    f"impossible values in a row do."
                )
            return
        per_chunk = float(total_us) / float(n_chunks)
        with self._lock:
            self._consecutive_invalid = 0
            if self._n_observations == 0:
                # First observation seeds the EWMA without dilution by the
                # conservative initial — that initial only governed cold-start.
                self._current = per_chunk
            else:
                self._current = self._alpha * per_chunk + (1.0 - self._alpha) * self._current
            self._last_observed_us = per_chunk
            self._n_observations += 1
            current = self._current
            n_obs = self._n_observations
        # Log outside the lock to keep critical section short.
        logger.info(
            "[xpool-cost] actuator EWMA update: observed=%.0fus/chunk "
            "(total=%.0fus over %d chunks), EWMA=%.0fus/chunk after %d obs",
            per_chunk, total_us, n_chunks, current, n_obs,
        )

    def seed_from_boot_probe(self, per_chunk_us: float) -> None:
        """Seed the cold-start estimate from the boot-time c^xfer probe
        (design.md §"Boot-time probe": `c^xfer` per page). Replaces
        the conservative 3000µs/chunk initial with a measured synthetic-
        transfer wall, so the fire-gate / Admitter cross-* pricing starts
        from a real number instead of the suppress-everything default.

        This is the boot constant the runtime EWMA `update()` then tracks
        against live fires, absorbing expected c^xfer drift (the value moving
        with GPU contention is by-design, not a fault). The invalid-streak guard
        RAISES when those live measurements are persistently invalid — a broken
        measurement path — not a drift detector. Distinct from `update()`:
        seeding does NOT count as an observation (`n_observations` stays
        0, so `is_calibrated` still gates on live fires), it just moves
        the cold-start baseline. No-op on a non-positive probe (fail to
        the conservative default rather than seed a bogus 0).
        """
        if not math.isfinite(per_chunk_us) or per_chunk_us <= 0:
            return
        with self._lock:
            self._initial = float(per_chunk_us)
            self._boot_seeded = True
            if self._n_observations == 0:
                self._current = float(per_chunk_us)
        logger.info(
            "[xpool-cost] c^xfer boot-probe seed: %.0f us/chunk "
            "(was conservative default; live EWMA still gates on fires)",
            per_chunk_us,
        )

    def reset(self) -> None:
        with self._lock:
            self._current = self._initial
            self._n_observations = 0
            self._last_observed_us = None
            self._boot_seeded = False


_runtime_actuator: Optional[RuntimeActuatorCost] = None


def get_runtime_actuator_cost() -> RuntimeActuatorCost:
    """Process-wide singleton. Initial value can be overridden via
    SGLANG_XPOOL_NB_CHUNK_COST_INIT_US (env name retains the legacy
    "CHUNK" spelling for back-compat; the value is per page).

    Default 3000us/page is the empirically-measured fire wall time on
    Qwen3.5-9B (D7 v5 measurement: 28 fires of 64 pages each averaged
    1234us/page = cap_barrier 344us + unmap 781us + map 109us). The
    3000us default applies a 2.4× safety factor over the measured mean,
    so the conservative gate suppresses spurious fires but doesn't stall
    the first legitimate fire on cold-start. Other hardware (different
    GPU SKUs, larger models, higher contention) may need tuning via the
    env var. Boot-time empirical calibration would
    auto-tune this; deferred since the static default works for our
    validation hardware.

    Prior default was 10000us/chunk, calibrated when MemPool was the
    arena path (~3x slower per pytorch issue 165419). After the
    from_blob switch (commits cd3902bcc6 + 241463552d), 3000us is
    realistic, not conservative."""
    global _runtime_actuator
    if _runtime_actuator is None:
        initial = float(
            os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_INIT_US", "3000")
        )
        alpha = float(os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_EWMA_ALPHA", "0.3"))
        _runtime_actuator = RuntimeActuatorCost(initial_us=initial, alpha=alpha)
        logger.info(
            "[xpool-cost] RuntimeActuatorCost initialized: initial=%.0fus/page "
            "(D7-measured + 2.4x safety; suppresses spurious fires but "
            "lets first legitimate fire through) alpha=%.2f",
            initial, alpha,
        )
    return _runtime_actuator


def reset_runtime_actuator_cost() -> None:
    global _runtime_actuator
    _runtime_actuator = None


class BootProbedMigrateCost:
    """Per-slot mamba migration wall, populated by a boot-time probe
    (`migrate_probe.measure_mamba_migrate`). design.md §"Shared cost
    model" specifies `c_m(X) ≈ X / side_stream_bw + per-slot constant`;
    since each migration moves exactly one mamba slot's worth of state,
    the probe collapses to a single scalar `per_slot_us`.

    Cold-start contract: until `set_mamba(per_slot_us)` is called,
    `mamba_per_slot_us` is `+inf` so `CostModel.c_migrate_us(...)`
    returns `+inf` and the Admitter's migrate candidates remain
    infeasible.

    Difference vs `RuntimeActuatorCost`: `RuntimeActuatorCost` cold-
    starts at a conservative 3000 µs (soft gate via `is_calibrated`)
    because c^xfer drifts under live traffic and must absorb EWMA
    updates; `c_m` is a fixed-hardware constant with a deterministic
    boot probe, so cold-start is hard `+inf` (no drift detector, no
    EWMA, single one-shot set). The two gates differ by design.

    Boot probe is run once by `BudgetAgent._ensure_actuator_chain`
    immediately after the actuator chain is built; the value stays
    fixed for the engine's lifetime (steady-state deterministic
    prediction, per design.md §"Why exact c^evict").

    KV-side migration is NOT supported (KV pool has no `migrate_slot`
    primitive); `c_migrate_us(pool='kv', ...)` returns +inf.
    """

    def __init__(self) -> None:
        self._mamba_per_slot_us: float = float("inf")
        self._calibrated: bool = False
        # c_m is a fixed-hardware constant (no drift — see class docstring),
        # so a calibrated value can be pinned offline via env, mirroring the
        # SGLANG_CSIGMA_* env-precedence path for κ_i. When pinned, the boot
        # probe in BudgetAgent skips re-measuring (env wins). A profile from
        # dev/eval/cost_model/calibrate_profile.sh emits this export.
        self._env_pinned: bool = False
        env_us = os.environ.get("SGLANG_CM_MAMBA_PER_SLOT_US")
        if env_us is not None:
            val = float(env_us)
            if val <= 0:
                raise ValueError(
                    "SGLANG_CM_MAMBA_PER_SLOT_US must be positive, got "
                    f"{env_us!r}"
                )
            self._mamba_per_slot_us = val
            self._calibrated = True
            self._env_pinned = True
            logger.info(
                "BootProbedMigrateCost: c_m pinned from "
                "SGLANG_CM_MAMBA_PER_SLOT_US=%.1f µs/slot (env-precedence; "
                "the boot probe will not overwrite it)",
                val,
            )

    @property
    def mamba_per_slot_us(self) -> float:
        return self._mamba_per_slot_us

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_env_pinned(self) -> bool:
        """True when c_m was pinned via SGLANG_CM_MAMBA_PER_SLOT_US; the
        boot probe defers to it (env-precedence, like κ_i's curves)."""
        return self._env_pinned

    def set_mamba(self, per_slot_us: float) -> None:
        if per_slot_us <= 0:
            raise ValueError(
                f"BootProbedMigrateCost: per_slot_us must be positive, "
                f"got {per_slot_us!r}"
            )
        self._mamba_per_slot_us = float(per_slot_us)
        self._calibrated = True
        logger.info(
            "BootProbedMigrateCost: mamba per-slot wall set to %.1f µs "
            "(c_migrate_us(N slots) = N × %.1f µs)",
            per_slot_us, per_slot_us,
        )

    def reset(self) -> None:
        self._mamba_per_slot_us = float("inf")
        self._calibrated = False


_migrate_cost: Optional[BootProbedMigrateCost] = None


def get_migrate_cost() -> BootProbedMigrateCost:
    """Process-wide singleton."""
    global _migrate_cost
    if _migrate_cost is None:
        _migrate_cost = BootProbedMigrateCost()
    return _migrate_cost


def reset_migrate_cost() -> None:
    global _migrate_cost
    _migrate_cost = None



class CostModel:
    """Unified per-arrival cost facade for the Admitter (design.md §358).

    A thin facade over `get_cost_curves()` (Stage-0 calibrated re-prefill
    curves) + `get_runtime_actuator_cost()` (EWMA over committed fires).
    Multiple `CostModel()` instances share the same underlying singletons.

    Provides the four cost functions the Admitter needs:

    - `c_xfer_us(n_pages)`: wall-time to transfer n_pages via actuator
    - `c_evict_us(pool, x_tokens)`: expected recompute if evicting cheapest
      x_tokens worth of cache (Phase 3 — currently raises NotImplementedError)
    - `c_recompute_us(pool, s_tokens)`: re-prefill wall for length s
    - `w_q_us()`: SLO penalty per req of queue wait

    Cold-start is cost-driven (no separate warm-up gate): until the c^xfer
    EWMA has enough observations, `get_runtime_actuator_cost()` returns the
    conservative initial value, which makes cross-* candidates lose the
    Admitter's min-cost compare on their own.
    """

    def __init__(self):
        # Per-pool radix-cache references for the c^evict predictor.
        # Plugged in by `set_evict_cache(pool, cache)` at scheduler
        # init. Default empty → `c_evict_us` returns +inf (no
        # own-evict candidate). Each cache exposes a
        # `predict_evict_cost_us(num_tokens, pool) -> float` method
        # that walks evictable leaves under the active policy.
        self._evict_caches: dict = {}

    def c_xfer_us(self, n_pages: int) -> float:
        """Expected wall-time (µs) to move `n_pages` pages cross-pool.
        Scales linearly with n_pages over the EWMA per-page cost."""
        if n_pages <= 0:
            return 0.0
        return float(n_pages) * get_runtime_actuator_cost().current_us

    def c_migrate_us(self, pool: str, n_slots: int) -> float:
        """Expected wall (µs) to migrate `n_slots` slots in `pool` via
        the side-stream slot-state copy (`MambaPool.migrate_slot`'s data
        body). Scales linearly over the boot-probed per-slot wall.
        Implements design.md §"Shared cost model" `c_m(X)`.

        Returns +inf for 'kv' (no KV-side migrate primitive — known
        semantic, not an error) and during mamba cold-start (probe
        hasn't run yet) — fail-closed so the Admitter's migrate
        candidates stay infeasible until the cost is known.

        Raises ValueError on any other pool name so caller typos crash
        loudly instead of silently returning +inf — mirrors
        `c_recompute_us`'s pool-string discipline.
        """
        if pool == "kv":
            return float("inf")
        if pool != "mamba":
            raise ValueError(
                f"c_migrate_us: unknown pool {pool!r} (expected "
                f"'mamba' or 'kv')"
            )
        if n_slots <= 0:
            return 0.0
        return float(n_slots) * get_migrate_cost().mamba_per_slot_us

    def c_evict_us(self, pool: str, x_tokens: int) -> float:
        """Expected recompute cost (µs) if we evict the cheapest blocks
        totaling `x_tokens` tokens from `pool` cache. Implements
        design.md §"Shared cost model" `c^evict_i(X)` — exact walk
        of the radix tree at decision time under the active eviction
        policy.

        Cold-start: returns +inf until a cache is plugged in via
        `set_evict_cache(pool, cache)` — fail-closed so the Admitter
        doesn't pick own-evict on an uninitialised cost model.

        Raises ValueError on unknown pool name (typos crash loudly;
        mirrors `c_recompute_us` / `c_migrate_us` discipline).
        """
        if pool not in ("kv", "mamba"):
            raise ValueError(
                f"c_evict_us: unknown pool {pool!r} (expected "
                f"'kv' or 'mamba')"
            )
        # Evicting 0 tokens costs nothing — independent of whether a cache
        # is wired. Must precede the cache-None check so a zero-drain
        # candidate (cumulative free→drain→migrate with no drain part)
        # is never poisoned to +inf when the cache is absent.
        if int(x_tokens) <= 0:
            return 0.0
        cache = self._evict_caches.get(pool)
        if cache is None:
            return float("inf")
        # Pass `pool` through: on a hybrid model the SAME
        # MambaRadixCache instance is wired for both pools and
        # dispatches the full-tree (kv) vs mamba walk on this arg.
        return cache.predict_evict_cost_us(int(x_tokens), pool=pool)

    def set_evict_cache(self, pool: str, cache) -> None:
        """Plug in (or replace) the per-pool radix-cache reference
        used by the c^evict predictor. `cache` must expose
        `predict_evict_cost_us(num_tokens, pool) -> float`. Called
        once at scheduler init when the radix cache is available; the
        same instance lives for the engine's lifetime. On a hybrid
        model the same MambaRadixCache is wired for both pools and
        dispatches the full-tree vs mamba walk on the `pool` arg.
        """
        if pool not in ("kv", "mamba"):
            raise ValueError(
                f"set_evict_cache: unknown pool {pool!r}"
            )
        self._evict_caches[pool] = cache

    def c_recompute_us(self, pool: str, s_tokens: int) -> float:
        """Re-prefill wall (µs) for a sequence of length `s_tokens` in
        `pool` ('kv' or 'mamba'). Reads from CostCurves (Stage-0)."""
        curves = get_cost_curves()
        if pool == "kv":
            return curves.c_kv_ms(int(s_tokens)) * 1000.0
        elif pool == "mamba":
            return curves.c_m_ms(int(s_tokens)) * 1000.0
        else:
            raise ValueError(f"unknown pool: {pool!r}")

    def w_q_us(self) -> float:
        """SLO penalty (µs) per req of queue wait. Default 100 µs,
        env override SGLANG_XPOOL_QUEUE_WAIT_US (matches existing
        pressure_adapter convention so the two stay synchronised)."""
        return float(os.environ.get("SGLANG_XPOOL_QUEUE_WAIT_US", "100"))

    def update_xfer(self, total_us: float, n_chunks: int) -> None:
        """Producer side: feed an observed fire's wall-time + chunk count
        into the c^xfer EWMA. Called from the worker after each non-aborted
        actuator fire (Phase 1 wiring in agent.py)."""
        get_runtime_actuator_cost().update(total_us, n_chunks)


_cost_model: Optional[CostModel] = None


def get_cost_model() -> CostModel:
    """Process-wide singleton facade (thin wrapper, all state lives in
    underlying CostCurves + RuntimeActuatorCost singletons)."""
    global _cost_model
    if _cost_model is None:
        _cost_model = CostModel()
    return _cost_model


def reset_cost_model() -> None:
    """Test helper. Resets the facade and the underlying singletons."""
    global _cost_model
    _cost_model = None
    reset_cost_curves()
    reset_runtime_actuator_cost()
    reset_migrate_cost()
