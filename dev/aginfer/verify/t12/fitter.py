"""T12 h_τ(occ) shape fitter (PLAN §1).

Three candidate functional forms for the per-(tier, subpool) holding
cost curve ``h_τ_sp(occ)`` (DESIGN §7).  Spec uses a linear
placeholder ``α × occ``; T12 falsifies that against two convex
alternatives.

Models
------

``linear``      ``y = α × occ``                  (1 param, the spec placeholder)
``power``       ``y = α × occ^γ``                (2 params; γ > 1 = right-tail-heavy)
``hyperbolic``  ``y = α / (1 − occ)``            (1 param; diverges as occ → 1)

Each fit minimises sum-of-squares.  Comparison metrics: RMSE, MAE,
R², and AIC (Akaike Information Criterion, adjusted for parameter
count).  Pick by AIC — adjusts for the extra DoF the 2-parameter
power model has over linear / hyperbolic.

This module is workload-agnostic: no benchmark-specific assumptions
baked in.  Feed it ``(occ, marginal_V_u)`` pairs from any source
(real scenario logs OR synthetic ground-truth for testing).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------- model functions


def _linear(occ: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * occ


def _power(occ: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    return alpha * np.power(occ, gamma)


def _hyperbolic(occ: np.ndarray, alpha: float) -> np.ndarray:
    # Guard against occ == 1.0 inputs — caller is expected to clamp,
    # but the fitter is defensive.
    return alpha / (1.0 - np.clip(occ, 0.0, 0.9999))


@dataclass(frozen=True)
class FitResult:
    """Result of fitting one shape to one (tier, subpool) bucket."""
    shape: str                  # "linear" / "power" / "hyperbolic"
    params: Tuple[float, ...]   # alpha, [gamma]
    n_samples: int
    rmse: float
    mae: float
    r_squared: float
    aic: float                  # lower = better; comparable across nested models


def _aic(n: int, rss: float, k: int) -> float:
    """Akaike Information Criterion for Gaussian residuals.

    ``k`` = parameter count.  ``+2k`` penalises extra params; lower
    AIC = better fit-after-penalty.

    Edge case: clean ground-truth data → RSS ≈ 0 → ``log(0)`` would
    raise.  We floor at a tiny positive value so a perfect fit gets
    a very-large-negative AIC (still the best by comparison).
    """
    if n <= k:
        return math.inf
    # Constant offset (n * log(2π) + n) dropped — only relative AIC matters.
    return n * math.log(max(rss / n, 1e-300)) + 2 * k


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_one(
    shape: str,
    occ: Sequence[float],
    y: Sequence[float],
) -> FitResult:
    """Fit one of {linear, power, hyperbolic} to ``(occ, y)`` data."""
    occ_arr = np.asarray(occ, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(occ_arr)
    if n < 2:
        raise ValueError(f"need >= 2 samples to fit; got {n}")

    if shape == "linear":
        params, _ = curve_fit(
            _linear, occ_arr, y_arr,
            p0=[1.0], bounds=([-np.inf], [np.inf]),
        )
        y_pred = _linear(occ_arr, *params)
        k = 1
    elif shape == "power":
        params, _ = curve_fit(
            _power, occ_arr, y_arr,
            p0=[1.0, 1.5],
            bounds=([-np.inf, 0.5], [np.inf, 10.0]),  # γ ∈ [0.5, 10]
            maxfev=5000,
        )
        y_pred = _power(occ_arr, *params)
        k = 2
    elif shape == "hyperbolic":
        params, _ = curve_fit(
            _hyperbolic, occ_arr, y_arr,
            p0=[1.0], bounds=([-np.inf], [np.inf]),
        )
        y_pred = _hyperbolic(occ_arr, *params)
        k = 1
    else:
        raise ValueError(f"unknown shape {shape!r}")

    resid = y_arr - y_pred
    rss = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(rss / n))
    mae = float(np.mean(np.abs(resid)))
    return FitResult(
        shape=shape,
        params=tuple(float(p) for p in params),
        n_samples=n,
        rmse=rmse, mae=mae,
        r_squared=_r_squared(y_arr, y_pred),
        aic=_aic(n, rss, k),
    )


def fit_all(
    occ: Sequence[float], y: Sequence[float],
) -> Dict[str, FitResult]:
    """Fit all 3 candidate shapes; return keyed by shape name.

    Caller picks by AIC (smaller = better) or by domain knowledge.
    Shapes that fail to converge are omitted; downstream picker
    handles missing keys."""
    out: Dict[str, FitResult] = {}
    for shape in ("linear", "power", "hyperbolic"):
        try:
            out[shape] = fit_one(shape, occ, y)
        except Exception as exc:  # noqa: BLE001
            # Convergence failures, ill-conditioned data, etc. — log
            # by omission so the picker sees a partial set.
            continue
    return out


def best_by_aic(fits: Dict[str, FitResult]) -> str:
    """Pick the shape with the LOWEST AIC.  Ties broken in favour of
    the simpler model (fewer parameters): linear / hyperbolic (1
    param) over power (2 params)."""
    if not fits:
        raise ValueError("no fits to choose from")
    # AIC alone handles the parameter-count penalty, but if AICs
    # are within ε (numerical tie), prefer simpler.
    ranked = sorted(fits.items(),
                    key=lambda kv: (kv[1].aic,
                                    len(kv[1].params)))
    return ranked[0][0]


# ---------------------------------------------------------- log-line parser


def parse_t12_log_lines(lines: Sequence[str]) -> Dict[
    Tuple[str, str], List[Tuple[float, float]]
]:
    """Parse structured aginfer log lines into per-(tier, subpool)
    buckets of ``(occ, marginal_v_u)`` pairs.

    Expected line format (emitted by daemon when
    ``AGINFER_T12_CALIBRATION_LOG=1`` — pending T11 instrumentation):

        aginfer_metric event=t12_calibration tier=HBM subpool=kv \\
            occ=0.62 marginal_v_u=-1.234 n_units=42

    Any line not starting with ``aginfer_metric event=t12_calibration``
    is skipped.  Required fields: tier, subpool, occ, marginal_v_u.
    Missing fields → line dropped (logged via returning a smaller
    bucket; callers can compare ``len(parsed)`` against ``len(lines)``
    if they need to detect drops).
    """
    out: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    for raw in lines:
        s = raw.strip()
        if "aginfer_metric event=t12_calibration" not in s:
            continue
        fields: Dict[str, str] = {}
        for tok in s.split():
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            fields[k] = v
        try:
            tier = fields["tier"]
            sp = fields["subpool"]
            occ = float(fields["occ"])
            mv = float(fields["marginal_v_u"])
        except (KeyError, ValueError):
            continue
        out.setdefault((tier, sp), []).append((occ, mv))
    return out


__all__ = [
    "FitResult",
    "fit_one", "fit_all", "best_by_aic",
    "parse_t12_log_lines",
]
