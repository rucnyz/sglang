"""T12 verify — h_τ(occ) shape fitter correctness (PLAN §1).

Like T13, T12 has two halves:

  (1) Fitter / picker harness for the 3 candidate shapes (linear /
      power / hyperbolic).  Stays workload-agnostic.
  (2) Real-data calibration on scenario-run logs.

Half (2) needs the calibration log line emitted by the daemon during
a real scenario run.  That instrumentation lands with **T11** (the
estimator-replacement task that owns the scenario-run data path —
see PLAN §1).  Until T11 lands, this verify exercises ONLY half (1)
against synthetic ground truth, which proves the fitter recovers
known shapes under noise.

Stage list:

  A. Recovery — given data drawn from each true shape, the picker
     returns that shape (AIC).
     A0 linear → linear
     A1 power(γ=2.5) → power
     A2 hyperbolic → hyperbolic

  B. Robustness — noise / partial-converge / ill-formed inputs
     B0 moderate Gaussian noise (σ = 5% of signal) preserves recovery
     B1 fit_one raises on < 2 samples
     B2 fit_all silently omits shapes that fail to converge
     B3 best_by_aic picks the smaller-param model on AIC ties

  C. Log parser — the future daemon log format
     C0 parse_t12_log_lines extracts the expected fields, groups by
        (tier, subpool); ignores foreign lines
     C1 malformed lines are silently dropped (no crash, no partial
        record)

Usage:
    python dev/aginfer/verify/t12/verify.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from fitter import (  # noqa: E402
    FitResult,
    best_by_aic,
    fit_all,
    fit_one,
    parse_t12_log_lines,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---- ground-truth generators ----


def _gen_linear(occ: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    return alpha * occ


def _gen_power(occ: np.ndarray, alpha: float = 1.5, gamma: float = 2.5) -> np.ndarray:
    return alpha * occ ** gamma


def _gen_hyperbolic(occ: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return alpha / (1.0 - np.clip(occ, 0.0, 0.99))


def _grid(n: int = 60, hi: float = 0.95) -> np.ndarray:
    """Even occ grid on [0.05, hi]; avoids the 0 and 1 boundaries
    (linear/power are degenerate at 0; hyperbolic at 1)."""
    return np.linspace(0.05, hi, n)


def _add_noise(y: np.ndarray, sigma_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = float(np.mean(np.abs(y))) * sigma_frac
    return y + rng.normal(0.0, scale, size=y.shape)


# ============================================================ A. recovery


def stage_a0_recover_linear() -> None:
    occ = _grid()
    y_true = _gen_linear(occ, alpha=2.0)
    fits = fit_all(occ.tolist(), y_true.tolist())
    pick = best_by_aic(fits)
    if pick != "linear":
        raise StageFail(
            f"true=linear → picker chose {pick!r}; AICs="
            f"{{ {', '.join(f'{k}:{v.aic:.2f}' for k,v in fits.items())} }}"
        )
    # Sanity: fit recovers α within 1 %.
    alpha = fits["linear"].params[0]
    if not (1.98 <= alpha <= 2.02):
        raise StageFail(f"α should ≈ 2.0; got {alpha}")


def stage_a1_recover_power_gamma_2p5() -> None:
    occ = _grid()
    y_true = _gen_power(occ, alpha=1.5, gamma=2.5)
    fits = fit_all(occ.tolist(), y_true.tolist())
    pick = best_by_aic(fits)
    if pick != "power":
        raise StageFail(
            f"true=power(γ=2.5) → picker chose {pick!r}; AICs="
            f"{{ {', '.join(f'{k}:{v.aic:.2f}' for k,v in fits.items())} }}"
        )
    alpha, gamma = fits["power"].params
    if not (1.40 <= alpha <= 1.60):
        raise StageFail(f"α should ≈ 1.5; got {alpha}")
    if not (2.4 <= gamma <= 2.6):
        raise StageFail(f"γ should ≈ 2.5; got {gamma}")


def stage_a2_recover_hyperbolic() -> None:
    # Hyperbolic curves get steep near occ=1; keep grid below 0.9 to
    # avoid numerical blow-up that would dominate residuals.
    occ = _grid(hi=0.85)
    y_true = _gen_hyperbolic(occ, alpha=0.1)
    fits = fit_all(occ.tolist(), y_true.tolist())
    pick = best_by_aic(fits)
    if pick != "hyperbolic":
        raise StageFail(
            f"true=hyperbolic → picker chose {pick!r}; AICs="
            f"{{ {', '.join(f'{k}:{v.aic:.2f}' for k,v in fits.items())} }}"
        )
    alpha = fits["hyperbolic"].params[0]
    if not (0.095 <= alpha <= 0.105):
        raise StageFail(f"α should ≈ 0.1; got {alpha}")


# ============================================================ B. robustness


def stage_b0_recovery_under_noise() -> None:
    """5 % Gaussian noise on each of the 3 shapes — picker still
    recovers the true shape (averaged over 5 seeds)."""
    cases: List[Tuple[str, Callable, Tuple[float, ...]]] = [
        ("linear",     _gen_linear,     (2.0,)),
        ("power",      _gen_power,      (1.5, 2.5)),
        ("hyperbolic", _gen_hyperbolic, (0.1,)),
    ]
    occ_lin = _grid()
    occ_hyp = _grid(hi=0.85)
    for true_shape, gen, params in cases:
        occ = occ_hyp if true_shape == "hyperbolic" else occ_lin
        y_clean = gen(occ, *params)
        hits = 0
        for seed in range(5):
            y_noisy = _add_noise(y_clean, sigma_frac=0.05, seed=seed)
            fits = fit_all(occ.tolist(), y_noisy.tolist())
            if best_by_aic(fits) == true_shape:
                hits += 1
        if hits < 4:  # >= 4/5 recoveries
            raise StageFail(
                f"true={true_shape}: only {hits}/5 noisy recoveries"
            )


def stage_b1_fit_one_too_few_samples() -> None:
    try:
        fit_one("linear", [0.5], [1.0])
    except ValueError:
        pass
    else:
        raise StageFail(
            "fit_one with n=1 must raise ValueError"
        )


def stage_b2_fit_all_omits_non_converge() -> None:
    """A pathological input (all zeros on a hyperbolic fit, etc.)
    should DROP the failing shape from the returned dict, not
    propagate the curve_fit exception."""
    occ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.zeros_like(occ)  # all zeros — degenerate for some shapes
    fits = fit_all(occ.tolist(), y.tolist())
    # We don't assert which shapes survive, just that fit_all does
    # not raise.  A real production caller would log the missing
    # shapes via len(fits) < 3.
    if not isinstance(fits, dict):
        raise StageFail(f"fit_all returned non-dict: {type(fits)}")


def stage_b3_best_by_aic_ties_prefer_simpler() -> None:
    """If two shapes have identical AICs (numeric tie), the picker
    breaks ties in favour of the model with fewer parameters
    (Occam's-razor default)."""
    # Fake two FitResults with the same AIC; one is "power" (2 params),
    # the other "linear" (1 param).  Picker must return "linear".
    fits = {
        "linear": FitResult(
            shape="linear", params=(1.0,), n_samples=10,
            rmse=0.1, mae=0.08, r_squared=0.99, aic=12.34,
        ),
        "power": FitResult(
            shape="power", params=(1.0, 1.0), n_samples=10,
            rmse=0.1, mae=0.08, r_squared=0.99, aic=12.34,
        ),
    }
    pick = best_by_aic(fits)
    if pick != "linear":
        raise StageFail(
            f"AIC tie → simpler model expected; got {pick!r}"
        )


# ============================================================ C. log parser


def stage_c0_parser_groups_by_tier_subpool() -> None:
    """Parser extracts the 4 required fields, groups by (tier,
    subpool), and ignores unrelated log lines."""
    lines = [
        "2026-06-01 01:00:00 INFO daemon: started",  # ignored
        "aginfer_metric event=t12_calibration tier=HBM subpool=kv "
        "occ=0.62 marginal_v_u=-1.234 n_units=42",
        "aginfer_metric event=t12_calibration tier=HBM subpool=kv "
        "occ=0.71 marginal_v_u=-0.812 n_units=39",
        "aginfer_metric event=t12_calibration tier=DRAM subpool=kv "
        "occ=0.30 marginal_v_u=-2.0 n_units=11",
        "aginfer_metric event=apply_failed reason=race_lost",  # ignored
    ]
    parsed = parse_t12_log_lines(lines)
    if set(parsed.keys()) != {("HBM", "kv"), ("DRAM", "kv")}:
        raise StageFail(f"groups: {set(parsed.keys())}")
    if len(parsed[("HBM", "kv")]) != 2:
        raise StageFail(
            f"HBM/kv samples: {parsed[('HBM','kv')]}"
        )
    if len(parsed[("DRAM", "kv")]) != 1:
        raise StageFail(
            f"DRAM/kv samples: {parsed[('DRAM','kv')]}"
        )
    # Spot-check values.
    occ, mv = parsed[("HBM", "kv")][0]
    if not math.isclose(occ, 0.62) or not math.isclose(mv, -1.234):
        raise StageFail(f"first HBM/kv: ({occ}, {mv})")


def stage_c1_parser_drops_malformed() -> None:
    """Lines that match the prefix but have missing / malformed
    required fields are silently dropped — no crash, no half-
    populated record."""
    lines = [
        "aginfer_metric event=t12_calibration tier=HBM occ=0.5",  # no subpool
        "aginfer_metric event=t12_calibration tier=HBM subpool=kv occ=NaNxx marginal_v_u=-1",  # malformed occ
        "aginfer_metric event=t12_calibration tier=HBM subpool=kv occ=0.5 marginal_v_u=-1",  # OK
    ]
    parsed = parse_t12_log_lines(lines)
    # Only one valid line — only (HBM, kv) bucket with one point.
    if parsed != {("HBM", "kv"): [(0.5, -1.0)]}:
        raise StageFail(f"parsed: {parsed}")


# ============================================================ run


_STAGES = [
    ("A0 recover linear from clean data",         stage_a0_recover_linear),
    ("A1 recover power(γ=2.5) from clean data",   stage_a1_recover_power_gamma_2p5),
    ("A2 recover hyperbolic from clean data",     stage_a2_recover_hyperbolic),
    ("B0 recover under 5% Gaussian noise (4/5 seeds)", stage_b0_recovery_under_noise),
    ("B1 fit_one < 2 samples raises",             stage_b1_fit_one_too_few_samples),
    ("B2 fit_all omits non-converging shapes",    stage_b2_fit_all_omits_non_converge),
    ("B3 best_by_aic ties prefer simpler model",  stage_b3_best_by_aic_ties_prefer_simpler),
    ("C0 parser groups by (tier, subpool)",       stage_c0_parser_groups_by_tier_subpool),
    ("C1 parser drops malformed lines",           stage_c1_parser_drops_malformed),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(f"\nT12 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT12 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
