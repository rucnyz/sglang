"""5th-round numeric-robustness audit (Phase 2): NaN/inf/negative inputs
must not poison the cost model.

Root cause found by the audit: every guard was written `x <= 0` /
`coeff < 0`, but `nan <= 0` is False in Python, so NaN slipped through
all of them. These tests pin the NaN-safe + finite-validation fixes:

1. The cost-curve LOADER (`_try_load_env` / `_try_load_json`) is the
   production-reachable HIGH: κ_i comes ONLY from SGLANG_CSIGMA_*
   (the in-engine probe is a no-op), and `calibrate_kappa.py` emits
   raw `np.polyfit` output — a thin/noisy offline fit can produce a
   negative or NaN/inf alpha. The loader must reject it → builtin.
2. `CostCurves.c_kv_ms` / `c_m_ms` must be NaN-safe (L=nan → constant
   term, not polynomial → `max(0,nan)=0` silent-wrong).
3. `fit_cost_curves` must RAISE on NaN-poisoned coeffs (not ship them).
4. `RuntimeActuatorCost.update` / `seed_from_boot_probe` must reject
   NaN/inf (a single NaN otherwise permanently poisons the EWMA).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

_KV = ("SGLANG_CSIGMA_KV_ALPHA", "SGLANG_CSIGMA_KV_BETA",
       "SGLANG_CSIGMA_KV_GAMMA", "SGLANG_CSIGMA_M_ALPHA",
       "SGLANG_CSIGMA_M_BETA", "SGLANG_CSIGMA_LSTAR")


def _clear():
    for k in list(os.environ):
        if k.startswith("SGLANG_CSIGMA_"):
            del os.environ[k]


def _reset():
    from sglang.srt.budgeter import cost_model
    cost_model.reset_cost_curves()


def test_1_loader_rejects_nonfinite_negative_alpha():
    """A profile (or stale/edited .sh) with NaN/inf/negative kv_alpha must
    NOT install a poisoned curve — fall back to builtin. κ_i comes ONLY
    from SGLANG_CSIGMA_* in production (in-engine probe is a no-op), so the
    loader is the last line of defense."""
    from sglang.srt.budgeter import cost_model
    for bad in ("nan", "inf", "-1e9", "Infinity", "NaN"):
        _clear(); _reset()
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = bad
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "5.0"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "0.0"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "0.0"
        c = cost_model.get_cost_curves()
        assert c.source == "builtin_default", (
            f"kv_alpha={bad!r} must be rejected → builtin fallback, got "
            f"source={c.source} kv_alpha={c.kv_alpha}"
        )
        assert math.isfinite(c.kv_alpha) and c.kv_alpha >= 0
    _clear(); _reset()
    print("  PASS  1  loader rejects NaN/inf/negative kv_alpha → builtin fallback")


def test_2_c_kv_ms_nan_safe():
    """c_kv_ms/c_m_ms with L=nan must take the constant-term branch (not
    run the polynomial → max(0,nan)=0 silent-wrong), and per-token must
    not emit NaN."""
    from sglang.srt.budgeter.cost_model import BUILTIN_DEFAULT as c
    assert c.c_kv_ms(float("nan")) == c.kv_gamma, c.c_kv_ms(float("nan"))
    assert c.c_m_ms(float("nan")) == c.m_beta, c.c_m_ms(float("nan"))
    assert not math.isnan(c.c_kv_per_token_us(float("nan"))), "per-token NaN"
    assert not math.isnan(c.c_m_per_token_us(float("nan"))), "per-token NaN"
    print("  PASS  2  c_kv_ms/c_m_ms NaN-safe (L=nan → constant term, no NaN)")


def test_3_fit_rejects_nan_coeff():
    """A NaN sample wall poisons np.polyfit → all coeffs NaN. The
    negative-coeff guard `coeff < 0` misses it (nan<0 is False); the fit
    must RAISE, not ship a NaN curve."""
    from sglang.srt.budgeter.cost_model import fit_cost_curves
    kv = [(256, 10.0), (1024, float("nan")), (4096, 200.0)]
    m = [(256, 5.0), (4096, 50.0)]
    raised = False
    try:
        fit_cost_curves(kv, m, source="test")
    except Exception:
        raised = True
    assert raised, "fit_cost_curves must raise on NaN-poisoned coeffs"
    print("  PASS  3  fit_cost_curves raises on NaN coeff (no silent NaN curve)")


def test_4_actuator_rejects_nonfinite():
    """A single NaN/inf observation otherwise permanently poisons the EWMA
    (nan<=0 is False so the `<=0` guard misses it; then α·x+(1-α)·nan stays
    NaN forever while is_calibrated still flips True)."""
    from sglang.srt.budgeter.cost_model import RuntimeActuatorCost
    a = RuntimeActuatorCost(initial_us=3000.0)
    a.update(float("nan"), 1)
    assert math.isfinite(a.current_us), f"NaN update poisoned EWMA: {a.current_us}"
    a.update(float("inf"), 1)
    assert math.isfinite(a.current_us), f"inf update poisoned EWMA: {a.current_us}"
    a.seed_from_boot_probe(float("nan"))
    assert math.isfinite(a.current_us), f"NaN seed poisoned: {a.current_us}"
    a.seed_from_boot_probe(float("inf"))
    assert math.isfinite(a.current_us), f"inf seed poisoned: {a.current_us}"
    # a good update still lands
    a.update(1500.0, 1)
    assert abs(a.current_us - 1500.0) < 1e-6, a.current_us
    print("  PASS  4  RuntimeActuatorCost rejects NaN/inf update+seed (no poison)")


def main():
    tests = [test_1_loader_rejects_nonfinite_negative_alpha,
             test_2_c_kv_ms_nan_safe, test_3_fit_rejects_nan_coeff,
             test_4_actuator_rejects_nonfinite]
    print(f"\nnumeric-robustness tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"numeric: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
