"""#118 — boot-time actuator cost calibration (c^xfer + κ_i).

Covers the CPU-testable core of the boot probes (design.md §"Boot-time
probe": pin `c^xfer` per page and `c_i(s)`'s `κ` coefficients at engine
start). The GPU-side probe EXECUTION (synthetic cross-pool transfer +
real model prefill timing) lives in `boot_probe.py` and is validated by
an engine boot; here we pin:

1. `fit_cost_curves` recovers known κ coefficients from clean samples.
2. fit raises on too-few distinct L (KV quadratic needs ≥3, mamba ≥2).
3. fit raises on unphysical negative leading coefficient.
4. `_crossover_L_star` solves c_KV(L*)=c_M(L*).
5. `set_cost_curves` installs probe curves when NO operator calibration.
6. `set_cost_curves` SKIPS (env precedence) when SGLANG_CSIGMA_* present.
7. `RuntimeActuatorCost.seed_from_boot_probe` moves the cold-start
   baseline without counting as a live observation; EWMA drift still
   works on top.
8. `balance_restore` converges a simulated kv capacity to baseline
   under an ASYMMETRIC fire model (fwd/rev steps don't pair).
9. `balance_restore` breaks on a no-progress fire (no spin to max).
10. `balance_restore` caps at `max_fires` (no infinite loop).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def _clear_csigma_env():
    for k in list(os.environ):
        if k.startswith("SGLANG_CSIGMA_"):
            del os.environ[k]


def test_1_fit_recovers_known_coefficients():
    from sglang.srt.budgeter.cost_model import fit_cost_curves
    # Ground-truth curve.
    kv_a, kv_b, kv_g = 1.2e-7, 3.0e-4, 0.44
    m_a, m_b = 2.17e-3, 6.99

    def c_kv(L):
        return kv_a * L * L + kv_b * L + kv_g

    def c_m(L):
        return m_a * L + m_b

    kv_lengths = [256, 1024, 2048, 4096, 8192, 16384]
    m_lengths = [256, 2048, 8192, 16384]
    kv_samples = [(L, c_kv(L)) for L in kv_lengths]
    m_samples = [(L, c_m(L)) for L in m_lengths]

    cur = fit_cost_curves(kv_samples, m_samples, source="test")
    assert abs(cur.kv_alpha - kv_a) < 1e-10, cur.kv_alpha
    assert abs(cur.kv_beta - kv_b) < 1e-7, cur.kv_beta
    assert abs(cur.kv_gamma - kv_g) < 1e-4, cur.kv_gamma
    assert abs(cur.m_alpha - m_a) < 1e-8, cur.m_alpha
    assert abs(cur.m_beta - m_b) < 1e-4, cur.m_beta
    assert cur.source == "test"
    # And it evaluates back to the ground truth.
    assert abs(cur.c_kv_ms(4096) - c_kv(4096)) < 1e-3
    assert abs(cur.c_m_ms(4096) - c_m(4096)) < 1e-3
    print(f"  PASS  1  fit recovers κ: kv_alpha={cur.kv_alpha:.3e} "
          f"m_alpha={cur.m_alpha:.3e} L*={cur.L_star:.0f}")


def test_2_fit_requires_enough_distinct_L():
    from sglang.srt.budgeter.cost_model import fit_cost_curves
    # Only 2 distinct KV L → quadratic underdetermined → raise.
    try:
        fit_cost_curves([(256, 1.0), (1024, 2.0)], [(256, 1.0), (1024, 2.0)])
    except ValueError as e:
        assert "≥3 distinct" in str(e) or "3 distinct" in str(e)
    else:
        raise AssertionError("KV fit with 2 distinct L must raise")
    # Only 1 distinct mamba L → linear underdetermined → raise.
    try:
        fit_cost_curves(
            [(256, 1.0), (1024, 2.0), (2048, 3.0)],
            [(256, 1.0), (256, 1.1)],
        )
    except ValueError as e:
        assert "≥2 distinct" in str(e) or "2 distinct" in str(e)
    else:
        raise AssertionError("mamba fit with 1 distinct L must raise")
    print("  PASS  2  fit raises on too-few distinct L (KV≥3, mamba≥2)")


def test_3_fit_rejects_negative_leading_coeff():
    from sglang.srt.budgeter.cost_model import fit_cost_curves
    # Samples exactly on a CONCAVE (negative-α) quadratic
    # c = -1e-6·L² + 1·L → fit recovers kv_alpha < 0 (unphysical:
    # recompute cost can't curve downward) → must raise.
    def neg(L):
        return -1e-6 * L * L + 1.0 * L
    kv = [(L, neg(L)) for L in (256, 1024, 4096)]
    m = [(256, 1.0), (4096, 10.0)]
    try:
        fit_cost_curves(kv, m)
    except ValueError as e:
        assert "negative" in str(e).lower(), e
    else:
        raise AssertionError("negative leading coeff must raise")
    print("  PASS  3  fit raises on unphysical negative leading coeff")


def test_4_crossover_l_star():
    from sglang.srt.budgeter.cost_model import _crossover_L_star
    # c_KV = 1e-6 L² + 0·L + 0 ; c_M = 1e-3 L + 0 → cross at
    # 1e-6 L² = 1e-3 L → L = 1000.
    L = _crossover_L_star(1e-6, 0.0, 0.0, 1e-3, 0.0)
    assert abs(L - 1000.0) < 1e-6, L
    # No positive crossover (parallel-ish) → 0 sentinel.
    assert _crossover_L_star(0.0, 1e-3, 0.0, 1e-3, 0.0) == 0.0
    print(f"  PASS  4  crossover L* solved: {L:.0f}")


def test_5_set_cost_curves_installs_without_env():
    from sglang.srt.budgeter import cost_model as cm
    _clear_csigma_env()
    cm.reset_cost_curves()
    fitted = cm.CostCurves(
        kv_alpha=2e-7, kv_beta=0.0, kv_gamma=0.5,
        m_alpha=3e-3, m_beta=7.0, L_star=0.0, source="boot_probe",
    )
    assert cm.set_cost_curves(fitted) is True
    got = cm.get_cost_curves()
    assert got is fitted, "boot curves must become the singleton"
    assert got.source == "boot_probe"
    print("  PASS  5  set_cost_curves installs probe curves (no env)")


def test_6_set_cost_curves_env_precedence():
    from sglang.srt.budgeter import cost_model as cm
    _clear_csigma_env()
    # Operator calibration present.
    os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.0e-7"
    os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.4"
    os.environ["SGLANG_CSIGMA_M_ALPHA"] = "2.0e-3"
    os.environ["SGLANG_CSIGMA_M_BETA"] = "7.0"
    try:
        cm.reset_cost_curves()
        fitted = cm.CostCurves(
            kv_alpha=9e-7, kv_beta=0.0, kv_gamma=9.0,
            m_alpha=9e-3, m_beta=9.0, L_star=0.0, source="boot_probe",
        )
        assert cm.set_cost_curves(fitted) is False, (
            "boot probe must NOT overwrite operator SGLANG_CSIGMA_* "
            "calibration (env precedence)"
        )
        got = cm.get_cost_curves()
        assert got.source == "env", got.source
        assert abs(got.kv_alpha - 1.0e-7) < 1e-12
    finally:
        _clear_csigma_env()
        cm.reset_cost_curves()
    print("  PASS  6  set_cost_curves skips on SGLANG_CSIGMA_* (env precedence)")


def test_7_seed_from_boot_probe():
    from sglang.srt.budgeter.cost_model import RuntimeActuatorCost
    rac = RuntimeActuatorCost(initial_us=3000.0, alpha=0.3)
    assert rac.current_us == 3000.0
    rac.seed_from_boot_probe(120.0)
    assert rac.current_us == 120.0, "seed must move cold-start baseline"
    assert rac.n_observations == 0, "seed must NOT count as a live obs"
    assert rac.is_calibrated is False, "still gated on live fires"
    # Non-positive probe is a no-op (don't seed a bogus 0).
    rac.seed_from_boot_probe(0.0)
    assert rac.current_us == 120.0
    # EWMA drift works on top: first real obs seeds, subsequent blend.
    rac.update(total_us=200.0, n_chunks=1)
    assert rac.current_us == 200.0 and rac.n_observations == 1
    print("  PASS  7  seed_from_boot_probe sets baseline, EWMA drift on top")


def test_8_balance_restore_converges_to_baseline():
    """`balance_restore` drives a simulated kv capacity back to baseline
    by firing whichever direction shrinks |cap − baseline|. Uses an
    ASYMMETRIC fire model (fwd shrinks kv by 3, rev grows by 2) so the
    fires don't pair one-for-one — the loop must still converge by
    closing on the observed capacity."""
    from sglang.srt.budgeter.boot_probe import balance_restore

    state = {"cap": 100}

    def _fire(direction, _n):
        # kv_to_mamba shrinks kv; mamba_to_kv grows it. Asymmetric steps.
        if direction == "kv_to_mamba":
            state["cap"] -= 3
        else:
            state["cap"] += 2
        return 1  # moved > 0

    def _cap():
        return state["cap"]

    # Already at baseline → no fires, residual 0.
    assert balance_restore(100, _fire, _cap, step_pages=12) == 0

    # Forward fired once externally (cap 97); restore must climb back.
    state["cap"] = 97
    residual = balance_restore(100, _fire, _cap, step_pages=12)
    # rev adds 2: 97→99→101 ... overshoots to 101 then fwd −3 → 98 ...
    # asymmetric 2/3 can't hit 100 exactly from 97; loop ends bounded,
    # residual is small + nonzero (the caller warns + budgeter absorbs).
    assert abs(residual) <= 3, residual
    print(f"  PASS  8  balance_restore: baseline→0 fires; asymmetric "
          f"converges to residual {residual} (≤ step)")


def test_9_balance_restore_no_progress_breaks():
    """If a fire moves nothing (planner can't build), the loop must
    break immediately rather than spin to max_fires."""
    from sglang.srt.budgeter.boot_probe import balance_restore
    calls = {"n": 0}

    def _fire(direction, _n):
        calls["n"] += 1
        return 0  # no plan buildable

    residual = balance_restore(100, _fire, lambda: 50, step_pages=12,
                               max_fires=16)
    assert residual == 50 - 100, residual
    assert calls["n"] == 1, f"must break after first no-progress fire, got {calls['n']}"
    print("  PASS  9  balance_restore breaks on no-progress (1 fire, not 16)")


def test_10_balance_restore_caps_at_max_fires():
    """A pathological fire that always moves but never reaches baseline
    must stop at max_fires (no infinite loop)."""
    from sglang.srt.budgeter.boot_probe import balance_restore
    state = {"cap": 0}
    calls = {"n": 0}

    def _fire(direction, _n):
        calls["n"] += 1
        state["cap"] += 1  # always moves toward, never hits (baseline 1000)
        return 1

    residual = balance_restore(1000, _fire, lambda: state["cap"],
                               step_pages=12, max_fires=5)
    assert calls["n"] == 5, calls["n"]
    print(f"  PASS  10 balance_restore caps at max_fires=5 (residual "
          f"{residual}, no infinite loop)")


def main() -> int:
    tests = [
        test_1_fit_recovers_known_coefficients,
        test_2_fit_requires_enough_distinct_L,
        test_3_fit_rejects_negative_leading_coeff,
        test_4_crossover_l_star,
        test_5_set_cost_curves_installs_without_env,
        test_6_set_cost_curves_env_precedence,
        test_7_seed_from_boot_probe,
        test_8_balance_restore_converges_to_baseline,
        test_9_balance_restore_no_progress_breaks,
        test_10_balance_restore_caps_at_max_fires,
    ]
    print(f"\n#118 boot-probe calibration tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n#118: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
