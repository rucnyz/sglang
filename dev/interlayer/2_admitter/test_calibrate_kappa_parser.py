"""#265 follow-up (Phase-2 audit) — unit-test calibrate_kappa.py's
log parser + fit, which previously had ZERO coverage (only the GPU
path in calibrate.sh exercised them).

The parser is the load-bearing fix for an earlier off-by-one: bench
prints a compile-WARMUP line first, so row order does not map 1:1 to
the sweep; L is recovered from `round(latency · throughput)` instead.
These tests feed synthetic bench-log text (warmup row + steady rows +
an OOM-truncated tail) and pin:

1. parse_bench_logs recovers the correct L per row via throughput.
2. collapse drops the compile-warmup sample for the first length and
   the steady median survives (the "is the warmup-drop sound for
   REPEATS>1?" question — verified by construction here).
3. fit recovers KNOWN quadratic coefficients to tolerance.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

_KAPPA = "/scratch/yuzhou/projects/sglang/dev/eval/cost_model/calibrate_kappa.py"


def _load():
    spec = importlib.util.spec_from_file_location("calibrate_kappa", _KAPPA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Ground-truth curve (ms): c(L) = a·L² + b·L + g.
_A, _B, _G = 1.0e-7, 2.0e-2, 5.0
_LS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _c_ms(L):
    return _A * L * L + _B * L + _G


def _row(L, ms):
    lat = ms / 1000.0
    thrpt = L / lat
    return f"Prefill. latency: {lat:.6f} s, throughput: {thrpt:.2f} token/s"


# Lengths that would be FALSELY recovered as `L` if any of the noise lines'
# `latency · throughput` leaked through the parser. None of these may appear
# in `by_L` — the `Prefill\. latency:` anchor must reject them.
_NOISE_L = [777, 888, 999]


def _noise_lines():
    """Lines real `bench_one_batch` interleaves that ALSO contain
    `latency: ... throughput: ...` but are NOT prefill-forward rows.
    The `_PREFILL_RE` `Prefill\\. latency:` anchor must reject every one
    of these — otherwise decode/total walls would corrupt the κ_i fit."""
    return [
        # per-step decode line
        f"Decode 5. latency: {777 / 7e4:.6f} s, throughput: 70000.00 token/s",
        # median-decode summary line
        f"Decode. median latency: {888 / 8e4:.6f} s, throughput: 80000.00 token/s",
        # end-to-end total line
        f"Total. latency: {999 / 9e4:.6f} s, throughput: 90000.00 token/s",
    ]


def _synth_log(warmup_inflate=3.0, truncate_tail=True, noise=True):
    """One bench run: a compile-warmup row at the first length (inflated
    latency) + steady rows, optionally interleaved with the Decode/Total
    noise lines real bench emits. Optionally drop the last length to mimic
    an end-of-sweep OOM truncation."""
    lines = ["Benchmark ..."]
    first = _LS[0]
    lines.append(_row(first, _c_ms(first) * warmup_inflate))  # warmup
    if noise:
        lines += _noise_lines()
    use = _LS[:-1] if truncate_tail else _LS
    for L in use:
        lines.append(_row(L, _c_ms(L)))  # steady
        if noise:
            lines += _noise_lines()
    return "\n".join(lines) + "\n"


def _write(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f.write(text)
    f.close()
    return f.name


def test_1_parse_recovers_L_via_throughput():
    m = _load()
    path = _write(_synth_log(truncate_tail=False))
    try:
        by_L = m.parse_bench_logs([path])
    finally:
        os.remove(path)
    # Every swept length recovered exactly from round(lat·thrpt); the
    # warmup row lands in the first length's bucket (2 samples there).
    assert set(by_L) == set(_LS), f"recovered L set {sorted(by_L)} != {_LS}"
    assert len(by_L[_LS[0]]) == 2, "warmup + steady → 2 samples at first L"
    assert all(len(by_L[L]) == 1 for L in _LS[1:]), "1 sample at other L"
    # Decode/Total noise lines also carry `latency:…throughput:…` but the
    # `Prefill\. latency:` anchor must reject them — their lengths (777/888/
    # 999) must NOT appear, and no extra buckets may sneak in.
    for nL in _NOISE_L:
        assert nL not in by_L, (
            f"noise length {nL} leaked into by_L — Prefill anchor failed; "
            f"recovered {sorted(by_L)}"
        )
    print("  PASS  1  parse recovers L via round(latency·throughput); "
          "warmup lands in first-L bucket; Decode/Total noise rejected")


def test_2_collapse_drops_warmup_keeps_steady_median():
    m = _load()
    # Multi-run (REPEATS=3) so the first length holds 3 warmup + 3 steady.
    text = "".join(_synth_log(truncate_tail=False) for _ in range(3))
    path = _write(text)
    try:
        by_L = m.parse_bench_logs([path])
        Ls, ms = m.collapse(by_L)
    finally:
        os.remove(path)
    first = _LS[0]
    # The steady value is _c_ms(first); the warmup is 3× that. The
    # collapsed value must be the steady one, NOT skewed up by warmups.
    idx = Ls.index(first)
    steady = _c_ms(first)
    assert abs(ms[idx] - steady) < 0.5 * steady, (
        f"first-L collapsed to {ms[idx]:.2f} ms; steady is {steady:.2f} ms "
        f"— warmup samples leaked into the median"
    )
    print(f"  PASS  2  collapse keeps steady median at first L "
          f"({ms[idx]:.2f}≈{steady:.2f} ms) despite 3 warmup samples")


def test_3_fit_recovers_known_quadratic():
    m = _load()
    text = "".join(_synth_log(truncate_tail=False) for _ in range(3))
    path = _write(text)
    try:
        by_L = m.parse_bench_logs([path])
        Ls, ms = m.collapse(by_L)
        kv_alpha, kv_beta, kv_gamma, quad_rms, lin_rms, cub_rms = m.fit(Ls, ms)
    finally:
        os.remove(path)
    assert abs(kv_alpha - _A) < 1e-9, f"kv_alpha {kv_alpha:.3e} != {_A:.3e}"
    assert abs(kv_beta - _B) < 1e-5, f"kv_beta {kv_beta:.5f} != {_B}"
    assert abs(kv_gamma - _G) < 1e-1, f"kv_gamma {kv_gamma:.3f} != {_G}"
    assert quad_rms < 1.0, f"clean synthetic data should fit tight: {quad_rms}"
    print(f"  PASS  3  fit recovers known quad: a={kv_alpha:.3e} "
          f"b={kv_beta:.5f} g={kv_gamma:.3f} (RMS {quad_rms:.3f}ms)")


def test_4_parse_tolerates_oom_truncated_sweep():
    m = _load()
    # Truncated sweep: last length never emitted (engine OOM'd).
    path = _write(_synth_log(truncate_tail=True))
    try:
        by_L = m.parse_bench_logs([path])
    finally:
        os.remove(path)
    assert _LS[-1] not in by_L, "truncated last length must be absent"
    assert set(by_L) == set(_LS[:-1]), "all completed lengths recovered"
    print("  PASS  4  parse tolerates OOM-truncated tail (absent length, "
          "no crash)")


def main():
    tests = [
        test_1_parse_recovers_L_via_throughput,
        test_2_collapse_drops_warmup_keeps_steady_median,
        test_3_fit_recovers_known_quadratic,
        test_4_parse_tolerates_oom_truncated_sweep,
    ]
    print(f"\ncalibrate_kappa.py parser/fit tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"parser: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
