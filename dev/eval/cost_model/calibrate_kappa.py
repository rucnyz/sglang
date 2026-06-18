"""Offline κ_i calibration (#118): fit the recompute-cost curve
`c_i(L)` coefficients from measured prefill-forward wall vs sequence
length L, and emit `SGLANG_CSIGMA_*` export lines.

WHY OFFLINE (not an in-engine boot probe): κ_i is a pure
kernel-launch + FLOPs property of (model, hardware) — it does NOT
drift across boots/deployments the way c^xfer (driver page-table, HBM
thermal) and c_m (side-stream contention) do, so it only needs
calibrating once per (model, GPU). An in-engine boot probe would also
have to split the per-stack `c_KV` (quadratic) / `c_M` (linear) wall
out of a single hybrid forward — which requires model-specific
module-level profiling (fragile, not general). This offline tool sits
outside the engine and stays model-agnostic.

WHAT IT CONSUMES: the stdout of `sglang.bench_one_batch`, which times
the PURE prefill GPU forward (CUDA-synchronize bracketed, no HTTP /
tokenizer / scheduler / decode in the measured window). We do NOT
subtract any "fixed-overhead floor": bench already excludes the
non-compute pipeline, and the residual per-forward cost (kernel-launch
× layers, ~16 ms, visible as the L≤256 plateau) is real recompute cost
that belongs in `kv_gamma` — every re-prefill of an evicted prefix pays
it. Sweeping L from tiny into the multi-chunk regime is ESSENTIAL
because chunked prefill flattens attention's L² into per-chunk
increments, so the quadratic only emerges past the chunk size
(empirically Qwen3.5-9B: ~linear ≤8k, L²-term ≈30% by 98k; kv_alpha
≈1.1e-7 agrees across bench / HTTP / builtin to 3 sources).

ALIGNMENT (the subtle part): `bench_one_batch` prints one COMPILE
WARMUP line (the first input-len, re-run) followed by the steady-state
lines, so row-order does NOT map 1:1 to the input-len list. We recover
each row's true L from `L = round(latency · throughput)` (throughput is
`tok/s` on the same line) instead of positional indexing — this is
immune to the warmup row and to any sweep that OOM-truncates early. Per
L we take the MEDIAN across runs (drops the compile-inflated first
sample for the warmup length too).

HYBRID CAVEAT: on a hybrid model a single prefill mixes the attention
stack (L²) and the mamba stack (L), and the total wall can't be
uniquely split into `c_KV` and `c_M`. We fit the TOTAL as the KV curve
(kv_α/β/γ) and set m_α=m_β=0. This is functionally exact for the
cross-pool system: every radix-cache entry binds KV+mamba, so evicting
a leaf always recomputes the whole prefix and `c^evict` uses
`c_kv + c_m = total + 0 = total`. (A KV-only model gives m=0 anyway.)

USAGE: `calibrate.sh` runs the bench sweep (N repeats) and pipes the
logs here. Or directly: `python calibrate_kappa.py <bench_log...>
[--model M] [--device D]`. The `export SGLANG_CSIGMA_*` lines go to
STDOUT (eval-able); diagnostics + plot path go to STDERR.
"""
import json
import math
import os
import re
import sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# bench_one_batch prints e.g.
#   "Prefill. latency: 0.20897 s, throughput:  39202.30 token/s"
_PREFILL_RE = re.compile(
    r"Prefill\. latency:\s*([0-9.]+)\s*s,\s*throughput:\s*([0-9.]+)"
)

# A length is in the "floor plateau" (compute negligible vs the
# per-forward kernel-launch cost) when small; report it separately as a
# sanity anchor for kv_gamma but still fit it (it IS recompute cost).
FLOOR_L_MAX = 256


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def parse_bench_logs(paths):
    """Map L -> list of prefill-wall-ms samples, recovering L per row
    from `round(latency · throughput)` so the compile-warmup row and any
    OOM-truncated sweep can't misalign the data.

    bench prints throughput as `input_len / latency`, so
    `latency · throughput == input_len` and `round(...)` recovers the
    true L exactly — no aliasing in practice (the swept lengths are far
    apart and throughput is derived from the same latency). A
    pathological collision would merge two lengths into one bucket; the
    downstream `<3 distinct L` guard in `main` catches the degenerate
    case loudly."""
    by_L = {}
    for path in paths:
        with open(path) as f:
            text = f.read()
        for lat_s, thrpt in _PREFILL_RE.findall(text):
            lat = float(lat_s)
            L = int(round(lat * float(thrpt)))
            by_L.setdefault(L, []).append(lat * 1000.0)  # -> ms
    return by_L


def collapse(by_L):
    """Per L: median across samples. The first swept length also carries
    the compile-warmup sample (bench re-runs input-len[0] once per run,
    inflated): across N repeats its bucket holds N warmup + N steady.
    Sorted, the N steady samples occupy the lower half, so the median
    lands on steady regardless of N; dropping the single largest is
    belt-and-suspenders for the N=1 case (1 warmup + 1 steady → steady).
    """
    Ls = sorted(by_L)
    ms = []
    for L in Ls:
        s = sorted(by_L[L])
        steady = s[:-1] if len(s) > 1 else s
        ms.append(_median(steady))
    return Ls, ms


def fit(Ls, ms):
    """Fit c_KV(L) = kv_alpha·L² + kv_beta·L + kv_gamma (ms). kv_gamma
    captures the per-forward kernel-launch floor (the L≤256 plateau),
    which is real recompute cost, so it is NOT forced through zero."""
    import numpy as np
    L = np.array(Ls, dtype=float)
    y = np.array(ms, dtype=float)
    kv_alpha, kv_beta, kv_gamma = (float(x) for x in np.polyfit(L, y, 2))
    pred = kv_alpha * L * L + kv_beta * L + kv_gamma
    quad_rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    b1, c1 = np.polyfit(L, y, 1)
    lin_rms = float(np.sqrt(np.mean((y - (b1 * L + c1)) ** 2)))
    c3 = np.polyfit(L, y, 3)
    cub_rms = float(np.sqrt(np.mean((y - np.polyval(c3, L)) ** 2)))
    return kv_alpha, kv_beta, kv_gamma, quad_rms, lin_rms, cub_rms


def plot(Ls, ms, kv_alpha, kv_beta, kv_gamma, quad_rms, lin_rms,
         floor_ms, model, device):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        L = np.array(Ls, dtype=float)
        b1, c1 = np.polyfit(L, np.array(ms), 1)
        xs = np.linspace(0, L.max(), 500)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, zoom in ((a1, False), (a2, True)):
            ax.scatter(L, ms, c="tab:blue", s=42, zorder=3,
                       label="measured (pure forward)")
            ax.plot(xs, b1 * xs + c1, "g--", lw=1,
                    label=f"linear (RMS {lin_rms:.1f}ms)")
            ax.plot(xs, kv_alpha * xs * xs + kv_beta * xs + kv_gamma, "r--",
                    lw=1.5, label=(f"quad {kv_alpha:.2e}L^2"
                                   f"+{kv_beta:.4f}L+{kv_gamma:.0f} "
                                   f"(RMS {quad_rms:.1f}ms)"))
            if floor_ms is not None:
                ax.axhline(floor_ms, color="gray", ls=":", lw=1,
                           label=f"per-forward floor {floor_ms:.0f}ms")
            ax.set_xlabel("prefill length L")
            ax.set_ylabel("prefill forward wall (ms)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_title("zoom (floor + chunk onset)" if zoom
                         else f"{model} / {device} (throughput-aligned)")
        a2.set_xlim(0, min(2700, L.max()))
        a2.set_ylim(0, 75)
        plt.tight_layout()
        path = f"{OUT_DIR}/kappa_fit.png"
        plt.savefig(path, dpi=120)
        _log(f"saved {path}")
    except Exception as e:  # plotting is diagnostic-only
        _log(f"plot skipped ({type(e).__name__}: {e})")


def main(argv):
    paths, model, device = [], "unknown-model", "unknown-device"
    i = 0
    bad_flag = None
    while i < len(argv):
        if argv[i] in ("--model", "--device"):
            # A trailing flag with no value must not IndexError with a bare
            # traceback — fall through to the usage message instead.
            if i + 1 >= len(argv):
                bad_flag = argv[i]
                break
            if argv[i] == "--model":
                model = argv[i + 1]
            else:
                device = argv[i + 1]
            i += 2
        else:
            paths.append(argv[i]); i += 1
    if bad_flag is not None or not paths:
        if bad_flag is not None:
            _log(f"error: {bad_flag} requires a value")
        _log("usage: calibrate_kappa.py <bench_log...> [--model M] "
             "[--device D]")
        return 1

    by_L = parse_bench_logs(paths)
    Ls, ms = collapse(by_L)
    if len(Ls) < 3:
        _log(f"ERROR: only {len(Ls)} distinct lengths parsed — need ≥3 "
             "for a quadratic fit. Aborting.")
        return 1

    floor = [m for L, m in zip(Ls, ms) if L <= FLOOR_L_MAX]
    floor_ms = _median(floor) if floor else None
    for L, m in zip(Ls, ms):
        _log(f"L={L:7d}  prefill_wall={m:9.2f} ms  (n={len(by_L[L])})")
    if floor_ms is not None:
        _log(f"per-forward floor (L≤{FLOOR_L_MAX} plateau) = {floor_ms:.1f} ms")

    kv_alpha, kv_beta, kv_gamma, quad_rms, lin_rms, cub_rms = fit(Ls, ms)
    _log(f"\nfit: c_KV(L) = {kv_alpha:.4e}·L² + {kv_beta:.5f}·L + "
         f"{kv_gamma:.3f}   (ms)")
    _log(f"  quad RMS={quad_rms:.2f}ms | linear RMS={lin_rms:.2f}ms | "
         f"cubic RMS={cub_rms:.2f}ms")
    if lin_rms < quad_rms * 1.15:
        _log("  NOTE: linear ≈ quadratic on this range — L too small to "
             "expose the L² term; extend the sweep into the multi-chunk "
             "regime before trusting kv_alpha.")
    if cub_rms < quad_rms * 0.5:
        _log("  WARNING: cubic halves the RMS — quadratic may be "
             "under-fitting; inspect kappa_fit.png residuals.")

    # Refuse to emit a poisoned curve: a thin / noisy sweep can fit a
    # negative or non-finite leading coeff (a curve that goes negative or
    # NaN at large L). CostCurves.__post_init__ rejects it at load time
    # → builtin fallback; catch it HERE so the operator sees it at
    # calibration time instead of shipping a profile the engine ignores.
    if not all(math.isfinite(c) for c in (kv_alpha, kv_beta, kv_gamma)):
        _log(f"ERROR: non-finite κ_i fit (kv_alpha={kv_alpha}, "
             f"kv_beta={kv_beta}, kv_gamma={kv_gamma}) — refusing to emit. "
             "Re-run with more / cleaner samples.")
        return 1
    if kv_alpha < 0:
        _log(f"ERROR: negative kv_alpha={kv_alpha:.3e} — the recompute curve "
             "would go negative at large L (unphysical; too few/noisy "
             "samples). Refusing to emit; extend the sweep.")
        return 1

    record = {
        "model": model, "device": device, "floor_ms": floor_ms,
        "fit": {
            "c_kv": {"alpha_ms_per_tok2": kv_alpha,
                     "beta_ms_per_tok": kv_beta, "gamma_ms": kv_gamma},
            "c_m": {"alpha_ms_per_tok": 0.0, "beta_ms": 0.0},
            "crossover_L_star": 0.0,
            "quad_rms_ms": quad_rms, "linear_rms_ms": lin_rms,
            "cubic_rms_ms": cub_rms,
        },
        "L": Ls, "prefill_ms": ms,
    }
    json.dump(record, open(f"{OUT_DIR}/kappa_fit.json", "w"), indent=2)
    _log(f"saved {OUT_DIR}/kappa_fit.json")
    plot(Ls, ms, kv_alpha, kv_beta, kv_gamma, quad_rms, lin_rms,
         floor_ms, model, device)

    # Eval-able export lines → STDOUT (everything else went to stderr).
    print(f"export SGLANG_CSIGMA_KV_ALPHA={kv_alpha:.6e}")
    print(f"export SGLANG_CSIGMA_KV_BETA={kv_beta:.6e}")
    print(f"export SGLANG_CSIGMA_KV_GAMMA={kv_gamma:.6e}")
    print("export SGLANG_CSIGMA_M_ALPHA=0.0")
    print("export SGLANG_CSIGMA_M_BETA=0.0")
    print("export SGLANG_CSIGMA_LSTAR=0.0")
    print(f"export SGLANG_CSIGMA_MODEL={model}")
    print(f"export SGLANG_CSIGMA_DEVICE={device}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
