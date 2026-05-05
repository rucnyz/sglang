"""
Fit c_σ(L) parametric curves from the microbench data.

Hypotheses:
  c_KV(L) = a_KV * L + b_KV   (FA prefill: O(L^2) total → per-token grows linearly)
            wait — total scales L^2, so the *total stack* time is α*L^2 + β*L.
  c_M(L)  = α_M * L + β_M     (GDN: O(L) total + per-chunk fixed overhead)

We fit on STACK-LEVEL totals (10 attn + 30 linear layers, ms) since that's
what the budgeter consumes. Output: (α, β) per pool, plus the crossover
length L* where c_M(L*) = c_KV(L*).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-json",
        default="dev/eval/cost_model/recovery_cost_NVIDIA_H200.json",
    )
    ap.add_argument("--out-dir", default="dev/eval/cost_model")
    args = ap.parse_args()

    with open(args.in_json) as f:
        data = json.load(f)
    rows = data["rows"]
    L = np.array([r["L"] for r in rows], dtype=float)
    c_kv = np.array([r["tot_kv_ms"] for r in rows])
    c_m = np.array([r["tot_m_ms"] for r in rows])

    # c_KV(L) = α_KV·L² + β_KV·L + γ_KV   (attn: L² compute + linear + per-layer launch)
    A_kv = np.stack([L * L, L, np.ones_like(L)], axis=1)
    coef_kv, *_ = np.linalg.lstsq(A_kv, c_kv, rcond=None)
    alpha_kv2, beta_kv, gamma_kv = coef_kv

    # c_M(L) = α_M · L + β_M  (linear scan + fixed chunk-setup overhead)
    A_m = np.stack([L, np.ones_like(L)], axis=1)
    coef_m, *_ = np.linalg.lstsq(A_m, c_m, rcond=None)
    alpha_m, beta_m = coef_m

    # crossover: α_kv2·L² + β_kv·L + γ_kv = α_m·L + β_m
    # → α_kv2·L² + (β_kv - α_m)·L + (γ_kv - β_m) = 0
    a, b, c = alpha_kv2, beta_kv - alpha_m, gamma_kv - beta_m
    disc = b * b - 4 * a * c
    if disc >= 0 and a > 0:
        L_star = (-b + np.sqrt(disc)) / (2 * a)
    else:
        L_star = float("nan")

    L_grid = np.geomspace(64, 32768, 200)
    c_kv_pred = alpha_kv2 * L_grid * L_grid + beta_kv * L_grid + gamma_kv
    c_m_pred = alpha_m * L_grid + beta_m

    # Print fit
    print("=" * 70)
    print("Stage-0 calibration outputs (fit to recovery_cost microbench)")
    print("=" * 70)
    print(f"c_KV(L) = α_KV·L² + β_KV·L + γ_KV   [stack-level, ms]")
    print(f"  α_KV = {alpha_kv2:.4e} ms / token²   (L² attn compute)")
    print(f"  β_KV = {beta_kv:.4e} ms / token     (linear-in-L overhead)")
    print(f"  γ_KV = {gamma_kv:.4e} ms             (kernel launch / fixed)")
    print(f"c_M(L)  = α_M·L + β_M               [stack-level, ms]")
    print(f"  α_M  = {alpha_m:.4e} ms / token   (asymptotic per-token scan)")
    print(f"  β_M  = {beta_m:.4e} ms             (chunk setup overhead)")
    print()
    print(f"Crossover L* = {L_star:.1f} tokens   (c_M = c_KV)")
    print(f"  At L < L*: c_M dominates (cheaper to evict KV, expensive to evict mamba)")
    print(f"  At L > L*: c_KV dominates (cheaper to evict mamba, expensive to evict KV)")
    print()

    # residuals + ratios at sampled points
    print(f"{'L':>6}  {'c_KV obs':>10}{'c_KV fit':>10}{'c_M obs':>10}{'c_M fit':>10}{'ratio':>8}")
    for r, Lv in zip(rows, L):
        kv_fit = alpha_kv2 * Lv * Lv + beta_kv * Lv + gamma_kv
        m_fit = alpha_m * Lv + beta_m
        ratio = r["tot_m_ms"] / r["tot_kv_ms"]
        print(
            f"{int(Lv):>6}  "
            f"{r['tot_kv_ms']:>10.3f}{kv_fit:>10.3f}{r['tot_m_ms']:>10.3f}{m_fit:>10.3f}{ratio:>7.2f}x"
        )

    # Plot fit overlay
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(L, c_kv, "o", color="C0", ms=8, label=r"$c_{KV}$ measured")
    ax.loglog(L_grid, c_kv_pred, "-", color="C0", alpha=0.6,
              label=rf"$c_{{KV}}(L) = {alpha_kv2:.2e}\,L^2 + {beta_kv:.2e}\,L + {gamma_kv:.2e}$")
    ax.loglog(L, c_m, "s", color="C3", ms=8, label=r"$c_M$ measured")
    ax.loglog(L_grid, c_m_pred, "-", color="C3", alpha=0.6,
              label=rf"$c_M(L) = {alpha_m:.2e}\,L + {beta_m:.2e}$")
    if not np.isnan(L_star):
        ax.axvline(L_star, color="gray", ls="--", lw=0.8)
        ax.text(L_star * 1.05, c_m.max() * 0.5,
                f"$L^*$ = {L_star:.0f}", color="gray", rotation=90, va="center")
    ax.set_xlabel("Recovery length L (tokens)")
    ax.set_ylabel("Stack-level recovery wall-clock (ms)")
    ax.set_title(
        f"Parametric cost model: $c_{{KV}}(L)$ vs $c_M(L)$ on Qwen3.5-35B-A3B / H200 BF16"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "cost_curves_fit.png")
    out_pdf = os.path.join(args.out_dir, "cost_curves_fit.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nwrote {out_png}\nwrote {out_pdf}")

    # Dump fit params for budgeter consumption
    fit_path = os.path.join(args.out_dir, "cost_curve_fit.json")
    with open(fit_path, "w") as f:
        json.dump(
            dict(
                model="Qwen3.5-35B-A3B",
                device=data["device"],
                dtype=data["dtype"],
                c_kv=dict(form="alpha*L**2 + beta*L + gamma",
                          alpha_ms_per_tok2=alpha_kv2,
                          beta_ms_per_tok=beta_kv,
                          gamma_ms=gamma_kv),
                c_m=dict(form="alpha*L + beta",
                         alpha_ms_per_tok=alpha_m,
                         beta_ms=beta_m),
                crossover_L_star=L_star,
            ),
            f,
            indent=2,
        )
    print(f"wrote {fit_path}")


if __name__ == "__main__":
    main()
