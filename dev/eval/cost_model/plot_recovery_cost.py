"""
Plot recovery-cost microbench: per-token cost and stack-level ratio vs L.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-json",
        default="dev/eval/cost_model/recovery_cost_NVIDIA_H200.json",
    )
    parser.add_argument("--out-dir", default="dev/eval/cost_model")
    args = parser.parse_args()

    with open(args.in_json) as f:
        data = json.load(f)
    rows = data["rows"]
    Ls = [r["L"] for r in rows]
    attn_per_tok = [r["attn_us_per_tok"] for r in rows]
    gdn_per_tok = [r["gdn_us_per_tok"] for r in rows]
    tot_kv = [r["tot_kv_ms"] for r in rows]
    tot_m = [r["tot_m_ms"] for r in rows]
    ratio = [r["stack_ratio"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Per-token kernel cost (one-layer, log-log)
    ax = axes[0]
    ax.loglog(Ls, attn_per_tok, "o-", color="C0", label="FA prefill (per attn layer)")
    ax.loglog(Ls, gdn_per_tok, "s-", color="C3", label="GDN prefill (per linear layer)")
    ax.set_xlabel("Recovery length L (tokens)")
    ax.set_ylabel("Per-token kernel cost (µs)")
    ax.set_title("Per-layer per-token recovery cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # 2. Stack-level total recovery (10 attn layers vs 30 linear layers)
    ax = axes[1]
    ax.loglog(Ls, tot_kv, "o-", color="C0", label=r"$c_{KV}$: 10 attn layers")
    ax.loglog(Ls, tot_m, "s-", color="C3", label=r"$c_M$: 30 linear layers")
    ax.set_xlabel("Recovery length L (tokens)")
    ax.set_ylabel("Total recovery wall-clock (ms)")
    ax.set_title("Stack-level recovery cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # 3. Ratio
    ax = axes[2]
    ax.semilogx(Ls, ratio, "D-", color="C2", lw=2)
    ax.axhspan(10, 30, alpha=0.15, color="C1", label="paper claim 10-30x")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Recovery length L (tokens)")
    ax.set_ylabel(r"$c_M / c_{KV}$  (stack-level wall-clock ratio)")
    ax.set_title(r"Cross-pool cost asymmetry")
    ax.grid(True, which="both", alpha=0.3)
    for L, r in zip(Ls, ratio):
        ax.annotate(f"{r:.1f}x", (L, r), textcoords="offset points", xytext=(4, 6))
    ax.legend()
    ax.set_ylim(0.5, max(ratio) * 1.5)

    fig.suptitle(
        f"Qwen3.5-35B-A3B recovery cost on {data['device']} (BF16): "
        f"c_M vs c_KV per-token wall-clock"
    )
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "recovery_cost.png")
    out_pdf = os.path.join(args.out_dir, "recovery_cost.pdf")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
