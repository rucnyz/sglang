"""
Illustrative figure: pool-utilization "bubbles" in opposite directions on
two complementary agent workloads. Synthetic data, not eval —the goal is
to make the static-partition argument visually obvious.

Two panels side-by-side:
  Left (long-horizon agent):
      paged-KV  ↑↑ (climbs as system prompt + multi-turn history accumulates)
      recurrent ─→ (stays low; few concurrent sessions, each holds 1 slot)
      Bubble = the wide white band between the two lines, KV-side filling.

  Right (agent swarm / fan-out):
      recurrent ↑↑ (spikes to near-saturation as N sub-agents launch)
      paged-KV  ─→ (stays low; sub-agent prompts are short)
      Bubble = the wide white band, recurrent-side filling.

The two bubbles point in opposite directions — same engine, same static
partition, but the "slack pool" flips. That's why a single deploy-time
ratio can't be right for both.

Output: dev/figures/bubble_two_workloads.{png,pdf}.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Deterministic noise for shape reproducibility.
rng = np.random.default_rng(seed=20260503)


def long_horizon_curves(t: np.ndarray):
    """KV climbs ~10% → ~95% over the run; recurrent stays low ~25-35%."""
    # Saturating exponential climb with mild noise.
    kv = 0.05 + 0.92 * (1.0 - np.exp(-t / 0.42))
    kv += rng.normal(0.0, 0.012, size=t.size)
    kv = np.clip(kv, 0.0, 1.0)

    # Recurrent: fast ramp into the 25-35% band, then quasi-stable.
    rc = 0.30 * (1.0 - np.exp(-t / 0.06)) + 0.05
    # Small breathing modulation (active sessions starting/finishing).
    rc += 0.04 * np.sin(t * 6.0) + rng.normal(0.0, 0.015, size=t.size)
    rc = np.clip(rc, 0.0, 1.0)
    return kv, rc


def swarm_curves(t: np.ndarray):
    """Recurrent spikes to ~95% almost immediately; KV stays in 10-20% band."""
    # Recurrent: sharp ramp to high saturation, then jittery hold.
    rc = 0.95 * (1.0 - np.exp(-t / 0.04)) + 0.02
    rc += 0.025 * np.sin(t * 18.0) + rng.normal(0.0, 0.012, size=t.size)
    rc = np.clip(rc, 0.0, 1.0)

    # KV: low band, slowly drifting because some sub-agents return slightly
    # longer answers.
    kv = 0.12 + 0.05 * (1.0 - np.exp(-t / 0.3)) + 0.04 * np.sin(t * 4.0)
    kv += rng.normal(0.0, 0.012, size=t.size)
    kv = np.clip(kv, 0.0, 1.0)
    return kv, rc


def _shade_bubble(ax, t, top, bottom, label):
    """Fill the bubble region between top (high line) and bottom (low line)
    with a soft hatch + arrow annotation."""
    ax.fill_between(t, top, bottom, color="#dddddd", alpha=0.55, zorder=0)
    # Arrow + label centered on the bubble.
    mid_t = t[int(len(t) * 0.55)]
    mid_y = (top[int(len(t) * 0.55)] + bottom[int(len(t) * 0.55)]) / 2
    ax.annotate(
        label,
        xy=(mid_t, mid_y),
        xytext=(mid_t, mid_y),
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", alpha=0.85),
    )


def main(out_dir: str = "dev/figures"):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)

    # Common time axis — abstract "phase index" 0..1 so it's not eval-dependent.
    t = np.linspace(0.0, 1.0, 600)

    # ---------- Left: long-horizon ----------
    ax = axes[0]
    kv_lh, rc_lh = long_horizon_curves(t)
    # Shade between the high (KV) and low (recurrent) lines as the bubble.
    _shade_bubble(ax, t, kv_lh, rc_lh, "bubble")
    ax.plot(t, kv_lh, color="#222222", lw=2.2, label="paged-KV", zorder=3)
    ax.plot(t, rc_lh, color="#666666", lw=1.8, ls="--", label="recurrent slots", zorder=3)
    # 100% capacity ceiling.
    ax.axhline(1.0, color="#aaaaaa", lw=0.7, ls=":", zorder=1)
    ax.text(0.985, 1.005, "capacity", ha="right", va="bottom", fontsize=9,
            color="#888888")
    ax.set_title("Long-horizon agent", fontsize=13)
    ax.set_xlabel("time", fontsize=11)
    ax.set_ylabel("pool utilization", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.07)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9)

    # ---------- Right: swarm ----------
    ax = axes[1]
    kv_sw, rc_sw = swarm_curves(t)
    # Recurrent is the high line, KV is the low line — bubble flips direction.
    _shade_bubble(ax, t, rc_sw, kv_sw, "bubble")
    ax.plot(t, kv_sw, color="#222222", lw=2.2, label="paged-KV", zorder=3)
    ax.plot(t, rc_sw, color="#666666", lw=1.8, ls="--", label="recurrent slots", zorder=3)
    ax.axhline(1.0, color="#aaaaaa", lw=0.7, ls=":", zorder=1)
    ax.text(0.985, 1.005, "capacity", ha="right", va="bottom", fontsize=9,
            color="#888888")
    ax.set_title("Agent swarm (fan-out)", fontsize=13)
    ax.set_xlabel("time", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.07)
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center right", fontsize=10, frameon=True, framealpha=0.9)

    fig.suptitle(
        "Static HBM partition leaves a bubble — direction flips with workload",
        fontsize=13.5, y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(out_dir, "bubble_two_workloads.png")
    out_pdf = os.path.join(out_dir, "bubble_two_workloads.pdf")
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
