"""Phase 1 dashboard renderer.

Reads a JSONL produced by sample_metrics.py and renders a 3x2 grid PNG showing
per-pool pressure + throughput/queue + eviction-rate over time.

Usage:
    python dev/1/dashboard.py <sampling.jsonl> [--out <path.png>] [--phase-marks t1=labelA,t2=labelB]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def col(rows, key, default=0.0):
    return [(r.get(key) if r.get(key) is not None else default) for r in rows]


def rate(rows, key, dt_smooth=5.0):
    """Convert cumulative counter to per-second rate, smoothed across ~dt_smooth seconds.

    Treats missing samples (key absent or None) as 0 so derivatives stay finite when
    a counter only starts firing partway through the run."""
    vals = [r.get(key) for r in rows]
    vals = [0.0 if v is None else v for v in vals]
    ts = [r["ts"] for r in rows]
    out = [0.0] * len(rows)
    for i in range(1, len(rows)):
        # Find a sample ~dt_smooth seconds ago for stable rate
        j = i - 1
        while j > 0 and ts[i] - ts[j] < dt_smooth:
            j -= 1
        dt = max(0.001, ts[i] - ts[j])
        out[i] = max(0.0, (vals[i] - vals[j]) / dt)
    return out


def parse_phase_marks(s: str | None, t0: float) -> list[tuple[float, str]]:
    """`--phase-marks 60=B,120=C` -> [(t0+60, 'B'), (t0+120, 'C')]"""
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        sec, _, label = tok.partition("=")
        try:
            out.append((t0 + float(sec), label))
        except ValueError:
            pass
    return out


def draw_phase_marks(ax, marks):
    for t, label in marks:
        ax.axvline(t, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.text(t, ax.get_ylim()[1] * 0.95, label, fontsize=8, ha="left", va="top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--phase-marks", default=None,
                    help="comma-separated 'sec=label' marks relative to t0, e.g. '60=B,120=C'")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rows = load(args.jsonl)
    if not rows:
        sys.exit(f"no data in {args.jsonl}")
    t0 = rows[0]["ts"]
    t = [r["ts"] - t0 for r in rows]
    marks = parse_phase_marks(args.phase_marks, 0)

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    # 1) Paged KV pool: stacked area showing used + evictable as fraction of max
    ax = axes[0, 0]
    kv_max = col(rows, "kv_max")
    used_frac = [
        (rows[i].get("kv_used", 0) / kv_max[i]) if kv_max[i] else 0
        for i in range(len(rows))
    ]
    evict_frac = [
        (rows[i].get("kv_evictable", 0) / kv_max[i]) if kv_max[i] else 0
        for i in range(len(rows))
    ]
    ax.stackplot(t, used_frac, evict_frac, labels=["active KV", "prefix-cached (evictable)"],
                 colors=["tab:blue", "tab:cyan"], alpha=0.7)
    ax.set_ylabel("fraction of paged KV pool")
    ax.set_title("Paged KV pool occupancy")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    draw_phase_marks(ax, marks)

    # 2) DeltaNet/SSM pool
    ax = axes[0, 1]
    ax.plot(t, col(rows, "mamba_usage"), color="tab:red", lw=2, label="mamba_usage")
    ax.set_ylabel("fraction")
    ax.set_title("DeltaNet / SSM state pool")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    draw_phase_marks(ax, marks)

    # 3) LoRA pool (only meaningful when --enable-lora)
    ax = axes[1, 0]
    ax.plot(t, col(rows, "lora_util"), color="tab:purple", lw=2, label="lora_util")
    lora_total = col(rows, "lora_total")
    if any(lora_total):
        ax2 = ax.twinx()
        ax2.plot(t, col(rows, "lora_used"), color="tab:gray", ls="--", lw=1, label="slots_used")
        ax2.set_ylabel("slots used", color="tab:gray")
        ax2.tick_params(axis="y", labelcolor="tab:gray")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax.text(0.5, 0.5, "(LoRA not enabled in this run)", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_ylabel("fraction (utilization)")
    ax.set_title("LoRA adapter cache")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    draw_phase_marks(ax, marks)

    # 4) Prefix cache hit rate
    ax = axes[1, 1]
    ax.plot(t, col(rows, "cache_hit_rate"), color="tab:orange", lw=2, label="cache_hit_rate")
    cached_rate = rate(rows, "cached_tokens_total")
    ax2 = ax.twinx()
    ax2.plot(t, cached_rate, color="tab:gray", ls="--", lw=1, label="cached tokens / s")
    ax.set_ylabel("hit rate")
    ax2.set_ylabel("rate (tok/s)", color="tab:gray")
    ax.set_title("Prefix cache")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    draw_phase_marks(ax, marks)

    # 5) Throughput + queue depth
    ax = axes[2, 0]
    ax.plot(t, col(rows, "gen_throughput"), color="tab:green", lw=2, label="gen tok/s")
    ax2 = ax.twinx()
    ax2.plot(t, col(rows, "num_running"), color="tab:blue", ls="--", lw=1, label="running")
    ax2.plot(t, col(rows, "num_queue"), color="tab:red", ls=":", lw=1, label="queued")
    ax.set_ylabel("gen throughput (tok/s)")
    ax2.set_ylabel("requests")
    ax.set_xlabel("time (s)")
    ax.set_title("Throughput & queue depth")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    draw_phase_marks(ax, marks)

    # 6) Eviction / retraction signals
    ax = axes[2, 1]
    evict_rate = rate(rows, "evicted_tokens_total")
    retr_rate = rate(rows, "num_retracted_total")
    ax.plot(t, evict_rate, color="tab:olive", lw=2, label="evicted tokens / s")
    ax.plot(t, col(rows, "num_retracted"), color="tab:red", lw=1.5, label="retracted (gauge)")
    ax.plot(t, col(rows, "num_paused"), color="tab:orange", ls="--", lw=1, label="paused (gauge)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("count or rate")
    ax.set_title("Eviction / retraction / pause")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    draw_phase_marks(ax, marks)

    fig.suptitle(args.title or f"Phase 1 dashboard — {Path(args.jsonl).name}",
                 fontsize=11, y=1.00)
    plt.tight_layout()

    out = args.out or str(Path(args.jsonl).with_suffix(".png"))
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
