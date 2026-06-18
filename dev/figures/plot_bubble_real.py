"""
Real baseline-data version of the long-horizon vs swarm "bubble" figure.
Reads the two CSVs produced by bench_two_workloads_baseline.sh and plots
the time series of paged-KV + recurrent-slot pool utilization side-by-
side, shading the bubble between the two lines on each panel.

Inputs:
  dev/figures/data/baseline_longhorizon.csv (t,token_usage,mamba_usage,num_running)
  dev/figures/data/baseline_swarm.csv       (t,token_usage,mamba_usage,num_running)

Output:
  dev/figures/bubble_two_workloads_real.{png,pdf}
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: str):
    """Legacy: Prometheus-sampled CSV. Note that sglang:full_token_usage
    and sglang:mamba_usage SUBTRACT cached/evictable size from the
    numerator, so these values understate the real pool fill (active +
    cache combined). Prefer the budgeter JSONL when available."""
    if not os.path.exists(path):
        return None
    t, kv, m, nr = [], [], [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                t.append(float(row["t"]))
                kv.append(float(row["token_usage"]))
                m.append(float(row["mamba_usage"]))
                nr.append(int(float(row["num_running"])))
            except (ValueError, KeyError):
                continue
    if not t:
        return None
    return np.array(t), np.array(kv), np.array(m), np.array(nr)


def _read_budgeter_jsonl(path: str):
    """Per-tick snapshot from the budgeter agent. usage_kv_inst /
    usage_mamba_inst here are computed as (pool.size - available) /
    pool.size and DO include radix-cached prefix/snapshots — that's the
    real "pool fill" we want for the bubble figure. Returns None if the
    file is missing or empty."""
    import json
    if not os.path.exists(path):
        return None
    t, kv, m, nr = [], [], [], []
    t0 = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ts = r.get("ts")
        # Prefer pool_occupancy_* (computed unconditionally in _snapshot,
        # includes radix-cached prefix/snapshots — the real "occupied
        # fraction" for the bubble figure). Fall back to xpool_plan_usage_*_inst
        # (planner-side, only present when arena is enabled). Older runs
        # used pool_fill_* — kept as a third fallback.
        kv_v = r.get("pool_occupancy_kv")
        if kv_v is None:
            kv_v = r.get("pool_fill_kv")
        if kv_v is None:
            kv_v = r.get("xpool_plan_usage_kv_inst")
        m_v = r.get("pool_occupancy_mamba")
        if m_v is None:
            m_v = r.get("pool_fill_mamba")
        if m_v is None:
            m_v = r.get("xpool_plan_usage_mamba_inst")
        nr_v = r.get("num_running_reqs", 0)
        if hasattr(nr_v, "total"):
            nr_v = nr_v.total
        if ts is None or kv_v is None or m_v is None:
            continue
        if t0 is None:
            t0 = ts
        try:
            t.append(float(ts) - t0)
            kv.append(float(kv_v))
            m.append(float(m_v))
            nr.append(int(nr_v) if isinstance(nr_v, (int, float)) else 0)
        except (TypeError, ValueError):
            continue
    if not t:
        return None
    return np.array(t), np.array(kv), np.array(m), np.array(nr)


def _trim_idle(t, kv, m, nr, head_idle_secs=2.0):
    """Drop leading samples where there's no traffic so the X axis starts
    at the moment the bench began producing load."""
    busy = np.where(nr > 0)[0]
    if busy.size == 0:
        return t, kv, m, nr
    start = max(0, busy[0] - 1)
    t = t[start:] - t[start]
    return t, kv[start:], m[start:], nr[start:]


def _shade_bubble(ax, t, top, bottom, label=None):
    ax.fill_between(t, top, bottom, where=(top >= bottom),
                    color="#dddddd", alpha=0.55, zorder=0,
                    interpolate=True)
    # Label argument retained for API compat but no longer rendered;
    # the shaded region speaks for itself and earlier text annotations
    # were overlapping the legend / data lines.


def _plot_panel(ax, csv_path, title, kv_high_side, jsonl_path=None):
    """kv_high_side: True = paged-KV is the upper line on this panel
    (long-horizon); False = recurrent is upper (swarm).
    Prefers jsonl_path (budgeter snapshot, includes cached prefix in
    usage) over csv_path (Prometheus, active-only)."""
    data = None
    if jsonl_path:
        data = _read_budgeter_jsonl(jsonl_path)
    if data is None:
        data = _read_csv(csv_path)
    if data is None:
        ax.text(0.5, 0.5, f"missing\n{csv_path}", transform=ax.transAxes,
                ha="center", va="center", color="#aa0000")
        return
    t, kv, m, nr = _trim_idle(*data)
    if kv_high_side:
        _shade_bubble(ax, t, kv, m, "bubble")
    else:
        _shade_bubble(ax, t, m, kv, "bubble")
    ax.plot(t, kv, color="#222222", lw=2.0, label="paged-KV", zorder=3)
    ax.plot(t, m, color="#666666", lw=1.6, ls="--",
            label="recurrent slots", zorder=3)
    ax.axhline(1.0, color="#aaaaaa", lw=0.7, ls=":", zorder=1)
    ax.text(t[-1] * 0.985 if len(t) else 0.985, 1.005, "capacity",
            ha="right", va="bottom", fontsize=9, color="#888888")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("bench time (s)", fontsize=11)
    ax.set_ylim(0, 1.07)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dev/figures/data")
    ap.add_argument("--out-dir", default="dev/figures")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)

    _plot_panel(axes[0],
                os.path.join(args.data_dir, "baseline_longhorizon.csv"),
                "Long-horizon agent",
                kv_high_side=True,
                jsonl_path=os.path.join(args.data_dir,
                                        "baseline_longhorizon_budgeter.jsonl"))
    axes[0].set_ylabel("pool utilization", fontsize=11)
    axes[0].legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)

    _plot_panel(axes[1],
                os.path.join(args.data_dir, "baseline_swarm.csv"),
                "Agent swarm",
                kv_high_side=False,
                jsonl_path=os.path.join(args.data_dir,
                                        "baseline_swarm_budgeter.jsonl"))
    axes[1].legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)

    fig.tight_layout()

    out_png = os.path.join(args.out_dir, "bubble_two_workloads_real.png")
    out_pdf = os.path.join(args.out_dir, "bubble_two_workloads_real.pdf")
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
