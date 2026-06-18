#!/usr/bin/env python3
"""Extract pool waste over time from an sglang baseline server log.

Waste = the idle fraction of the pool that is NOT binding, while the other pool
is saturated. On the static boot split this is the memory the cross-pool layer
could reclaim. The signal is the per-batch usage sglang already logs, so no
instrumentation is needed:

  ... full token usage: <kv>, mamba usage: <mamba>, #running-req: <r>, #queue-req: <q>

Outputs (into --out):
  waste.csv      one row per batch tick: t_s, kv, mamba, running, queue
  waste.png      KV and mamba usage over wall-clock time, with queue depth
  summary.json   headline numbers (see below)

Headline numbers:
  kv_bound_frac      fraction of ticks KV is saturated (>= --bound)
  mamba_idle_at_kv   mean idle mamba (1 - mamba usage) during KV-bound ticks
                     = the borrowable mamba while KV is the wall (case 1)
  mamba_bound_frac   fraction of ticks mamba is saturated
  kv_idle_at_mamba   mean idle KV during mamba-bound ticks (case 2)
  queue_*            mean / p99 queue depth (the cost the waste imposes)

Note: mamba usage here is total occupancy (live + cached). Cached snapshots are
reclaimable, so 1 - usage is a LOWER bound on the truly borrowable mamba; the
live-only figure (design.md "Where we win") is larger.
"""
import argparse
import csv
import json
import os
import re
from datetime import datetime

LINE = re.compile(
    r"\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*?"
    r"full token usage: (?P<kv>[0-9.]+), mamba usage: (?P<mamba>[0-9.]+), "
    r"#running-req: (?P<run>\d+), #queue-req: (?P<q>\d+)"
)


MAMBA_POOL = re.compile(
    r"max_mamba_cache_size: (?P<m>\d+), conv_state size: (?P<conv>[\d.]+)GB, "
    r"ssm_state size: (?P<ssm>[\d.]+)GB"
)
KV_POOL = re.compile(r"K size: (?P<k>[\d.]+) GB, V size: (?P<v>[\d.]+) GB")


def parse_log(path):
    """Parse per-batch usage + boot-time pool sizes. mamba LIVE (active recurrent
    states) is approximated as #running-req / mamba_pool: each running request
    holds exactly one recurrent slot, so running-req is the live mamba count, and
    (mamba usage - live) is the reclaimable cached-snapshot fraction. Pool GB come
    from the boot allocation lines and turn an idle fraction into wasted bytes."""
    mamba_pool = None
    mamba_gb = kv_gb = None
    rows = []
    t0 = None
    with open(path) as f:
        for line in f:
            if mamba_pool is None:
                pm = MAMBA_POOL.search(line)
                if pm:
                    mamba_pool = int(pm["m"])
                    mamba_gb = float(pm["conv"]) + float(pm["ssm"])
            if kv_gb is None:
                pk = KV_POOL.search(line)
                if pk:
                    kv_gb = float(pk["k"]) + float(pk["v"])
            m = LINE.search(line)
            if not m:
                continue
            ts = datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S")
            if t0 is None:
                t0 = ts
            run = int(m["run"])
            rows.append({
                "t_s": (ts - t0).total_seconds(),
                "kv": float(m["kv"]),
                "mamba": float(m["mamba"]),
                "mamba_live": round(min(1.0, run / mamba_pool), 4) if mamba_pool else None,
                "running": run,
            })
    return rows, {"mamba_slots": mamba_pool, "mamba_gb": mamba_gb, "kv_gb": kv_gb}


def summarize(rows, pools):
    """Waste = the idle (borrowable) capacity of the non-bottleneck pool, in both
    occupancy fraction and absolute GB. Occupancy is the PEAK (max over active
    ticks): the highest each pool ever reaches, so the bottleneck pool reads near
    its true ceiling and the idle pool's peak bounds how little it ever needs. The
    wasted pool is the one with the lower peak; its wasted GB = pool_gb *
    (1 - peak), the capacity idle even at its busiest, which the cross-pool layer
    could lend. GB separates static splits: a small idle pool wastes little, an
    over-provisioned idle pool wastes a lot."""
    act = [r for r in rows if r["running"] > 0] or rows
    kv_occ = max(r["kv"] for r in act)
    mamba_occ = max(r["mamba"] for r in act)
    live = [r["mamba_live"] for r in act if r["mamba_live"] is not None]
    mamba_live_occ = max(live) if live else None
    wasted = "mamba" if mamba_occ <= kv_occ else "kv"
    kv_gb, mamba_gb = pools.get("kv_gb"), pools.get("mamba_gb")
    wasted_gb = None
    if wasted == "mamba" and mamba_gb is not None:
        wasted_gb = round(mamba_gb * (1 - mamba_occ), 2)
    elif wasted == "kv" and kv_gb is not None:
        wasted_gb = round(kv_gb * (1 - kv_occ), 2)
    return {
        "n_active_ticks": len(act),
        "kv_occ_peak": round(kv_occ, 3),
        "mamba_occ_peak": round(mamba_occ, 3),
        "mamba_live_occ_peak": round(mamba_live_occ, 3) if mamba_live_occ is not None else None,
        "wasted_pool": wasted,
        "wasted_gb": wasted_gb,
        "mamba_pool_slots": pools.get("mamba_slots"),
        "mamba_pool_gb": mamba_gb,
        "kv_pool_gb": kv_gb,
    }


def plot(rows, out_png, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t_s"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, [r["kv"] for r in rows], label="KV usage", color="C0")
    ax.plot(t, [r["mamba"] for r in rows], label="mamba usage (total)", color="C1")
    if any(r["mamba_live"] is not None for r in rows):
        ax.plot(t, [r["mamba_live"] for r in rows], label="mamba usage (live)",
                color="C2", linestyle="--", linewidth=0.9)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("pool usage")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    ax.set_title(f"pool occupancy over time: {label}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    rows, pools = parse_log(args.log)
    if not rows:
        raise SystemExit(f"no batch-usage lines parsed from {args.log}")
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "waste.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["t_s", "kv", "mamba", "mamba_live", "running"]
        )
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows, pools)
    summary["label"] = args.label
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot(rows, os.path.join(args.out, "waste.png"), args.label or args.log)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
