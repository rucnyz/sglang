#!/usr/bin/env python3
"""Aggregate the 4-arm fairness matrix.

Pulls data from THREE separate matrix roots (the previous matrix +
the previous H'_now matrix + the new 4-arm matrix), maps each cycle
to its arm, computes Welch t-tests across pairs.

Arms:
  LRU      — F'_now           — run_LRU_now_*
  TA       — G_now            — run_TA_now_*
  OURS_inline — H'_now matrix — run_H_prime_now_matrix_*_cycle{1,2,3}/
  OURS_full   — matrix "ours" — run_K_full_matrix_*_cycle{2,4,6}_ours/

Usage:
    python parse_4arm.py
    (no args; paths hardcoded for this project)
"""
from __future__ import annotations
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

RESULTS = Path(
    "/scratch/yuzhou/projects/sglang/dev/aginfer/results"
)


def gather_durations(harbor_jobs_dir: Path):
    durations = []
    for run_dir in harbor_jobs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for inst_dir in run_dir.iterdir():
            res = inst_dir / "result.json"
            if not res.exists():
                continue
            try:
                r = json.loads(res.read_text())
            except Exception:
                continue
            if r.get("started_at") and r.get("finished_at"):
                s = datetime.fromisoformat(r["started_at"].replace("Z", "+00:00"))
                f = datetime.fromisoformat(r["finished_at"].replace("Z", "+00:00"))
                durations.append((f - s).total_seconds())
    return durations


def cycle_stats(cycle_dir: Path):
    hj = cycle_dir / "harbor_jobs"
    if not hj.exists():
        return None
    durs = gather_durations(hj)
    if not durs:
        return None
    return {
        "n": len(durs),
        "mean": statistics.mean(durs),
        "p50": statistics.median(durs),
        "stdev": statistics.stdev(durs) if len(durs) > 1 else 0.0,
        "p99": sorted(durs)[int(0.99 * len(durs))] if len(durs) > 2 else max(durs),
    }


def find_dirs(pattern: str):
    return sorted([d for d in RESULTS.glob(pattern) if d.is_dir()])


def arm_cycles_for_lru_ta(cfg: str):
    # New 4-arm cycles
    dirs = []
    for p in find_dirs(f"run_{cfg}_now_matrix_*"):
        dirs.append(p)
    return dirs


def main():
    arms = {}

    # 1. LRU — new
    arms["LRU"] = arm_cycles_for_lru_ta("LRU")

    # 2. TA  — new
    arms["TA"] = arm_cycles_for_lru_ta("TA")

    # 3. OURS_inline — from previous H'_now matrix.
    # Pattern `run_H_prime_now_matrix_*` matches BOTH the matrix root
    # and its sibling cycleN dirs (`...matrix_TAG_cycleN`).  Exclude
    # the latter by requiring no `_cycle\d+$` suffix.
    h_prime_root = [
        d for d in find_dirs("run_H_prime_now_matrix_*")
        if not re.search(r"_cycle\d+$", d.name)
    ]
    if h_prime_root:
        cycle_dirs = []
        for p in h_prime_root[-1].iterdir():
            if p.is_symlink() and re.match(r"cycle\d+_h_prime_now", p.name):
                cycle_dirs.append(p.resolve())
        if not cycle_dirs:
            tag = h_prime_root[-1].name.replace("run_H_prime_now_matrix_", "")
            for i in (1, 2, 3):
                p = RESULTS / f"run_H_prime_now_matrix_{tag}_cycle{i}"
                if p.exists():
                    cycle_dirs.append(p)
        arms["OURS_inline"] = cycle_dirs

    # 4. OURS_full — from previous matrix (cycles 2/4/6 are "ours")
    matrix_root = sorted(find_dirs("run_K_matrix_*"))
    if matrix_root:
        cycle_dirs = []
        for i in (2, 4, 6):
            for p in find_dirs(f"run_K_full_matrix_*_cycle{i}_ours"):
                cycle_dirs.append(p)
                break
        arms["OURS_full"] = cycle_dirs

    print("# T9 4-arm fairness matrix\n")
    print(f"Generated: {datetime.now().isoformat()}\n")

    summary = {}
    for arm_name, cycle_dirs in arms.items():
        print(f"## {arm_name} ({len(cycle_dirs)} cycles)\n")
        per_cycle_means = []
        for cd in cycle_dirs:
            s = cycle_stats(cd)
            if s is None:
                print(f"* {cd.name}: no data")
                continue
            print(f"* {cd.name}: n={s['n']} mean={s['mean']:.1f} p50={s['p50']:.1f} p99={s['p99']:.1f} stdev={s['stdev']:.1f}")
            per_cycle_means.append(s["mean"])
        if per_cycle_means:
            m = statistics.mean(per_cycle_means)
            sd = statistics.stdev(per_cycle_means) if len(per_cycle_means) > 1 else 0.0
            summary[arm_name] = {"n": len(per_cycle_means), "mean": m, "stdev": sd}
            print(f"\n**Across-cycle**: mean = {m:.1f} ± {sd:.1f} s (N={len(per_cycle_means)})\n")

    if len(summary) >= 2:
        print("## Pairwise Welch t-tests\n")
        names = list(summary)
        print("| arm A | arm B | A mean ± std | B mean ± std | Δ (A−B) | SE | z |")
        print("|---|---|---|---|---|---|---|")
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sa, sb = summary[a], summary[b]
                d = sa["mean"] - sb["mean"]
                se = (sa["stdev"] ** 2 / sa["n"] + sb["stdev"] ** 2 / sb["n"]) ** 0.5
                z = d / se if se > 0 else 0.0
                print(
                    f"| {a} | {b} | {sa['mean']:.1f} ± {sa['stdev']:.1f} | "
                    f"{sb['mean']:.1f} ± {sb['stdev']:.1f} | "
                    f"{d:+.1f} | {se:.1f} | {z:+.2f} |"
                )

    print("\n## Final ranking (by mean)\n")
    for arm in sorted(summary, key=lambda x: summary[x]["mean"]):
        print(f"* {arm}: {summary[arm]['mean']:.1f} ± {summary[arm]['stdev']:.1f} s")


if __name__ == "__main__":
    main()
