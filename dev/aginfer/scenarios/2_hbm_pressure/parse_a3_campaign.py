#!/usr/bin/env python3
"""#208 — aggregate the A3 (HBM-pressure) 5-arm + const-V_u campaign.

Globs the campaign's per-cycle result dirs by STAMP (the tag a3_campaign.sh
embeds), computes per-arm across-cycle mean ± std of the per-trial wall
time, and the pairwise Welch z table — same format as
scenarios/1_swebench_default/ANALYSIS.md.

Reuses parse_4arm's per-cycle duration parsing so the numbers are computed
identically to the swebench_default matrix.

Usage:
    python parse_a3_campaign.py <STAMP>          # e.g. 20260605_085244
    python parse_a3_campaign.py                  # reads /tmp/a3_campaign_stamp.txt
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
sys.path.insert(0, str(_SHARED))
from parse_4arm import RESULTS, cycle_stats  # noqa: E402

# arm label → results-dir prefix produced by the runners (RUN_K_RESULTS_TAG
# = a3camp_<stamp>_<arm>_cycle<i>); see a3_campaign.sh.
ARM_PREFIX = {
    "LRU": "run_LRU_now",
    "TA": "run_TA_now",
    "OURS_inline": "run_H_prime_now",
    "OURS_full": "run_K_a3",
    "const_V_u": "run_K_a3",
}
ARM_TAG = {  # the <arm> token inside the tag (run_k.sh arms share run_K_a3)
    "LRU": "lru",
    "TA": "ta",
    "OURS_inline": "ours_inline",
    "OURS_full": "ours_full",
    "const_V_u": "const_vu",
}


def arm_dirs(arm: str, stamp: str):
    pat = f"{ARM_PREFIX[arm]}_a3camp_{stamp}_{ARM_TAG[arm]}_cycle*"
    return sorted(d for d in RESULTS.glob(pat) if d.is_dir())


def main() -> int:
    if len(sys.argv) > 1:
        stamp = sys.argv[1]
    else:
        line = Path("/tmp/a3_campaign_stamp.txt").read_text().strip()
        stamp = line.split("=", 1)[1]
    print(f"# A3 (HBM-pressure) 5-arm + const-V_u campaign — stamp {stamp}\n")
    summary = {}
    for arm in ARM_PREFIX:
        dirs = arm_dirs(arm, stamp)
        print(f"## {arm} ({len(dirs)} cycles)\n")
        means = []
        for cd in dirs:
            s = cycle_stats(cd)
            if s is None:
                print(f"* {cd.name}: no data")
                continue
            print(f"* {cd.name}: n={s['n']} mean={s['mean']:.1f} "
                  f"p50={s['p50']:.1f} p99={s['p99']:.1f} stdev={s['stdev']:.1f}")
            means.append(s["mean"])
        if means:
            m = statistics.mean(means)
            sd = statistics.stdev(means) if len(means) > 1 else 0.0
            summary[arm] = {"n": len(means), "mean": m, "stdev": sd}
            print(f"\n**Across-cycle**: mean = {m:.1f} ± {sd:.1f} s "
                  f"(N={len(means)})\n")

    if len(summary) >= 2:
        print("## Pairwise Welch t-tests\n")
        names = list(summary)
        print("| arm A | arm B | A mean ± std | B mean ± std | Δ (A−B) | SE | z |")
        print("|---|---|---|---|---|---|---|")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = summary[a], summary[b]
                d = sa["mean"] - sb["mean"]
                se = (sa["stdev"] ** 2 / sa["n"] + sb["stdev"] ** 2 / sb["n"]) ** 0.5
                z = d / se if se > 0 else 0.0
                print(f"| {a} | {b} | {sa['mean']:.1f} ± {sa['stdev']:.1f} | "
                      f"{sb['mean']:.1f} ± {sb['stdev']:.1f} | {d:+.1f} | "
                      f"{se:.1f} | {z:+.2f} |")

    print("\n## Final ranking (by per-trial mean)\n")
    for arm in sorted(summary, key=lambda x: summary[x]["mean"]):
        print(f"* {arm}: {summary[arm]['mean']:.1f} ± {summary[arm]['stdev']:.1f} s")

    print("\n## Isolation decomposition\n")
    def delta(a, b):
        if a in summary and b in summary:
            return summary[a]["mean"] - summary[b]["mean"]
        return None
    rows = [
        ("V_u RANKING value (ours_full − const_V_u)", "OURS_full", "const_V_u"),
        ("multi-tier MACHINERY value (const_V_u − LRU)", "const_V_u", "LRU"),
        ("DAEMON value (ours_full − ours_inline)", "OURS_full", "OURS_inline"),
        ("vs BFD baseline (ours_full − TA)", "OURS_full", "TA"),
        ("vs LRU baseline (ours_full − LRU)", "OURS_full", "LRU"),
    ]
    for label, a, b in rows:
        d = delta(a, b)
        if d is not None:
            print(f"* {label}: {d:+.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
