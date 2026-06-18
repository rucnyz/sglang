"""Compare two sets of idle_no_regression runs (e.g. with-lock vs no-lock) and
report whether the TTFT difference is distinguishable from noise.

Usage:
  python noise_compare.py \\
    --with /tmp/d8b_9b_lock_r1 /tmp/d8b_9b_lock_r2 /tmp/d8b_9b_lock_r3 \\
    --without /tmp/d8b_9b_nolock_r1 /tmp/d8b_9b_nolock_r2 /tmp/d8b_9b_nolock_r3

Statistical test:
  Compute mean ± std of TTFT_off and TTFT_inter across the N reps for
  each group. The "regression delta" is the difference in (TTFT_inter
  − TTFT_off) between the two groups. If |delta| > 2 × pooled_std, the
  regression is real. Otherwise it's within noise.
"""
import argparse
import json
import math
import os
import sys


def load_pair(d):
    """Load (TTFT_off, TTFT_inter, throughput_off, throughput_inter)
    from one idle_no_regression run dir."""
    off = json.load(open(os.path.join(d, "off.bench.json")))
    inter = json.load(open(os.path.join(d, "inter.bench.json")))
    return (
        off["mean_ttft_ms"], inter["mean_ttft_ms"],
        off["output_throughput"], inter["output_throughput"],
    )


def mean_std(xs):
    n = len(xs)
    if n == 0: return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1: return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with", dest="with_", nargs="+", required=True,
                    help="dirs from runs WITH the lock")
    ap.add_argument("--without", nargs="+", required=True,
                    help="dirs from runs WITHOUT the lock")
    args = ap.parse_args()

    print(f"\n{'='*80}")
    print(f"alloc_lock — lock perf-regression noise floor")
    print(f"{'='*80}")

    print(f"\nN_with={len(args.with_)}  N_without={len(args.without)}\n")

    for label, dirs in [("WITH lock", args.with_),
                        ("WITHOUT lock", args.without)]:
        print(f"--- {label} ---")
        ttft_off, ttft_inter, tput_off, tput_inter = [], [], [], []
        deltas = []
        for d in dirs:
            try:
                to, ti, po, pi = load_pair(d)
            except Exception as e:
                print(f"  SKIP {d}: {e}")
                continue
            ttft_off.append(to); ttft_inter.append(ti)
            tput_off.append(po); tput_inter.append(pi)
            deltas.append((ti - to) / to * 100)
            print(f"  {os.path.basename(d):>30s}  off={to:6.2f}  "
                  f"inter={ti:6.2f}  Δ={(ti-to)/to*100:+6.3f}%  "
                  f"tput_off={po:6.1f}  tput_inter={pi:6.1f}")

        if ttft_off:
            mo, so = mean_std(ttft_off)
            mi, si = mean_std(ttft_inter)
            md, sd = mean_std(deltas)
            print(f"  {label:>30s}  off={mo:6.2f}±{so:.2f}  "
                  f"inter={mi:6.2f}±{si:.2f}  "
                  f"Δ={md:+6.3f}±{sd:.3f}%")
            # Store for cross-group comparison
            if label == "WITH lock":
                with_md, with_sd, with_n = md, sd, len(deltas)
            else:
                without_md, without_sd, without_n = md, sd, len(deltas)
        print()

    print(f"{'='*80}")
    print(f"VERDICT")
    print(f"{'='*80}")

    if 'with_md' not in dir() or 'without_md' not in dir():
        print("(insufficient data — need at least 1 rep per group)")
        return 1

    regression_pp = with_md - without_md
    # Pooled std of the difference (independent samples)
    pooled_se = math.sqrt(with_sd**2 / with_n + without_sd**2 / without_n)

    print(f"lock-induced Δ in TTFT regression: {regression_pp:+.3f} pp")
    print(f"pooled standard error             : {pooled_se:.3f} pp")
    print(f"|Δ| / SE                          : {abs(regression_pp)/max(pooled_se,1e-9):.2f}")

    if pooled_se < 1e-9:
        # N=1 reps, no σ info
        print(f"\nCannot compute significance with N=1; need ≥2 reps per group.")
        return 1
    if abs(regression_pp) < 2 * pooled_se:
        print(f"\n→ lock-induced regression is WITHIN noise (|Δ| < 2 × SE).")
        print(f"  Treat as no real loss; document and move on.")
        return 0
    else:
        print(f"\n→ lock-induced regression is REAL (|Δ| ≥ 2 × SE).")
        print(f"  Proceed to alloc_lock/TODO.md TODO 1 (finer CS) or TODO 2 (worker-active flag).")
        return 2


if __name__ == "__main__":
    sys.exit(main())
