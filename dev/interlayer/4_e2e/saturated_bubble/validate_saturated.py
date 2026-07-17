"""saturated_bubble — saturated single-pool: bubble harvest +10% throughput validator.

Parses {off,inter}.bench.json + inter.budgeter.jsonl from a saturated_bubble run.

Asserts (design.md §saturated_bubble):
  (a) inter completed_throughput >= off completed_throughput × 1.10
      — the +10% bound from the design's headline claim
  (b) inter total_completed >= off total_completed
      — sanity: harvest shouldn't reduce completion count
  (c) inter has ≥ 1 non-aborted fire
      — otherwise we're measuring noise, not the mechanism

Fail-closed: any check fails → FAIL, with diagnostic data.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    off_path   = os.path.join(args.out_dir, "off.bench.json")
    inter_path = os.path.join(args.out_dir, "inter.bench.json")
    budg_path  = os.path.join(args.out_dir, "inter.budgeter.jsonl")

    for p in (off_path, inter_path, budg_path):
        if not os.path.exists(p):
            print(f"FAIL: {p} missing")
            return 1

    off   = json.load(open(off_path))
    inter = json.load(open(inter_path))

    off_thru = float(off.get("completed_request_throughput") or
                     off.get("request_throughput") or 0.0)
    in_thru  = float(inter.get("completed_request_throughput") or
                     inter.get("request_throughput") or 0.0)
    off_n    = int(off.get("completed") or 0)
    in_n     = int(inter.get("completed") or 0)
    off_ttft = float(off.get("mean_ttft_ms") or 0.0)
    in_ttft  = float(inter.get("mean_ttft_ms") or 0.0)
    off_tpot = float(off.get("mean_tpot_ms") or 0.0)
    in_tpot  = float(inter.get("mean_tpot_ms") or 0.0)

    # Count non-aborted fires in inter budgeter log
    fires = 0
    aborted = 0
    with open(budg_path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("fire_completion") and r.get("fire_direction") and \
                    r.get("fire_direction") != "none":
                if r.get("fire_aborted"):
                    aborted += 1
                else:
                    fires += 1

    print("=" * 78)
    print("saturated_bubble — saturated single-pool: bubble harvest +10% throughput")
    print("=" * 78)
    print(f"  off  : completed={off_n:>5d}  throughput={off_thru:>6.2f} req/s  "
          f"TTFT={off_ttft:>7.1f}ms  TPOT={off_tpot:>5.2f}ms")
    print(f"  inter: completed={in_n:>5d}  throughput={in_thru:>6.2f} req/s  "
          f"TTFT={in_ttft:>7.1f}ms  TPOT={in_tpot:>5.2f}ms")
    print(f"  inter fires: {fires} non-aborted, {aborted} aborted")
    if off_thru > 0:
        thru_pct = (in_thru - off_thru) / off_thru * 100
        print(f"  throughput Δ: {thru_pct:+.2f}% (target: ≥ +10%)")
    if off_n > 0:
        n_pct = (in_n - off_n) / off_n * 100
        print(f"  completion Δ: {n_pct:+.2f}% (target: ≥ 0%)")
    if off_ttft > 0:
        ttft_pct = (in_ttft - off_ttft) / off_ttft * 100
        print(f"  TTFT Δ:       {ttft_pct:+.2f}% (informational)")

    all_ok = True

    # (a) +10% throughput
    if off_thru <= 0:
        print("\nFAIL (a): off baseline throughput = 0 (workload didn't run)")
        return 1
    target = off_thru * 1.10
    if in_thru >= target:
        print(f"\n(a) throughput ≥ +10% — PASS "
              f"({in_thru:.2f} ≥ {target:.2f})")
    else:
        all_ok = False
        print(f"\n(a) throughput ≥ +10% — FAIL "
              f"({in_thru:.2f} < {target:.2f}, deficit "
              f"{target - in_thru:.2f} req/s)")

    # (b) completion non-regression
    if in_n >= off_n:
        print(f"(b) completion ≥ off — PASS ({in_n} ≥ {off_n})")
    else:
        all_ok = False
        print(f"(b) completion ≥ off — FAIL ({in_n} < {off_n})")

    # (c) at least one fire
    if fires >= 1:
        print(f"(c) ≥1 non-aborted fire — PASS ({fires} fires)")
    else:
        all_ok = False
        print(f"(c) ≥1 non-aborted fire — FAIL (0 fires; can't claim "
              f"throughput Δ comes from mechanism)")

    print(f"\nD8: {'ALL PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
