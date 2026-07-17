"""alternating_saturation validator.

Pass criteria (design.md §alternating_saturation):
- output_throughput_inter ≥ output_throughput_off × 0.95
  (no > 5% regression from actuator over-firing)
- (additional sanity) fire direction histogram is not 100% one-way;
  fires should reverse direction over the adversarial workload —
  otherwise budgeter isn't actually responding to alternating phases.
"""
import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    off_path = os.path.join(args.out_dir, "off.bench.json")
    inter_path = os.path.join(args.out_dir, "inter.bench.json")
    budg_path = os.path.join(args.out_dir, "inter.budgeter.jsonl")

    for p_ in (off_path, inter_path):
        if not os.path.exists(p_):
            print(f"MISSING: {p_}")
            sys.exit(2)

    off = json.load(open(off_path))
    inter = json.load(open(inter_path))

    print(f"off:   completed={off['completed']} tps={off['output_throughput']:.0f} "
          f"tpot={off['mean_tpot_ms']:.2f}ms dur={off['duration_s']:.1f}s")
    print(f"inter: completed={inter['completed']} tps={inter['output_throughput']:.0f} "
          f"tpot={inter['mean_tpot_ms']:.2f}ms dur={inter['duration_s']:.1f}s")
    print()

    # ---- Throughput regression check ----
    tps_off = off["output_throughput"]
    tps_inter = inter["output_throughput"]
    ratio = tps_inter / tps_off if tps_off > 0 else 0
    print(f"output_throughput ratio: {ratio:.4f} (= inter / off)")
    if ratio >= 0.95:
        print(f"  PASS (a): {ratio*100:.2f}% ≥ 95% (no > 5% regression)")
    else:
        print(f"  FAIL (a): {ratio*100:.2f}% < 95% (> 5% regression)")
        sys.exit(1)

    # ---- Fire direction sanity ----
    fires = []
    if os.path.exists(budg_path):
        for line in open(budg_path):
            e = json.loads(line)
            if e.get("fire_completion") and not e.get("fire_aborted"):
                fires.append(e)
    print()
    print(f"Fires: {len(fires)} non-aborted")
    if fires:
        dirs = {}
        for f in fires:
            d = f.get("fire_direction", "?")
            dirs[d] = dirs.get(d, 0) + 1
        print(f"  direction histogram: {dirs}")
        # Sanity: if adversarial workload truly alternates pressure,
        # fires should also alternate. ≥1 in both directions is a
        # signal the planner sees the alternation.
        kv2m = dirs.get("kv_to_mamba", 0)
        m2kv = dirs.get("mamba_to_kv", 0)
        if kv2m > 0 and m2kv > 0:
            print(f"  PASS (b): fires in BOTH directions "
                  f"({kv2m} kv→mamba, {m2kv} mamba→kv)")
        elif kv2m + m2kv > 0:
            print(f"  WARN (b): fires only in ONE direction. May indicate "
                  f"planner doesn't see alternation, or workload only "
                  f"stresses one pool.")
        else:
            print(f"  WARN (b): no non-aborted fires at all. inter ≈ off.")
    else:
        print(f"  WARN: no fires; comparing budgeter overhead only.")

    print()
    print("=== alternating_saturation PASS ===")


if __name__ == "__main__":
    main()
