#!/usr/bin/env python3
"""Setting 4 — estimator accuracy analysis.

Compare the budgeter's V_σ' approximation (V_σ' ≈ usage_σ from
cross_pool_planner.py L26) against ground-truth marginal value
inferred from Sweep 1 throughput differences.

Sweep 1 measured input-throughput at 5 mamba_full_memory_ratio
settings on Qwen3.5-35B-A3B. mamba peak usage is saturated at 0.66
across all 5 points; the proxy V_mamba' = usage_mamba is therefore
flat. But throughput varies 1.91× across the sweep, meaning the true
V_mamba' is non-zero. The proxy can't see it.
"""

import json
import statistics
import sys


# Sweep 1 measurements (from RESULTS.md table; reproduced 2026-04-30).
SWEEP1 = [
    # (mamba_full_memory_ratio, input_TPS, mamba_peak, full_peak, mean_TTFT_s)
    (0.1, 4512, 0.66, 0.01, 38.91),
    (0.3, 6461, 0.66, 0.02, 21.37),
    (0.5, 7585, 0.66, 0.04, 14.94),
    (0.7, 7919, 0.66, 0.05, 13.27),
    (0.9, 8610, 0.66, 0.07, 10.40),
]

print("=== Setting 4: estimator (V≈usage) accuracy on Sweep 1 ===\n")
print(f"{'ratio':>6} {'TPS':>6} {'usage_mamba':>12} {'usage_kv':>10} {'V_mamba_true':>14} {'V_kv_true':>12}")
print('-' * 72)

# Compute ground truth marginal value V_σ' = ΔTPS / ΔPool_size.
# Pool size for mamba: ratio * total_mem ⇒ ΔPool_mamba ∝ Δratio.
# So per-Δratio: V_mamba' = ΔTPS / Δratio.
# For KV: V_kv' = ΔTPS / Δ(1-ratio) = -ΔTPS / Δratio (decreasing as more
# memory shifts to mamba). At every Sweep 1 point KV is below 7%
# utilization — KV is unsaturated, so the "true" V_kv' is small.

prev_ratio, prev_tps = SWEEP1[0][0], SWEEP1[0][1]
for ratio, tps, mb_use, kv_use, ttft in SWEEP1:
    if ratio == prev_ratio:
        v_mb = float('nan')
        v_kv = float('nan')
    else:
        v_mb = (tps - prev_tps) / (ratio - prev_ratio)
        # Symmetric: more memory to mamba is less to KV.
        v_kv = -v_mb
    print(f"{ratio:>6.1f} {tps:>6.0f} {mb_use:>12.3f} {kv_use:>10.3f} "
          f"{v_mb:>14.0f} {v_kv:>12.0f}")
    prev_ratio, prev_tps = ratio, tps

print()
print("Key observation:")
print(f"  - usage_mamba is FLAT at 0.66 across all 5 points (admission")
print(f"    ceiling — mamba pool is saturated regardless of allocation).")
print(f"  - But throughput climbs 4512 → 8610 (1.91x) as more memory")
print(f"    is given to mamba.")
print(f"  - True V_mamba' (= ΔTPS/Δratio) varies from ~1670 to ~9745;")
print(f"    Pearson correlation with the flat proxy = 0 (proxy is constant).")
print()
print("Implication: the V_σ' ≈ usage_σ approximation is *saturation-blind*.")
print("On admission-limited workloads it cannot distinguish 'needs more")
print("mamba' from 'has enough mamba'. The planner's threshold-with-")
print("hysteresis logic still works (it fires when usage crosses the high")
print("watermark), but the *direction* of allocation can't be derived")
print("from usage alone in this regime.")
print()
print("Where the proxy DOES work: when neither pool is saturated, usage")
print("does correlate with marginal value (the textbook regime). On the")
print("Phase 1+2+3 stress trace (Setting 3.C) usage_mamba varies 0.01–0.43;")
print("the planner fires 21 transfers in the right direction, all kv→mamba.")
print("That's the proxy working as intended.")

# Also load runtime usage time series from the 3.C runs to demonstrate
# the unsaturated regime.
import glob, os
print("\n=== usage time series from Setting 3.C runs ===")
for path in sorted(glob.glob("/tmp/setting3c_v2_*/L*_budgeter.jsonl"))[:3]:
    cell = os.path.basename(path).split("_budgeter")[0]
    kvs, mbs = [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                kv = d.get("xpool_plan_usage_kv_inst", 0)
                mb = d.get("xpool_plan_usage_mamba_inst", 0)
                if kv > 0 or mb > 0:
                    kvs.append(kv); mbs.append(mb)
            except: continue
    if kvs:
        print(f"  {cell}: kv [{min(kvs):.3f},{max(kvs):.3f}] "
              f"mamba [{min(mbs):.3f},{max(mbs):.3f}] (n={len(kvs)} ticks)")
