# alternating_saturation — adversarial alternating-saturation: no regression

What it tests (design.md §alternating_saturation): a workload that alternates between
KV-saturated and mamba-saturated phases at the tick-boundary period
forces the planner to fire constantly back-and-forth. Throughput
should NOT regress more than 5% vs `inter=off` baseline, even though
actuator work is high.

This is a **negative** test: passes by NOT regressing. Confirms the
planner's cooldown / hysteresis / persist-consec logic prevents
runaway firing under adversarial workloads.

## Driver

- `payload_adversarial.py` — custom HTTP dispatcher (sglang's
  `bench_serving` doesn't support phase-switching prompt mixes).
  Alternates every `--phase_s` seconds:
  - KV phase: long prompts (default 1024 input tokens) → high KV
    pressure → planner sees usage_kv_active climb
  - mamba phase: short prompts (default 64) → mamba pressure
  - At `--rps` requests per second
  - Total `--duration` seconds (default 300 = 5 min)
- `run_adversarial.sh` — wraps it in two phases (off, inter)
- `validate_adversarial.py` — asserts:
  - (a) `output_throughput_inter / output_throughput_off ≥ 0.95`
  - (b) fire direction histogram has BOTH `kv_to_mamba` and
    `mamba_to_kv` (sanity: planner actually saw the alternation)

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
GPU=3 PORT=30077 OUT_DIR=/tmp/d8c_run DURATION_S=300 PHASE_S=2.0 RPS=50 \
    bash dev/interlayer/4_e2e/alternating_saturation/run_adversarial.sh
```

Wall ≈ 11 min (2 × (50s boot + 300s drive + cleanup)).

## Tuning the workload

If the workload doesn't actually saturate either pool, the test is
weak (planner won't fire, comparison reduces to budgeter-tick noise).

To verify saturation: examine `inter.budgeter.jsonl` after a run.
Look at `pool_occupancy_kv` and `pool_occupancy_mamba` over time —
they should swing between high (saturated) and low (released) as
phases alternate.

If only one pool ever saturates, increase the dominant phase's RPS or
adjust `kv_input_len` / `mamba_input_len`.

## Result (2026-05-28)

### Run 1: weak workload (RPS=50, output=64)

```
off:   completed=15001 tps=3195 tpot=11.51ms dur=300.5s
inter: completed=15001 tps=3195 tpot=11.51ms dur=300.5s
ratio: 1.0000 → PASS (a) trivially  (no fires happened; planner never above threshold)
```

Workload didn't saturate either pool — no fires triggered. Strict
spec satisfied, but the test is meaningless because the adversarial
mechanism wasn't exercised. Tune `RPS`, `OUTPUT_LEN`, or
`SGLANG_XPOOL_MAMBA_HIGH` to actually trigger fires.

### Run 2: saturated workload (RPS=100, output=1024)

```
single rep: off=460, inter=414, ratio=0.901 → FAIL (a): 90.06% < 95%
```

Looked like a regression. But N=3 follow-up revealed it was noise:

```
N=3:
  off:   tps=872 ± 132 (per-rep: 972, 920, 723)
  inter: tps=927 ± 146 (per-rep: 852, 1095, 832)
  Δ throughput: +6.28% (within noise — per-rep std ~15%)
```

**At this workload, per-rep variance is ~15-16% of mean**, so anything
in [-20%, +20%] is statistically plausible. The single-rep -10% and
N=3 mean +6.28% are both consistent with "no real effect".

**Why so noisy:** queue length ~272-493 reqs (severely admission-
bound) → tiny differences in admission timing / radix-cache eviction
get amplified. Also, fires never trigger here — `usage_mamba_active`
caps at ~0.36 (admission limit × radix-cached subtraction) — so this
isn't actually testing the adversarial-thrash conjecture.

### Conclusion / next

alternating_saturation at admission-bound RPS=100 is NOT a usable test for the
"alternating-saturation no-thrash" conjecture because:
- Workload variance dominates any effect we'd want to measure
- Fires don't trigger at the planner's default thresholds

To make alternating_saturation meaningful, future work needs ONE of:
- Lower RPS (~60-70) just past saturation but not catastrophic queueing
- Lower `SGLANG_XPOOL_MAMBA_HIGH` (e.g. 0.30) so fires trigger at
  observable `usage_mamba_active` levels
- A different model with bigger per-req mamba footprint so the
  saturation envelope is sharper

Marked: **alternating_saturation spec-passing variant TBD**. Recommend addressing in
a future tuning pass or in cc_traces_headline (real CC traces, naturally varied).
