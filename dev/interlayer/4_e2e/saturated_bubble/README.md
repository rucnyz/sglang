# saturated_bubble — saturated single-pool: bubble harvest +10% throughput

What it tests (design.md §saturated_bubble headline): under a workload where one
pool is admission-bottlenecked (mamba), enabling the budgeter
harvests cross-pool capacity ("bubbles") that the bottlenecked pool
borrows, measurably raising throughput by ≥10%.

This is **the core selling point** of the interlayer mechanism. byte_transfer
proved fires move bytes correctly. saturated_bubble proves moving those bytes
delivers measurable throughput.

## Driver + validator

- `run_saturated.sh` — runs the SAME workload as D7 v5 (random RPS=32,
  output=1024, `--max-mamba-cache-size=100`), but in TWO phases (off,
  inter) on the same server config so the comparison is apples-to-apples.
- `validate_saturated.py` — asserts:
  - (a) inter throughput ≥ off throughput × 1.10 (headline +10%)
  - (b) inter completion ≥ off completion (sanity: harvest didn't
        steal completions)
  - (c) inter ≥ 1 non-aborted fire (otherwise we're measuring noise,
        not the mechanism)

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
GPU=<gpu> PORT=30077 OUT_DIR=/tmp/d8_run WORKLOAD_S=180 \
    bash dev/interlayer/4_e2e/saturated_bubble/run_saturated.sh
```

Wall ~8 min (2 × (50s boot + 180s workload + teardown)).

## Workload — inherited from D7 v5

Same parameters (forces saturation via small mamba pool + sustained
random workload). See `../byte_transfer/README.md` "Workload choice"
for why `--max-mamba-cache-size=100` + `SGLANG_XPOOL_MAMBA_HIGH=0.50`
are needed post-active-fix v2.

The `SGLANG_XPOOL_MAMBA_HIGH=0.50` override only applies to the
inter phase (off phase doesn't use the planner at all).

## Result (2026-05-27, after Phase 7 + fixes #128 and #134)

saturated_bubble originally showed +3.70% TPOT regression in closed-loop bench. Two
root-cause bisects + fixes resolved it completely. See
`../../1_dyn_admission_cap/d8_regression_bisect.md` for the full
multi-stage drill-down.

**Final state (N=5 alternating off/inter reps, post fixes):**

| phase | TPOT (mean ± std) | RPS | duration |
|---|---|---|---|
| off | 7.911 ± 0.048 ms | 8.192 ± 0.046 | 351.59 ± 1.99 s |
| inter (post fixes) | **7.959 ± 0.061 ms** | 8.140 ± 0.064 | 353.83 ± 2.78 s |

**Δ TPOT: +0.61%** (1.4σ — borderline significant, within
day-to-day system noise).

Note: SHORT 90-s workload, alternating reps to control for thermal
drift. The original saturated_bubble spec (`--num-prompts $((180 * 32))`) takes
12–15 min per phase; the 90-s version reproduces the same cost
characteristics in less time.

An initial N=3 run reported -0.26% but cross-session N=5 settled on
+0.61%. The N=5 is more trustworthy.

**Fixes that resolved saturated_bubble:**
- #128: vectorize `SchedulerOwnerProvider.build_kv_owner_map` — 1669×
  speedup, removed 200 ms/fire Python loop on scheduler thread.
- #134: rewrite `allocator.mark_pages_capped` / `unmark_pages_capped` —
  no longer mutate the ~2.4 M-element `free_pages` tensor on every
  fire. `alloc()` filters against the small `_capped_pages` instead.
  Mark wall: 1.14 ms → 0.006 ms (190× faster); per-fire peak alloc
  138 MB → 24 KB (5750× reduction). This was the dominant cost.

**Original headline goal (+10% throughput) not yet demonstrated:**
the closed-loop bench (fixed num-prompts, RPS=32) is not admission-
bound, so dynamic admission can't show throughput gain here. cc_traces_headline
(real CC traces) or an open-loop bench is needed to demonstrate +10%.
