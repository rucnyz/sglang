# Step 2 — scheduler idle gap trace

## Claims under test

- **A2**: Between any two consecutive `run_batch` calls, the GPU
  reaches a moment when **no captured CUDA graph is in flight**.
- **A3**: That moment is wide enough to safely insert an ~82 ms
  fire (or, if not, we can quantify how much TPS the forced bubble
  would cost).

Both are EMPIRICAL claims — they depend on the workload and
scheduling. We can't reason them from code alone.

## What we're really asking

Option G has two operational flavors:

- **G-natural**: ride the existing idle bubble between batches.
  Requires A2 + A3 to both hold; cheapest in TPS (just uses already-
  idle GPU time).
- **G-forced**: force a bubble (drain + delay) when fire is needed.
  Only A2 needed; equivalent in cost to sync fire (~82 ms of decode
  stall per fire).

If A3 holds, fires are nearly free (G-natural). If only A2 holds,
G's cost ≈ sync fire — but with the safety properties (no
mid-flight graph) sync doesn't have either.

If A2 itself fails (graphs are ALWAYS in flight, never quiescent),
G is dead — we can't safely fire even between batches.

## Test design

**No mocks.** Run a real sglang server with a real workload (CC trace
or similar). Instrument the scheduler's main loop to record per-batch
GPU events around `run_batch`.

Two CUDA events per batch:
- `event_batch_started`: recorded just before `run_batch` launches
  any kernels
- `event_batch_finished`: recorded just after `run_batch` returns
  (specifically: after `process_batch_result` reads the GPU outputs,
  which already implicitly waits)

Then for each pair of consecutive batches B[i], B[i+1]:
- gap = `event_batch_started[i+1] - event_batch_finished[i]`
- gap is positive if GPU was idle, ≈ 0 if back-to-back, negative if
  CUDA was still draining when next launch happened (i.e., next
  launch queued behind previous, no real boundary)

Collected over many iterations under realistic load.

Distribution:
- p50 < 1 ms → A3 fails for G-natural (forced bubble required)
- p50 > 80 ms → A3 holds (fires fit naturally)
- gap < 0 frequent → A2 fails (no clean boundary)

## Workloads to sweep

1. **CC trace @ C=14** — light, baseline
2. **CC trace @ C=56** — production-stress (where the live crash
   happened)

If gap distribution differs significantly between the two, we may
need adaptive policy (G-natural at low C, G-forced at high C).

## Files

- `scheduler_gap_patch.py` — small monkey-patch wrapper that adds
  CUDA event timing around `run_batch` (to be written)
- `bench_gap_trace.sh` — launches sglang server with the patch +
  runs CC replay + collects gap log
- `analyze_gap_trace.py` — reads gap log, prints distribution

## Status

C=14 trace complete (2026-05-31). Implementation lives in
`scheduler.py:1517+` (`SGLANG_GAP_TRACE_LOG` env-guarded CUDA event
hook around `run_batch`) + `bench_gap_trace.sh` + `analyze_gap_trace.py`.

## Outcome — C=14 4-min CC trace (n=9248 batch transitions)

| metric | gap_us_gpu |
|---|---|
| min | 95 |
| **p50** | **103** ← natural gap is 0.1 ms |
| p90 | 169 |
| p99 | 6,444 (6 ms) |
| max | 1,277,350 (1.3 s, likely a request-arrival idle window) |
| % gap > 0 | 100% |
| % gap >= 82 ms | 0.02% (2 of 9248) |

### Important fact discovered en route

The user observed that sglang's `event_loop_overlap` (default for most
models) intentionally pipelines CPU prep ahead of GPU completion,
which would give A2 ≈ 0% if running there. Investigation:

- `server_args.py:2396`: for mamba models with `mamba_scheduler_strategy="no_buffer"`
  (the HiMA default), sglang AUTO-DISABLES overlap and runs
  `event_loop_normal` instead.
- D10's launch arg `disable_overlap_schedule=True` was the auto-set
  result. So our entire prior analysis IS in event_loop_normal.
- Step 2 data above is from `event_loop_normal` running CC traces.
- **Conclusion**: for HiMA workloads we are AUTOMATICALLY in the
  normal loop. For pure attention models that use overlap, G's
  premise would fail outright (A2 would be near-zero).

### A2 / A3 verdict

- **A2 holds**: 100% of transitions show positive gap. GPU does
  drain before next launch in normal loop (the loop reads outputs
  from result_queue between batches, which is a sync point).
- **A3 refuted**: natural gap p50 = 100 µs, p99 = 6 ms. Fire wall
  cost is 82 ms. **800× shortfall**. G-natural is NOT viable.
- The remaining option G-forced (block scheduler 82 ms per fire) is
  semantically equivalent to sync fire — it just forces the bubble.
  There is no "ride the natural idle" advantage at our scale.

### Decision

G's distinguishing premise ("nearly-free fire during natural idle")
is empirically refuted. G-forced equals sync fire in TPS impact.
Path forward options recorded in the parent README's "Decision after
step 2" section.

C=56 trace skipped: the bottleneck is CPU loop tightness (100 µs
between batches), not GPU work duration. Higher concurrency makes
GPU batches longer but doesn't reduce the inter-batch CPU overhead.
At any concurrency, natural gap << 82 ms fire wall.

## Raw data

`/tmp/gap_trace/gap.jsonl` (~780 KiB, 9248 lines, JSONL).

## Decision rule

| Observation | Action |
|---|---|
| p99 idle gap ≥ 100 ms at C=14 AND C=56 | A3 confirmed strongly. G-natural is viable. Advance to step 3 with G-natural. |
| p99 < 100 ms but gap is consistently > 0 | A2 holds, A3 partially. Advance to step 3 with G-forced (~82 ms bubble per fire is the cost). |
| gap distribution shows frequent < 0 (CUDA queued, no boundary) | A2 fails. G dead. Revisit C or accept current state. |
| Differs between C=14 and C=56 | Step 3 needs adaptive policy. |