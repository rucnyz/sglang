# batch_boundary_fire — option G: fire only between scheduler batches

> **Status (2026-05-31)**: scoping / step 1. Sequenced TDD, not yet integrated.

## Why this exists

Live D10@C=56 N=3 v4 (post-#173, post-#174) confirmed cross-pool m2k
fires cause `cudaErrorIllegalAddress` (Triton-wrapped as "Pointer
argument cannot be accessed from Triton"). The crash is consistent:
2 per inter cell × 3 runs.

Independent bench (`dev/interlayer/bench/bench_graph_unmap_race.py`)
empirically reproduced the underlying mechanism: a captured CUDA
graph whose recorded base pointer covers a VA range that is
subsequently `cuMemUnmap`'d will fail on replay with the same
`cudaErrorIllegalAddress`. The `torch.cuda.synchronize()` at
`xpool_actuator.py:339` only drains *in-flight* kernels — it does
NOT prevent the **next replay** from referencing the unmapped range.

Two prior options were considered:

- **A. drain-then-remap** (vllm sleep/wake pattern): scheduler-block
  ~82 ms per fire. Equivalent to sync fire. Already partially
  implemented as the in-fire safety net.
- **C. snapshot + redirect**: copy slot contents to a still-mapped
  position, rewrite `state_indices_list`, then unmap. ~1 ms GPU but
  ~600 LOC across 5 files and tight scheduler/allocator/arena coupling.

A subagent code-review of sglang + vllm (`subagent` report,
2026-05-31) recommended **G as the architecturally cleanest path
forward**:

> "C's 1 ms marginal win is dwarfed by 600 LOC of plumbing. C =
> (cap_barrier protection) + (1 ms memcpy backup) for cases where
> in-flight requests hold tail slots. G eliminates the race at the
> source: fire only when no decode is mid-step. C should NOT be
> chosen first — it's a `G + C` upgrade if `G`'s bubble proves
> unacceptable."

## The G idea in one paragraph

The scheduler's main loop (`Scheduler.event_loop_normal`) processes
batches sequentially. Between batches, NO captured CUDA graph is
replaying. If a fire (cap_barrier + cuMemUnmap + cuMemMap) happens
at that moment:

1. No graph is touching mamba/KV VA — unmap is safe (no
   in-flight read).
2. The next batch is built AFTER the fire — its
   `state_indices_list` is constructed from the post-fire allocator,
   which excludes capped slots. Replay of any captured graph for
   that batch is safe by construction.

The cost is conceptually equal to sync fire (~82 ms), but folded
into the natural idle gap between batches rather than blocking the
scheduler's CPU thread at an arbitrary moment.

## Falsifiable assumptions

We must validate each of these by REAL (not mock) experiment before
moving on:

| Assumption | Validation step |
|---|---|
| (A1) Fire-between-replays does NOT crash a captured graph whose state_indices is updated post-fire | step 1 |
| (A2) Scheduler has a stable batch boundary where no captured graph is in-flight | step 2 |
| (A3) The idle gap at batch boundary is wide enough to swallow ~82 ms fire without unacceptable bubble | step 2 |
| (A4) Queueing fires from Budgeter and draining at boundary preserves the cost-model assumption (fire happens within "reasonable" wall-time of decision) | step 3 |
| (A5) Replacing async-worker dispatch with boundary dispatch in the real actuator chain produces NO crashes under live workload | step 4 |
| (A6) End-to-end D10@C=56 with G achieves at minimum neutral throughput vs off and zero CUDA Graph crashes | step 5 |

## Steps

| step | folder | claim | test type |
|---|---|---|---|
| 1 ✓ | [`step1_boundary_safety_invariant/`](step1_boundary_safety_invariant/) | A1 (torch.gather): real CUDA graph + real cuMemUnmap → no crash if state_indices update happens between replays | standalone Python + real CUDA |
| 1b ✓ | [`step1b_triton_kernel_safety/`](step1b_triton_kernel_safety/) | A1 extended to real production Triton kernel (`fused_recurrent_gated_delta_rule_packed_decode`) | real kernel + real cuMemUnmap |
| 2 | step2_scheduler_idle_gap_trace/ | A2, A3: real sglang server + real workload, measure batch-boundary idle | server instrumentation |
| 3 | step3_fire_queue_at_boundary/ | A4: real Budgeter + queued fire + boundary drain | unit + integration |
| 4 | step4_actuator_integration/ | A5: real cross-pool fire on real server | full server smoke |
| 5 | step5_live_d10_validation/ | A6: live D10 sweep | N=3 sweep |

Each step's folder has its own README that:
- Restates the claim being tested
- Lists the test files and how to run them
- Records the OUTCOME (pass/fail/observations) once executed
- Decides whether to advance to the next step or backtrack

**No mocks**. Every test runs against real CUDA primitives, real
sglang components, or real server processes. The whole point of
this folder is to incrementally land the G hypothesis under real
conditions, not to assert behavior in vacuum.

## What this folder will NOT do

- Implement Option F (shared fixed VA) — too invasive (~2000 LOC).
- Implement Option C (snapshot+redirect) — only if G fails on A3 (idle gap too small).
- Touch the cost model (`c_actuator_us` EWMA) — separate fix tracked in
  `dev/interlayer/bench/` notes. G's GPU cost is the same as sync,
  so the existing EWMA stays valid.

## Cross-references

- Architectural observation: [`../UNIFIED_CAP_ARCHITECTURE.md`](../UNIFIED_CAP_ARCHITECTURE.md)
- Cost benches that motivate G: [`../bench/bench_cumem_costs.py`](../bench/bench_cumem_costs.py),
  [`../bench/bench_graph_unmap_race.py`](../bench/bench_graph_unmap_race.py),
  [`../bench/bench_snapshot_cost.py`](../bench/bench_snapshot_cost.py)
- D10 history that surfaced the race:
  [`../verify/D10/README.md`](../verify/D10/README.md)
