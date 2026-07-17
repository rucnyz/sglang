# 0_zero_blocking_fire — combine A (stream isolation) + C (predictive snapshot)

> **Status (2026-05-31)**: active line of work. Replaces archived
> [`../archive/0_batch_boundary_fire/`](../archive/0_batch_boundary_fire/) (G,
> falsified by step 2: natural idle gap p50 = 100 µs, p99 = 6 ms ≪ 82 ms fire).

## What we're building

A fire path that does **zero work on the decode stream**. Specifically:

1. **Predictive snapshot**: Budgeter, on every tick, copies the
   contents of the oldest K mamba slots into a **pre-reserved
   "shadow ring"** of N mapped chunks. Copy runs on a SIDE stream,
   overlapping with whatever decode is doing on the decode stream.
2. **Index swap**: when fire fires, the actuator only has to update
   `req.mamba_pool_idx` for the affected reqs so the NEXT replay's
   `state_indices_list.copy_(mamba_indices)`
   (`hybrid_linear_attn_backend.py:520`) routes through the shadow
   slot.
3. **Stream-isolated unmap**: now that nothing reads the original
   src slot's VA, we call `cuMemSetAccess(NONE)` from the decode
   stream's perspective and `cuMemUnmap` on the side stream. CUDA
   spec says the unmap synchronizes only against streams that have
   access — decode stream has been revoked, so decode runs
   concurrently.

The result: **fire's marginal cost on the decode stream is ~0**.
The snapshot copy is amortized into prior decode batches via the
side stream; the unmap doesn't block decode at all.

This is the architecturally cleanest path discovered in the
2026-05-31 subagent investigation (`subagent` report, Section 4):
"C+A combination ... fire decision → unmap path becomes ~1.8 ms";
the additional pre-snapshot folds that ~3 ms copy into prior
batches, giving 0 ms on the decode stream.

## Why this is the ideal architecture

The original async fire design assumed worker-thread unmap doesn't
slow the scheduler. That assumption was false because all CUDA work
shared one stream (the default). With proper stream isolation, the
assumption can be made TRUE.

The original sync fire design accepted ~82 ms scheduler block per
fire. The cost model has to charge that wall every fire, suppressing
~all marginal fires.

With C+A, the cost model can honestly charge 0 ms — every fire that
clears the cost gate by ANY margin is net positive. This is the
first time since the project began that the cost model is consistent
with the actuator's actual behavior.

## Falsifiable assumptions

| Assumption | Validation step |
|---|---|
| (A1) `cuMemUnmap` on side stream WITH `cuMemSetAccess(NONE)` on decode stream's access actually allows concurrent decode kernel (subagent's S2 hypothesis) | step 1 |
| (A2) `MambaPool.migrate_slot` + `state_indices` re-routing is byte-correct under a captured graph replay (i.e., post-migrate reads see migrated bytes) | step 2 |
| (A3) Shadow ring is wide enough that snapshot can stay ahead of fire demand under realistic load (no starvation) | step 3 |
| (A4) Per-tick shadow snapshot on side stream does not measurably slow the decode stream (target: < 0.5 ms TPS impact per snapshot) | step 4 |
| (A5) End-to-end fire path (snapshot → index swap → side-stream unmap) survives a live cross-pool fire with no crash + no decode-stream stall | step 5 |
| (A6) D10@C=56 N=3 with C+A fire achieves at minimum a net positive on mean_ttft / output_tps vs off baseline | step 6 |

## Steps

| step | folder | claim | test type |
|---|---|---|---|
| 1 ✓ | [`step1_stream_isolated_unmap/`](step1_stream_isolated_unmap/) | A1: side-thread unmap + setAccess revoke → concurrent decode does not crash and does not stall (PROVEN 2026-05-31 at production scope: real Triton kernel, captured graph, SAME-VA reservation, worker thread; deltas all < 0.15 ms vs baseline 53.5 ms; revoke confirmed real via fault-on-read) | standalone Python + real CUDA + real Triton kernel |
| 2 ✓ | [`step2_migrate_slot_replay_invariant/`](step2_migrate_slot_replay_invariant/) | A2: migrate_slot followed by state_indices rewrite preserves captured-graph correctness (PROVEN 2026-05-31: diff 0.000000 byte-exact; side-stream copy during decode delta -0.01 ms) | real Triton kernel + captured graph |
| 3 | step3_shadow_ring_planner/ | A3: shadow ring sizing model + planner picks snapshot targets without starving | real Budgeter + workload trace |
| 4 | step4_side_stream_snapshot_overhead/ | A4: per-tick snapshot side-stream cost on decode stream | real server, instrumented |
| 5 | step5_full_fire_integration/ | A5: full snapshot → swap → unmap path on real cross-pool fire | full server smoke |
| 6 | step6_live_d10_validation/ | A6: live D10 sweep | N=3 |

## What we will NOT do

- **Re-attempt G** (batch-boundary fire) — empirically refuted by
  archive/0_batch_boundary_fire/step2. The natural idle gap is too
  small at any concurrency.
- **Force-block scheduler 82 ms per fire** — equivalent to sync, no
  win.
- **Rebuild captured graphs per fire** — costs more than the fire it
  would replace.

## Cross-references

- Falsified G investigation: [`../archive/0_batch_boundary_fire/`](../archive/0_batch_boundary_fire/)
- Cost benches: [`../bench/`](../bench/) (fire wall = 82 ms; snapshot
  copy = 1.13–3 ms; graph-unmap race repro)
- Architectural observation: [`../UNIFIED_CAP_ARCHITECTURE.md`](../UNIFIED_CAP_ARCHITECTURE.md)
- Live data that surfaced the race: [`../verify/D10/README.md`](../verify/D10/README.md)
- Existing infrastructure we reuse:
  - `python/sglang/srt/mem_cache/memory_pool.py:738-820` MambaPool.migrate_slot
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:484-520` per-replay index copy
  - `python/sglang/srt/arena/xpool_actuator.py` fire flow
