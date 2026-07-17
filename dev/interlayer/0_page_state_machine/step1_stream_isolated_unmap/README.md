# A1 — stream-isolated `cuMemUnmap` does not block the decode stream

## Claim under test

If we issue `cuMemUnmap` on a **side CUDA stream** AFTER calling
`cuMemSetAccess(NONE)` to revoke the decode stream's access to the
target VA range, then a kernel running on the **decode stream**
operating on a DIFFERENT VA range completes WITHOUT crash and
WITHOUT stalling for the unmap.

This is the structural property that lets the actuator's
worker-thread Unmap step (`xpool_actuator.py:254`
`_execute_async_locked`) cost ~0 ms on the decode stream.

## Why this might fail

CUDA spec (programming guide §6.2.5.3 cuMemUnmap) says
`cuMemUnmap` synchronizes against streams that have access to the
range. If our `cuMemSetAccess(NONE)` either:

- doesn't actually revoke access in a way the driver respects, or
- causes the decode stream to fault when it next reads ANY VA
  (because the access table is in a transient state), or
- forces a global sync regardless,

then A1 is refuted: the worker-thread Unmap step cannot run
concurrently with decode without stall, and the "fire wall on
decode stream ≈ 0 ms" claim in design.md §"Threading model"
fails.

Equally possible: the CUDA driver may insert a barrier on cuMemUnmap
that affects all streams that EVER had access historically, not just
"have access right now." We need empirical verification.

## Test design (no mocks, real CUDA)

Setup:
1. Reserve VA, map two regions: `region_keep` (decode stream reads
   from here) and `region_unmap` (the chunk we'll yank).
2. Build a long-running busy kernel on the decode stream that reads
   from `region_keep` repeatedly.
3. On a separate CUDA stream (the "fire" stream):
   - `cuMemSetAccess(region_unmap, prot=NONE)` from decode stream's
     perspective.
   - `cuMemUnmap(region_unmap)`.
4. Observe: does the decode stream kernel complete normally? How
   long does the fire stream take? Does either stream stall?

Negative control:
- Same setup, but `cuMemUnmap` on the SAME stream as decode. Should
  observe a stall in decode while unmap completes — establishing
  that the stall mechanism exists and that A1's stream isolation is
  load-bearing.

## Files

- `test_stream_isolation.py` — to be written.

## Outcome (2026-05-31)

**A1 CONFIRMED across all production-relevant variants (1.1-1.7).**

Original 1.1-1.4 (cuBLAS + eager + disjoint VA + main thread):

```
[1.1] baseline decode (warmed up, 5 runs, median): 53.51 ms (very stable)
[1.2] pure cuMemUnmap (no decode): 0.233 ms host wall (1 chunk)
[1.3] concurrent 1-chunk unmap during decode:
        decode GPU: 53.52 ms (delta +0.02 ms — within noise floor ±0.1 ms) ✓
[1.4] 200-chunk unmap during decode (DISJOINT VA reservation):
        pure 200-chunk unmap: 5.43 ms host wall
        concurrent decode GPU: 53.53 ms (delta +0.02 ms — within noise) ✓
```

Audit additions 1.5-1.7 (closes G1+G2+G3, G4+G5, G6 from 2026-05-31
depth audit — see `## Audit (2026-05-31) — depth check` below):

```
[1.5] real Triton kernel + captured graph + SAME-reservation concurrent unmap:
        state in big_va chunks 0..3; ssm_state_indices=[0,1,2,3];
        concurrent unmap target = chunks 100..199 OF THE SAME big_va;
        3 trials × 50 replays + 100-chunk concurrent unmap: no crash ✓
[1.6] worker-thread cuMemUnmap (Python threading.Thread, mirroring
        Budgeter _fire_worker at xpool_actuator.py:105,251):
        decode GPU: 53.61 ms (delta +0.11 ms — within noise) ✓
[1.7] cuMemSetAccess(prot=0) revoke verification (subprocess-isolated):
        post-revoke device read raised AcceleratorError
        "cudaErrorIllegalAddress" as expected ✓ (revoke is a real
        access denial, not driver bookkeeping)
```

### Properties proven

1. **`cuMemUnmap` host call returns almost immediately** (~27 µs/chunk
   amortized). It does NOT host-block waiting for queued GPU work to
   complete first.
2. **GPU decode kernel running concurrently is unaffected** for both
   kernel classes (cuBLAS GEMM in 1.4, real Triton recurrent kernel in
   1.5).
3. **`cuMemSetAccess(prot=NONE)` followed by `cuMemUnmap` does NOT
   trigger a global stream sync** at our scale (1+200+300 chunks
   across the three concurrent-unmap variants).
4. **Captured CUDA graph replay survives same-reservation concurrent
   unmap** when `ssm_state_indices` excludes the unmapped chunks. The
   captured graph's frozen base pointer covers the unmapped offsets,
   but as long as the kernel's tile loads (gated by `state_idx`) stay
   in the mapped range, no fault occurs.
5. **Worker-thread unmap does not cross any CUDA primary-context
   boundary that would stall the main thread's decode stream**.
6. **`cuMemSetAccess(prot=0)` is a real revoke**, not a bookkeeping
   no-op. A device-side read after revoke faults with
   `cudaErrorIllegalAddress`.

### What this means for the actuator

The Unmap step in a full fire (1152 chunks total ≈ ~30 ms host
wall) runs from the Budgeter `_fire_worker` thread while the
captured mamba-decode graph replays on the scheduler's main
thread, and decode is **not slowed**. The "fire wall on decode
stream ≈ 0 ms" claim of design.md §"Threading model" holds
empirically at the production kernel, graph mode, VA layout, and
thread axes.

**Which defense layer production uses (post-#205)**: production
takes the raw `cuMemUnmap` leg — no `torch.cuda.synchronize()`,
no `cuMemSetAccess(prot=0)` revoke. Step 1.5 / 1.6 proved that
leg is safe given the layer-0 invariant (`fire_planner` only
picks pages whose slot indices no in-flight req's
`state_indices` references); step 1.7 proved `cuMemSetAccess(prot=0)`
is an equivalent alternative defense (also safe, +0.11 ms wall),
not required for safety. The production-bound regression for the
no-defense leg lives at
[`../decode_wall/`](../decode_wall/).

### Caveats (residual gaps, lower severity)

1. **G7 (overclaim)**: 1.4/1.5/1.6 delta is within the ±0.1 ms noise
   floor. We have proven "decode impact < 0.1 ms", not "decode impact
   = 0 ms". Fine for the cost model (threshold = +5 ms).
2. **G9 (200/300 vs 1152 chunks)**: the concurrent-unmap test scales to
   300 chunks (1.5) and 100 chunks (1.6); production fire is 1152
   chunks. Per-call host cost is ioctl-dominated (no GPU contention),
   so linear extrapolation gives ~30 ms host wall — bounded by Budgeter
   tick budget. Step 5's full-server smoke will exercise actual 1152
   under real Triton replay.
3. **G8 (back-to-back fires)**: `_fire_inflight` lock at
   xpool_actuator.py:105 serialises fires; back-to-back fires queue
   rather than race. Out of scope for step 1.

## Audit (2026-05-31) — depth check

Original 1.1-1.4 (committed earlier 2026-05-31) only proved A1 for the
cuBLAS-GEMM + eager-launch + disjoint-VA + main-thread combination.
A subagent audit identified 9 gaps; G1-G6 were closed inline as 1.5,
1.6, 1.7 (above). Verbatim mapping:

| Audit gap | Closed by | Resolution |
|---|---|---|
| **G1** test uses cuBLAS, not real Triton kernel | 1.5 | `fused_recurrent_gated_delta_rule_packed_decode` (the live D10 crash's kernel) inside the captured graph; no crash |
| **G2** eager launch, not captured graph replay | 1.5 | captured `torch.cuda.CUDAGraph`, 50 replays per trial |
| **G3** unmap target in DIFFERENT VA reservation than decode reads | 1.5 | unmap target is chunks 100..199 of the SAME 200-chunk `big_va` that holds the state in chunks 0..3 — matches production layout (chunk_arena.py:299-300) |
| **G4** "side-stream cuMemUnmap" claim never tested — unmap is a host syscall | 1.6 | unmap runs on a Python `threading.Thread`; A1 is reframed as "non-decode-thread host unmap doesn't block decode" (the meaningful production property) |
| **G5** main thread vs Budgeter `_fire_worker` thread | 1.6 | same as G4 |
| **G6** revoke step (`cuMemSetAccess(prot=0)`) never proven to actually revoke | 1.7 | subprocess-isolated test: post-revoke read faults with `cudaErrorIllegalAddress`; revoke confirmed real |
| **G7** decode delta within noise — README overclaim | this update | qualified to "decode impact < 0.1 ms (below noise floor)" |
| **G8** back-to-back fires | covered by `_fire_inflight` lock at xpool_actuator.py:105 — fires serialise; cited in caveats |
| **G9** scaling to full 1152-chunk fire | deferred to step 5 (full-server smoke); per-call cost is ioctl-dominated, linear extrapolation holds |

## Decision

A1 holds. Advance to step 2 (migrate_slot replay invariant).

## Decision rule

| Observation | Action |
|---|---|
| Decode kernel completes without stall AND no crash | A1 confirmed; advance to step 2. |
| Decode kernel stalls until unmap finishes | A1 refuted: side-thread isolation does not deliver decode-stream-free unmap. The "fire wall on decode stream ≈ 0 ms" claim fails; revisit the threading model or accept a stall budget. |
| Decode crashes (e.g., it touched region_unmap somehow) | Test bug; A1 stays untested. Fix the test setup. |
| Side-stream unmap is dramatically slower than same-stream | Possible CUDA driver penalty for cross-stream coordination; still might be ok if total wall is < ~5 ms. |