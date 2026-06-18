# Step 1 — boundary safety invariant

## Claim (A1)

If a `cuMemUnmap` lands BETWEEN two captured CUDA graph replays AND
the consuming `state_indices` tensor is updated AFTER the unmap to
exclude the unmapped slot, the SECOND replay completes without
`cudaErrorIllegalAddress`.

Equivalently: **the captured graph's recorded base pointer + the
runtime index tensor is enough to avoid the crash**, as long as no
graph is in flight during the unmap AND the index tensor doesn't
reference any unmapped slot at the next replay.

This is the core hypothesis of Option G. Step 1 isolates and proves it
in pure CUDA primitives, with no sglang state machine involved.

## Why this matters

The negative version of this claim is already empirically proven by
[`../../bench/bench_graph_unmap_race.py`](../../bench/bench_graph_unmap_race.py):
unmap → replay the SAME graph that touches the unmapped VA → crash.

What step 1 must prove is the POSITIVE version: **with a runtime
index tensor that selectively reads only mapped positions, the
second replay is safe even though the underlying VA has a hole in
it**. If this fails, the entire G approach is dead and we have to
revisit the architecture before any further work.

## Test design (no mocks)

The test uses raw CUDA via sglang's ctypes wrappers
(`python/sglang/srt/arena/chunk_arena.py`). No sglang state machine,
no mocks, no stubs. Just CUDA primitives + torch tensors backed by
real VMM memory.

Scenario:
1. Reserve VA for N=4 chunks of 2 MiB each.
2. Map physical chunks at slots 0..3.
3. Build a `state` tensor over the whole VA (similar to mamba's
   `intermediate_ssm`).
4. Write known sentinel values into each slot.
5. Capture a CUDA graph that does `out = state[indices].sum()`
   where `indices` is a small int64 tensor (e.g., size 2). This
   mirrors mamba's runtime-supplied `state_indices_list[bs-1]`.
6. Replay the graph with `indices = [0, 2]` (safe positions);
   verify the result.
7. Without re-capturing: replay again with `indices = [0, 3]` after
   physically `cuMemUnmap`ing slot 1. Verify the result is correct
   AND no crash occurs. (Slot 1 is unmapped, but indices [0, 3] do
   NOT reference it.)
8. Then update `indices = [0, 3]` to be `[1, 3]` and replay — expect
   the crash to reappear, since index 1 now points to an unmapped
   slot. This is the NEGATIVE control proving the index, not the
   graph capture, is the gating factor.

## Files

- [`test_boundary_safety.py`](test_boundary_safety.py) — runnable
  test. Exit 0 = invariant holds.

## How to run

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=3 .venv/bin/python \
  dev/interlayer/0_batch_boundary_fire/step1_boundary_safety_invariant/test_boundary_safety.py
```

## Outcome (2026-05-31, deepened after audit)

A subagent audit (`subagent` report, 2026-05-31) flagged that the
initial version of this test (only `torch.gather` over a single VA,
no remap, no chunk-grouped slots) was a "happy path" that could
pass while production still crashes. Five gaps were identified;
three (G2/G4/G5) were patched inline; the most important (G1, real
Triton kernel) is the subject of step1b/.

Final run after deepening:

```
[1.1] baseline replay (indices=[0,2]): out=2.0 == 2.0 ✓
[1.2] cuMemUnmap'd slot 1
[1.3] safe-indices replay (indices=[0,3]) after unmap: out=3.0 ✓
[1.5] remap-then-replay (slot 1 remapped, contents=99): out=102.0 ✓
[1.6] chunk-grouped safe replay (chunk 2 unmapped, indices in chunk 0): out=3.0 ✓
[1.7] chunk-grouped UNSAFE replay (index in unmapped chunk) CRASHED ✓
```

Five properties proven for **torch.gather-class kernels**:

1. Captured CUDA graph + runtime gather is **index-gated**: the
   graph does NOT eagerly validate the captured VA span; it only
   faults on actual dereference of unmapped offset.
2. `cuMemUnmap` does not invalidate the captured graph wholesale —
   it remains replayable; what matters is the runtime input.
3. **Remap-then-replay sees NEW contents** via the SAME captured
   pointer (1.5: slot 1 remapped to handle with sentinel 99,
   captured graph reads 99 via the same `state` tensor address).
4. **Chunk-grouped slots**: unmap of chunk K kills ALL logical
   slots backed by chunk K; logical slots in other chunks stay
   safe (1.6 with SLOTS_PER_CHUNK=4).
5. **Crashes are reproducible by intent** (1.7) — confirms tests
   aren't accidentally avoiding faults via cache or padding.

## Known open: G1 — Triton block-load kernel

The production crash happens inside
`fused_recurrent_gated_delta_rule_packed_decode_kernel`
(`python/sglang/srt/layers/attention/fla/fused_recurrent.py:186-265`).
That kernel does `tl.load(p_h0, mask=mask_h, other=0)` over a
[BV, BK] tile (line 232). Reading the code, the load is index-
gated by `state_idx` (line 222) so the invariant SHOULD extend —
but Triton's vectorized load semantics need empirical verification.
Step 1b covers this.

## Decision

torch.gather path: invariant confirmed. **Do NOT advance to step 2
until step 1b verifies the same for the real Triton kernel.**

## Decision rule

- **All assertions pass** → assumption A1 confirmed, advance to step 2.
- **Step 7 (safe-indices replay-after-unmap) crashes** → A1
  refuted; the captured graph reads MORE memory than just the
  indexed positions (e.g., due to prefetch or Triton's bounds
  check). G is then NOT viable as proposed; we need to revisit
  (likely C with index rewrite for full coverage, or D / E).
- **Step 8 (unsafe-indices) does NOT crash** → unexpected, suggests
  CUDA tolerates unmapped reads silently (possible with caching).
  Step 1 needs revision.
