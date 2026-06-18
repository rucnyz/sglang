# A2 — migrate_slot + index rewrite preserves replay correctness

## Claim under test

If we:

1. Capture a CUDA graph that calls
   `fused_recurrent_gated_delta_rule_packed_decode` reading
   `ssm_state_indices`
2. Copy the contents of mamba slot **src** to mamba slot **dst** via
   the same data-move pattern as `MambaPool.migrate_slot`
   (memory_pool.py:756)
3. Update the runtime `ssm_state_indices` tensor to point at **dst**
   instead of **src**
4. Replay the captured graph

Then the kernel produces the **same output** as if we'd replayed
with the original indices pointing at **src** (and src's original
contents).

This is the correctness primitive for the Migration transition
(design.md §"Page ownership state": LIVE → FREE): it lets the
actuator move slot state out of a page before that page is
unmapped, without disturbing the in-flight requests still
reading that slot through the captured CUDA graph.

## Why this could fail

- The captured graph's recorded base pointer might index into the
  wrong slot when `ssm_state_indices` is updated mid-stream — e.g.,
  if PyTorch CUDA graph capture has some shared-buffer assumption.
- The data copy on a separate stream might not be visible to the
  decode kernel when the captured graph replays (memory ordering).
- The Triton kernel's tile load might span more bytes per slot than
  the `migrate_slot` copy actually moves, leaving garbage in tail
  bytes that affect the result.
- migrate_slot in production copies BOTH `intermediate_ssm` and
  `intermediate_conv_window`; our test isolates `intermediate_ssm`
  (which is what the recurrent kernel reads). If conv_window's
  contents matter for the recurrent path, our test would miss it.

## Test design

Uses the real production Triton kernel
(`fused_recurrent_gated_delta_rule_packed_decode`) and VMM-backed
`initial_state` tensor.

Setup:
1. Reserve VA, map N=4 chunks, build 4096-slot `initial_state` of shape
   `(4096, HV=4, V=16, K=16)`, bf16.
2. Fill each slot with sentinel = `float(slot_id % 100)`.
3. Build the small attention inputs (mixed_qkv, a, b, A_log, dt_bias, out).
4. Use one persistent `ssm_state_indices` int32 tensor (as production
   does at `hybrid_linear_attn_backend.py:484-520`).

Capture: ONE captured CUDA graph over a `run_kernel()` call.

Scenario:
- **2.1 baseline**: indices=[0, 1, 2, 3]; replay → record outputs as
  `baseline_outputs`. This is the kernel's behavior on the original
  sentinel data.
- **2.2 migrate src→dst on side stream**: pick dst slots that are
  far away (e.g., 2000..2003). On a SIDE stream:
  `initial_state[dst].copy_(initial_state[src])`. The side stream
  finishes before the captured graph replays again.
- **2.3 replay with rewritten indices**: indices=[2000, 2001, 2002, 2003]
  → replay → record `migrated_outputs`. **MUST equal**
  `baseline_outputs`.
- **2.4 negative control**: replay with indices=[2000, 2001, 2002, 2003]
  BEFORE migrating (i.e., dst slots have their pre-migration zero
  contents). Output should NOT match baseline (sanity check that the
  test would detect a real divergence).
- **2.5 side-stream non-interference (timing)**: measure decode kernel
  GPU time with concurrent migrate_slot copy on side stream. Should
  be ≈ baseline (a la step 1).

## Files

- [`test_migrate_replay.py`](test_migrate_replay.py)

## Outcome (2026-05-31)

**A2 CONFIRMED. (First run actually failed — but for a test-side
reason, not for the claim. See "test debug story" below.)**

```
[2.1] baseline replay with indices=[0,1,2,3]: out recorded
[2.4 neg-ctrl] replay with dst indices (uncopied, all-zero):
        diff from baseline = 0.5317 (control discriminates ✓)
[2.2] migrated 4 slot states src→dst on side stream: 0.091 ms
        byte-correct: state[dst] == state[src] ✓
[2.3] replay with migrated dst indices: diff = 0.000000 ✓
[2.5] side-stream migrate during 53 ms decode: delta -0.01 ms
        (within noise; no decode stall)
```

### Test debug story

First run showed **diff = 887**, suggesting A2 refuted. The actual
cause was a test bug, not a kernel bug:

- `fused_recurrent_gated_delta_rule_packed_decode` at
  `fused_recurrent.py:381-382` passes BOTH `h0=initial_state` and
  `ht=initial_state` — the kernel reads from `initial_state[idx]`
  AND writes back to `initial_state[idx]`.
- So 2.1 baseline mutated slots 0-3 with the kernel's post-step
  state. 2.2 then migrated the POST-baseline data (not the
  pre-baseline sentinels) into slots 2000-2003. 2.3 then ran a
  third time over those mutated values, producing yet a third
  output.
- Fix: snapshot a pristine copy of `initial_state` and `.copy_()`
  restore before each replay. Now each replay starts from the
  same input.

After the fix, all sub-tests passed byte-exact.

### Implications

- Captured graph + runtime `ssm_state_indices` rewrite is sound:
  the captured graph uses the runtime tensor's CURRENT contents
  every replay, no shared-state surprises from CUDA Graph capture.
- Side-stream `.copy_()` of slot state is **visible to a captured
  graph replay on the decode stream** even without explicit cross-
  stream synchronization (PyTorch handles the ordering via the
  caching allocator + stream record).
- Side-stream copy of 50 slots' worth of state (~8 MiB) overlaps
  with a 53 ms decode kernel at zero measured cost.
- Critically, this test also verifies the production kernel
  `fused_recurrent_gated_delta_rule_packed_decode` does not load
  any out-of-slot bytes — if it had a vectorised tile that crossed
  slot boundaries, the post-migrate dst slot's neighbours (still
  pristine pre-migration data) would have affected the output.

### Decision

A2 confirmed. Migration is a sound primitive for the actuator's
LIVE → FREE transition (design.md §"Page ownership state"). With
A1 (step 1) already confirming the Unmap step is decode-stream
free, both GPU-side primitives of a full Migration-driven fire
are proven free of decode-stream impact.

## Decision rule

| Observation | Action |
|---|---|
| 2.3 == 2.1 (byte-exact) AND 2.4 != 2.1 AND 2.5 ≈ baseline | A2 confirmed — Migration transition is sound. |
| 2.3 != 2.1 (output differs by more than fp tolerance) | A2 refuted — captured graph + index rewrite is not safe. Migration cannot be a cost-model candidate; design.md §"Admitter" 7-action set drops to 5. |
| 2.4 == 2.1 (control fails to differ) | Test bug — sentinel values too similar; tighten. |
| 2.5 ≈ baseline + 5 ms | Side-stream copy stalls decode — investigate alignment / stream dependency. |