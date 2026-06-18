# Step 1b — Triton block-load kernel safety

## Why this exists

Audit of step 1 (`subagent`, 2026-05-31) flagged:

> "G1 (biggest gap): toy kernel uses `torch.gather` of 1 element;
> production uses Triton **block load `tl.load(p_h0, mask=mask_h)`**
> of `[BV,BK]≈[32,256]` floats per `(state_idx, hv)` from
> `fused_recurrent_gated_delta_rule_packed_decode_kernel`
> (`fused_recurrent.py:230-232`). If Triton prefetches/vectorizes
> ACROSS a slot stride boundary into a hole — even though `state_idx`
> itself points to a mapped slot — kernel faults. Slot stride is
> `HV*V*K * 4B`, easily MBs; one bad page anywhere in the slot would
> fault."

Step 1 proved the invariant for the `torch.gather` kernel class.
Step 1b is the **focused empirical test** of the same invariant for
the production Triton kernel.

## Claim under test

Same as step 1's A1 with the kernel substituted:

> If a `cuMemUnmap` lands BETWEEN two captured CUDA graph replays
> that call `fused_recurrent_gated_delta_rule_packed_decode`, AND
> the runtime `ssm_state_indices` tensor excludes any slot whose
> chunk has been unmapped, the SECOND replay completes without
> `cudaErrorIllegalAddress`.

Code reading (`fused_recurrent.py:222-232`):

```python
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
...
p_h0 = h0 + state_idx * stride_init_state_token
p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
```

`state_idx` is a runtime scalar. All subsequent loads are at addresses
derived from `state_idx`. **In principle** this should be safe as long
as the slot indexed by `state_idx` is fully mapped. But Triton's
vectorized tile load (BV × BK = 32 × 128 typically = 16 KB) could
have hardware behaviors (HBM page-table walks, masked-but-fetched
prefetch, etc.) that fault on adjacent unmapped pages even with
mask_h excluding them. The only way to know is to run it.

## Test design

Use the REAL `fused_recurrent_gated_delta_rule_packed_decode`
function from `python/sglang/srt/layers/attention/fla/fused_recurrent.py`.
Allocate `initial_state` as a **VMM-backed tensor** so we can call
`cuMemUnmap` on individual slots.

Parameters (mirror Qwen3-Next sizing where possible):
- HV = 4, V = 16, K = 16 (small so test is fast)
- dtype = bfloat16
- Per-slot state = `HV * V * K * 2 = 2048 bytes` → many slots per chunk
- chunk_bytes = 2 MiB → slots_per_chunk = 1024

Setup:
1. Reserve VA for N=4 chunks
2. Map all 4 chunks
3. Build `initial_state` as `tensor_from_va(va_base, (n_slots, HV, V, K), bf16)`
4. Fill slots 0..n_slots-1 with known sentinels
5. Allocate small `mixed_qkv`, `a`, `b`, `A_log`, `dt_bias`, `out` tensors

Capture a CUDA graph that calls the real function with `ssm_state_indices`
pointing to slots 0, 1, 2, 3 (one per batch element). Verify output
sentinel propagates correctly (baseline 1.1).

Then:
- 1.2: cuMemUnmap chunk 2 (kills slots in chunk 2)
- 1.3: replay graph with `ssm_state_indices` = slots from chunks 0, 1, 3 → must NOT crash
- 1.4 (negative control, LAST step): replay with `ssm_state_indices` including a slot from chunk 2 → expect crash

## Files

- [`test_triton_kernel_safety.py`](test_triton_kernel_safety.py) —
  written and runnable.

## Outcome (2026-05-31)

**A1 CONFIRMED for Triton block-load kernel path.**

```
[1b.1] eager run with ssm_state_indices=[0,1,2,3]: out shape OK ✓
[1b.2] captured CUDA graph
[1b.3] baseline replay matches eager ✓
[1b.4] cuMemUnmap'd chunk 2 → slots [2048, 3072) unmapped
[1b.5] SAFE replay (indices=[0,1,2,3072]) after chunk 2 unmap: no crash ✓
[1b.6] UNSAFE replay (idx 2500 in unmapped chunk 2) CRASHED as expected ✓
```

The production Triton kernel `fused_recurrent_gated_delta_rule_packed_decode`
(the EXACT kernel from the live D10@C=56 N=3 v3 crash stack trace)
was:
1. Captured into a CUDA graph
2. Replayed once normally → matches eager
3. One backing chunk cuMemUnmap'd
4. Replayed AGAIN with `ssm_state_indices` excluding slots in the
   unmapped chunk → **no crash, correct semantics**
5. Replayed with an unsafe index → crashes as expected (proves the
   test would have caught a fault if one had occurred at step 4)

This refutes the audit's G1 concern: Triton's `tl.load(p_h0,
mask=mask_h, other=0)` does NOT vectorize / prefetch across the
slot stride boundary in a way that touches adjacent unmapped pages.
The `mask_h` derived from `mask_v[:, None] & mask_k[None, :]` (line
220) is honored — pages outside the masked region are not read.

## Decision

A1 holds for both the toy `torch.gather` kernel (step 1) and the
production `fused_recurrent_gated_delta_rule_packed_decode` kernel
(this step). **Advance to step 2** (measure scheduler batch-boundary
idle gap in real workload).

## Decision rule

- All assertions pass → step 1 + 1b complete, A1 fully confirmed for
  production kernel class. **Advance to step 2.**
- Step 1.3 crashes (safe state_indices but kernel still touches
  unmapped page) → A1 refuted for Triton. **G is not viable** as
  proposed. Must revisit:
  - (a) Whether `mask_h` is honored by Triton's load — if not, we
    need either a custom kernel that does element-by-element load
    (no vectorize), OR Option C (per-slot snapshot) becomes the only
    safe path.
  - (b) Whether unmap targeting can be made chunk-aligned so the
    kernel's prefetch never crosses an unmap boundary.
- Step 1.4 does NOT crash → unsafe access silently returns garbage
  via mask=0 fallback — surprising but not catastrophic (would mean
  the captured graph survives but reads wrong data; correctness
  problem rather than crash). Needs separate test.
