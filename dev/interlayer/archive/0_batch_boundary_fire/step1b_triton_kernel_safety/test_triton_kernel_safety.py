"""Step 1b — Triton block-load kernel safety under VA hole.

Real CUDA + real production Triton kernel
(`fused_recurrent_gated_delta_rule_packed_decode_kernel` at
`python/sglang/srt/layers/attention/fla/fused_recurrent.py:186-265`).

Hypothesis under test: Triton's `tl.load(p_h0, mask=mask_h, other=0)`
of a [BV, BK] tile is index-gated by `state_idx`, so as long as
`ssm_state_indices` points only to mapped slots, replaying a
captured graph after a `cuMemUnmap` of an OTHER slot is safe.

Falsification scenarios:
- Triton prefetches / vectorizes outside the masked region → fault
  even though `mask_h` excludes the unmapped offsets.
- Triton's address calculation does a single big load that spans
  the unmapped page boundary regardless of mask → fault.
- Triton's compile-time tile sizing causes the load to span multiple
  slots → fault even when the indexed slot is mapped.

If any of those scenarios trigger, the test at step 1b.3 (safe
indices, after unmap, real Triton kernel) will fail with
`cudaErrorIllegalAddress`. That falsifies A1 for the production
kernel class.

Run:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python \\
    dev/interlayer/0_batch_boundary_fire/step1b_triton_kernel_safety/test_triton_kernel_safety.py
"""
from __future__ import annotations

import ctypes
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

import sglang.srt.arena.chunk_arena as ca
from sglang.srt.arena.from_blob_ext import tensor_from_va
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

DEVICE = 0
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()
ca.CUDA.cuInit(0)


# ---- CUDA primitives (same pattern as step 1) ----

def alloc_handles(n: int) -> list[int]:
    prop = ca._CUmemAllocationProp()
    prop.type = ca.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = DEVICE
    out = []
    for _ in range(n):
        h = ctypes.c_ulonglong()
        ca._check(
            ca.CUDA.cuMemCreate(ctypes.byref(h), CHUNK_BYTES, ctypes.byref(prop), 0),
            "cuMemCreate",
        )
        out.append(h.value)
    return out


def reserve_va(total_bytes: int) -> int:
    va = ctypes.c_ulonglong()
    ca._check(
        ca.CUDA.cuMemAddressReserve(ctypes.byref(va), total_bytes, CHUNK_BYTES, 0, 0),
        "cuMemAddressReserve",
    )
    return va.value


def map_handle(va_base: int, slot: int, handle: int) -> None:
    va = va_base + slot * CHUNK_BYTES
    ca._check(ca.CUDA.cuMemMap(va, CHUNK_BYTES, 0, handle, 0), "cuMemMap")


def unmap_chunk(va_base: int, slot: int) -> None:
    va = va_base + slot * CHUNK_BYTES
    ca._check(ca.CUDA.cuMemUnmap(va, CHUNK_BYTES), "cuMemUnmap")


def set_access(va_base: int, n_slots: int) -> None:
    desc = ca._CUmemAccessDesc()
    desc.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = DEVICE
    desc.flags = 3
    ca._check(
        ca.CUDA.cuMemSetAccess(
            va_base, n_slots * CHUNK_BYTES, ctypes.byref(desc), 1
        ),
        "cuMemSetAccess",
    )


# ---- Kernel sizing ----
# Per-slot mamba state = HV × V × K × bf16. We choose dims that give
# multiple slots per 2 MiB chunk so the unmap unit is meaningful.
#
# HV=4, V=16, K=16, bf16 → 4*16*16*2 = 2048 bytes = 2 KiB per slot
# 2 MiB chunk / 2 KiB per slot = 1024 slots per chunk.

H = 2            # number of heads (must satisfy HV % H == 0 in kernel)
HV = 4
V = 16
K = 16
DTYPE = torch.bfloat16
DTYPE_BYTES = 2

BYTES_PER_SLOT = HV * V * K * DTYPE_BYTES   # 2048
CHUNK_BYTES = 2 * 1024 * 1024                # 2 MiB
SLOTS_PER_CHUNK = CHUNK_BYTES // BYTES_PER_SLOT  # 1024

N_CHUNKS = 4
N_SLOTS = N_CHUNKS * SLOTS_PER_CHUNK         # 4096


def build_initial_state_vmm() -> tuple[int, list[int], torch.Tensor]:
    """Reserve VA + map N_CHUNKS chunks; return VA base, handles list,
    and a torch tensor view of shape (N_SLOTS, HV, V, K) over the VA.
    Each slot's state lives at a fixed offset; slot s sits in chunk
    s // SLOTS_PER_CHUNK."""
    handles = alloc_handles(N_CHUNKS)
    va_base = reserve_va(N_CHUNKS * CHUNK_BYTES)
    for slot_chunk in range(N_CHUNKS):
        map_handle(va_base, slot_chunk, handles[slot_chunk])
    set_access(va_base, N_CHUNKS)
    n_bf16 = (N_CHUNKS * CHUNK_BYTES) // DTYPE_BYTES
    flat = tensor_from_va(va_base, (n_bf16,), DTYPE, DEVICE)
    state = flat.view(N_SLOTS, HV, V, K)
    return va_base, handles, state


def main() -> int:
    print("=== Step 1b: Triton kernel safety under VA hole ===\n")
    print(f"Kernel: fused_recurrent_gated_delta_rule_packed_decode")
    print(f"Sizing: HV={HV} V={V} K={K} dtype={DTYPE} "
          f"per-slot={BYTES_PER_SLOT}B  N_SLOTS={N_SLOTS} "
          f"({SLOTS_PER_CHUNK}/chunk × {N_CHUNKS} chunks)\n")

    va_base, handles, initial_state = build_initial_state_vmm()
    # Fill each slot's state with sentinel = slot_id (in bf16 — note
    # bf16 only has ~7 bit mantissa for ints, so use values < 128).
    for s in range(min(N_SLOTS, 128)):
        initial_state[s].fill_(float(s % 100))
    # Higher slots get 0 — we'll only index into the first few anyway.
    torch.cuda.synchronize()
    print(f"[setup] initial_state at 0x{initial_state.data_ptr():x}, "
          f"shape={tuple(initial_state.shape)}, "
          f"stride[0]={initial_state.stride(0)} (= HV*V*K bf16 elements)")

    # Allocate other inputs.
    B = 4
    qk_dim = H * K
    qkv_dim = qk_dim * 2 + HV * V
    mixed_qkv = torch.zeros(B, qkv_dim, dtype=DTYPE, device=f"cuda:{DEVICE}")
    a = torch.zeros(B, HV, dtype=DTYPE, device=f"cuda:{DEVICE}")
    b = torch.zeros(B, HV, dtype=DTYPE, device=f"cuda:{DEVICE}")
    A_log = torch.zeros(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    dt_bias = torch.zeros(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    out = torch.zeros(B, 1, HV, V, dtype=DTYPE, device=f"cuda:{DEVICE}")

    # Persistent ssm_state_indices tensor — same pattern as production
    # `state_indices_list[bs-1]` (`hybrid_linear_attn_backend.py:484`).
    ssm_state_indices = torch.zeros(B, dtype=torch.int32, device=f"cuda:{DEVICE}")

    def run_kernel():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=1.0,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=False,
        )

    # Warm up + JIT compile before capture.
    ssm_state_indices.copy_(
        torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=f"cuda:{DEVICE}")
    )
    run_kernel()
    torch.cuda.synchronize()
    baseline_out = out.clone()
    print(f"[1b.1] eager run with ssm_state_indices=[0,1,2,3]: "
          f"out shape OK, no fault ✓")

    # ---- Capture CUDA graph ----
    stream = torch.cuda.Stream(device=DEVICE)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            run_kernel()
    torch.cuda.synchronize()
    print(f"[1b.2] captured CUDA graph")

    # Baseline replay (no unmap yet) — must reproduce eager result.
    ssm_state_indices.copy_(
        torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=f"cuda:{DEVICE}")
    )
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, baseline_out), \
        "graph replay diverged from eager BEFORE unmap"
    print(f"[1b.3] baseline replay matches eager ✓")

    # ---- Unmap chunk 2 (kills slots [2*SLOTS_PER_CHUNK,
    #                                  3*SLOTS_PER_CHUNK))
    # i.e., slots [2048, 3072). Safe slots: 0..2047 (in chunks 0-1)
    # plus 3072..4095 (in chunk 3).
    torch.cuda.synchronize()
    unmap_chunk(va_base, 2)
    print(f"[1b.4] cuMemUnmap'd chunk 2 → slots "
          f"[{2*SLOTS_PER_CHUNK}, {3*SLOTS_PER_CHUNK}) are now in "
          f"unmapped VA")

    # ---- Step 1b.5: replay with SAFE indices (none in chunk 2) ----
    # Use slots [0, 1, 2, 3072]. Slot 3072 is in chunk 3 (mapped).
    safe_indices = torch.tensor([0, 1, 2, 3072], dtype=torch.int32,
                                  device=f"cuda:{DEVICE}")
    ssm_state_indices.copy_(safe_indices)
    safe_crashed = False
    try:
        g.replay()
        torch.cuda.synchronize()
    except Exception as e:
        safe_crashed = True
        print(f"[1b.5] SAFE replay (indices=[0,1,2,3072]) CRASHED: "
              f"{type(e).__name__}: {str(e)[:200]}")
        print()
        print("=== A1 REFUTED FOR TRITON ===")
        print("Triton's tl.load with mask_h is NOT enough to prevent "
              "the fault — the captured graph faults even when "
              "state_indices points to mapped slots. Likely cause: "
              "Triton's block load reads ACROSS the slot stride into "
              "an adjacent unmapped page. Option G is NOT viable for "
              "this kernel class. Revisit: either custom non-vectorized "
              "load kernel, or Option C (per-slot snapshot).")
        return 2
    print(f"[1b.5] SAFE replay (indices=[0,1,2,3072]) after chunk 2 "
          f"unmap: no crash ✓")

    # Sanity: the output for slot 3072 should differ from the eager
    # baseline (we put 0 in slot 3072 — slot < 100 sentinel). But
    # since the kernel does math on mixed_qkv (all zeros) and state
    # (zeros for slot 3072), the result is just dependent on the
    # algebra; we don't compare values here — only that no crash.

    # ---- Step 1b.6: replay with UNSAFE indices (one in chunk 2) ----
    # MUST be LAST step — crashes corrupt the CUDA context.
    unsafe_indices = torch.tensor([0, 1, 2500, 3072], dtype=torch.int32,
                                    device=f"cuda:{DEVICE}")
    ssm_state_indices.copy_(unsafe_indices)
    unsafe_crashed = False
    try:
        g.replay()
        torch.cuda.synchronize()
        print(f"[1b.6] WARNING: UNSAFE replay (idx 2500 in unmapped chunk "
              f"2) did NOT crash. Triton may be silently padding via "
              f"mask_h. Inconclusive — would not detect production bug.")
        return 3
    except Exception as e:
        unsafe_crashed = True
        print(f"[1b.6] UNSAFE replay (idx 2500 in unmapped chunk 2) "
              f"CRASHED as expected: {type(e).__name__}: {str(e)[:200]}")

    print()
    print("=== A1 (Triton block-load kernel path) CONFIRMED ===")
    print(f"`fused_recurrent_gated_delta_rule_packed_decode` was replayed")
    print(f"successfully via a captured CUDA graph after one of its "
          f"backing chunks was cuMemUnmap'd, provided `ssm_state_indices` "
          f"excluded slots whose chunk was the unmapped one. Indexing into "
          f"the unmapped chunk reproducibly crashed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
