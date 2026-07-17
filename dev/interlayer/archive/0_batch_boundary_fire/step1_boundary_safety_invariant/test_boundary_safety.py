"""Step 1 — boundary safety invariant.

Real CUDA + torch only. No mocks. No sglang state machine. Proves
(or refutes) that with a runtime-supplied index tensor that excludes
unmapped positions, a captured CUDA graph can be replayed safely
even after the underlying VA has been partially unmapped.

Run:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python \\
    dev/interlayer/0_batch_boundary_fire/step1_boundary_safety_invariant/test_boundary_safety.py

Exit 0 = invariant confirmed. Non-zero = invariant refuted; see
README's decision rule.
"""
from __future__ import annotations

import ctypes
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

import sglang.srt.arena.chunk_arena as ca
from sglang.srt.arena.from_blob_ext import tensor_from_va

DEVICE = 0
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()
ca.CUDA.cuInit(0)


# ---- raw CUDA helpers (copy of pattern in bench/) ----

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
        ca.CUDA.cuMemSetAccess(va_base, n_slots * CHUNK_BYTES, ctypes.byref(desc), 1),
        "cuMemSetAccess",
    )


# ---- test params ----

N_CHUNKS = 4
CHUNK_BYTES = 2 * 1024 * 1024  # 2 MiB

# Each "slot" we'll index has size = CHUNK_BYTES / 4 floats. Within a
# slot, the kernel reads a contiguous span; this mirrors mamba's
# per-slot state vector. The runtime `indices` tensor picks WHICH
# slots to read.


def main() -> int:
    print("=== Step 1: boundary safety invariant ===\n")

    # ---- Setup: reserve VA, map 4 chunks, fill with known data ----
    handles = alloc_handles(N_CHUNKS)
    va_base = reserve_va(N_CHUNKS * CHUNK_BYTES)
    for slot in range(N_CHUNKS):
        map_handle(va_base, slot, handles[slot])
    set_access(va_base, N_CHUNKS)

    n_floats_total = (N_CHUNKS * CHUNK_BYTES) // 4
    state = tensor_from_va(va_base, (n_floats_total,), torch.float32, DEVICE)
    # Initialise each slot with a sentinel value equal to its slot id
    # (so we can tell which slot the kernel actually read).
    floats_per_slot = CHUNK_BYTES // 4
    for slot in range(N_CHUNKS):
        state[slot * floats_per_slot : (slot + 1) * floats_per_slot] = float(slot)
    torch.cuda.synchronize()
    print(f"[setup] reserved VA at 0x{va_base:x}, mapped 4 chunks, each "
          f"filled with its slot id")

    # ---- Build a captured CUDA graph that reads indexed slots ----
    # The kernel is `state.gather(0, indices_in_floats)` — for each
    # slot id i in `indices`, read state[i * floats_per_slot] (just the
    # first float of the slot is enough to demonstrate access).
    indices = torch.zeros(2, dtype=torch.int64, device=f"cuda:{DEVICE}")
    out = torch.zeros(1, dtype=torch.float32, device=f"cuda:{DEVICE}")

    # The captured ops: scale indices to float offsets, gather, sum.
    # Using torch.gather over a flattened view: indices_in_floats =
    # indices * floats_per_slot.
    def kernel_op():
        idx_in_floats = indices * floats_per_slot
        out.copy_(state.gather(0, idx_in_floats).sum().reshape(1))

    # Warm-up the kernel + JIT before capture
    indices.copy_(torch.tensor([0, 2], dtype=torch.int64, device=f"cuda:{DEVICE}"))
    kernel_op()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream(device=DEVICE)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            kernel_op()
    torch.cuda.synchronize()
    print(f"[capture] graph captured. Kernel reads state.gather(indices * "
          f"{floats_per_slot}).sum()")

    # ---- Step 1.1: baseline replay with indices [0, 2] (safe slots) ----
    indices.copy_(torch.tensor([0, 2], dtype=torch.int64, device=f"cuda:{DEVICE}"))
    g.replay()
    torch.cuda.synchronize()
    got = out.item()
    expected = 0.0 + 2.0  # slot 0's sentinel + slot 2's sentinel
    assert got == expected, f"baseline: got {got}, expected {expected}"
    print(f"[1.1] baseline replay (indices=[0,2]): out={got} == {expected} ✓")

    # ---- Step 1.2: unmap slot 1 — between replays, no graph in flight ----
    torch.cuda.synchronize()  # drain (this is the "G window")
    unmap_chunk(va_base, 1)
    print(f"[1.2] cuMemUnmap'd slot 1. Slot 1's VA range "
          f"[0x{va_base + CHUNK_BYTES:x}..0x{va_base + 2*CHUNK_BYTES:x}) "
          f"is now unbacked.")

    # ---- Step 1.3: replay with SAFE indices [0, 3] — slot 1 not in indices ----
    # If A1 holds, the kernel reads only slots 0 and 3 (both still
    # mapped) and we get sentinel_0 + sentinel_3 = 0 + 3 = 3.
    indices.copy_(torch.tensor([0, 3], dtype=torch.int64, device=f"cuda:{DEVICE}"))
    safe_crashed = False
    safe_got = None
    try:
        g.replay()
        torch.cuda.synchronize()
        safe_got = out.item()
    except Exception as e:
        safe_crashed = True
        print(f"[1.3] REPLAY CRASHED with safe indices [0,3]: "
              f"{type(e).__name__}: {str(e)[:200]}")
    if not safe_crashed:
        expected_safe = 0.0 + 3.0
        assert safe_got == expected_safe, (
            f"safe indices replay returned wrong value: got {safe_got}, "
            f"expected {expected_safe}"
        )
        print(f"[1.3] safe-indices replay (indices=[0,3]) after unmap: "
              f"out={safe_got} == {expected_safe} ✓ — **A1 SUPPORTED**")

    # NOTE: deliberate unsafe access (cudaErrorIllegalAddress) poisons
    # the CUDA context for the rest of the process — no more CUDA ops
    # work after it. So we run all the POSITIVE sub-tests first, and
    # save the negative controls for the very end.

    if not safe_crashed:

        # ---- Step 1.5: remap-then-replay (audit G5) ------------------
        # cap_barrier in production unmaps SRC chunks, then maps NEW
        # handles to those same VA positions for DST. Make sure that
        # a captured graph whose pointer is stable across an unmap +
        # map cycle sees the NEW chunk's contents (not stale bytes).
        new_handle = alloc_handles(1)[0]
        map_handle(va_base, 1, new_handle)
        set_access(va_base, N_CHUNKS)
        state[1 * floats_per_slot : 2 * floats_per_slot] = 99.0
        torch.cuda.synchronize()
        indices.copy_(torch.tensor([1, 3], dtype=torch.int64,
                                    device=f"cuda:{DEVICE}"))
        g.replay()
        torch.cuda.synchronize()
        expected_remapped = 99.0 + 3.0
        assert out.item() == expected_remapped, (
            f"remap-then-replay: got {out.item()}, expected {expected_remapped}. "
            f"Captured graph may have cached contents instead of address."
        )
        print(f"[1.5] remap-then-replay (indices=[1,3], slot 1 has new "
              f"handle, contents=99): out={out.item()} == {expected_remapped} ✓")

        # ---- Step 1.6: chunk-holds-many-slots (audit G4) -------------
        # Production chunk_bytes is 32 MiB but a chunk holds many slots
        # (slots_per_chunk = chunk_bytes / per_token_bytes). The
        # invariant we need is: unmapping chunk K kills ALL slots in K,
        # and the test must verify the captured graph can still touch
        # other-chunk slots safely. Re-parameterize over the existing
        # VA with logical slots smaller than physical chunks.
        SLOTS_PER_CHUNK = 4
        floats_per_logical_slot = (CHUNK_BYTES // 4) // SLOTS_PER_CHUNK
        # Re-init sentinels at logical-slot granularity. Slot s sits in
        # chunk s // SLOTS_PER_CHUNK.
        for ls in range(N_CHUNKS * SLOTS_PER_CHUNK):
            state[ls * floats_per_logical_slot:
                  (ls + 1) * floats_per_logical_slot] = float(ls)
        torch.cuda.synchronize()

        indices2 = torch.zeros(2, dtype=torch.int64, device=f"cuda:{DEVICE}")
        out2 = torch.zeros(1, dtype=torch.float32, device=f"cuda:{DEVICE}")

        def kernel_op2():
            idx_floats = indices2 * floats_per_logical_slot
            out2.copy_(state.gather(0, idx_floats).sum().reshape(1))

        indices2.copy_(torch.tensor([0, 3], dtype=torch.int64,
                                     device=f"cuda:{DEVICE}"))
        kernel_op2()
        torch.cuda.synchronize()

        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(g2, stream=stream):
                kernel_op2()
        torch.cuda.synchronize()

        # Unmap chunk 2 (originally slot 2 in 1.2 was already unmapped
        # then remapped in 1.5; now we use chunk 2 which is still
        # pristine). That kills logical slots [8, 9, 10, 11].
        unmap_chunk(va_base, 2)
        # Safe indices: logical slots both in chunks 0 or 3.
        # slot 0 → chunk 0 (mapped). slot 3 → chunk 0 (mapped, ls=3 is
        # the last slot of chunk 0 since SLOTS_PER_CHUNK=4).
        indices2.copy_(torch.tensor([0, 3], dtype=torch.int64,
                                     device=f"cuda:{DEVICE}"))
        g2.replay()
        torch.cuda.synchronize()
        expected_cs = 0.0 + 3.0
        assert out2.item() == expected_cs, (
            f"chunk-slot safe replay: got {out2.item()}, expected {expected_cs}"
        )
        print(f"[1.6] chunk-slot safe replay (chunk 2 unmapped, indices "
              f"[0,3] in chunk 0): out={out2.item()} == {expected_cs} ✓")

        # Negative control for chunk-slot variant: index a logical slot
        # in the unmapped chunk.
        indices2.copy_(torch.tensor([8, 13], dtype=torch.int64,
                                     device=f"cuda:{DEVICE}"))
        chunk_unsafe_crashed = False
        try:
            g2.replay()
            torch.cuda.synchronize()
            print(f"[1.7] WARNING: chunk-slot UNSAFE replay (idx 8 → "
                  f"unmapped chunk 2) did NOT crash. out={out2.item()}. "
                  f"Stale cache?")
            return 4
        except Exception as e:
            chunk_unsafe_crashed = True
            print(f"[1.7] chunk-slot UNSAFE replay (indices=[8,13], slot 8 "
                  f"in unmapped chunk 2) CRASHED as expected: "
                  f"{type(e).__name__}: {str(e)[:140]}")

        # If we got here: 1.3 + 1.4 + 1.5 + 1.6 + 1.7 all confirm the
        # invariant under three perturbations: index-only, remap+replay,
        # and chunk-grouped-slots. That's the full positive confirmation
        # for the TORCH.GATHER code path. Triton block-load is still an
        # open question — see step1b/.
        print()
        print("=== A1 (torch.gather kernel path) CONFIRMED ===")
        print("Captured CUDA graphs can be safely replayed across a "
              "cuMemUnmap, provided the runtime index tensor excludes any "
              "slot whose chunk has been unmapped. Additional invariants "
              "verified: (1.5) post-fire dst remap is visible via the "
              "same captured pointer; (1.6) chunk-grouped slots — unmap "
              "of chunk K kills all slots in K, others stay safe; (1.7) "
              "indexing into unmapped chunk faults as expected.")
        print()
        print("Implication for G: if the scheduler fires at a batch "
              "boundary (no graph in flight) AND the cap_barrier removes "
              "unmapped slots from the allocator's free list BEFORE the "
              "next batch builds its state_indices_list, no captured-graph "
              "replay can crash IN THIS KERNEL CLASS (torch.gather).")
        print()
        print("OPEN: step1b/ must verify the same for the real Triton "
              "block-load kernel (`fused_recurrent_gated_delta_rule_packed"
              "_decode`) which has prefetch / vectorization semantics not "
              "exercised here.")
        return 0

    # Step 1.3 crashed — A1 is refuted.
    print()
    print("=== A1 REFUTED ===")
    print("Even with safe indices [0, 3], the captured graph crashed on "
          "replay-after-unmap. This means the captured graph touches MORE "
          "than just the indexed slots (possibly: bounds check, prefetch, "
          "or Triton-side pointer validation against the full VA range). "
          "G as proposed will NOT work. We must either (i) make the "
          "captured graph only touch indexed positions, or (ii) switch to "
          "C with full per-fire snapshot.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
