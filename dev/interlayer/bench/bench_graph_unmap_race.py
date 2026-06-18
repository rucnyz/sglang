"""Reproduce the CUDA graph + cuMemUnmap crash in minimal form.

Setup:
  1. Reserve VA, map chunks to populate a tensor
  2. Capture a CUDA graph that reads from that tensor (Triton-like kernel)
  3. While the graph is "replayable" (capture is done), unmap a chunk
  4. Replay the graph
  5. Observe: does it crash? What's the error?

This isolates the "captured graph + unmap" interaction from all sglang
state — pure CUDA / torch primitives.

Usage:
  .venv/bin/python dev/interlayer/bench/bench_graph_unmap_race.py
"""
from __future__ import annotations

import ctypes
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import sglang.srt.arena.chunk_arena as ca

import torch

DEVICE = 0
CHUNK_BYTES = 2 * 1024 * 1024
N_CHUNKS = 4  # small for demo
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()
ca.CUDA.cuInit(0)


def alloc_handles(n):
    prop = ca._CUmemAllocationProp()
    prop.type = ca.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = DEVICE
    out = []
    for _ in range(n):
        h = ctypes.c_ulonglong()
        ca._check(ca.CUDA.cuMemCreate(ctypes.byref(h), CHUNK_BYTES,
                                       ctypes.byref(prop), 0),
                  "cuMemCreate")
        out.append(h.value)
    return out


def reserve_va(total_bytes):
    va = ctypes.c_ulonglong()
    ca._check(ca.CUDA.cuMemAddressReserve(
        ctypes.byref(va), total_bytes, CHUNK_BYTES, 0, 0),
        "cuMemAddressReserve")
    return va.value


def map_handle(va_base, slot, handle):
    va = va_base + slot * CHUNK_BYTES
    ca._check(ca.CUDA.cuMemMap(va, CHUNK_BYTES, 0, handle, 0), "cuMemMap")


def set_access(va_base, n_slots):
    desc = ca._CUmemAccessDesc()
    desc.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = DEVICE
    desc.flags = 3
    ca._check(ca.CUDA.cuMemSetAccess(
        va_base, n_slots * CHUNK_BYTES, ctypes.byref(desc), 1),
        "cuMemSetAccess")


def unmap_chunk(va_base, slot):
    va = va_base + slot * CHUNK_BYTES
    ca._check(ca.CUDA.cuMemUnmap(va, CHUNK_BYTES), "cuMemUnmap")


def main():
    handles = alloc_handles(N_CHUNKS)
    va_base = reserve_va(N_CHUNKS * CHUNK_BYTES)
    for slot in range(N_CHUNKS):
        map_handle(va_base, slot, handles[slot])
    set_access(va_base, N_CHUNKS)
    torch.cuda.synchronize()

    # Build a torch tensor over the VA (each chunk = N tokens × hidden_dim
    # of fp32, here we just treat it as a 1-D float buffer).
    n_floats = (N_CHUNKS * CHUNK_BYTES) // 4
    from sglang.srt.arena.from_blob_ext import tensor_from_va
    big = tensor_from_va(va_base, (n_floats,), torch.float32, DEVICE)
    big.fill_(1.0)
    torch.cuda.synchronize()
    print(f"Tensor view: shape={big.shape} dtype={big.dtype} "
          f"data_ptr=0x{big.data_ptr():x}")

    # ---- BASELINE: eager call, no graph ----
    # Read the tensor — should give all-ones sum.
    s = big.sum().item()
    print(f"\n[1] Eager sum (no graph): {s} (expected: {n_floats}.0)")
    assert s == n_floats, "baseline tensor is not all-ones"

    # ---- Capture a CUDA graph that reads the tensor ----
    out = torch.zeros(1, device=f"cuda:{DEVICE}")
    stream = torch.cuda.Stream(device=DEVICE)
    torch.cuda.synchronize()

    print(f"\n[2] Capturing CUDA graph that reads tensor...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            # Recorded ops: sum over `big`, write to `out`.
            # Note: this is a cuBLAS/cuDNN reduction, not Triton. The
            # live Triton-specific wording "cannot be accessed from
            # Triton" won't appear here, but the underlying
            # `cudaErrorIllegalAddress` will (it's what Triton wraps).
            out.copy_(big.sum().reshape(1))
    torch.cuda.synchronize()
    n_nodes = "n/a"
    try:
        n_nodes = g.num_nodes()
    except Exception:
        pass
    print(f"    captured graph num_nodes={n_nodes}")

    # Replay the graph; should give correct sum
    g.replay()
    torch.cuda.synchronize()
    print(f"[3] Graph replay BEFORE unmap: out={out.item()} (expected: {n_floats}.0)")
    if out.item() != n_floats:
        print("    UNEXPECTED: replay should match eager")

    # ---- Now unmap one chunk (in the MIDDLE of the tensor) ----
    target_slot = 1  # middle chunk
    print(f"\n[4] Unmapping chunk slot {target_slot} (VA at offset "
          f"{target_slot * CHUNK_BYTES // (1024**2)} MiB)...")
    # Drain any in-flight first
    torch.cuda.synchronize()
    unmap_chunk(va_base, target_slot)
    print(f"    cuMemUnmap returned successfully")

    # ---- Replay the SAME captured graph, which still references the tensor ----
    print(f"\n[5] Replaying captured graph AFTER unmap (expect crash)...")
    try:
        g.replay()
        torch.cuda.synchronize()
        print(f"    NO CRASH: out={out.item()} (expected n_floats but VA "
              f"partly unmapped — surprising)")
    except Exception as e:
        print(f"    CAUGHT: {type(e).__name__}: {str(e)[:300]}")

    # ---- Compare: eager call AFTER unmap ----
    print(f"\n[6] Eager sum AFTER unmap (no graph)...")
    try:
        s2 = big.sum().item()
        print(f"    NO CRASH: eager sum={s2}")
    except Exception as e:
        print(f"    CAUGHT: {type(e).__name__}: {str(e)[:300]}")

    # ---- Repair: re-map the chunk into the same VA position ----
    print(f"\n[7] cuMemMap-ing chunk back to same VA position (vLLM-style "
          f"VA-stable rebind)...")
    map_handle(va_base, target_slot, handles[target_slot])
    set_access(va_base, N_CHUNKS)
    torch.cuda.synchronize()

    print(f"\n[8] Replay captured graph after rebind...")
    try:
        g.replay()
        torch.cuda.synchronize()
        print(f"    OK: out={out.item()} (expected: {n_floats}.0)")
    except Exception as e:
        print(f"    CAUGHT: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
