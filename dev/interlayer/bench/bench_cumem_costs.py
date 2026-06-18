"""Measure pure cuMemUnmap + cuMemMap timing for the chunk sizes
we actually use in HiMA fires. Validates the claim "fire ≈ 30ms".

Setup:
  - 48 chunks (= 1 typical m2k fire)
  - chunk_bytes = 2 MiB (= sglang config in D10 runs, per
    `MultiTensorArena initialized: chunk_bytes=2097152`)
  - 24 sub-pools × 48 chunks each (mirrors 24-layer mamba pool)
  - 1 device, no concurrency

Usage:
  .venv/bin/python dev/interlayer/bench/bench_cumem_costs.py
"""
from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

# Use sglang's existing ctypes wrappers
import sglang.srt.arena.chunk_arena as ca

# ---- params (mirror production D10@C=56) ----
N_CHUNKS = 48              # one m2k fire's worth
N_SUBPOOLS = 24            # 24 layers
CHUNK_BYTES = 2 * 1024 * 1024  # 2 MiB, sglang default
DEVICE = 0
N_TRIALS = 20              # how many fire-equivalent rounds to time

# ---- raw CUDA setup ----
ca.CUDA.cuInit(0)
dev = ctypes.c_int(0)
ca.CUDA.cuDeviceGet(ctypes.byref(dev), DEVICE)
# Need a context; use torch to create one
import torch
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()


def _check(rc, what):
    ca._check(rc, what)


def alloc_handles(n: int) -> list:
    """Create N physical handles for chunks."""
    prop = ca._CUmemAllocationProp()
    prop.type = ca.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = DEVICE
    handles = []
    for _ in range(n):
        h = ctypes.c_ulonglong()
        _check(ca.CUDA.cuMemCreate(ctypes.byref(h), CHUNK_BYTES,
                                    ctypes.byref(prop), 0),
               "cuMemCreate")
        handles.append(h.value)
    return handles


def reserve_va(total_bytes: int) -> int:
    """Reserve VA range of total_bytes; returns base pointer."""
    va = ctypes.c_ulonglong()
    _check(ca.CUDA.cuMemAddressReserve(
        ctypes.byref(va), total_bytes, CHUNK_BYTES, 0, 0),
        "cuMemAddressReserve")
    return va.value


def map_handle(va_base: int, slot: int, handle: int) -> None:
    va = va_base + slot * CHUNK_BYTES
    _check(ca.CUDA.cuMemMap(va, CHUNK_BYTES, 0, handle, 0), "cuMemMap")


def set_access(va_base: int, n_slots: int) -> None:
    """Grant RW to all currently-mapped slots."""
    desc = ca._CUmemAccessDesc()
    desc.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = DEVICE
    desc.flags = 3  # CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _check(ca.CUDA.cuMemSetAccess(
        va_base, n_slots * CHUNK_BYTES, ctypes.byref(desc), 1),
        "cuMemSetAccess")


def unmap_range(va_base: int, slots: list) -> None:
    """Unmap a set of slots (one cuMemUnmap call per slot)."""
    for s in slots:
        va = va_base + s * CHUNK_BYTES
        _check(ca.CUDA.cuMemUnmap(va, CHUNK_BYTES), "cuMemUnmap")


def main():
    # Allocate enough physical handles to populate all sub-pools once,
    # PLUS 48 spare handles for the "destination" side of cross-pool.
    n_total_chunks = N_SUBPOOLS * N_CHUNKS + N_SUBPOOLS * N_CHUNKS
    handles = alloc_handles(n_total_chunks)
    print(f"Allocated {n_total_chunks} chunks ({n_total_chunks * CHUNK_BYTES // (1024**2)} MiB)")

    # Two arenas, each with N_SUBPOOLS sub-pools of N_CHUNKS slots.
    # Each sub-pool has its own VA reservation.
    src_arenas = []
    dst_arenas = []
    for _ in range(N_SUBPOOLS):
        src_arenas.append(reserve_va(N_CHUNKS * CHUNK_BYTES))
        dst_arenas.append(reserve_va(N_CHUNKS * CHUNK_BYTES))

    # Initially: map first half of handles into src arena, leave dst empty.
    free_handles = list(range(n_total_chunks))
    for sp_idx in range(N_SUBPOOLS):
        for slot in range(N_CHUNKS):
            h_idx = free_handles.pop(0)
            map_handle(src_arenas[sp_idx], slot, handles[h_idx])
        set_access(src_arenas[sp_idx], N_CHUNKS)
    # dst stays unmapped; we'll map handles into it after unmap.

    torch.cuda.synchronize()
    print(f"Initial mapping complete. Running {N_TRIALS} fire trials...")

    unmap_only = []
    map_only = []
    full_roundtrip = []

    # For each trial, we do a "fake fire":
    #   1. unmap N_CHUNKS slots from src across all N_SUBPOOLS
    #   2. map those handles into dst (all N_SUBPOOLS, same N_CHUNKS)
    # Then we reverse (so subsequent trials have something to unmap).

    # Track which arena currently holds the handles. Start: src.
    src_has_handles = True

    for trial in range(N_TRIALS):
        if src_has_handles:
            from_arenas, to_arenas = src_arenas, dst_arenas
        else:
            from_arenas, to_arenas = dst_arenas, src_arenas

        torch.cuda.synchronize()

        # Phase 1: unmap N_CHUNKS slots from each src sub-pool
        t0 = time.monotonic_ns()
        for sp_idx in range(N_SUBPOOLS):
            unmap_range(from_arenas[sp_idx], list(range(N_CHUNKS)))
        torch.cuda.synchronize()
        t1 = time.monotonic_ns()

        # Phase 2: map them into dst sub-pools
        # Reuse the same handle slot indices (they're free now)
        for sp_idx in range(N_SUBPOOLS):
            for slot in range(N_CHUNKS):
                h_global = sp_idx * N_CHUNKS + slot
                map_handle(to_arenas[sp_idx], slot, handles[h_global])
        for sp_idx in range(N_SUBPOOLS):
            set_access(to_arenas[sp_idx], N_CHUNKS)
        torch.cuda.synchronize()
        t2 = time.monotonic_ns()

        unmap_us = (t1 - t0) // 1000
        map_us = (t2 - t1) // 1000
        unmap_only.append(unmap_us)
        map_only.append(map_us)
        full_roundtrip.append(unmap_us + map_us)

        src_has_handles = not src_has_handles
        print(f"  trial {trial:2d}: unmap={unmap_us:6d} us  map={map_us:6d} us  total={unmap_us+map_us:6d} us")

    def _stats(name, vals):
        srt = sorted(vals)
        print(f"  {name:18s} p50={srt[len(srt)//2]/1000:6.2f} ms "
              f"p99={srt[int(0.99*len(srt))]/1000:6.2f} ms "
              f"mean={statistics.mean(vals)/1000:6.2f} ms")

    print(f"\n=== {N_TRIALS} trials, {N_SUBPOOLS} sub-pools × {N_CHUNKS} chunks × {CHUNK_BYTES//1024} KiB ===")
    _stats("unmap only", unmap_only)
    _stats("map only", map_only)
    _stats("full roundtrip", full_roundtrip)
    bytes_moved = N_SUBPOOLS * N_CHUNKS * CHUNK_BYTES
    median_us = sorted(full_roundtrip)[len(full_roundtrip)//2]
    print(f"  effective bandwidth: {bytes_moved / (median_us * 1024):.2f} MiB/sec")


if __name__ == "__main__":
    main()
