"""
Microbench: cuMemUnmap + cuMemMap remap on the SAME VA — CUDA graph compatibility.

Questions answered:
  1. Can you remap a new physical handle to a VA after cuMemUnmap?
  2. Does a CUDA graph captured against that VA still work after remap?

Run with: CUDA_VISIBLE_DEVICES=3 /scratch/yuzhou/projects/sglang/.venv/bin/python dev_vmm_remap_bench.py
"""
import ctypes
import os
import sys

import torch

CUDA = ctypes.CDLL("libcuda.so")

CU_SUCCESS = 0
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1

_HANDLE = ctypes.c_ulonglong
_DPTR   = ctypes.c_ulonglong


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType",      ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage",                ctypes.c_ushort),
        ("reserved",             ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type",                  ctypes.c_int),
        ("requestedHandleTypes",  ctypes.c_int),
        ("location",              _CUmemLocation),
        ("win32HandleMetaData",   ctypes.c_void_p),
        ("allocFlags",            _CUmemAllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


CUDA.cuInit.argtypes                    = [ctypes.c_uint]
CUDA.cuDeviceGet.argtypes               = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
CUDA.cuMemGetAllocationGranularity.argtypes = [
    ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_int]
CUDA.cuMemAddressReserve.argtypes       = [
    ctypes.POINTER(_DPTR), ctypes.c_size_t, ctypes.c_size_t, _DPTR, ctypes.c_ulonglong]
CUDA.cuMemAddressFree.argtypes          = [_DPTR, ctypes.c_size_t]
CUDA.cuMemCreate.argtypes               = [
    ctypes.POINTER(_HANDLE), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong]
CUDA.cuMemRelease.argtypes              = [_HANDLE]
CUDA.cuMemMap.argtypes                  = [
    _DPTR, ctypes.c_size_t, ctypes.c_size_t, _HANDLE, ctypes.c_ulonglong]
CUDA.cuMemUnmap.argtypes                = [_DPTR, ctypes.c_size_t]
CUDA.cuMemSetAccess.argtypes            = [
    _DPTR, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
CUDA.cuGetErrorString.argtypes          = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
CUDA.cuCtxSynchronize.argtypes          = []
CUDA.cuCtxSynchronize.restype           = ctypes.c_int


def _check(rc: int, tag: str) -> None:
    if rc != CU_SUCCESS:
        msg = ctypes.c_char_p()
        CUDA.cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"{tag} failed rc={rc}: {msg.value.decode() if msg.value else '?'}")


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
    torch.cuda.init()
    dev_idx = torch.cuda.current_device()    # logical 0 after CVD mask
    print(f"Device: {torch.cuda.get_device_name(dev_idx)}  (logical index {dev_idx})")

    # --- granularity ---
    prop = _CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev_idx

    gran = ctypes.c_size_t()
    _check(CUDA.cuMemGetAllocationGranularity(
        ctypes.byref(gran), ctypes.byref(prop), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
        "cuMemGetAllocationGranularity")
    chunk_size = gran.value          # use 1 granule per chunk (smallest possible)
    n_chunks   = 4
    total_va   = chunk_size * n_chunks
    print(f"Granularity: {gran.value >> 20} MiB  |  chunk_size={chunk_size}  n_chunks={n_chunks}")

    # --- reserve VA ---
    va = _DPTR(0)
    _check(CUDA.cuMemAddressReserve(ctypes.byref(va), total_va, 0, 0, 0), "cuMemAddressReserve")
    va_base = va.value
    print(f"VA base: 0x{va_base:016x}  total={total_va >> 20} MiB")

    # access descriptor
    desc = _CUmemAccessDesc()
    desc.location.type  = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id    = dev_idx
    desc.flags          = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    def map_chunk(chunk_idx: int, handle_val: int) -> None:
        va_c = va_base + chunk_idx * chunk_size
        _check(CUDA.cuMemMap(va_c, chunk_size, 0, handle_val, 0),
               f"cuMemMap chunk={chunk_idx}")
        _check(CUDA.cuMemSetAccess(va_c, chunk_size, ctypes.byref(desc), 1),
               f"cuMemSetAccess chunk={chunk_idx}")

    def unmap_chunk(chunk_idx: int) -> None:
        va_c = va_base + chunk_idx * chunk_size
        _check(CUDA.cuMemUnmap(va_c, chunk_size), f"cuMemUnmap chunk={chunk_idx}")

    # --- create handles ---
    # H0..H3: initial set  H4..H5: replacement handles for remap tests
    N_HANDLES = 6
    handles = []
    for i in range(N_HANDLES):
        h = _HANDLE(0)
        _check(CUDA.cuMemCreate(ctypes.byref(h), chunk_size, ctypes.byref(prop), 0),
               f"cuMemCreate h{i}")
        handles.append(h.value)
    print(f"Created {N_HANDLES} physical handles")

    # --- map chunks 0-3 with handles 0-3 ---
    for i in range(n_chunks):
        map_chunk(i, handles[i])
    print("Mapped chunks 0-3 with handles 0-3")

    # Build a torch tensor that covers the full VA reservation.
    # Elements = int32, so each chunk holds chunk_size//4 int32 values.
    import sys
    sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
    from sglang.srt.arena.from_blob_ext import tensor_from_va

    elems_per_chunk = chunk_size // 4
    total_elems = elems_per_chunk * n_chunks
    t = tensor_from_va(
        va=va_base,
        sizes=(total_elems,),
        dtype=torch.int32,
        device_index=dev_idx,
    )
    # Write sentinel values to each chunk region
    for i in range(n_chunks):
        t[i * elems_per_chunk : (i+1) * elems_per_chunk] = (i + 1) * 111
    torch.cuda.synchronize()
    vals = [t[i * elems_per_chunk].item() for i in range(n_chunks)]
    print(f"After initial write, chunk sentinels: {vals}  (expect [111,222,333,444])")
    assert vals == [111, 222, 333, 444], f"unexpected: {vals}"

    # ===================================================================
    # TEST 1: remap chunk 2 — unmap, map new handle, write, read back
    # ===================================================================
    print("\n--- TEST 1: remap chunk 2 (eager write after remap) ---")
    torch.cuda.synchronize()
    unmap_chunk(2)
    print("  cuMemUnmap chunk 2: OK")
    map_chunk(2, handles[4])          # remap with handle H4
    print("  cuMemMap chunk 2 -> H4: OK")
    t[2 * elems_per_chunk : 3 * elems_per_chunk] = 999
    torch.cuda.synchronize()
    v = t[2 * elems_per_chunk].item()
    print(f"  Write 999 to chunk 2 after remap -> read back: {v}")
    assert v == 999, f"TEST 1 FAILED: got {v}"
    print("  TEST 1 PASSED: remap + eager write works")

    # ===================================================================
    # TEST 2: CUDA graph captured BEFORE remap — replay AFTER remap
    # Target: write value 777 to chunk 1 via a captured graph
    # Then unmap chunk 1, remap with H5, replay the graph
    # If the VA is stable, the graph's captured ptr still points to chunk 1
    # and the replay should work on the NEW physical pages.
    # ===================================================================
    print("\n--- TEST 2: CUDA graph replay after remap ---")

    # Prepare a result tensor (small, standard allocation) to receive output
    out = torch.zeros(1, dtype=torch.int32, device="cuda")

    # Capture a graph: write 777 into chunk 1, then read [0] back to `out`
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        t[1 * elems_per_chunk : 2 * elems_per_chunk] = 777
        out[0] = t[1 * elems_per_chunk]

    print("  CUDA graph captured (writes 777 to chunk 1, reads back to out[0])")

    # Replay before remap — sanity check
    g.replay()
    torch.cuda.synchronize()
    v_before = out[0].item()
    print(f"  Replay before remap: out={v_before}  (expect 777)")
    assert v_before == 777, f"sanity replay failed: {v_before}"

    # Now remap chunk 1: unmap H1, map H5
    torch.cuda.synchronize()
    unmap_chunk(1)
    print("  cuMemUnmap chunk 1: OK")
    map_chunk(1, handles[5])
    print("  cuMemMap chunk 1 -> H5: OK")
    # New physical pages are zeroed (cuMemCreate zero-initializes)
    v_post_remap_raw = t[1 * elems_per_chunk].item()
    print(f"  Read chunk 1 after remap (before replay): {v_post_remap_raw}  (expect 0)")

    # Replay graph — it captured a raw VA ptr; new physical pages are now there
    g.replay()
    torch.cuda.synchronize()
    v_after = out[0].item()
    print(f"  Replay after remap: out={v_after}  (expect 777)")
    assert v_after == 777, f"TEST 2 FAILED: graph replay after remap: {v_after}"
    print("  TEST 2 PASSED: CUDA graph replay works after physical remap")

    # ===================================================================
    # TEST 3: unmap chunk 3 (WHILE graph captured against it exists)
    # — graph is NOT replayed against unmapped VA — just verifying that
    # the VA itself is accessible after a second remap cycle
    # ===================================================================
    print("\n--- TEST 3: second remap cycle on chunk 2 ---")
    unmap_chunk(2)
    map_chunk(2, handles[1])   # reuse H1 (freed from earlier unmap)
    t[2 * elems_per_chunk] = 42
    torch.cuda.synchronize()
    v3 = t[2 * elems_per_chunk].item()
    print(f"  Second remap cycle, write 42 -> read back: {v3}")
    assert v3 == 42, f"TEST 3 FAILED: {v3}"
    print("  TEST 3 PASSED")

    # cleanup
    for i in range(n_chunks):
        try:
            unmap_chunk(i)
        except RuntimeError:
            pass  # already unmapped
    for h in handles:
        CUDA.cuMemRelease(h)
    CUDA.cuMemAddressFree(va_base, total_va)

    print("\n=== ALL TESTS PASSED ===")
    print("CONCLUSION:")
    print("  1. cuMemUnmap + cuMemMap to the SAME VA: WORKS")
    print("  2. CUDA graph replay after remap to NEW physical handle: WORKS")
    print("  => ideal fire (VMM remap without new allocation) IS compatible with CUDA graphs")


if __name__ == "__main__":
    main()
