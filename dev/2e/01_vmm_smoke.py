"""
Phase 2e.1.a — CUDA VMM smoke test.

Goal: prove that on this box (H200, CUDA 13.2) we can:
  1. reserve a contiguous VA range,
  2. create independent physical-memory handles,
  3. map each handle into different offsets of the VA range,
  4. write region-distinguishable data through CUDA,
  5. unmap one handle and remap it to a different offset,
  6. read back and verify the data follows the physical mapping (not the VA).

This is the minimum kernel of Layer 2's chunk-bitmap shared-arena allocator.
No PyTorch dependency on purpose — keeps the test isolated to the Driver API.

Run: CUDA_VISIBLE_DEVICES=2 python dev/2e/01_vmm_smoke.py
"""

import ctypes
import sys

CUDA = ctypes.CDLL("libcuda.so")

# CUresult is int.
CU_SUCCESS = 0

# Set argtypes/restype for every Driver-API call we use. ctypes defaults integer
# args to c_int (32-bit), which silently truncates the 64-bit pointer / handle
# args (CUdeviceptr, CUmemGenericAllocationHandle) — yielding "invalid argument"
# errors at runtime that would otherwise look like nothing happened.
CUDA.cuInit.argtypes = [ctypes.c_uint]
CUDA.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
CUDA.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
CUDA.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
CUDA.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
CUDA.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
CUDA.cuCtxSynchronize.argtypes = []

# 64-bit handles & VAs.
_HANDLE = ctypes.c_ulonglong  # CUmemGenericAllocationHandle
_DPTR = ctypes.c_ulonglong  # CUdeviceptr

CUDA.cuMemAddressReserve.argtypes = [
    ctypes.POINTER(_DPTR), ctypes.c_size_t, ctypes.c_size_t, _DPTR, ctypes.c_ulonglong
]
CUDA.cuMemAddressFree.argtypes = [_DPTR, ctypes.c_size_t]
CUDA.cuMemCreate.argtypes = [
    ctypes.POINTER(_HANDLE), ctypes.c_size_t,
    ctypes.c_void_p,  # CUmemAllocationProp*
    ctypes.c_ulonglong,
]
CUDA.cuMemRelease.argtypes = [_HANDLE]
CUDA.cuMemMap.argtypes = [_DPTR, ctypes.c_size_t, ctypes.c_size_t, _HANDLE, ctypes.c_ulonglong]
CUDA.cuMemUnmap.argtypes = [_DPTR, ctypes.c_size_t]
CUDA.cuMemSetAccess.argtypes = [
    _DPTR, ctypes.c_size_t,
    ctypes.c_void_p,  # CUmemAccessDesc*
    ctypes.c_size_t,
]
CUDA.cuMemGetAllocationGranularity.argtypes = [
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_void_p,  # CUmemAllocationProp*
    ctypes.c_int,
]
CUDA.cuMemsetD8_v2.argtypes = [_DPTR, ctypes.c_ubyte, ctypes.c_size_t]
CUDA.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, _DPTR, ctypes.c_size_t]

# CUmemAllocationType
CU_MEM_ALLOCATION_TYPE_PINNED = 1

# CUmemLocationType
CU_MEM_LOCATION_TYPE_DEVICE = 1

# CUmemAccess_flags
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

# CUmemAllocationGranularity_flags
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


def check(rc, what):
    if rc != CU_SUCCESS:
        msg = ctypes.c_char_p()
        CUDA.cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"{what} failed: {rc} {msg.value.decode() if msg.value else ''}")


def cu_init():
    check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(CUDA.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    name = (ctypes.c_char * 128)()
    check(CUDA.cuDeviceGetName(name, 128, dev), "cuDeviceGetName")
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    check(CUDA.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    print(f"device 0 = {name.value.decode()}, {total.value >> 30} GiB total, {free.value >> 30} GiB free")
    return dev.value, ctx


def make_prop(device_id):
    p = CUmemAllocationProp()
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED
    p.requestedHandleTypes = 0
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = device_id
    return p


def query_granularity(prop):
    g = ctypes.c_size_t()
    check(
        CUDA.cuMemGetAllocationGranularity(
            ctypes.byref(g), ctypes.byref(prop), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        ),
        "cuMemGetAllocationGranularity",
    )
    return g.value


def reserve_va(size):
    ptr = ctypes.c_ulonglong(0)
    check(
        CUDA.cuMemAddressReserve(ctypes.byref(ptr), size, 0, 0, 0),
        "cuMemAddressReserve",
    )
    return ptr.value


def create_handle(size, prop):
    h = ctypes.c_ulonglong(0)
    check(
        CUDA.cuMemCreate(ctypes.byref(h), size, ctypes.byref(prop), 0),
        "cuMemCreate",
    )
    return h.value


def map_at(va, size, offset_into_handle, handle):
    check(
        CUDA.cuMemMap(va, size, offset_into_handle, handle, 0),
        f"cuMemMap(va=0x{va:x}, size={size})",
    )


def unmap(va, size):
    check(CUDA.cuMemUnmap(va, size), f"cuMemUnmap(va=0x{va:x}, size={size})")


def grant_access(va, size, device_id):
    desc = CUmemAccessDesc()
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device_id
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(
        CUDA.cuMemSetAccess(va, size, ctypes.byref(desc), 1),
        "cuMemSetAccess",
    )


def memset_d8(va, value, n):
    check(CUDA.cuMemsetD8_v2(va, ctypes.c_ubyte(value), n), f"cuMemsetD8(0x{value:02x})")
    check(CUDA.cuCtxSynchronize(), "cuCtxSynchronize")


def memcpy_dtoh(va, n_bytes):
    buf = (ctypes.c_ubyte * n_bytes)()
    check(CUDA.cuMemcpyDtoH_v2(buf, va, n_bytes), "cuMemcpyDtoH")
    return buf


def main():
    print("== Phase 2e.1.a: VMM smoke test ==")
    dev_id, _ctx = cu_init()
    prop = make_prop(dev_id)
    g_min = ctypes.c_size_t()
    check(
        CUDA.cuMemGetAllocationGranularity(
            ctypes.byref(g_min), ctypes.byref(prop), CU_MEM_ALLOC_GRANULARITY_MINIMUM
        ),
        "cuMemGetAllocationGranularity(MINIMUM)",
    )
    g_rec = query_granularity(prop)
    print(f"allocation granularity: minimum={g_min.value >> 20} MiB, recommended={g_rec >> 20} MiB")

    # Use 4 chunks of recommended granularity.
    chunk = g_rec
    n_chunks = 4
    arena = chunk * n_chunks
    print(f"arena: {n_chunks} x {chunk >> 20} MiB = {arena >> 20} MiB")

    va = reserve_va(arena)
    print(f"reserved VA = 0x{va:x}")

    # Step 1: create handles A, B, C, D.
    handles = [create_handle(chunk, prop) for _ in range(n_chunks)]
    print(f"created {n_chunks} physical handles: {[hex(h) for h in handles]}")

    # Step 2: map A,B,C,D at chunk 0,1,2,3 of the VA range.
    for i, h in enumerate(handles):
        map_at(va + i * chunk, chunk, 0, h)
    grant_access(va, arena, dev_id)
    print("mapped all 4 handles + granted RW access")

    # Step 3: write distinguishable bytes into each chunk via cuMemsetD8.
    pattern = [0xAA, 0xBB, 0xCC, 0xDD]
    for i, p in enumerate(pattern):
        memset_d8(va + i * chunk, p, chunk)
    print(f"wrote patterns {[hex(p) for p in pattern]} into chunks 0..3")

    # Step 4: read back the first byte of each chunk; should equal the pattern.
    buf = memcpy_dtoh(va, n_chunks * 16)  # first 16 bytes of each chunk's region
    for i, p in enumerate(pattern):
        b = buf[i * chunk] if False else buf[i * 16] if i * 16 < len(buf) else None
        # buf is a flat read of first n_chunks*16 bytes from VA[0..n_chunks*16),
        # which is all in chunk 0. So that's not what we want — read each chunk individually.
    # Read each chunk's leading byte separately.
    for i, p in enumerate(pattern):
        bb = memcpy_dtoh(va + i * chunk, 1)
        assert bb[0] == p, f"chunk {i} expected 0x{p:02x}, got 0x{bb[0]:02x}"
    print("verified initial mapping: each chunk holds its own pattern")

    # Step 5: unmap chunk 1 (handle B). Then unmap chunk 3 (handle D).
    # Then map handle B into chunk 3.
    unmap(va + 1 * chunk, chunk)  # detach handle B from VA[1]
    unmap(va + 3 * chunk, chunk)  # detach handle D from VA[3]
    print("unmapped chunks 1 and 3")

    map_at(va + 3 * chunk, chunk, 0, handles[1])  # B now lives at VA[3]
    grant_access(va + 3 * chunk, chunk, dev_id)
    print("re-mapped handle B (was at VA[1]) to VA[3]")

    # Step 6: chunk 3 should now read 0xBB (B's old data), NOT 0xDD (D's data).
    bb = memcpy_dtoh(va + 3 * chunk, 1)
    assert bb[0] == 0xBB, f"chunk 3 after remap expected 0xBB (handle B's data), got 0x{bb[0]:02x}"
    print(f"verified chunk 3 after remap: 0x{bb[0]:02x} (= handle B's pattern, NOT 0xDD)")

    # Step 7: write 0x11 into chunk 3 via VA[3]. Then map B back to VA[1] and confirm it sees 0x11.
    memset_d8(va + 3 * chunk, 0x11, chunk)
    print("wrote 0x11 into VA[3] (which is handle B's mapping)")

    unmap(va + 3 * chunk, chunk)  # detach B from VA[3]
    map_at(va + 1 * chunk, chunk, 0, handles[1])  # remap B to VA[1]
    grant_access(va + 1 * chunk, chunk, dev_id)
    bb = memcpy_dtoh(va + 1 * chunk, 1)
    assert bb[0] == 0x11, f"VA[1] after remap-back expected 0x11, got 0x{bb[0]:02x}"
    print(f"verified handle B's data follows physical handle, not VA: 0x{bb[0]:02x}")

    # Cleanup.
    unmap(va + 0 * chunk, chunk)
    unmap(va + 1 * chunk, chunk)
    unmap(va + 2 * chunk, chunk)
    for h in handles:
        check(CUDA.cuMemRelease(h), f"cuMemRelease(0x{h:x})")
    check(CUDA.cuMemAddressFree(va, arena), "cuMemAddressFree")
    print("cleanup complete")
    print("\n== PASSED: VMM unmap+remap semantics work end-to-end ==")


if __name__ == "__main__":
    sys.exit(main() or 0)
