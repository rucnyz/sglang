"""
Phase 2e.1.b — bridge VMM-backed VA to a torch.Tensor.

We prove the soft-cap property end-to-end:
    1. Reserve a VA range, create physical handles, map them all.
    2. Hand the arena to a tiny C allocator (`arena.so`) loaded as a
       torch.cuda.memory.CUDAPluggableAllocator.
    3. Allocate a torch.Tensor inside that pool. Verify the tensor's
       data_ptr() falls inside our reserved VA range.
    4. Write through the tensor; read back via cuMemcpyDtoH at the same VA.
       (Roundtrip: PyTorch -> our VA.)
    5. Remap the physical handle behind the tensor's VA: unmap the chunk
       under VA[0], map a different physical handle there. Read from
       PyTorch the same tensor; expect to see the *new* handle's bytes.
       (This is the cross-pool transfer_chunks property.)
    6. Write through the tensor again, then unmap and remap that handle
       to a different VA, and verify the data follows the physical handle.

Run: CUDA_VISIBLE_DEVICES=2 python dev/2e/02_torch_bridge.py
"""

import ctypes
import os
import sys

CUDA = ctypes.CDLL("libcuda.so")

CU_SUCCESS = 0
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
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


_HANDLE = ctypes.c_ulonglong
_DPTR = ctypes.c_ulonglong

CUDA.cuInit.argtypes = [ctypes.c_uint]
CUDA.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
CUDA.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
CUDA.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
CUDA.cuCtxSynchronize.argtypes = []
CUDA.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
CUDA.cuMemGetAllocationGranularity.argtypes = [
    ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_int
]
CUDA.cuMemAddressReserve.argtypes = [
    ctypes.POINTER(_DPTR), ctypes.c_size_t, ctypes.c_size_t, _DPTR, ctypes.c_ulonglong
]
CUDA.cuMemAddressFree.argtypes = [_DPTR, ctypes.c_size_t]
CUDA.cuMemCreate.argtypes = [
    ctypes.POINTER(_HANDLE), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong
]
CUDA.cuMemRelease.argtypes = [_HANDLE]
CUDA.cuMemMap.argtypes = [_DPTR, ctypes.c_size_t, ctypes.c_size_t, _HANDLE, ctypes.c_ulonglong]
CUDA.cuMemUnmap.argtypes = [_DPTR, ctypes.c_size_t]
CUDA.cuMemSetAccess.argtypes = [_DPTR, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
CUDA.cuMemsetD8_v2.argtypes = [_DPTR, ctypes.c_ubyte, ctypes.c_size_t]
CUDA.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, _DPTR, ctypes.c_size_t]


def check(rc, what):
    if rc != CU_SUCCESS:
        msg = ctypes.c_char_p()
        CUDA.cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"{what} failed: {rc} {msg.value.decode() if msg.value else ''}")


def cu_setup():
    check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    return dev.value


def make_prop(dev_id):
    p = CUmemAllocationProp()
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED
    p.requestedHandleTypes = 0
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = dev_id
    return p


def grant(va, size, dev_id):
    desc = CUmemAccessDesc()
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = dev_id
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(CUDA.cuMemSetAccess(va, size, ctypes.byref(desc), 1), "cuMemSetAccess")


def memset_d8(va, value, n):
    check(CUDA.cuMemsetD8_v2(va, ctypes.c_ubyte(value), n), f"cuMemsetD8(0x{value:02x})")
    check(CUDA.cuCtxSynchronize(), "cuCtxSynchronize")


def read_byte(va):
    buf = (ctypes.c_ubyte * 1)()
    check(CUDA.cuMemcpyDtoH_v2(buf, va, 1), "cuMemcpyDtoH")
    return buf[0]


def main():
    print("== Phase 2e.1.b: torch.Tensor on VMM arena ==")

    # ---- Step 1: prep the CUDA driver primary context. We do NOT create our
    # own context here, because PyTorch will use the runtime API which expects
    # the runtime's primary context. Instead, just init the driver.
    check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    dev_id = dev.value

    # Force PyTorch to set up its primary context first, so subsequent
    # driver-API calls see the same context.
    import torch
    assert torch.cuda.is_available(), "no CUDA"
    print(f"torch={torch.__version__} cuda={torch.version.cuda} dev_count={torch.cuda.device_count()}")
    _ = torch.empty(1, device="cuda")  # forces primary-context init
    torch.cuda.synchronize()
    print(f"primary context initialized on device {torch.cuda.current_device()}")

    # ---- Step 2: VMM arena setup.
    prop = make_prop(dev_id)
    g = ctypes.c_size_t()
    check(CUDA.cuMemGetAllocationGranularity(
        ctypes.byref(g), ctypes.byref(prop), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED), "granularity")
    # PyTorch's caching allocator fetches segments around 20 MiB; use 32 MiB
    # chunks so a single tensor request fits in one chunk.
    chunk = g.value * 16  # = 32 MiB on H200 (granularity is 2 MiB)
    n_chunks = 4
    arena_size = chunk * n_chunks

    va = _DPTR(0)
    check(CUDA.cuMemAddressReserve(ctypes.byref(va), arena_size, 0, 0, 0), "cuMemAddressReserve")
    arena_base = va.value
    print(f"arena: {n_chunks} x {chunk >> 20} MiB at VA 0x{arena_base:x}")

    handles = []
    for _ in range(n_chunks):
        h = _HANDLE(0)
        check(CUDA.cuMemCreate(ctypes.byref(h), chunk, ctypes.byref(prop), 0), "cuMemCreate")
        handles.append(h.value)

    for i, h in enumerate(handles):
        check(CUDA.cuMemMap(arena_base + i * chunk, chunk, 0, h, 0), f"cuMemMap[{i}]")
    grant(arena_base, arena_size, dev_id)
    print(f"all {n_chunks} chunks mapped")

    # Pre-fill chunks with distinguishable patterns.
    patterns = [0xAA, 0xBB, 0xCC, 0xDD]
    for i, p in enumerate(patterns):
        memset_d8(arena_base + i * chunk, p, chunk)
    print(f"prefilled chunks with {[hex(p) for p in patterns]}")

    # ---- Step 3: hand arena to C allocator.
    arena_so_path = os.path.join(os.path.dirname(__file__), "arena.so")
    arena_lib = ctypes.CDLL(arena_so_path)
    arena_lib.arena_init.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t]
    arena_lib.arena_init(arena_base, chunk, n_chunks)

    # ---- Step 4: register as torch CUDAPluggableAllocator.
    from torch.cuda.memory import CUDAPluggableAllocator
    plug = CUDAPluggableAllocator(arena_so_path, "arena_malloc", "arena_free")
    pool = torch.cuda.MemPool(allocator=plug.allocator())
    print("CUDAPluggableAllocator + MemPool registered")

    # ---- Step 5: allocate a tensor inside the pool. data_ptr should be VA[0].
    # Allocate a tensor smaller than chunk so PyTorch's caching allocator
    # doesn't try to grab >chunk-size segments. The segment grab will still
    # be ~20 MiB which fits inside a 32 MiB chunk.
    with torch.cuda.use_mem_pool(pool):
        t = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")  # 1 MiB tensor
    torch.cuda.synchronize()
    t_ptr = t.data_ptr()
    print(f"tensor.data_ptr() = 0x{t_ptr:x}")
    assert t_ptr == arena_base, f"expected tensor at 0x{arena_base:x}, got 0x{t_ptr:x}"
    print("tensor.data_ptr() == arena_base, GOOD")

    # ---- Step 6: tensor sees chunk 0's prefilled bytes (0xAA).
    first_byte = t[0].item()
    print(f"tensor[0] = 0x{first_byte:02x} (expected 0xAA)")
    assert first_byte == 0xAA, f"expected 0xAA, got 0x{first_byte:02x}"

    # Write through the tensor.
    t.fill_(0x42)
    torch.cuda.synchronize()
    print(f"tensor.fill_(0x42); tensor[0]=0x{t[0].item():02x}")

    # Verify by reading the same VA via driver API.
    b = read_byte(arena_base)
    assert b == 0x42, f"VA[0] expected 0x42, got 0x{b:02x}"
    print(f"VA[0] via cuMemcpyDtoH = 0x{b:02x}, GOOD (PyTorch write reaches our VA)")

    # ---- Step 7: REMAP. The soft-cap property:
    # tensor's data_ptr stays at arena_base, but we swap the physical handle
    # under it. The tensor must now read different bytes.
    #
    # Plan: unmap chunk 0 from VA[0]; unmap chunk 1 from VA[1]; map handle 1
    # to VA[0]. After this, VA[0] has handle 1's bytes (0xBB).
    print("\n-- remap: swap handle 1 into VA[0] --")
    check(CUDA.cuMemUnmap(arena_base + 0 * chunk, chunk), "unmap VA[0]")
    check(CUDA.cuMemUnmap(arena_base + 1 * chunk, chunk), "unmap VA[1]")
    check(CUDA.cuMemMap(arena_base + 0 * chunk, chunk, 0, handles[1], 0), "map handle1 -> VA[0]")
    grant(arena_base + 0 * chunk, chunk, dev_id)

    # Tensor data_ptr unchanged. Bytes underneath are now handle 1's (0xBB).
    assert t.data_ptr() == arena_base, "tensor data_ptr changed unexpectedly"
    new_byte = t[0].item()
    print(f"after remap: tensor[0] = 0x{new_byte:02x} (expected 0xBB, handle 1's pattern)")
    assert new_byte == 0xBB, f"expected 0xBB, got 0x{new_byte:02x}"
    print("data follows physical handle, NOT the VA. GOOD.")

    # ---- Step 8: write through the tensor again. It should land in handle 1.
    t.fill_(0x77)
    torch.cuda.synchronize()
    print(f"tensor.fill_(0x77) (writes into handle 1, currently at VA[0])")

    # Now move handle 1 back to VA[1]. Unmap from VA[0], map to VA[1].
    check(CUDA.cuMemUnmap(arena_base + 0 * chunk, chunk), "unmap VA[0] (handle 1)")
    check(CUDA.cuMemMap(arena_base + 1 * chunk, chunk, 0, handles[1], 0), "map handle 1 -> VA[1]")
    grant(arena_base + 1 * chunk, chunk, dev_id)
    via_va1 = read_byte(arena_base + 1 * chunk)
    print(f"VA[1] via cuMemcpyDtoH = 0x{via_va1:02x} (expected 0x77, handle 1's new content)")
    assert via_va1 == 0x77, f"VA[1] expected 0x77, got 0x{via_va1:02x}"
    print("PyTorch's write through old VA[0] persisted on handle 1, visible at new VA[1]. GOOD.")

    # ---- Cleanup. Map handle 0 back to VA[0] so cleanup is symmetric.
    check(CUDA.cuMemMap(arena_base + 0 * chunk, chunk, 0, handles[0], 0), "remap handle0 -> VA[0]")
    grant(arena_base + 0 * chunk, chunk, dev_id)
    for i in range(n_chunks):
        check(CUDA.cuMemUnmap(arena_base + i * chunk, chunk), f"unmap VA[{i}]")
    for h in handles:
        check(CUDA.cuMemRelease(h), "cuMemRelease")
    check(CUDA.cuMemAddressFree(arena_base, arena_size), "cuMemAddressFree")
    print("\n== PASSED: torch.Tensor + VMM remap end-to-end ==")


if __name__ == "__main__":
    sys.exit(main() or 0)
