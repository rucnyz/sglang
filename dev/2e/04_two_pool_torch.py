"""
Phase 2e.2.b — two PyTorch tensors on shared-arena pools, with cross-pool transfer.

Two pools (A and B) share one VA arena. Each has its own MemPool +
CUDAPluggableAllocator (pool0_malloc/free, pool1_malloc/free in arena_multi.so).
A torch.Tensor allocated inside pool A's MemPool lands in pool A's VA sub-range;
similarly for B. We then run `transfer_chunks(B, A, 1)` and verify:

1. Both tensors' `data_ptr()` are unchanged (the soft-cap property).
2. Both tensors' contents are unchanged (the moved chunk was a different slot).
3. The newly-arrived chunk in A's sub-range carries B's evicted bytes.

Run: CUDA_VISIBLE_DEVICES=2 python dev/2e/04_two_pool_torch.py
"""

import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from chunk_arena import CUDA, ChunkArena, _check  # noqa: E402

CUDA.cuMemsetD8_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_ubyte, ctypes.c_size_t]
CUDA.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]


def memset_d8(va: int, value: int, n: int) -> None:
    _check(CUDA.cuMemsetD8_v2(va, ctypes.c_ubyte(value), n), f"memset 0x{value:02x}")
    _check(CUDA.cuCtxSynchronize(), "ctx sync")


def read_byte(va: int) -> int:
    buf = (ctypes.c_ubyte * 1)()
    _check(CUDA.cuMemcpyDtoH_v2(buf, va, 1), "cuMemcpyDtoH")
    return buf[0]


def main() -> int:
    print("== Phase 2e.2.b: torch.Tensor in two pools + transfer ==")

    _check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    _check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    dev_id = dev.value

    import torch
    _ = torch.empty(1, device="cuda")  # primary context init
    torch.cuda.synchronize()
    print(f"torch={torch.__version__}, dev_count={torch.cuda.device_count()}")

    # PyTorch's caching allocator pulls ~20 MiB segments. Use 32 MiB chunks.
    chunk = 32 * 1024 * 1024
    arena = ChunkArena(
        device_id=dev_id,
        chunk_size=chunk,
        n_handles=4,
        pool_capacities=[("A", 4), ("B", 4)],
    )
    print(f"arena base=0x{arena.va_base:x} A=0x{arena.pool_va_base('A'):x} "
          f"B=0x{arena.pool_va_base('B'):x}")

    # Initial mapping: 2 chunks each.
    arena.grow("A", 2)
    arena.grow("B", 2)
    print(f"initial: A={arena.pool_mapped_chunks('A')} B={arena.pool_mapped_chunks('B')} "
          f"free_handles={arena.free_handle_count()}")

    a_va = arena.pool_va_base("A")
    b_va = arena.pool_va_base("B")
    memset_d8(a_va + 0 * chunk, 0xA0, chunk)
    memset_d8(a_va + 1 * chunk, 0xA1, chunk)
    memset_d8(b_va + 0 * chunk, 0xB0, chunk)
    memset_d8(b_va + 1 * chunk, 0xB1, chunk)

    # ---- C side: tell each pool's allocator about its VA + capacity.
    so_path = os.path.join(os.path.dirname(__file__), "arena_multi.so")
    multi_lib = ctypes.CDLL(so_path)
    multi_lib.multi_init.argtypes = [
        ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t]
    multi_lib.multi_set_capacity.argtypes = [ctypes.c_int, ctypes.c_size_t]

    multi_lib.multi_init(0, a_va, chunk, arena.pool_mapped_chunks("A"))
    multi_lib.multi_init(1, b_va, chunk, arena.pool_mapped_chunks("B"))

    # ---- PyTorch: two MemPools, one per pool.
    from torch.cuda.memory import CUDAPluggableAllocator
    plug_a = CUDAPluggableAllocator(so_path, "pool0_malloc", "pool0_free")
    plug_b = CUDAPluggableAllocator(so_path, "pool1_malloc", "pool1_free")
    pool_torch_a = torch.cuda.MemPool(allocator=plug_a.allocator())
    pool_torch_b = torch.cuda.MemPool(allocator=plug_b.allocator())
    print("two CUDAPluggableAllocators + MemPools registered")

    # ---- Allocate one tensor in each pool. 1 MiB tensor; PyTorch grabs a
    # segment of size <= chunk into pool 0/1's first slot.
    with torch.cuda.use_mem_pool(pool_torch_a):
        t_a = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    with torch.cuda.use_mem_pool(pool_torch_b):
        t_b = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    print(f"t_a.data_ptr() = 0x{t_a.data_ptr():x} (A.va_base=0x{a_va:x})")
    print(f"t_b.data_ptr() = 0x{t_b.data_ptr():x} (B.va_base=0x{b_va:x})")
    assert a_va <= t_a.data_ptr() < a_va + chunk, "t_a not in A's chunk 0"
    assert b_va <= t_b.data_ptr() < b_va + chunk, "t_b not in B's chunk 0"
    print("each tensor lands in its own pool's first chunk, GOOD")

    # Tensors initially see the prefilled bytes.
    assert t_a[0].item() == 0xA0, f"t_a[0]=0x{t_a[0].item():02x} expected 0xA0"
    assert t_b[0].item() == 0xB0, f"t_b[0]=0x{t_b[0].item():02x} expected 0xB0"

    # Write through tensors.
    t_a.fill_(0x42)
    t_b.fill_(0x77)
    torch.cuda.synchronize()
    print(f"after fill_: t_a[0]=0x{t_a[0].item():02x}, t_b[0]=0x{t_b[0].item():02x}")
    assert t_a[0].item() == 0x42 and t_b[0].item() == 0x77

    # ---- Cross-pool transfer: move 1 chunk from B to A.
    a_ptr_before = t_a.data_ptr()
    b_ptr_before = t_b.data_ptr()
    n = arena.transfer_chunks("B", "A", 1)
    assert n == 1
    print(f"transfer B->A: A={arena.pool_mapped_chunks('A')} B={arena.pool_mapped_chunks('B')}")

    # Tell the C allocators about new capacities (post-transfer).
    multi_lib.multi_set_capacity(0, arena.pool_mapped_chunks("A"))
    multi_lib.multi_set_capacity(1, arena.pool_mapped_chunks("B"))

    # ---- Verify: tensor pointers are stable.
    assert t_a.data_ptr() == a_ptr_before, f"t_a.data_ptr() drifted: {hex(a_ptr_before)} -> {hex(t_a.data_ptr())}"
    assert t_b.data_ptr() == b_ptr_before, f"t_b.data_ptr() drifted: {hex(b_ptr_before)} -> {hex(t_b.data_ptr())}"
    print(f"tensor data_ptrs unchanged: t_a=0x{t_a.data_ptr():x}, t_b=0x{t_b.data_ptr():x}")

    # ---- Verify: tensor contents unchanged (their slot 0 chunks were not touched).
    a0 = t_a[0].item()
    b0 = t_b[0].item()
    print(f"after transfer: t_a[0]=0x{a0:02x} (expected 0x42), t_b[0]=0x{b0:02x} (expected 0x77)")
    assert a0 == 0x42, f"t_a[0] changed: 0x{a0:02x}"
    assert b0 == 0x77, f"t_b[0] changed: 0x{b0:02x}"

    # ---- Verify: the moved chunk in A's slot 2 carries B's evicted bytes.
    moved_byte = read_byte(a_va + 2 * chunk)
    print(f"A.va[slot=2] (the moved chunk) = 0x{moved_byte:02x} (expected 0xB1, B's slot-1 pattern)")
    assert moved_byte == 0xB1

    # ---- Cleanup. Tensors first (release their references), then arena.
    del t_a, t_b
    del pool_torch_a, pool_torch_b
    arena.cleanup()
    print("\n== PASSED: cross-pool transfer preserves tensor identity ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
