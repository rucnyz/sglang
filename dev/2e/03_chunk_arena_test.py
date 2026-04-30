"""
Phase 2e.2.a — ChunkArena unit test.

Two pools (A and B) on a shared VA arena. Walk through:
  1. Initial mapping: A gets handles 0,1; B gets handles 2,3.
  2. Memset distinguishable patterns.
  3. transfer_chunks(B -> A, 1): B's tail handle moves into A's next slot.
     Verify content of A's new slot equals B's evicted handle's pattern.
  4. transfer_chunks(A -> B, 1): handle moves back. Verify content
     round-trips.

Run: CUDA_VISIBLE_DEVICES=2 python dev/2e/03_chunk_arena_test.py
"""

import ctypes
import sys

# Make sure we can import the module from this dir.
import os
sys.path.insert(0, os.path.dirname(__file__))

from chunk_arena import CUDA, ChunkArena, _check, _DPTR  # noqa: E402

CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


def _additional_argtypes():
    CUDA.cuMemsetD8_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_ubyte, ctypes.c_size_t]
    CUDA.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]


_additional_argtypes()


def memset_d8(va: int, value: int, n: int) -> None:
    _check(CUDA.cuMemsetD8_v2(va, ctypes.c_ubyte(value), n), f"memset 0x{value:02x}")
    _check(CUDA.cuCtxSynchronize(), "ctx sync")


def read_byte(va: int) -> int:
    buf = (ctypes.c_ubyte * 1)()
    _check(CUDA.cuMemcpyDtoH_v2(buf, va, 1), "cuMemcpyDtoH")
    return buf[0]


def main() -> int:
    print("== Phase 2e.2.a: ChunkArena two-pool transfer ==")

    _check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    _check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    dev_id = dev.value

    # Force the runtime to set up its primary context.
    import torch
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    chunk = 2 * 1024 * 1024  # 2 MiB (= H200 granularity)
    arena = ChunkArena(
        device_id=dev_id,
        chunk_size=chunk,
        n_handles=4,
        pool_capacities=[("A", 4), ("B", 4)],  # 4 slots each, 8 slots total VA, 4 handles
    )
    print(f"arena va_base=0x{arena.va_base:x}, A va=0x{arena.pool_va_base('A'):x}, "
          f"B va=0x{arena.pool_va_base('B'):x}")

    # -- Step 1: initial mapping. Give 2 chunks to each pool.
    a_grew = arena.grow("A", 2)
    b_grew = arena.grow("B", 2)
    assert a_grew == 2 and b_grew == 2, f"grew A={a_grew}, B={b_grew}"
    assert arena.pool_mapped_chunks("A") == 2
    assert arena.pool_mapped_chunks("B") == 2
    assert arena.free_handle_count() == 0
    print(f"after initial grow: A={arena.pool_mapped_chunks('A')} chunks, "
          f"B={arena.pool_mapped_chunks('B')} chunks, free handles={arena.free_handle_count()}")

    # -- Step 2: memset distinguishable patterns into each chunk.
    # We expect A's slots 0,1 to hold handles 3,2 (LIFO from the free list)
    # and B's slots 0,1 to hold handles 1,0. But the test does NOT assume
    # this — we just write through A's two slots and B's two slots and
    # check what moves where.
    a_va = arena.pool_va_base("A")
    b_va = arena.pool_va_base("B")
    memset_d8(a_va + 0 * chunk, 0xA0, chunk)
    memset_d8(a_va + 1 * chunk, 0xA1, chunk)
    memset_d8(b_va + 0 * chunk, 0xB0, chunk)
    memset_d8(b_va + 1 * chunk, 0xB1, chunk)
    print("wrote A[0]=0xA0 A[1]=0xA1 B[0]=0xB0 B[1]=0xB1")

    # Sanity read.
    assert read_byte(a_va + 0 * chunk) == 0xA0
    assert read_byte(a_va + 1 * chunk) == 0xA1
    assert read_byte(b_va + 0 * chunk) == 0xB0
    assert read_byte(b_va + 1 * chunk) == 0xB1
    print("readback confirmed")

    # -- Step 3: transfer 1 chunk from B to A.
    # tail-evict policy: B's last mapped slot is slot 1 (with content 0xB1).
    # That handle's bytes should appear in A's first free slot (slot 2).
    n = arena.transfer_chunks("B", "A", 1)
    assert n == 1, f"expected 1 transferred, got {n}"
    assert arena.pool_mapped_chunks("A") == 3
    assert arena.pool_mapped_chunks("B") == 1
    print(f"transfer B->A: A={arena.pool_mapped_chunks('A')}, B={arena.pool_mapped_chunks('B')}")

    # A's slot 2 should now read 0xB1 (handle that used to be at B's slot 1).
    a2 = read_byte(a_va + 2 * chunk)
    print(f"A[2] after transfer = 0x{a2:02x} (expected 0xB1)")
    assert a2 == 0xB1, f"expected 0xB1 at A[2], got 0x{a2:02x}"

    # B's slot 0 still has 0xB0 (was not touched).
    b0 = read_byte(b_va + 0 * chunk)
    assert b0 == 0xB0
    print(f"B[0] still 0x{b0:02x}, GOOD")

    # -- Step 4: write through A's slot 2 and transfer back.
    memset_d8(a_va + 2 * chunk, 0x77, chunk)
    print("wrote A[2]=0x77")
    n = arena.transfer_chunks("A", "B", 1)
    assert n == 1
    assert arena.pool_mapped_chunks("A") == 2
    assert arena.pool_mapped_chunks("B") == 2

    # Where did the handle go? B's first free slot was 1 (slot 0 still
    # holds the original handle). So B[1] should now read 0x77.
    b1 = read_byte(b_va + 1 * chunk)
    print(f"B[1] after transfer back = 0x{b1:02x} (expected 0x77)")
    assert b1 == 0x77, f"expected 0x77 at B[1], got 0x{b1:02x}"
    print("data round-tripped via the physical handle. GOOD.")

    # -- Step 5: A's pool VA base is unchanged across transfers.
    assert arena.pool_va_base("A") == a_va
    assert arena.pool_va_base("B") == b_va
    print(f"VA bases stable: A=0x{arena.pool_va_base('A'):x} B=0x{arena.pool_va_base('B'):x}")

    arena.cleanup()
    print("\n== PASSED: two-pool transfer_chunks works end-to-end ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
