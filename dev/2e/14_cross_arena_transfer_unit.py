"""
Phase 2e.5.6.1 — SharedHandlePool + cross_arena_transfer unit test.

Two ChunkArenas, separate VA reservations, but both sharing one
SharedHandlePool. We verify:

  1. Both arenas can grow their pools using handles from the shared pool.
  2. cross_arena_transfer(A.poolX, B.poolY, n) actually unmaps n chunks
     from A.poolX, leaving them free in the shared pool, then maps them
     into B.poolY.
  3. The bytes follow the physical handle: write a pattern through A's
     view, transfer the chunk to B, read through B's view → original
     pattern. Then write a new pattern through B, transfer back, read
     through A → new pattern.
  4. VA bases of all pools (A.poolX.va_base, B.poolY.va_base) are stable
     across all transfers.
  5. PyTorch tensors built atop two MultiTensorArenas (a small KV-like
     and a small mamba-like) survive a cross-arena transfer with stable
     data_ptr.

Pass: all six checks succeed.

Run:
  cd /scratch/yuzhou/projects/sglang
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
    .venv/bin/python -u dev/2e/14_cross_arena_transfer_unit.py
"""
from __future__ import annotations
import ctypes
import sys

import torch

from sglang.srt.arena.chunk_arena import (
    ChunkArena,
    SharedHandlePool,
    cross_arena_transfer,
    CUDA,
    _DPTR,
)
from sglang.srt.arena.multi_tensor_arena import MultiTensorArena


CHUNK_SIZE = 2 * 1024 * 1024  # 2 MiB == H200 VMM granularity (cheapest test)

# Module-level register so VMM-backed objects survive past test-function
# return; we hard-exit via os._exit before any destructor runs.
_KEEPALIVE: list = []

# Driver-API symbols for raw byte writes/reads through the VA range. The
# `_v2` suffix is the modern CUDA driver entry-point name; the un-suffixed
# name is not always exported by libcuda.so.
_LIBCUDA = ctypes.CDLL("libcuda.so")
_LIBCUDA.cuMemsetD8_v2.argtypes = [_DPTR, ctypes.c_ubyte, ctypes.c_size_t]
_LIBCUDA.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, _DPTR, ctypes.c_size_t]


def _write_byte(va: int, value: int, nbytes: int) -> None:
    rc = _LIBCUDA.cuMemsetD8_v2(va, value, nbytes)
    if rc != 0:
        raise RuntimeError(f"cuMemsetD8_v2 va={va:#x} value={value} nbytes={nbytes} rc={rc}")


def _read_byte(va: int, nbytes: int) -> int:
    buf = (ctypes.c_ubyte * nbytes)()
    rc = _LIBCUDA.cuMemcpyDtoH_v2(buf, va, nbytes)
    if rc != 0:
        raise RuntimeError(f"cuMemcpyDtoH_v2 va={va:#x} nbytes={nbytes} rc={rc}")
    return int(buf[0])


def test_basic_cross_arena_transfer() -> None:
    print("== Test 1: basic cross-arena transfer ==")
    pool = SharedHandlePool(device_id=0, chunk_size=CHUNK_SIZE, n_handles=4)

    arena_A = ChunkArena(
        device_id=0, chunk_size=CHUNK_SIZE,
        n_handles=0,  # ignored when external_handle_pool is provided
        pool_capacities=[("kv0", 4)],
        external_handle_pool=pool,
    )
    arena_B = ChunkArena(
        device_id=0, chunk_size=CHUNK_SIZE,
        n_handles=0,
        pool_capacities=[("mamba0", 4)],
        external_handle_pool=pool,
    )
    print(f"  A.va_base=0x{arena_A.va_base:x}, B.va_base=0x{arena_B.va_base:x}")
    print(f"  initial: shared free={pool.free_count()}")
    assert pool.free_count() == 4

    # Grow A by 2 chunks — pool should drop to 2.
    n = arena_A.grow("kv0", 2)
    assert n == 2, f"expected to grow A by 2, got {n}"
    assert pool.free_count() == 2, f"shared free should be 2, got {pool.free_count()}"
    print(f"  after A.grow(2): shared free={pool.free_count()}")

    # Write distinguishable patterns into A's two slots. shrink() uses
    # tail eviction, so slot 1 is the one that will leave A on transfer.
    A_va_slot0 = arena_A.pool_va_base("kv0")
    A_va_slot1 = A_va_slot0 + CHUNK_SIZE
    _write_byte(A_va_slot0, 0xA0, 8)
    _write_byte(A_va_slot1, 0xA1, 8)
    print(f"  wrote A.slot0=0xA0, A.slot1=0xA1")
    assert _read_byte(A_va_slot0, 8) == 0xA0
    assert _read_byte(A_va_slot1, 8) == 0xA1

    # Transfer A.kv0 → B.mamba0, 1 chunk. tail-evicted = slot 1's handle.
    # The handle carrying 0xA1 should land in B.mamba0's first free slot.
    moved = cross_arena_transfer(arena_A, "kv0", arena_B, "mamba0", 1)
    assert moved == 1
    print(f"  cross_arena_transfer(A.kv0 → B.mamba0, 1): moved={moved}")
    assert arena_A.pool_mapped_chunks("kv0") == 1
    assert arena_B.pool_mapped_chunks("mamba0") == 1
    assert pool.free_count() == 2

    # Read through B.slot0 — should see 0xA1 (= the tail-evicted handle's bytes).
    B_va_slot0 = arena_B.pool_va_base("mamba0")
    val = _read_byte(B_va_slot0, 8)
    assert val == 0xA1, f"after transfer, B.slot0 should hold 0xA1, got {val:#x}"
    print(f"  B.slot0 reads {val:#x} — bytes followed the handle. GOOD.")

    # A.slot0 (handle untouched by transfer) should still hold 0xA0.
    val = _read_byte(A_va_slot0, 8)
    assert val == 0xA0, f"A.slot0 should be unaffected (0xA0), got {val:#x}"
    print(f"  A.slot0 still reads {val:#x}, untouched. GOOD.")

    # Write a new pattern through B, transfer back, A should see new pattern
    # at the slot that gets the handle (= A's first free slot = slot 1).
    _write_byte(B_va_slot0, 0xCC, 8)
    moved_back = cross_arena_transfer(arena_B, "mamba0", arena_A, "kv0", 1)
    assert moved_back == 1
    val = _read_byte(A_va_slot1, 8)
    assert val == 0xCC, f"after transfer back, A.slot1 should be 0xCC, got {val:#x}"
    print(f"  transferred back, A.slot1 reads {val:#x}. GOOD.")

    # VA bases stable.
    assert arena_A.pool_va_base("kv0") == A_va_slot0
    assert arena_B.pool_va_base("mamba0") == B_va_slot0
    print("  VA bases unchanged across transfers. GOOD.")

    # Same-pool guard.
    raised = False
    try:
        cross_arena_transfer(arena_A, "kv0", arena_A, "kv0", 1)
    except ValueError:
        raised = True
    assert raised, "cross_arena_transfer(A,A) should raise"
    print("  cross_arena_transfer(A,A) correctly raises.")

    # Different-pool-object guard.
    other_pool = SharedHandlePool(device_id=0, chunk_size=CHUNK_SIZE, n_handles=2)
    arena_C = ChunkArena(
        device_id=0, chunk_size=CHUNK_SIZE, n_handles=0,
        pool_capacities=[("p", 2)], external_handle_pool=other_pool,
    )
    raised = False
    try:
        cross_arena_transfer(arena_A, "kv0", arena_C, "p", 1)
    except ValueError:
        raised = True
    assert raised, "cross_arena_transfer with different SharedHandlePools should raise"
    print("  cross_arena_transfer with disjoint pools correctly raises.")

    arena_C.cleanup()
    other_pool.cleanup()
    arena_B.cleanup()
    arena_A.cleanup()
    pool.cleanup()
    print("  cleanup ok")
    print("PASS Test 1\n")


def test_legacy_self_owned_path() -> None:
    print("== Test 2: legacy self-owned ChunkArena still works ==")
    a = ChunkArena(
        device_id=0, chunk_size=CHUNK_SIZE,
        n_handles=4,
        pool_capacities=[("p", 4)],
        # external_handle_pool=None  (default)
    )
    assert a._external_pool is None
    assert len(a._handles) == 4
    n = a.grow("p", 2)
    assert n == 2
    assert a.pool_mapped_chunks("p") == 2
    raised = False
    try:
        # Attempting cross-arena transfer with no shared pool must raise.
        b = ChunkArena(
            device_id=0, chunk_size=CHUNK_SIZE, n_handles=4,
            pool_capacities=[("q", 4)],
        )
        cross_arena_transfer(a, "p", b, "q", 1)
        b.cleanup()
    except ValueError:
        raised = True
        b.cleanup()
    assert raised
    print("  legacy mode preserved + cross-arena correctly refuses without shared pool")
    a.cleanup()
    print("PASS Test 2\n")


def test_multi_tensor_arenas_share_pool() -> None:
    print("== Test 3: two MultiTensorArenas with shared handle pool ==")
    # Small KV-like: 2 layers × 2 kinds = 4 sub-pools.
    # Small mamba-like: 2 layers × 1 kind = 2 sub-pools.
    # Per-token shape (8, 16), bf16 → 256 B/token. PyTorch's caching allocator
    # grabs ≥ 20 MiB segments, so chunks must be at least 20 MiB. Use 64 MiB
    # chunks (matching the engine's KV/mamba arena defaults) → 262144
    # tokens/chunk; max_tokens = 1 chunk worth.
    n_layers_kv, n_kinds_kv = 2, 2
    n_layers_mamba, n_kinds_mamba = 2, 1
    chunk_bytes = 64 * 1024 * 1024
    per_token_shape = (8, 16)
    dtype = torch.bfloat16
    per_token_bytes = 8 * 16 * 2  # bf16
    tokens_per_chunk = chunk_bytes // per_token_bytes  # = 262144
    # max_tokens = 2 chunks worth of VA per sub-pool (so each has a free slot
    # available to receive a transferred handle). init = 1 chunk worth.
    max_tokens = 2 * tokens_per_chunk
    init_tokens = 1 * tokens_per_chunk

    n_kv_subpools = n_layers_kv * n_kinds_kv
    n_mamba_subpools = n_layers_mamba * n_kinds_mamba
    # Each sub-pool gets 1 chunk at init (= n_subpools handles total),
    # plus headroom for transfer demos.
    n_total_chunks_needed = (n_kv_subpools + n_mamba_subpools) * 1 + 2

    pool = SharedHandlePool(
        device_id=torch.cuda.current_device(),
        chunk_size=chunk_bytes,
        n_handles=n_total_chunks_needed,
    )

    kv = MultiTensorArena(
        device_id=torch.cuda.current_device(),
        n_layers=n_layers_kv, n_kinds=n_kinds_kv,
        per_token_shape=per_token_shape, dtype=dtype,
        max_tokens=max_tokens, init_tokens=init_tokens,
        chunk_bytes=chunk_bytes,
        external_handle_pool=pool,
        subpool_offset=0,
    )
    mamba = MultiTensorArena(
        device_id=torch.cuda.current_device(),
        n_layers=n_layers_mamba, n_kinds=n_kinds_mamba,
        per_token_shape=per_token_shape, dtype=dtype,
        max_tokens=max_tokens, init_tokens=init_tokens,
        chunk_bytes=chunk_bytes,
        external_handle_pool=pool,
        subpool_offset=n_kv_subpools,    # KV took 0..n_kv-1; mamba starts at n_kv
    )
    print(f"  shared pool free after both inited: {pool.free_count()}")
    assert pool.free_count() == n_total_chunks_needed - n_kv_subpools - n_mamba_subpools

    # Ensure each tensor lives at a distinct VA.
    seen = set()
    for li in range(n_layers_kv):
        for ki in range(n_kinds_kv):
            ptr = kv.tensor(li, ki).data_ptr()
            assert ptr not in seen, f"KV ({li},{ki}) at duplicate VA {ptr:#x}"
            seen.add(ptr)
    for li in range(n_layers_mamba):
        for ki in range(n_kinds_mamba):
            ptr = mamba.tensor(li, ki).data_ptr()
            assert ptr not in seen, f"Mamba ({li},{ki}) at duplicate VA {ptr:#x}"
            seen.add(ptr)
    print(f"  all {len(seen)} sub-tensors at distinct VAs. GOOD.")

    # Snapshot data_ptr()s so we can confirm they don't move.
    kv_ptrs = [
        kv.tensor(li, ki).data_ptr()
        for li in range(n_layers_kv) for ki in range(n_kinds_kv)
    ]
    mamba_ptrs = [
        mamba.tensor(li, ki).data_ptr()
        for li in range(n_layers_mamba) for ki in range(n_kinds_mamba)
    ]

    # Do a cross-arena transfer at the underlying ChunkArena level.
    # MultiTensorArena exposes the underlying ChunkArena as `_arena`.
    # Each MTA's first sub-pool name is `_pool_name(0)` = `sub{subpool_offset}`.
    # KV with offset 0 → "sub0"; mamba with offset 4 → "sub4".
    kv_name = kv._pool_name(0)
    mamba_name = mamba._pool_name(0)
    print(f"  KV first sub-pool name: {kv_name}, mamba first sub-pool name: {mamba_name}")
    moved = cross_arena_transfer(kv._arena, kv_name, mamba._arena, mamba_name, 1)
    assert moved == 1
    print(f"  cross_arena_transfer(KV.{kv_name} → mamba.{mamba_name}, 1): moved={moved}")
    print(f"    kv._arena.pool_mapped_chunks({kv_name!r}) = {kv._arena.pool_mapped_chunks(kv_name)}")
    print(f"    mamba._arena.pool_mapped_chunks({mamba_name!r}) = {mamba._arena.pool_mapped_chunks(mamba_name)}")
    assert kv._arena.pool_mapped_chunks(kv_name) == 0
    assert mamba._arena.pool_mapped_chunks(mamba_name) == 2

    # Tensor data_ptrs should be stable across the transfer.
    new_kv_ptrs = [
        kv.tensor(li, ki).data_ptr()
        for li in range(n_layers_kv) for ki in range(n_kinds_kv)
    ]
    new_mamba_ptrs = [
        mamba.tensor(li, ki).data_ptr()
        for li in range(n_layers_mamba) for ki in range(n_kinds_mamba)
    ]
    assert new_kv_ptrs == kv_ptrs, "KV data_ptrs moved across transfer!"
    assert new_mamba_ptrs == mamba_ptrs, "Mamba data_ptrs moved across transfer!"
    print("  all sub-tensor data_ptrs stable across cross-arena transfer. GOOD.")

    # Transfer back.
    moved_back = cross_arena_transfer(mamba._arena, mamba_name, kv._arena, kv_name, 1)
    assert moved_back == 1
    assert kv._arena.pool_mapped_chunks(kv_name) == 1
    assert mamba._arena.pool_mapped_chunks(mamba_name) == 1
    print("  transferred back. GOOD.")

    # Skip cleanup: the PyTorch MemPool destructor sequence interacts badly
    # with our VMM unmap (known issue, see 2e.4.c). Stash the live objects
    # in a module-level register so Python doesn't GC them at function
    # return — main() will os._exit before any destructor runs.
    _KEEPALIVE.extend([kv, mamba, pool])
    print("PASS Test 3\n")


def main() -> int:
    # Initialize the driver and force the PyTorch primary context to come up
    # before any cuMemsetD8_v2 / cuMemcpyDtoH_v2 calls. Otherwise those raw
    # driver-API calls fail with CUDA_ERROR_INVALID_CONTEXT (rc=201).
    CUDA.cuInit(0)
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    test_basic_cross_arena_transfer()
    test_legacy_self_owned_path()
    test_multi_tensor_arenas_share_pool()
    print("== ALL PASS: SharedHandlePool + cross_arena_transfer ready ==")
    return 0


if __name__ == "__main__":
    rc = main()
    # Hard-exit to skip PyTorch MemPool destructors (known buggy with VMM-
    # backed pools, see 2e.4.c). Doesn't affect test validity — the runtime
    # behavior was fully exercised before this point.
    import os
    os._exit(rc)
