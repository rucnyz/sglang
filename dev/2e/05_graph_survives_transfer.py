"""
Phase 2e.2.c — captured CUDA graph survives a cross-pool transfer.

This validates the paper's §4.4 soft-cap claim end-to-end: any CUDA graph
captured against a tensor that lives in the always-mapped (static-min)
portion of a pool's VA range remains valid across `transfer_chunks` calls
that operate in the soft (above-static-min) portion.

Setup:
  - Pool A has 2 chunks mapped (slots 0, 1). Slot 0 is "always-mapped" —
    we will not unmap it. Tensor T_A lives in slot 0. Slot 1 is "soft."
  - Pool B has 2 chunks mapped. Slot 0 holds T_B; slot 1 is "soft."
  - We capture a CUDA graph that increments T_A by 1.
  - Replay the graph once: T_A goes from N to N+1.
  - `transfer_chunks(B, A, 1)`: B's slot 1 (soft) moves to A's slot 2 (soft).
    Neither A's slot 0 nor B's slot 0 are touched.
  - Replay the graph again: T_A goes from N+1 to N+2. The graph is still
    valid because the bytes it references were never unmapped.

Run: CUDA_VISIBLE_DEVICES=2 python dev/2e/05_graph_survives_transfer.py
"""

import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from chunk_arena import CUDA, ChunkArena, _check  # noqa: E402

CUDA.cuMemsetD8_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_ubyte, ctypes.c_size_t]


def memset_d8(va: int, value: int, n: int) -> None:
    _check(CUDA.cuMemsetD8_v2(va, ctypes.c_ubyte(value), n), f"memset 0x{value:02x}")
    _check(CUDA.cuCtxSynchronize(), "ctx sync")


def main() -> int:
    print("== Phase 2e.2.c: CUDA graph survives cross-pool transfer ==")

    _check(CUDA.cuInit(0), "cuInit")
    dev = ctypes.c_int(-1)
    _check(CUDA.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    dev_id = dev.value

    import torch
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    chunk = 32 * 1024 * 1024
    arena = ChunkArena(
        device_id=dev_id,
        chunk_size=chunk,
        n_handles=4,
        pool_capacities=[("A", 4), ("B", 4)],
    )
    arena.grow("A", 2)
    arena.grow("B", 2)
    a_va = arena.pool_va_base("A")
    b_va = arena.pool_va_base("B")
    memset_d8(a_va + 0 * chunk, 0xA0, chunk)
    memset_d8(a_va + 1 * chunk, 0xA1, chunk)
    memset_d8(b_va + 0 * chunk, 0xB0, chunk)
    memset_d8(b_va + 1 * chunk, 0xB1, chunk)

    so_path = os.path.join(os.path.dirname(__file__), "arena_multi.so")
    multi_lib = ctypes.CDLL(so_path)
    multi_lib.multi_init.argtypes = [
        ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t]
    multi_lib.multi_set_capacity.argtypes = [ctypes.c_int, ctypes.c_size_t]
    multi_lib.multi_init(0, a_va, chunk, arena.pool_mapped_chunks("A"))
    multi_lib.multi_init(1, b_va, chunk, arena.pool_mapped_chunks("B"))

    from torch.cuda.memory import CUDAPluggableAllocator
    plug_a = CUDAPluggableAllocator(so_path, "pool0_malloc", "pool0_free")
    plug_b = CUDAPluggableAllocator(so_path, "pool1_malloc", "pool1_free")
    pool_a = torch.cuda.MemPool(allocator=plug_a.allocator())
    pool_b = torch.cuda.MemPool(allocator=plug_b.allocator())

    # Allocate tensors in each pool's static-min region (= slot 0 in this test).
    # We use uint32 so increments don't overflow during the experiment.
    n_elem = 256 * 1024  # 1 MiB / 4 bytes
    with torch.cuda.use_mem_pool(pool_a):
        t_a = torch.zeros(n_elem, dtype=torch.int32, device="cuda")
    with torch.cuda.use_mem_pool(pool_b):
        t_b = torch.zeros(n_elem, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()

    print(f"t_a at 0x{t_a.data_ptr():x}, t_b at 0x{t_b.data_ptr():x}")
    assert a_va <= t_a.data_ptr() < a_va + chunk
    assert b_va <= t_b.data_ptr() < b_va + chunk

    # ---- Capture a graph that increments t_a by 1.
    # Use a dedicated stream (CUDA graph capture requirement).
    stream = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()

    # Warmup on the stream first.
    with torch.cuda.stream(stream):
        t_a += 1
    torch.cuda.synchronize()
    print(f"after warmup: t_a[0]={t_a[0].item()}")
    assert t_a[0].item() == 1

    # Capture. Reset t_a to a known value first.
    t_a.zero_()
    torch.cuda.synchronize()

    with torch.cuda.graph(g, stream=stream):
        t_a += 1
    print("graph captured")

    # ---- First replay (no transfer yet).
    g.replay()
    torch.cuda.synchronize()
    v1 = t_a[0].item()
    print(f"after first replay: t_a[0]={v1} (expected 1)")
    assert v1 == 1

    # ---- Cross-pool transfer in the soft region.
    # B's slot 1 (containing 0xB1) → A's slot 2 (newly mapped). Neither A's
    # slot 0 (where t_a lives) nor B's slot 0 (where t_b lives) is touched.
    print("-- transfer B->A (1 chunk, soft region) --")
    n = arena.transfer_chunks("B", "A", 1)
    assert n == 1
    multi_lib.multi_set_capacity(0, arena.pool_mapped_chunks("A"))
    multi_lib.multi_set_capacity(1, arena.pool_mapped_chunks("B"))
    print(f"after transfer: A={arena.pool_mapped_chunks('A')} B={arena.pool_mapped_chunks('B')}")

    # Tensor pointers stable.
    assert t_a.data_ptr() == a_va
    assert t_b.data_ptr() == b_va

    # ---- Replay the captured graph again, after the transfer.
    g.replay()
    torch.cuda.synchronize()
    v2 = t_a[0].item()
    print(f"after replay post-transfer: t_a[0]={v2} (expected 2)")
    assert v2 == 2, f"graph replay broke after transfer: t_a[0]={v2}"

    # And one more, just to be sure the graph is still good.
    g.replay()
    torch.cuda.synchronize()
    v3 = t_a[0].item()
    print(f"after second replay post-transfer: t_a[0]={v3} (expected 3)")
    assert v3 == 3

    # ---- Free the graph BEFORE arena cleanup so the captured stream
    # doesn't outlive the mappings.
    del g
    del t_a, t_b
    del pool_a, pool_b
    arena.cleanup()

    print("\n== PASSED: captured CUDA graph survives cross-pool transfer ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
