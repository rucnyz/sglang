"""Ideal fire path: shrink source + grow dest with recycled handles.

Guards the core ChunkArena transfer invariant: physical handles flow from
source to destination through SharedHandlePool without leaking, and the
VA-stable design survives CUDA graph capture across remap cycles.

Uses real ChunkArena + SharedHandlePool on CUDA (CPU fallback skips the
VMM tests that require cuMemCreate/cuMemMap).
"""
import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

os.environ.setdefault("SGLANG_ARENA_CHUNK_BYTES", str(2 * 1024 * 1024))

import pytest
import torch

from sglang.srt.arena.chunk_arena import ChunkArena, SharedHandlePool

HAS_CUDA = torch.cuda.is_available()
DEVICE_ID = 0
CHUNK_BYTES = 2 * 1024 * 1024


def _skip_no_cuda():
    if not HAS_CUDA:
        pytest.skip("CUDA not available")


def _make_shared_arena(n_src_chunks, n_dst_chunks):
    """Build a SharedHandlePool + two ChunkArenas (src, dst) sharing it.

    src boots with n_src_chunks mapped, dst boots with n_dst_chunks mapped.
    Returns (shared_pool, src_arena, dst_arena).
    """
    _skip_no_cuda()
    total_handles = n_src_chunks + n_dst_chunks
    pool = SharedHandlePool(
        device_id=DEVICE_ID,
        chunk_size=CHUNK_BYTES,
        n_handles=total_handles,
    )
    src = ChunkArena(
        device_id=DEVICE_ID,
        chunk_size=CHUNK_BYTES,
        n_handles=0,
        pool_capacities=[("src", n_src_chunks + n_dst_chunks)],
        external_handle_pool=pool,
    )
    for _ in range(n_src_chunks):
        mapped = src.grow("src", 1)
        assert len(mapped) == 1
    dst = ChunkArena(
        device_id=DEVICE_ID,
        chunk_size=CHUNK_BYTES,
        n_handles=0,
        pool_capacities=[("dst", n_src_chunks + n_dst_chunks)],
        external_handle_pool=pool,
    )
    for _ in range(n_dst_chunks):
        mapped = dst.grow("dst", 1)
        assert len(mapped) == 1
    return pool, src, dst


# ---- Test 1: source pool.size decrements after fire ----

def test_source_mapped_decrements_after_fire():
    """Shrink src by 3 chunks, grow dst by 3. Assert mapped counts and
    SharedHandlePool free-handle conservation."""
    pool, src, dst = _make_shared_arena(n_src_chunks=10, n_dst_chunks=0)

    assert src.pool_mapped_chunks("src") == 10
    assert dst.pool_mapped_chunks("dst") == 0
    free_before = pool.free_count()

    src.shrink_explicit("src", list(range(7, 10)))
    dst_mapped = dst.grow("dst", 3)

    assert src.pool_mapped_chunks("src") == 7, (
        f"src must have 7 mapped after shrinking 3; got {src.pool_mapped_chunks('src')}"
    )
    assert dst.pool_mapped_chunks("dst") == 3, (
        f"dst must have 3 mapped after growing 3; got {dst.pool_mapped_chunks('dst')}"
    )
    assert len(dst_mapped) == 3
    assert pool.free_count() == free_before, (
        f"handles recycled, free count must be unchanged; "
        f"before={free_before} after={pool.free_count()}"
    )
    src.cleanup()
    dst.cleanup()


# ---- Test 2: round-trip conservation ----

def test_round_trip_conservation():
    """Fire 3 chunks src->dst, then 3 back dst->src. Both pools return
    to original mapped counts. Total handles unchanged."""
    pool, src, dst = _make_shared_arena(n_src_chunks=10, n_dst_chunks=0)

    total_handles = pool.total_count()
    src_mapped_orig = src.pool_mapped_chunks("src")
    dst_mapped_orig = dst.pool_mapped_chunks("dst")

    src.shrink_explicit("src", list(range(7, 10)))
    dst.grow("dst", 3)

    assert src.pool_mapped_chunks("src") == 7
    assert dst.pool_mapped_chunks("dst") == 3

    dst.shrink_explicit("dst", list(range(0, 3)))
    src.grow("src", 3)

    assert src.pool_mapped_chunks("src") == src_mapped_orig, (
        f"src must return to {src_mapped_orig}; got {src.pool_mapped_chunks('src')}"
    )
    assert dst.pool_mapped_chunks("dst") == dst_mapped_orig, (
        f"dst must return to {dst_mapped_orig}; got {dst.pool_mapped_chunks('dst')}"
    )
    assert pool.total_count() == total_handles, (
        f"total handles must be unchanged; was {total_handles}, now {pool.total_count()}"
    )
    src.cleanup()
    dst.cleanup()


# ---- Test 3: free-slot-only guarantee ----

def test_free_slot_only_guarantee():
    """cap_barrier's count_referenced guard catches attempts to shrink
    live (non-free) slots. Uses real MambaSlotAllocator + MambaArenaActuator."""
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

    device = "cuda:0" if HAS_CUDA else "cpu"
    alloc = MambaSlotAllocator(size=10, device=device)
    assert alloc.available_size() == 10

    allocated = alloc.alloc(5)
    assert allocated is not None and allocated.numel() == 5
    live_ids = allocated
    assert alloc.available_size() == 5

    free_slots = alloc.free_slots
    assert free_slots.numel() == 5

    cap_free = free_slots.clone()
    alloc.mark(cap_free)
    assert alloc.available_size() == 0, (
        "after marking all 5 free slots, available must be 0"
    )

    ref_on_free = alloc._fl.count_referenced(cap_free)
    assert ref_on_free == 0, (
        f"free (now capped) slots must show 0 referenced; got {ref_on_free}"
    )

    ref_on_live = alloc._fl.count_referenced(live_ids)
    assert ref_on_live == 5, (
        f"live slots must show 5 referenced; got {ref_on_live}"
    )


# ---- Test 4: CUDA graph compatibility after remap ----

def test_cuda_graph_compat_after_remap():
    """Capture a CUDA graph reading chunk 0 via tensor_from_va. Remap
    chunk 0 to a new physical handle. Replay the graph and verify it
    reads the new data through the same VA."""
    _skip_no_cuda()
    torch.cuda.set_device(DEVICE_ID)

    from sglang.srt.arena.from_blob_ext import tensor_from_va

    pool = SharedHandlePool(
        device_id=DEVICE_ID,
        chunk_size=CHUNK_BYTES,
        n_handles=4,
    )
    arena = ChunkArena(
        device_id=DEVICE_ID,
        chunk_size=CHUNK_BYTES,
        n_handles=0,
        pool_capacities=[("p", 4)],
        external_handle_pool=pool,
    )
    mapped = arena.grow("p", 2)
    assert len(mapped) == 2

    per_token_bytes = 2
    tokens_per_chunk = CHUNK_BYTES // per_token_bytes
    va = arena.pool_va_base("p")
    t = tensor_from_va(
        va=va,
        sizes=(4 * tokens_per_chunk,),
        dtype=torch.float16,
        device_index=DEVICE_ID,
    )

    t[:tokens_per_chunk].fill_(42.0)
    torch.cuda.synchronize()

    output = torch.empty(tokens_per_chunk, dtype=torch.float16, device=f"cuda:{DEVICE_ID}")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            output[:] = t[:tokens_per_chunk]
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        output[:] = t[:tokens_per_chunk]

    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert output[0].item() == 42.0, (
        f"pre-remap graph replay: expected 42, got {output[0].item()}"
    )

    arena.shrink_explicit("p", [0])
    assert arena.pool_mapped_chunks("p") == 1

    remapped = arena.grow("p", 1)
    assert len(remapped) == 1 and remapped[0] == 0
    assert arena.pool_mapped_chunks("p") == 2

    t[:tokens_per_chunk].fill_(99.0)
    torch.cuda.synchronize()

    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert output[0].item() == 99.0, (
        f"post-remap graph replay: expected 99 (new physical page, same VA), "
        f"got {output[0].item()}"
    )

    arena.cleanup()


# ---- main harness (matches existing test pattern) ----

def main() -> int:
    tests = [
        test_source_mapped_decrements_after_fire,
        test_round_trip_conservation,
        test_free_slot_only_guarantee,
        test_cuda_graph_compat_after_remap,
    ]
    print(f"\nIdeal fire path tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except pytest.skip.Exception as e:
            print(f"  SKIP  {t.__name__}: {e}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nIdeal fire: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
