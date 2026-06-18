"""cuda_graph_safety — CUDA graph safety post-transfer.

Tests the CORE invariant that makes interlayer work: captured CUDA
graphs reference (a) pool VA base addresses (stable for arena
lifetime), (b) page_table tensors (kernel input, NOT graph-baked).
After cuMemUnmap+cuMemMap moves physical handles between pools, the
SAME graph re-played with an updated page_table reads from the new
mappings.

Sub-tests:

  1. Intra-arena: capture graph, transfer adds chunks, replay reads
     correct values across all slots (incl. full-chunk byte check on
     new slot to catch granularity-mismatch bugs). VA/data_ptr
     stability asserted inline (merged from old test_2).
  2. Cross-arena (production KV↔mamba path): same scenario via
     cross_arena_transfer.
  3. Shrink+regrow round-trip: snapshot the original handle_idx, force
     it through the free-list, assert a DIFFERENT handle_idx is bound
     to the same VA after regrow — then verify graph replay reads via
     the NEW handle (proves kernel doesn't bake physical handle).
"""
import sys
import torch

from sglang.srt.arena.chunk_arena import (
    SharedHandlePool, ChunkArena, cross_arena_transfer,
)
from sglang.srt.arena.from_blob_ext import tensor_from_va


CHUNK_SIZE = 2 * 1024 * 1024
DEVICE     = 0


def _fpc():
    return CHUNK_SIZE // 4


def _make_pool(n):
    return SharedHandlePool(DEVICE, CHUNK_SIZE, n_handles=n)


def _make_arena(pool, caps):
    return ChunkArena(DEVICE, CHUNK_SIZE,
                      n_handles=sum(c for _, c in caps),
                      pool_capacities=caps,
                      external_handle_pool=pool)


def _tensor_over(arena, pool_name, cap):
    return tensor_from_va(arena.pool_va_base(pool_name),
                          [cap * _fpc()], torch.float32, DEVICE)


def _capture_indexed_read(t_src: torch.Tensor, page_table: torch.Tensor,
                           output: torch.Tensor) -> torch.cuda.CUDAGraph:
    """Capture a CUDAGraph that runs output.copy_(t_src[page_table])."""
    for _ in range(3):           # warmup (CUDA graph allocator state)
        output.copy_(t_src[page_table])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output.copy_(t_src[page_table])
    torch.cuda.synchronize()
    return g


# ---------- sub-tests ----------

def test_1_intra_arena_graph_replay_after_transfer():
    """Capture graph reading 12 logical slots; only 8 mapped at capture.
    Transfer adds 4 chunks. Update page_table. Replay reads all 12 OK.

    Also asserts VA + data_ptr stability across the entire scenario.
    Final check: full-chunk byte integrity of one newly-mapped slot
    (catches a CUDA-driver-mapping-wrong-page-size bug).
    """
    INIT_A, INIT_B, MOVE = 12, 8, 4
    N_LOGICAL = INIT_B + MOVE   # 12

    pool = _make_pool(32)
    try:
        arena = _make_arena(pool, [("A", 16), ("B", 16)])
        try:
            arena.grow("A", INIT_A)
            arena.grow("B", INIT_B)
            t_B = _tensor_over(arena, "B", 16)
            fpc = _fpc()

            # Sentinel: first float of each mapped chunk
            for i in range(INIT_B):
                t_B[i * fpc].fill_(float(100 + i))
            torch.cuda.synchronize()

            # Snapshot VA / data_ptr (merged-in from old test_2)
            va_before = arena.pool_va_base("B")
            ptr_before = t_B.data_ptr()
            assert ptr_before == va_before

            # page_table: 12 entries; last 4 dummy (will be updated)
            page_table = torch.tensor(
                [i * fpc for i in range(INIT_B)] + [0] * MOVE,
                dtype=torch.int64, device='cuda')
            output = torch.zeros(N_LOGICAL, device='cuda')

            g = _capture_indexed_read(t_B, page_table, output)

            # Pre-transfer sanity replay
            output.zero_(); g.replay(); torch.cuda.synchronize()
            for i in range(INIT_B):
                assert output[i].item() == float(100 + i)
            for i in range(INIT_B, N_LOGICAL):
                assert output[i].item() == 100.0   # dummy → slot 0

            # === Transfer A → B (4 chunks)
            assert arena.transfer_chunks("A", "B", MOVE) == MOVE
            torch.cuda.synchronize()

            # Write sentinels to NEW chunks
            for i in range(INIT_B, INIT_B + MOVE):
                t_B[i * fpc].fill_(float(100 + i))
            torch.cuda.synchronize()

            # Update page_table values in place
            page_table.copy_(torch.tensor(
                [i * fpc for i in range(N_LOGICAL)],
                dtype=torch.int64, device='cuda'))

            # Replay — same graph, new page_table values, new mappings
            output.zero_(); g.replay(); torch.cuda.synchronize()
            for i in range(N_LOGICAL):
                v = output[i].item()
                expected = float(100 + i)
                assert v == expected, f"output[{i}]={v}, expected {expected}"

            # VA / data_ptr unchanged (merged-in invariant)
            assert arena.pool_va_base("B") == va_before
            assert t_B.data_ptr() == ptr_before

            # Full-chunk byte integrity on one NEW slot — catches
            # granularity-mismatch bugs (cuMemMap mapped 4 KiB instead
            # of 2 MiB)
            new_slot = INIT_B + MOVE - 1   # slot 11
            t_B[new_slot * fpc:(new_slot + 1) * fpc].fill_(777.0)
            torch.cuda.synchronize()
            full_chunk = t_B[new_slot * fpc:(new_slot + 1) * fpc]
            if not torch.all(full_chunk == 777.0):
                first_bad = (full_chunk != 777.0).nonzero(as_tuple=True)[0][0].item()
                raise AssertionError(
                    f"new slot {new_slot}: byte offset {first_bad * 4} mismatched. "
                    f"cuMemMap may have mapped less than chunk_size.")
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


def test_2_cross_arena_graph_replay_after_transfer():
    """Same scenario as test_1 but via cross_arena_transfer — the
    production KV↔mamba path. This test covers the production path;
    without it, only the simplified intra-arena path is proven
    graph-safe."""
    INIT_KV, INIT_MAMBA, MOVE = 12, 8, 4
    N_LOGICAL = INIT_MAMBA + MOVE

    pool = _make_pool(32)
    try:
        arena_KV = _make_arena(pool, [("kv", 16)])
        arena_M  = _make_arena(pool, [("mamba", 16)])
        try:
            arena_KV.grow("kv", INIT_KV)
            arena_M.grow("mamba", INIT_MAMBA)
            t_M = _tensor_over(arena_M, "mamba", 16)
            fpc = _fpc()

            for i in range(INIT_MAMBA):
                t_M[i * fpc].fill_(float(200 + i))
            torch.cuda.synchronize()

            page_table = torch.tensor(
                [i * fpc for i in range(INIT_MAMBA)] + [0] * MOVE,
                dtype=torch.int64, device='cuda')
            output = torch.zeros(N_LOGICAL, device='cuda')
            g = _capture_indexed_read(t_M, page_table, output)

            # === Cross-arena transfer KV → mamba
            granted = cross_arena_transfer(arena_KV, "kv",
                                             arena_M, "mamba", MOVE)
            assert granted == MOVE
            torch.cuda.synchronize()

            for i in range(INIT_MAMBA, INIT_MAMBA + MOVE):
                t_M[i * fpc].fill_(float(200 + i))
            torch.cuda.synchronize()

            page_table.copy_(torch.tensor(
                [i * fpc for i in range(N_LOGICAL)],
                dtype=torch.int64, device='cuda'))

            output.zero_(); g.replay(); torch.cuda.synchronize()
            for i in range(N_LOGICAL):
                v = output[i].item()
                expected = float(200 + i)
                assert v == expected, \
                    f"cross-arena: output[{i}]={v}, expected {expected}"
        finally:
            arena_M.cleanup(); arena_KV.cleanup()
    finally:
        pool.cleanup()


def test_3_same_va_new_handle_graph_reads_new_data():
    """Shrink+regrow puts a DIFFERENT physical handle at the SAME VA
    slot. Captured graph reads correct data via the new handle.

    Critically: this test VERIFIES the handle actually changed (snapshot
    old handle_idx, force it through the free-list, assert new
    handle_idx differs). Without that verification, the test would
    silently pass even if shrink was a no-op — read 44 after writing
    44 would succeed regardless.
    """
    pool = _make_pool(8)
    try:
        # Two pools so we can park the old handle in a different pool
        # to force a different handle_idx to come back to slot 3 on regrow
        arena = _make_arena(pool, [("A", 4), ("B", 4)])
        try:
            arena.grow("A", 4)   # uses 4 handles (likely 0..3 LIFO order)
            arena.grow("B", 4)   # uses next 4 handles
            t_B = _tensor_over(arena, "B", 4)
            fpc = _fpc()

            old_handle_idx = arena.pools["B"].mapped[3]
            assert old_handle_idx is not None
            print(f"      slot B[3] originally backed by handle "
                  f"#{old_handle_idx}")

            # Write sentinel + capture graph
            t_B[3 * fpc].fill_(33.0)
            torch.cuda.synchronize()
            idx = torch.tensor([3 * fpc], dtype=torch.int64, device='cuda')
            out = torch.zeros(1, device='cuda')
            g = _capture_indexed_read(t_B, idx, out)

            out.zero_(); g.replay(); torch.cuda.synchronize()
            assert out[0].item() == 33.0   # baseline sanity

            # Shrink B by 1 (tail-evict frees slot 3's handle to pool.free)
            arena.shrink("B", 1)
            torch.cuda.synchronize()
            assert arena.pools["B"].mapped[3] is None
            assert old_handle_idx in pool.free, \
                "old handle should be back in shared pool free list"

            # Force a different handle to be at the LIFO top so grow(B)
            # gets it instead of the original. Create a fresh handle via
            # pool.grow(1) — it appends to pool.free → becomes the
            # next-popped (LIFO).
            pool.grow(1)
            assert pool.free[-1] != old_handle_idx, \
                "new handle index should be different from original; " \
                "if equal, SharedHandlePool's handle-id allocation reused " \
                "an index (which the design says it shouldn't)"

            # Grow B → pops the new (different) handle into slot 3
            arena.grow("B", 1)
            torch.cuda.synchronize()
            new_handle_idx = arena.pools["B"].mapped[3]
            assert new_handle_idx is not None
            assert new_handle_idx != old_handle_idx, \
                f"slot B[3] got the SAME handle #{new_handle_idx} back — " \
                f"test premise broken; cannot verify 'new handle' claim"
            print(f"      slot B[3] now backed by handle #{new_handle_idx} "
                  f"(was #{old_handle_idx})")

            # Now write a different sentinel and verify graph reads NEW data
            # (which proves: same VA → new physical handle → graph reads it)
            t_B[3 * fpc].fill_(44.0)
            torch.cuda.synchronize()
            out.zero_(); g.replay(); torch.cuda.synchronize()
            assert out[0].item() == 44.0, (
                f"got {out[0].item()}, expected 44.0. Captured graph "
                f"reads via VA {arena.pool_va_base('B') + 3 * CHUNK_SIZE} "
                f"which is now backed by a different physical handle "
                f"(#{new_handle_idx}); if the kernel had baked the "
                f"original handle, this read would be stale.")
        finally:
            arena.cleanup()
    finally:
        pool.cleanup()


# ---------- runner ----------

def main():
    tests = [
        ("1 intra-arena graph + transfer + page_table update + full-chunk",
         test_1_intra_arena_graph_replay_after_transfer),
        ("2 cross_arena_transfer graph safety (KV↔mamba production path)",
         test_2_cross_arena_graph_replay_after_transfer),
        ("3 same VA, verified-different handle, graph reads new data",
         test_3_same_va_new_handle_graph_reads_new_data),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            import traceback
            traceback.print_exc()
    print(f"\ncuda_graph_safety: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
