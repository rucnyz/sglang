"""T3 part 2 unit test: ChunkArena.shrink_explicit unmaps the
specified slot indices, returns the count actually unmapped, and
silently skips out-of-range / not-currently-mapped slots.

Requires CUDA (chunk_arena uses cuMemCreate/Map directly).
"""

import sys
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.arena.chunk_arena import ChunkArena


def main():
    # Small arena: 1 pool, 8 slots, 2 MiB chunks. CUDA must be available.
    chunk_size = 2 * 1024 * 1024
    arena = ChunkArena(
        device_id=0,
        chunk_size=chunk_size,
        n_handles=8,
        pool_capacities=[("test_pool", 8)],
    )

    # Map all 8 slots.
    n_mapped = arena.grow("test_pool", 8)
    assert n_mapped == 8, f"grow expected 8, got {n_mapped}"
    print(f"[init] mapped 8 slots; free_handles={arena.free_handle_count()}")

    # shrink_explicit on slots [2, 5, 7] — all currently mapped.
    n_unmapped = arena.shrink_explicit("test_pool", [2, 5, 7])
    assert n_unmapped == 3, f"unmapped expected 3, got {n_unmapped}"
    print(f"[shrink_explicit [2,5,7]] unmapped={n_unmapped}; free_handles={arena.free_handle_count()}")
    assert arena.free_handle_count() == 3, \
        f"3 free handles after unmap: {arena.free_handle_count()}"

    # Try to shrink_explicit on slot already unmapped + out-of-range.
    # Should silently skip both.
    n_unmapped2 = arena.shrink_explicit("test_pool", [2, 999, -1, 0])
    # Slot 2 was already unmapped (skip), 999 oob (skip), -1 oob (skip), 0 mapped (unmap).
    assert n_unmapped2 == 1, f"unmapped expected 1, got {n_unmapped2}"
    print(f"[shrink_explicit [2,999,-1,0]] unmapped={n_unmapped2}; free_handles={arena.free_handle_count()}")

    # Try with a torch tensor input (used by the real wiring).
    import torch
    t = torch.tensor([1, 3, 6], dtype=torch.int64, device="cuda")
    n_unmapped3 = arena.shrink_explicit("test_pool", t)
    assert n_unmapped3 == 3, f"unmapped expected 3, got {n_unmapped3}"
    print(f"[shrink_explicit tensor[1,3,6]] unmapped={n_unmapped3}; free_handles={arena.free_handle_count()}")

    arena.cleanup()
    print("\nT3 shrink_explicit unit test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
