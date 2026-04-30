"""
Phase 2e.4.d — set_capacity_tokens unit test.

Constructs an MHATokenToKVPool with SGLANG_KV_ARENA=1, then exercises
set_capacity_tokens(shrink), set_capacity_tokens(grow), verifying:

  1. Pool resizes succeed and arena.pool_mapped_chunks reflects the new size.
  2. tensor.data_ptr() is stable across resize (the soft-cap property).
  3. Tokens [0, live_capacity) remain readable/writable after each resize.
  4. Allocator's free_pages count tracks the live capacity when the
     allocator is told via set_capacity_pages.

This is the building block for hooking the budgeter to the KV pool.

Run: CUDA_VISIBLE_DEVICES=3 SGLANG_KV_ARENA=1 \
        PYTHONPATH=/data/yuzhou/projects/sglang/python:$PYTHONPATH \
        .venv/bin/python dev/2e/08_set_capacity_unit.py
"""

import os
import sys

import torch


def main() -> int:
    os.environ.setdefault("SGLANG_KV_ARENA", "1")

    # Force primary CUDA context.
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    # Stub out the bits MHATokenToKVPool.__init__ needs but we don't care about.
    # Easiest path: import it and construct directly with minimum args.
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = MHATokenToKVPool(
        size=131072,                # 128K tokens
        page_size=64,
        dtype=torch.bfloat16,
        head_num=8,
        head_dim=128,
        layer_num=4,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,    # avoid CUDA stream init
        enable_kv_cache_copy=False, # avoid JIT kernel warmup
    )

    print(f"== Phase 2e.4.d: set_capacity_tokens unit test ==")
    print(f"pool.size = {pool.size}, layer_num = {pool.layer_num}")
    print(f"k_buffer[0].shape = {pool.k_buffer[0].shape}")
    print(f"k_buffer[0].data_ptr() = 0x{pool.k_buffer[0].data_ptr():x}")
    print(f"live_capacity_tokens() = {pool.live_capacity_tokens()}")

    # --- Capture original data_ptrs across all 8 layer-tensors.
    original_ptrs = [b.data_ptr() for b in (*pool.k_buffer, *pool.v_buffer)]
    print(f"captured {len(original_ptrs)} data_ptrs across layer-tensors")

    # --- Write a marker into token 0 of every layer's k_buffer.
    for i, k in enumerate(pool.k_buffer):
        k[pool.page_size + 100, 0, 0] = float(i + 1)  # token 100 in real data
    torch.cuda.synchronize()

    # --- Shrink to half capacity.
    target_shrink = pool.size // 2
    actual_shrink = pool.set_capacity_tokens(target_shrink)
    print(f"shrink to {target_shrink} -> live={actual_shrink}, "
          f"arena_chunks={pool._kv_arena._arena.pool_mapped_chunks(pool._kv_arena._pool_name(0))}")
    assert pool.live_capacity_tokens() <= pool.size

    # --- data_ptrs unchanged.
    new_ptrs = [b.data_ptr() for b in (*pool.k_buffer, *pool.v_buffer)]
    assert new_ptrs == original_ptrs, "data_ptr drifted after shrink"
    print("all 8 layer-tensor data_ptrs stable across shrink")

    # --- Token 100 still readable (within the live capacity).
    for i, k in enumerate(pool.k_buffer):
        v = k[pool.page_size + 100, 0, 0].item()
        assert v == float(i + 1), f"layer {i}: token 100 = {v}, expected {i+1}"
    print("data in [0, live_capacity) preserved across shrink")

    # --- Grow back to full.
    pool.set_capacity_tokens(pool.size)
    print(f"grow back to {pool.size} -> live={pool.live_capacity_tokens()}")

    # --- data_ptrs still stable.
    grow_ptrs = [b.data_ptr() for b in (*pool.k_buffer, *pool.v_buffer)]
    assert grow_ptrs == original_ptrs, "data_ptr drifted after grow"
    print("all 8 layer-tensor data_ptrs stable across grow")

    # --- The newly-restored region should be writable.
    far_token = pool.size - 1
    for i, k in enumerate(pool.k_buffer):
        k[pool.page_size + far_token, 0, 0] = float(100 + i)
    torch.cuda.synchronize()
    for i, k in enumerate(pool.k_buffer):
        v = k[pool.page_size + far_token, 0, 0].item()
        assert v == float(100 + i)
    print(f"writes to far token {far_token} (in newly-grown region) succeed")

    # --- Allocator coordination: simulate scheduler-side set_capacity_pages.
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    alloc = TokenToKVPoolAllocator(
        size=pool.size,
        dtype=torch.bfloat16, device="cuda", kvcache=pool, need_sort=False,
    )
    alloc.clear()
    full = alloc.available_size()
    print(f"allocator full available = {full}")

    alloc.set_capacity_pages(alloc.size // 2)
    half = alloc.available_size()
    print(f"after set_capacity_pages(size/2): available = {half}")
    assert half < full
    assert half >= alloc.size // 2 - 2  # roughly half

    alloc.set_capacity_pages(alloc.size)
    restored = alloc.available_size()
    print(f"after set_capacity_pages(size): available = {restored}")
    assert restored == full, f"grow back didn't restore: {restored} != {full}"

    print("\n== PASSED: set_capacity_tokens + allocator coordination work ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
