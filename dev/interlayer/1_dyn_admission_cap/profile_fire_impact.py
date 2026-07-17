"""Profile the GPU-level impact of a cross-pool fire on decode-like work.

Hypothesis: cuMemMap/cuMemUnmap/cuMemSetAccess invalidate TLB or
otherwise stall the GPU pipeline, slowing decode kernels for some
iterations after a fire.

Test: in a tight loop, run "decode-like" GPU work (matmul +
gather/scatter on a representative-sized tensor). Periodically inject
a cuMemMap/Unmap on a different VA range (no-op semantically — just
re-map a chunk).

Measure: iteration time for N iters before fire, during fire, and N
after. If the post-fire iteration time spikes, that's the cost.

This is isolated from sglang's scheduler / planner / async work so we
can attribute timing changes purely to the cuMem ops.
"""
import ctypes
import statistics
import sys
import time

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
torch.cuda.set_device(0)
DEV = "cuda:0"


def decode_like_iter(req_to_token, kv_pool, conv_state, indices):
    """Mimic a single decode iteration's main memory accesses.
    - Read req_to_token at batch indices (mimics attention KV index lookup)
    - Read KV pool tokens
    - Update conv_state at batch indices
    """
    # 1. Gather req_to_token[indices] — mimics fa3 backend's metadata.page_table
    page_table = req_to_token[indices]  # shape (batch, max_context_len)
    # 2. Read KV pool at the gathered indices (mimics attention)
    kv_read = kv_pool[page_table[:, :128]]  # small slice
    _ = kv_read.sum()
    # 3. Update conv_state — mimics mamba decode state write
    conv_state[indices] += 1
    torch.cuda.synchronize()


def main():
    from sglang.srt.arena.chunk_arena import ChunkArena, SharedHandlePool

    BATCH = 33
    MAX_CTX = 262144
    KV_TOKENS = 100_000

    print("Allocating decode tensors...")
    req_to_token = torch.zeros(BATCH * 4, MAX_CTX, dtype=torch.int32, device=DEV)
    kv_pool = torch.randn(KV_TOKENS, 256, dtype=torch.bfloat16, device=DEV)
    conv_state = torch.zeros(BATCH * 4, 8192, 3, dtype=torch.bfloat16, device=DEV)
    indices = torch.arange(BATCH, dtype=torch.int64, device=DEV)
    # Pre-fill req_to_token rows with valid KV indices
    req_to_token[indices] = torch.randint(
        0, KV_TOKENS, (BATCH, MAX_CTX), dtype=torch.int32, device=DEV
    )
    torch.cuda.synchronize()

    print("Setting up a separate cuMem arena for fire injection...")
    shared = SharedHandlePool(device_id=0, chunk_size=2 * 1024 * 1024, n_handles=32)
    fire_arena = ChunkArena(
        device_id=0, chunk_size=2 * 1024 * 1024, n_handles=0,
        pool_capacities=[("fire", 16)],
        external_handle_pool=shared,
    )
    fire_arena.grow("fire", 8)  # init 8 chunks mapped

    # Warmup
    print("Warmup decode_like iterations...")
    for _ in range(20):
        decode_like_iter(req_to_token, kv_pool, conv_state, indices)

    def time_iters(n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            decode_like_iter(req_to_token, kv_pool, conv_state, indices)
            ts.append(time.perf_counter() - t0)
        return ts

    print("\n=== Baseline: 30 iters, no fire ===")
    base = time_iters(30)
    base_mean = statistics.mean(base) * 1000
    base_sd = statistics.stdev(base) * 1000
    print(f"  baseline: {base_mean:.3f} ± {base_sd:.3f} ms")

    print("\n=== 10 iters → fire (cuMemMap+SetAccess) → 30 iters ===")
    pre = time_iters(10)
    pre_mean = statistics.mean(pre) * 1000

    print(f"  pre-fire: {pre_mean:.3f} ms")

    # Inject "fire" — unmap 4 chunks from fire arena, then re-grow them.
    # Same number of cuMemMap/Unmap/SetAccess as a real fire on a small pool.
    fire_t0 = time.perf_counter()
    fire_arena.shrink("fire", 4)  # unmap 4 chunks
    fire_arena.grow("fire", 4)    # re-map 4 chunks (different handles likely)
    torch.cuda.synchronize()
    fire_wall = (time.perf_counter() - fire_t0) * 1000
    print(f"  fire wall: {fire_wall:.2f} ms (unmap + map + setaccess on 4 chunks)")

    # Immediately after fire: 30 iters
    post = time_iters(30)
    # Split: first 5 (most affected) vs rest
    p1 = post[:5]
    p2 = post[5:]
    print(f"  post-fire iter 1-5:   {statistics.mean(p1)*1000:.3f} ± "
          f"{statistics.stdev(p1)*1000:.3f} ms")
    print(f"  post-fire iter 6-30:  {statistics.mean(p2)*1000:.3f} ± "
          f"{statistics.stdev(p2)*1000:.3f} ms")

    delta1 = (statistics.mean(p1)*1000 - base_mean) / base_mean * 100
    delta2 = (statistics.mean(p2)*1000 - base_mean) / base_mean * 100
    print(f"\n  iter 1-5 vs baseline:  Δ = {delta1:+.2f}%")
    print(f"  iter 6-30 vs baseline: Δ = {delta2:+.2f}%")

    # Run 3 fire-cycles to amplify any effect
    print("\n=== 3 fire-cycles in sequence ===")
    all_post = []
    for cycle in range(3):
        fire_arena.shrink("fire", 4)
        fire_arena.grow("fire", 4)
        torch.cuda.synchronize()
        cycle_post = time_iters(10)
        all_post.extend(cycle_post)
        cycle_mean = statistics.mean(cycle_post) * 1000
        print(f"  cycle {cycle+1} post-fire 10 iters: {cycle_mean:.3f} ms")
    all_mean = statistics.mean(all_post) * 1000
    delta_all = (all_mean - base_mean) / base_mean * 100
    print(f"  combined 30 post-fire iters: {all_mean:.3f} ms (Δ = {delta_all:+.2f}%)")

    fire_arena.cleanup()
    shared.cleanup()


if __name__ == "__main__":
    main()
