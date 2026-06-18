"""Profile: does cuMemMap/Unmap on an ARENA slow down decode kernels
that READ FROM that arena?

Difference from profile_cumem_decode_impact.py:
  - the "decode workload" reads from an arena-backed tensor (cuMemMap)
  - the cuMem operations modify chunks in THE SAME arena
  - this matches the real sglang pattern: KV pool reads + cross-pool
    fire that unmaps KV chunks

If the slowdown is from "scattered physical pages after cuMem reshuffle"
(TLB / MMU layout effect), this test should show it.
"""
import statistics
import sys

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
DEVICE = "cuda:0"
torch.cuda.set_device(0)


def main():
    from sglang.srt.arena.chunk_arena import ChunkArena, SharedHandlePool
    from sglang.srt.arena.from_blob_ext import tensor_from_va

    print("Allocating arena-backed KV-like tensor...")
    n_chunks = 4096        # ~8 GB
    chunk_bytes = 2 * 1024 * 1024
    shared = SharedHandlePool(device_id=0, chunk_size=chunk_bytes,
                              n_handles=n_chunks + 256)
    arena = ChunkArena(
        device_id=0, chunk_size=chunk_bytes, n_handles=0,
        pool_capacities=[("kv", n_chunks + 256)],
        external_handle_pool=shared,
    )
    arena.grow("kv", n_chunks)
    # build a flat bf16 view over the mapped range
    pool_state = arena.pools["kv"]
    n_bytes = n_chunks * chunk_bytes
    n_elems = n_bytes // 2
    kv_flat = tensor_from_va(pool_state.va_base, [n_elems],
                             torch.bfloat16, device_index=0)
    # Reshape to (rows, hidden)
    hidden = 4096
    n_elems = n_bytes // 2
    rows = n_elems // hidden
    kv = kv_flat[:rows * hidden].view(rows, hidden)
    print(f"  arena KV: {rows}x{hidden} bf16 = {rows*hidden*2/1024**3:.1f} GiB "
          f"({n_chunks} chunks)")

    # Fill with random data via a temporary CUDA tensor
    init = torch.randn_like(kv)
    kv.copy_(init)
    del init

    # Decode work
    batch = 33
    seq = 8192
    indices = torch.randint(0, rows, (batch, seq), dtype=torch.int64,
                            device=DEVICE)
    weights = torch.randn(hidden, hidden, dtype=torch.bfloat16, device=DEVICE)

    def decode_iter():
        gathered = kv[indices.flatten()].view(batch, seq, hidden)
        out = (gathered @ weights).sum(dim=1)
        return out.sum()

    # Warmup
    for _ in range(20):
        _ = decode_iter()
    torch.cuda.synchronize()

    def measure(n):
        out = []
        for _ in range(n):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); _ = decode_iter(); e.record()
            torch.cuda.synchronize()
            out.append(s.elapsed_time(e))
        return out

    print("\n=== Baseline (no cuMem activity) ===")
    base = measure(200)
    base_mean = statistics.mean(base)
    base_std = statistics.stdev(base)
    print(f"  baseline n=200: mean={base_mean:.3f}ms ± {base_std:.3f}ms")

    # Test 1: shrink + grow the SAME chunks (no net change)
    def shrink_then_grow(n_pages):
        arena.shrink("kv", n_pages)
        arena.grow("kv", n_pages)
        torch.cuda.synchronize()

    print("\n=== shrink+grow same chunks (test pure cuMem churn) ===")
    for n_pages in [3, 48, 192]:
        # Pre-shrink any pending pages
        import time
        t0 = time.perf_counter()
        shrink_then_grow(n_pages)
        wall = (time.perf_counter() - t0) * 1000

        # Post measurements
        post = measure(200)
        post_mean = statistics.mean(post)
        post_std = statistics.stdev(post)
        delta = (post_mean - base_mean) / base_mean * 100
        print(f"  n_pages={n_pages:>3d}: fire_wall={wall:>5.1f}ms  "
              f"post mean={post_mean:.3f}ms ± {post_std:.3f}ms  "
              f"Δ={delta:+.2f}%")

    # Test 2: actually unmap + remap with different physical handles
    # (so VA may be re-laid)
    print("\n=== unmap to shared, then re-grow (different physical) ===")
    for n_pages in [3, 48, 192]:
        # Snapshot baseline before
        b2 = measure(50)
        b2_mean = statistics.mean(b2)
        # Move chunks out and back in
        arena.shrink("kv", n_pages)
        arena.grow("kv", n_pages)
        torch.cuda.synchronize()
        post = measure(200)
        post_mean = statistics.mean(post)
        delta = (post_mean - b2_mean) / b2_mean * 100
        print(f"  n_pages={n_pages:>3d}: pre={b2_mean:.3f}ms post={post_mean:.3f}ms Δ={delta:+.2f}%")

    arena.cleanup()
    shared.cleanup()


if __name__ == "__main__":
    main()
