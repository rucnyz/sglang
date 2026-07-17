"""Profile: does cuMemMap/Unmap actually slow down subsequent decode kernels?

Sets up:
  - A large KV-cache-like bf16 tensor on GPU (~25 GB), accessed via realistic
    decode-like pattern (gather + matmul-ish) using ~7 ms of GPU work per iter
  - A separate chunk_arena to perform cuMemMap/cuMemUnmap operations
    matching real fire patterns (16 sub-pools × ~3 chunks unmap/map each)

Measures via CUDA events (precise GPU-side timing):
  - N baseline iters → mean / p99 of decode GPU time
  - inject a "fire" (cuMemUnmap + cuMemMap matching real fire)
  - M iters immediately after → mean / p99 of decode GPU time

Hypothesis: post-fire iters 0..K have measurably higher GPU time than
baseline due to TLB / MMU perturbation. If so, K and Δ tell us the
diffuse-cost shape.

Run: .venv/bin/python dev/interlayer/1_dyn_admission_cap/profile_cumem_decode_impact.py
"""
import statistics
import sys

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

DEVICE = "cuda:0"
torch.cuda.set_device(0)


def make_decode_work(kv_size_gb=20):
    """Realistic decode-like GPU work.

    Mimics:
      - read from a big KV-like bf16 tensor (scattered indices to stress TLB)
      - small matmul to match decode forward compute cost
      - write a result back

    Calibrated to ~7 ms per call (matches sglang decode iter at bs=33).
    """
    n_bytes = int(kv_size_gb * 1024**3)
    n_elems = n_bytes // 2  # bf16 = 2 bytes
    # n_elems ≈ 10G elements; arrange as (rows, hidden)
    hidden = 4096
    rows = n_elems // hidden
    kv = torch.randn(rows, hidden, dtype=torch.bfloat16, device=DEVICE)
    print(f"  KV-like tensor: {rows}x{hidden} bf16 = "
          f"{rows*hidden*2/1024**3:.1f} GiB")

    # Decode batch: 33 reqs, each indexes ~30 random positions
    batch = 33
    seq_per_req = 8192
    indices = torch.randint(0, rows, (batch, seq_per_req),
                            dtype=torch.int64, device=DEVICE)
    weights = torch.randn(hidden, hidden, dtype=torch.bfloat16, device=DEVICE)

    def decode_iter():
        # 1. gather (TLB-stressing scatter read)
        gathered = kv[indices.flatten()].view(batch, seq_per_req, hidden)
        # 2. small matmul (matches ~softmax+matmul cost)
        out = (gathered @ weights).sum(dim=1)  # (batch, hidden)
        return out.sum()

    # Warmup
    for _ in range(10):
        _ = decode_iter()
    torch.cuda.synchronize()
    return decode_iter


def measure_iter_gpu_time_ms(decode_iter, n_iters):
    """Return list of per-iter GPU times in ms via CUDA events."""
    out = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = decode_iter()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end))
    return out


def main():
    from sglang.srt.arena.chunk_arena import ChunkArena, SharedHandlePool

    print("Setting up decode workload...")
    decode = make_decode_work(kv_size_gb=20)

    # Warmup more
    print("Warmup...")
    _ = measure_iter_gpu_time_ms(decode, 20)

    # Baseline: 100 iters, no fires
    print("\n=== Baseline (no cuMem) ===")
    base = measure_iter_gpu_time_ms(decode, 100)
    base_mean = statistics.mean(base)
    base_std = statistics.stdev(base)
    base_p99 = sorted(base)[int(0.99 * len(base))]
    print(f"  baseline: mean={base_mean:.3f}ms ± {base_std:.3f}ms p99={base_p99:.3f}ms")

    # Setup chunk_arena with N sub-pools mimicking real KV/mamba arena
    print("\n=== Setup chunk_arena (16 sub-pools × 8 chunks each) ===")
    shared = SharedHandlePool(device_id=0, chunk_size=2 * 1024 * 1024,
                              n_handles=256)
    n_subpools = 16
    arena = ChunkArena(
        device_id=0, chunk_size=2 * 1024 * 1024, n_handles=0,
        pool_capacities=[(f"sp_{i}", 16) for i in range(n_subpools)],
        external_handle_pool=shared,
    )
    # pre-map 8 chunks per sub-pool (so we have something to unmap)
    for i in range(n_subpools):
        arena.grow(f"sp_{i}", 8)

    # Mimic a real fire: unmap 3 chunks per sub-pool (16 × 3 = 48 unmaps),
    # then map them into the other side. Use grow/shrink to keep it simple.
    def one_fire():
        # 16 sub-pools × shrink 3 = 48 cuMemUnmap calls
        for i in range(n_subpools):
            arena.shrink(f"sp_{i}", 3)
        # 16 sub-pools × grow 3 = 48 cuMemMap calls
        for i in range(n_subpools):
            arena.grow(f"sp_{i}", 3)
        torch.cuda.synchronize()

    print("\n=== Inject one 'fire' (48 cuMemUnmap + 48 cuMemMap) ===")
    import time
    t0 = time.perf_counter()
    one_fire()
    fire_wall_ms = (time.perf_counter() - t0) * 1000
    print(f"  fire wall: {fire_wall_ms:.1f}ms")

    # Measure 100 iters immediately after fire. Break into chunks of 10 to
    # see if cost decays.
    print("\n=== Post-fire decode iter GPU time (binned) ===")
    print(f"  {'iter range':>12s}  {'mean ms':>9s}  {'p99 ms':>8s}  "
          f"{'Δ vs base':>10s}")
    for chunk_start in [0, 10, 20, 30, 50, 80]:
        chunk_size = 10 if chunk_start < 30 else (20 if chunk_start < 50 else 30)
        ms = measure_iter_gpu_time_ms(decode, chunk_size)
        mean = statistics.mean(ms)
        p99 = sorted(ms)[int(0.99 * len(ms))]
        delta = (mean - base_mean) / base_mean * 100
        print(f"  [{chunk_start:3d}..{chunk_start+chunk_size-1:3d}]"
              f"     {mean:>7.3f}    {p99:>7.3f}    {delta:+8.2f}%")

    # Repeat 3 more times to amortize variance
    print("\n=== Repeat: 3 more fire cycles, batched 100 iters each ===")
    for cycle in range(3):
        one_fire()
        ms = measure_iter_gpu_time_ms(decode, 100)
        mean = statistics.mean(ms)
        p99 = sorted(ms)[int(0.99 * len(ms))]
        delta = (mean - base_mean) / base_mean * 100
        print(f"  cycle {cycle+1}: mean={mean:.3f}ms p99={p99:.3f}ms "
              f"Δ={delta:+.2f}%")

    arena.cleanup()
    shared.cleanup()


if __name__ == "__main__":
    main()
