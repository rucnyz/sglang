"""Measure the DATA COPY cost of "snapshot C" approach.

Background:
  Option C ("snapshot + redirect"): before cuMemUnmap, copy the chunk
  contents from src VA to a holding buffer (still mapped, elsewhere)
  and update mamba_cache_indices to point there. The captured CUDA
  graph reads the holding buffer instead of the (about-to-be-unmapped)
  VA. After unmap+map, the chunks can move to dst.

  GPU cost of C = data copy + index-tensor update (small) + cap_barrier
                  + cuMemUnmap + cuMemMap (worst case)

  BUT if the snapshot target VA is *inside the pool's pre-reserved
  range* and is itself never unmapped during the fire, the cuMemUnmap
  is on the OLD position (already redirected away from). The captured
  graph reads only the new position, which stays mapped throughout.

  So the GPU-time "fire" reduces to:
    copy(src → holding) + small sync.

This bench measures the COPY cost for two regimes:

  A. Worst-case (whole-chunk snapshot): copy 24 sub-pools × 48 chunks
     × 2 MiB = 2.3 GiB. This is what you'd pay if you didn't know which
     slots are live.

  B. Active-only snapshot: copy only the slots that are CURRENTLY in
     use. With ~30 active reqs × per-slot-state ~256 KiB (typical
     Qwen mamba: d_inner=4096, d_state=128, fp16 × 24 layers ≈ 6 MiB
     per req actually but tightly packed = much less per "slot-equiv"),
     this is much smaller.

  We compare both to the measured cuMemUnmap+cuMemMap cost (82 ms p50
  from bench_cumem_costs.py).

Usage:
  .venv/bin/python dev/interlayer/bench/bench_snapshot_cost.py
"""
from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

DEVICE = 0
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()

# Mirror the cuMem bench layout for direct comparability.
N_SUBPOOLS = 24
N_CHUNKS = 48
CHUNK_BYTES = 2 * 1024 * 1024
TOTAL_BYTES_WORST = N_SUBPOOLS * N_CHUNKS * CHUNK_BYTES   # 2304 MiB = 2.25 GiB

# "Active-only" estimate: Qwen3.5-9B mamba per-slot state size.
# Looking at production logs:
#   intermediate_ssm: [n_layers, n_slots, d_inner * d_state * dtype_bytes]
#   intermediate_conv_window: smaller, conv pre-image
# For Qwen3.5-9B at fp16: ~256 KiB per slot per layer × 24 layers = 6 MiB per slot.
# With max_running_requests=256 but typical decode at C=56, ~56 active slots:
N_ACTIVE_SLOTS = 56
PER_SLOT_BYTES = 6 * 1024 * 1024   # 6 MiB across all 24 layers
TOTAL_BYTES_ACTIVE = N_ACTIVE_SLOTS * PER_SLOT_BYTES  # 336 MiB

N_TRIALS = 30
N_WARMUP = 3


def bench_copy(src: torch.Tensor, dst: torch.Tensor, label: str):
    """Time device-to-device copy."""
    assert src.shape == dst.shape
    torch.cuda.synchronize()

    times_us = []
    for trial in range(-N_WARMUP, N_TRIALS):
        torch.cuda.synchronize()
        t0 = time.monotonic_ns()
        dst.copy_(src)
        torch.cuda.synchronize()
        t1 = time.monotonic_ns()
        if trial >= 0:
            times_us.append((t1 - t0) // 1000)

    srt = sorted(times_us)
    p50 = srt[len(srt) // 2] / 1000
    p99 = srt[int(0.99 * len(srt))] / 1000
    mean = statistics.mean(times_us) / 1000
    bw = src.numel() * src.element_size() / (1024**3) / (p50 / 1000)
    print(f"  {label:30s}  p50={p50:6.2f} ms  p99={p99:6.2f} ms  "
          f"mean={mean:6.2f} ms  bw={bw:6.1f} GB/s")
    return p50, p99, mean


def main():
    print("=== Snapshot-C copy cost benchmarks (Qwen3.5-9B sizing) ===")
    print(f"N_TRIALS={N_TRIALS}  (N_WARMUP={N_WARMUP} discarded)\n")

    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"GPU memory: {free / 1024**3:.1f} GiB free of {total / 1024**3:.1f} GiB total\n")

    print(f"Regime A — WORST CASE (snapshot all touched chunks):")
    print(f"  size = {N_SUBPOOLS} sub-pools × {N_CHUNKS} chunks × "
          f"{CHUNK_BYTES // 1024} KiB = "
          f"{TOTAL_BYTES_WORST // (1024**2)} MiB")
    src_a = torch.empty(TOTAL_BYTES_WORST // 4, dtype=torch.float32, device=DEVICE)
    dst_a = torch.empty_like(src_a)
    src_a.fill_(1.5)
    p50_a, _, _ = bench_copy(src_a, dst_a, "whole-chunk snapshot (2.3 GiB)")
    del src_a, dst_a
    torch.cuda.empty_cache()

    print(f"\nRegime B — ACTIVE-ONLY (snapshot only live slot data):")
    print(f"  size = {N_ACTIVE_SLOTS} active slots × {PER_SLOT_BYTES // 1024} KiB "
          f"= {TOTAL_BYTES_ACTIVE // (1024**2)} MiB")
    src_b = torch.empty(TOTAL_BYTES_ACTIVE // 4, dtype=torch.float32, device=DEVICE)
    dst_b = torch.empty_like(src_b)
    src_b.fill_(1.5)
    p50_b, _, _ = bench_copy(src_b, dst_b, "active-only snapshot (336 MiB)")
    del src_b, dst_b
    torch.cuda.empty_cache()

    print(f"\nRegime C — MINIMAL (snapshot per-fire transfer ≈ a few slots):")
    n_per_fire_slots = 8
    bytes_per_fire = n_per_fire_slots * PER_SLOT_BYTES
    print(f"  size = {n_per_fire_slots} slots × {PER_SLOT_BYTES // 1024} KiB "
          f"= {bytes_per_fire // (1024**2)} MiB")
    src_c = torch.empty(bytes_per_fire // 4, dtype=torch.float32, device=DEVICE)
    dst_c = torch.empty_like(src_c)
    src_c.fill_(1.5)
    p50_c, _, _ = bench_copy(src_c, dst_c, "minimal snapshot (48 MiB)")

    print(f"\n=== Comparison to baseline cuMemUnmap+cuMemMap ===")
    print(f"  baseline (measured by bench_cumem_costs.py): p50=82 ms")
    print(f"  C worst-case:     p50={p50_a:6.2f} ms  ({82/p50_a:.1f}× faster)")
    print(f"  C active-only:    p50={p50_b:6.2f} ms  ({82/p50_b:.1f}× faster)")
    print(f"  C minimal:        p50={p50_c:6.2f} ms  ({82/p50_c:.1f}× faster)")

    print(f"\nKey caveat: this measures ONLY data-copy GPU cost. A full ")
    print(f"option-C implementation also needs:")
    print(f"  - atomic update of mamba_cache_indices for active batches")
    print(f"  - synchronization between fire and in-flight captured-graph replay")
    print(f"  - handling for races (req admitted during snapshot)")
    print(f"None of those are benched here. They are CPU-side overheads")
    print(f"(microseconds) but architectural complexity.")


if __name__ == "__main__":
    main()
