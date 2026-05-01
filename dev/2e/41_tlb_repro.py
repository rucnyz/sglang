"""TLB micro-bench: stream over a large cuMemMap'd range cold vs warm.

Allocates a chunk-arena-style range via the sglang chunk_arena C
extension, wraps as torch tensor via from_blob_ext, runs streaming sum()
kernels in cold-TLB and warm-TLB modes. Demonstrates the TLB-induced
cost without an inference engine in the loop.

Companion ncu invocation (proxy counters for Hopper, since direct TLB
metrics aren't exposed in the public PerfWorks catalog — verified via
`ncu --query-metrics --chip GH100 | grep -iE "tlb|page" → 0 hits`):
  sudo ncu --target-processes all \\
    --kernel-name regex:reduce \\
    --launch-skip 0 --launch-count 8 \\
    --replay-mode kernel --cache-control all --clock-control base \\
    --metrics dram__bytes_read.sum,dram__sectors_read.sum,\\
              lts__t_sectors_aperture_device.sum,\\
              lts__t_sector_hit_rate.pct,\\
              smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\\
              sm__cycles_elapsed.avg \\
    --csv --log-file /tmp/tlb_proxy.csv \\
    /scratch/yuzhou/projects/sglang/.venv/bin/python \\
    /scratch/yuzhou/projects/sglang/dev/2e/41_tlb_repro.py {cold|warm}
"""

import os
import sys
import torch

# Use sglang's already-debugged ctypes bindings to libcuda. They've been
# vetted by months of arena-tensor work and handle all the struct layouts.
from sglang.srt.arena.chunk_arena import ChunkArena
from sglang.srt.arena.from_blob_ext import tensor_from_va


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cold"
    assert mode in ("cold", "warm"), f"mode must be cold|warm, got {mode}"

    torch.cuda.set_device(0)
    torch.cuda.init()

    # 8 GiB allocation in 256 MiB chunks: matches the production arena's
    # chunk_size and produces 32 chunks. With 2 MiB-page granularity that's
    # 4096 page-table entries — well above the H200's per-SM TLB coverage.
    CHUNK_BYTES = 256 * 1024 * 1024
    N_CHUNKS = 32
    NBYTES = CHUNK_BYTES * N_CHUNKS    # 8 GiB
    DTYPE = torch.float32
    NUM_ELEMS = NBYTES // 4

    print(f"[tlb_repro] mode={mode} nbytes={NBYTES} ({NBYTES/(1<<30):.1f} GiB) "
          f"chunk_size={CHUNK_BYTES/(1<<20):.0f} MiB n_chunks={N_CHUNKS}")

    # ChunkArena reserves VA, allocates physical handles, maps them with
    # cuMemMap, calls cuMemSetAccess. Pool name is arbitrary.
    arena = ChunkArena(
        device_id=0,
        chunk_size=CHUNK_BYTES,
        n_handles=N_CHUNKS,
        pool_capacities=[("tlb_repro_pool", N_CHUNKS)],
    )
    arena.grow("tlb_repro_pool", N_CHUNKS)   # cuMemMap all N_CHUNKS into pool
    va = arena.pool_va_base("tlb_repro_pool")
    print(f"[tlb_repro] allocated VA=0x{va:x}")

    # Wrap as torch tensor via from_blob (no-op deleter — VMM owns lifetime)
    t = tensor_from_va(
        va=va,
        sizes=(NUM_ELEMS,),
        dtype=DTYPE,
        device_index=0,
    )
    print(f"[tlb_repro] tensor shape={tuple(t.shape)} dtype={t.dtype} ptr=0x{t.data_ptr():x}")

    if mode == "warm":
        print("[tlb_repro] WARMING: 3× full-tensor sum() to populate TLB")
        for _ in range(3):
            t.sum()
        torch.cuda.synchronize()

    # Timed measurement: 8 streaming reductions
    print("[tlb_repro] TIMED: 8× streaming sum() (Triton/CUDA reduce kernel)")
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(8)]
    torch.cuda.synchronize()
    for s, e in events:
        s.record()
        _ = t.sum()
        e.record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in events]
    print(f"[tlb_repro] mode={mode} per-launch times (ms):")
    for i, t_ms in enumerate(times_ms):
        print(f"           [{i}] {t_ms:.3f}")
    mean = sum(times_ms) / len(times_ms)
    print(f"[tlb_repro] mode={mode} mean={mean:.3f} ms "
          f"min={min(times_ms):.3f} max={max(times_ms):.3f}")
    print(f"[tlb_repro] STDOUT_RESULT mode={mode} mean_ms={mean:.4f}")


if __name__ == "__main__":
    main()
