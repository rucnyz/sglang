"""decode_wall — multi-sub-pool batched unmap + decode wall budget.

Pins design.md §fire_wall+decode_wall budget (decode-stream wall ≤ 0.10 ms during a fire,
~3-5× headroom over the stable 0.00–0.03 ms median across n=20 trials)
on the actual #205-edit code path: the per-sub-pool batched unmap
loop in `xpool_actuator._execute_async_locked`
(`for name in src_names: src._arena.shrink_explicit(name, ...)`).

This test extends test_no_crash.py (which covered a single-sub-pool
unmap) to mirror the production `_execute_async_locked` pattern:
worker thread iterates through N sub-pools, calling shrink_explicit
on each, while a heavy GEMM workload runs on the decode stream.

What's actually pinned
----------------------
1. The per-sub-pool batched unmap loop is the place where #205
   removed `torch.cuda.synchronize()` calls. After #205, the loop is
   just back-to-back ctypes cuMemUnmap calls with no defensive sync
   between iterations. We pin that this loop, executed from a worker
   Python thread, does not stall a heavy GEMM workload on the decode
   stream beyond the design's §fire_wall+decode_wall budget (≤ 0.10 ms).

2. The methodology mirrors 0_page_state_machine/step1.6 (which
   measured 4096×4096 × 20 GEMM as decode work + 100-chunk
   single-pool worker-thread unmap → delta +0.10 ms). This test
   scales the unmap dimension to multi-sub-pool to capture the
   production loop shape; the budget stays the same.

What this test does NOT pin
---------------------------
- The full `XPoolActuator._execute_async_locked` entry (which also
  does cap-barrier accounting, dst.grow, and dst cap-bump). Those
  are scheduler-thread phases, not worker-thread, and have separate
  budgets covered elsewhere. This test isolates the worker-thread
  unmap loop which is the only #205-edited piece.
- A real `MultiTensorArena` / `XPoolActuator` / `FirePlan` build-up.
  We drive the relevant code path directly via `ChunkArena.shrink_explicit`,
  which is exactly what `_execute_async_locked` ends up calling for
  each sub-pool name.

Pass criterion
--------------
- Across n=20 trials, median(decode_gpu_time_with_worker_unmap)
  - median(baseline_decode_gpu_time) ≤ 0.10 ms (design.md §fire_wall+decode_wall).
- Worker thread completes without raising.
- Fail criterion: delta exceeds 0.10 ms → §fire_wall+decode_wall budget violated.

Run
---
  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python \\
    dev/interlayer/0_page_state_machine/decode_wall/test_decode_wall.py
"""
from __future__ import annotations

import sys
import threading

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

import sglang.srt.arena.chunk_arena as ca
from sglang.srt.arena.chunk_arena import ChunkArena

DEVICE = 0
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()
ca.CUDA.cuInit(0)

CHUNK_BYTES = 2 * 1024 * 1024
N_SUBPOOLS = 4              # mirrors production n_src for a small hybrid model
SLOTS_PER_SUBPOOL = 200
N_HANDLES = N_SUBPOOLS * SLOTS_PER_SUBPOOL  # 800
POOL_NAMES = [f"subpool_{i}" for i in range(N_SUBPOOLS)]

# Per-pool unmap range — mirrors production where each sub-pool drops
# the same `per_src` page range during one fire.
UNMAP_RANGE = list(range(100, 200))  # 100 chunks per sub-pool


def main() -> int:
    print("=== decode_wall — multi-sub-pool batched unmap + decode wall budget ===")
    print()

    arena = ChunkArena(
        device_id=DEVICE,
        chunk_size=CHUNK_BYTES,
        n_handles=N_HANDLES,
        pool_capacities=[(name, SLOTS_PER_SUBPOOL) for name in POOL_NAMES],
    )
    for name in POOL_NAMES:
        n_mapped = len(arena.grow(name, SLOTS_PER_SUBPOOL))  # #213: grow returns list[int]
        assert n_mapped == SLOTS_PER_SUBPOOL, (name, n_mapped)
    print(f"[setup] {N_SUBPOOLS} sub-pools × {SLOTS_PER_SUBPOOL} slots "
          f"= {N_HANDLES} chunks mapped")

    # ---- Decode workload: 20× GEMM 4096×4096 fp32 (same as step 1.6).
    decode_stream = torch.cuda.Stream(device=DEVICE)
    MAT_N = 4096
    A = torch.randn(MAT_N, MAT_N, dtype=torch.float32, device=f"cuda:{DEVICE}")
    B = torch.randn(MAT_N, MAT_N, dtype=torch.float32, device=f"cuda:{DEVICE}")
    C = torch.empty(MAT_N, MAT_N, dtype=torch.float32, device=f"cuda:{DEVICE}")
    LOOPS = 20

    def launch_decode_work():
        with torch.cuda.stream(decode_stream):
            for _ in range(LOOPS):
                torch.matmul(A, B, out=C)

    # Warmup cuBLAS JIT.
    for _ in range(3):
        launch_decode_work()
    decode_stream.synchronize()

    # ---- Baseline decode wall (no concurrent unmap).
    baseline_runs = []
    for _ in range(5):
        decode_stream.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(decode_stream)
        launch_decode_work()
        e.record(decode_stream)
        decode_stream.synchronize()
        baseline_runs.append(s.elapsed_time(e))
    baseline_ms = sorted(baseline_runs)[len(baseline_runs) // 2]
    print(f"[baseline] decode GPU time (no fire), median of 5: "
          f"{baseline_ms:.2f} ms (runs: {[f'{x:.2f}' for x in baseline_runs]})")
    print()

    # ---- Concurrent runs: worker iterates through ALL sub-pools.
    # Mirrors _execute_async_locked's:
    #   src_names = self._all_subpool_names(src)
    #   for name in src_names:
    #       unmapped_total += src._arena.shrink_explicit(name, src_unmap_list)
    n_trials = 20
    with_unmap_runs = []
    fault = []

    def worker_unmap_loop():
        try:
            for name in POOL_NAMES:
                arena.shrink_explicit(name, UNMAP_RANGE)
        except Exception as e:
            fault.append(("worker", type(e).__name__, str(e)[:200]))

    for trial in range(n_trials):
        if trial > 0:
            # Restore the unmapped chunks for the next trial.
            for name in POOL_NAMES:
                n_re = len(arena.grow(name, len(UNMAP_RANGE)))  # #213
                assert n_re == len(UNMAP_RANGE), (name, n_re)
            torch.cuda.synchronize()

        decode_stream.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(decode_stream)
        launch_decode_work()
        # Worker fires AFTER the decode work is queued but while the
        # GPU is processing it. Mirrors production where _fire_worker
        # runs concurrently with active decode.
        worker_t = threading.Thread(
            target=worker_unmap_loop, name=f"d2b_unmap_worker_{trial}",
        )
        worker_t.start()
        e.record(decode_stream)
        worker_t.join(timeout=30.0)
        decode_stream.synchronize()

        if worker_t.is_alive():
            fault.append(("worker", "TIMEOUT", "did not finish in 30 s"))
            break
        if fault:
            break

        gpu_ms = s.elapsed_time(e)
        with_unmap_runs.append(gpu_ms)
        delta = gpu_ms - baseline_ms
        print(f"[trial {trial}] decode GPU time with {N_SUBPOOLS}-sub-pool "
              f"× {len(UNMAP_RANGE)}-chunk worker unmap: {gpu_ms:.2f} ms "
              f"(delta {delta:+.2f} ms vs baseline)")

    print()
    if fault:
        print(f"=== decode_wall: FAILED — {len(fault)} fault(s) ===")
        for src, typ, msg in fault:
            print(f"  [{src}] {typ}: {msg}")
        return 1

    with_unmap_median = sorted(with_unmap_runs)[len(with_unmap_runs) // 2]
    delta_median = with_unmap_median - baseline_ms
    # n=20 trials keep the median stable at ~0.00-0.03 ms across runs;
    # 0.10 ms gives ~3-5× headroom over that median. A reintroduced
    # `torch.cuda.synchronize()` adds ≥ 50 µs per iter and pushes the
    # median past this budget, which is the failure-mode signal this
    # test exists to catch.
    budget_ms = 0.10  # design.md §"fire_wall_curve + decode_wall" (decode-stream-wall half)

    print(f"  baseline median:                     {baseline_ms:.2f} ms")
    print(f"  with-unmap median (n={n_trials}):              "
          f"{with_unmap_median:.2f} ms")
    print(f"  delta median:                        {delta_median:+.2f} ms")
    print(f"  design.md §fire_wall+decode_wall budget:                ≤ {budget_ms:.2f} ms")
    print()

    if delta_median > budget_ms:
        print(f"=== decode_wall: FAILED ===")
        print(f"Multi-sub-pool batched unmap exceeded §fire_wall+decode_wall decode-stream "
              f"wall budget (delta {delta_median:.2f} ms > "
              f"{budget_ms:.2f} ms). The worker-thread cuMemUnmap loop "
              f"is stalling the decode stream more than design allows.")
        return 1

    print(f"=== decode_wall: PASS ===")
    print(f"Production multi-sub-pool fire loop ({N_SUBPOOLS}-sub-pool × "
          f"{len(UNMAP_RANGE)}-chunk unmap) fits §fire_wall+decode_wall budget — decode-stream "
          f"wall delta {delta_median:.2f} ms ≤ {budget_ms:.2f} ms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
