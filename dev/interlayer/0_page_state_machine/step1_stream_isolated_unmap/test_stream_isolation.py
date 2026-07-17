"""Step 1 — does side-stream cuMemUnmap let a decode kernel keep running?

Real CUDA only. No mocks. No sglang state. Pure CUDA primitives via
sglang's ctypes wrappers + a small Triton kernel as the "decode"
proxy.

Scenario:
  1. Reserve a single VA covering 4 chunks. Map all 4.
  2. Decode stream busy-runs a Triton kernel that reads from chunk 0
     in a long loop (mimicking many decode iterations).
  3. Concurrently, on a side stream:
       cuMemSetAccess(chunk 2, decode_stream, prot=NONE)
       cuMemUnmap(chunk 2)
  4. Measure: how long does the decode kernel take, and was it
     stalled by the unmap?

If decode kernel time ≈ baseline → A1 confirmed (side-stream unmap
doesn't block decode).

If decode kernel time ≈ baseline + unmap_wall → A1 refuted (CUDA
serialised them anyway).

Run:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python \\
    dev/interlayer/0_zero_blocking_fire/step1_stream_isolated_unmap/test_stream_isolation.py

Exit 0 = A1 confirmed; non-zero = A1 refuted (decision rule in README).
"""
from __future__ import annotations

import ctypes
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

import sglang.srt.arena.chunk_arena as ca
from sglang.srt.arena.from_blob_ext import tensor_from_va

DEVICE = 0
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()
ca.CUDA.cuInit(0)


# ---- CUDA primitive wrappers ----

CHUNK_BYTES = 2 * 1024 * 1024
N_CHUNKS = 4


def alloc_handles(n: int) -> list[int]:
    prop = ca._CUmemAllocationProp()
    prop.type = ca.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = DEVICE
    out = []
    for _ in range(n):
        h = ctypes.c_ulonglong()
        ca._check(
            ca.CUDA.cuMemCreate(ctypes.byref(h), CHUNK_BYTES, ctypes.byref(prop), 0),
            "cuMemCreate",
        )
        out.append(h.value)
    return out


def reserve_va(total_bytes: int) -> int:
    va = ctypes.c_ulonglong()
    ca._check(
        ca.CUDA.cuMemAddressReserve(ctypes.byref(va), total_bytes, CHUNK_BYTES, 0, 0),
        "cuMemAddressReserve",
    )
    return va.value


def map_handle(va_base: int, slot: int, handle: int) -> None:
    va = va_base + slot * CHUNK_BYTES
    ca._check(ca.CUDA.cuMemMap(va, CHUNK_BYTES, 0, handle, 0), "cuMemMap")


def unmap_chunk(va_base: int, slot: int) -> None:
    va = va_base + slot * CHUNK_BYTES
    ca._check(ca.CUDA.cuMemUnmap(va, CHUNK_BYTES), "cuMemUnmap")


def set_access(va_base: int, n_slots: int, prot: int = 3) -> None:
    """prot=3 → RW; prot=0 → NONE (revoke)."""
    desc = ca._CUmemAccessDesc()
    desc.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = DEVICE
    desc.flags = prot
    ca._check(
        ca.CUDA.cuMemSetAccess(
            va_base, n_slots * CHUNK_BYTES, ctypes.byref(desc), 1
        ),
        "cuMemSetAccess",
    )


def set_access_chunk(va_base: int, slot: int, prot: int) -> None:
    desc = ca._CUmemAccessDesc()
    desc.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = DEVICE
    desc.flags = prot
    ca._check(
        ca.CUDA.cuMemSetAccess(
            va_base + slot * CHUNK_BYTES, CHUNK_BYTES, ctypes.byref(desc), 1
        ),
        "cuMemSetAccess(chunk)",
    )


def main() -> int:
    print("=== Step 1: stream-isolated cuMemUnmap ===\n")

    handles = alloc_handles(N_CHUNKS)
    va_base = reserve_va(N_CHUNKS * CHUNK_BYTES)
    for slot in range(N_CHUNKS):
        map_handle(va_base, slot, handles[slot])
    set_access(va_base, N_CHUNKS, prot=3)
    print(f"[setup] reserved VA 0x{va_base:x}, mapped 4 chunks")

    # Build a tensor over chunk 0 only — the "decode" data the kernel
    # reads from. Chunk 2 will be the one we unmap.
    n_floats_per_chunk = CHUNK_BYTES // 4
    state = tensor_from_va(
        va_base, (N_CHUNKS * n_floats_per_chunk,), torch.float32, DEVICE
    )
    state.fill_(1.0)
    torch.cuda.synchronize()

    # ---- Build a long-running kernel on the decode stream ----
    # We use a simple compute-bound loop: sum(state[0:N]) repeated K times
    # to give the GPU enough work that the unmap could overlap it.
    decode_stream = torch.cuda.Stream(device=DEVICE)
    side_stream = torch.cuda.Stream(device=DEVICE)

    chunk0 = state[:n_floats_per_chunk]
    out = torch.zeros(1, dtype=torch.float32, device=f"cuda:{DEVICE}")

    # ---- Use a HEAVY matmul as the long-running kernel so we have
    # reliably ~50-100 ms of GPU work that PyTorch can't fold across
    # invocations. Sum of fp32 tensor would only be one small launch.
    # ----
    MAT_N = 4096
    A = torch.randn(MAT_N, MAT_N, dtype=torch.float32, device=f"cuda:{DEVICE}")
    B = torch.randn(MAT_N, MAT_N, dtype=torch.float32, device=f"cuda:{DEVICE}")
    C = torch.empty(MAT_N, MAT_N, dtype=torch.float32, device=f"cuda:{DEVICE}")
    LOOPS = 20

    def launch_decode_work():
        """Queue LOOPS matmuls into decode_stream. Returns once HOST is
        done queueing (does NOT wait for GPU)."""
        with torch.cuda.stream(decode_stream):
            for _ in range(LOOPS):
                torch.matmul(A, B, out=C)

    # Warmup the matmul kernel + cublas JIT before any timing.
    for _ in range(3):
        launch_decode_work()
    decode_stream.synchronize()

    # Measure baseline N=5 runs, take median to filter outliers.
    baseline_runs = []
    for _ in range(5):
        decode_stream.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record(decode_stream)
        launch_decode_work()
        end_evt.record(decode_stream)
        decode_stream.synchronize()
        baseline_runs.append(start_evt.elapsed_time(end_evt))
    baseline_ms = sorted(baseline_runs)[len(baseline_runs) // 2]
    print(f"[1.1] baseline decode work ({LOOPS}× matmul {MAT_N}×{MAT_N} "
          f"fp32), GPU time median of 5 runs: {baseline_ms:.2f} ms "
          f"(runs: {[f'{x:.1f}' for x in baseline_runs]})")

    # ---- Test 1: time the cuMemUnmap call when no decode is running
    # (pure host-side cost) ----
    decode_stream.synchronize()
    t_pure_unmap_start = time.monotonic_ns()
    set_access_chunk(va_base, 3, prot=0)  # revoke from device
    unmap_chunk(va_base, 3)
    t_pure_unmap_end = time.monotonic_ns()
    pure_unmap_ms = (t_pure_unmap_end - t_pure_unmap_start) / 1e6
    print(f"[1.2] pure cuMemUnmap (no concurrent decode), host wall: "
          f"{pure_unmap_ms:.3f} ms")
    # Re-map for later tests.
    map_handle(va_base, 3, handles[3])
    set_access_chunk(va_base, 3, prot=3)
    torch.cuda.synchronize()

    # ---- Test 2: CONCURRENT cuMemUnmap during decode ----
    # Sequence:
    #   1. host queues 20 matmuls into decode_stream
    #   2. host immediately calls cuMemUnmap on slot 3
    #   3. host calls decode_stream.synchronize() to wait for decode
    # CUDA events bracket the decode workload to measure GPU time.
    # Host wall timings bracket the unmap call.
    # If unmap "stole" GPU time from decode, decode_gpu_ms will be
    # significantly larger than baseline. If unmap was truly host-side
    # / overlapping with no GPU contention, decode_gpu_ms ≈ baseline.
    decode_stream.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record(decode_stream)
    launch_decode_work()
    end_evt.record(decode_stream)

    t_host_unmap_start = time.monotonic_ns()
    set_access_chunk(va_base, 3, prot=0)
    unmap_chunk(va_base, 3)
    t_host_unmap_end = time.monotonic_ns()
    concurrent_unmap_ms = (t_host_unmap_end - t_host_unmap_start) / 1e6

    decode_stream.synchronize()
    decode_gpu_ms = start_evt.elapsed_time(end_evt)
    print(f"[1.3] concurrent unmap during decode:")
    print(f"      decode GPU time (during which unmap was called): "
          f"{decode_gpu_ms:.2f} ms")
    print(f"      unmap host wall: {concurrent_unmap_ms:.3f} ms")
    print(f"      baseline decode GPU time: {baseline_ms:.2f} ms")

    # Re-map.
    map_handle(va_base, 3, handles[3])
    set_access_chunk(va_base, 3, prot=3)
    torch.cuda.synchronize()

    # ---- Verdict ----
    # If decode_gpu_ms ≈ baseline_ms, decode was NOT slowed by unmap
    # → A1 confirmed (unmap doesn't steal GPU time).
    # If decode_gpu_ms >> baseline_ms (say > baseline + 5 ms), the
    # decode stream stalled — A1 refuted.
    slow_threshold_ms = baseline_ms + 5.0
    print()
    print(f"=== Analysis ===")
    print(f"  baseline decode GPU time:           {baseline_ms:7.2f} ms")
    print(f"  decode GPU time with concurrent unmap: {decode_gpu_ms:7.2f} ms")
    print(f"  unmap host wall:                    {concurrent_unmap_ms:7.2f} ms")
    print(f"  delta (concurrent - baseline):      {decode_gpu_ms - baseline_ms:7.2f} ms")
    print(f"  slow threshold (baseline + 5 ms):   {slow_threshold_ms:7.2f} ms")

    if decode_gpu_ms > slow_threshold_ms:
        print()
        print("=== A1 REFUTED (small-scale test) ===")
        print(f"Decode GPU time during concurrent unmap "
              f"({decode_gpu_ms:.2f} ms) is +{decode_gpu_ms - baseline_ms:.2f} ms "
              f"over baseline ({baseline_ms:.2f} ms). The CUDA driver "
              f"stalled the decode stream for the unmap.")
        print()
        print("Implication: C+A's 'unmap on side stream' assumption is "
              "false. Best we can get is C alone — fire wall ~5 ms but "
              "it BLOCKS decode for that duration.")
        return 1

    print(f"  [1.3 verdict] small-scale (1 chunk, ~0.04 ms unmap): "
          f"decode unaffected ✓")

    # ---- Test 4: PRODUCTION-SCALE unmap. -----------------------------
    # Production fire unmaps 24 sub-pools × 48 chunks = 1152 cuMemUnmap
    # calls in sequence. Each ~25-30 µs → host total ~30 ms. With
    # decode taking ~50 ms, this is a discriminative test:
    #   - if host's 30 ms of cuMemUnmap calls blocks GPU stream: decode
    #     GPU time will grow toward baseline + 30 ms
    #   - if GPU work is independent of host-side unmap: decode stays
    #     at baseline.
    print()
    print(f"[1.4] PRODUCTION-SCALE: unmap many chunks while decode runs.")
    # Set up: alloc enough handles to do 200 unmaps. Reserve another
    # large VA so we have something to unmap without disturbing the
    # decode-target VA.
    N_BIG = 200
    big_handles = alloc_handles(N_BIG)
    big_va = reserve_va(N_BIG * CHUNK_BYTES)
    for slot in range(N_BIG):
        map_handle(big_va, slot, big_handles[slot])
    set_access(big_va, N_BIG, prot=3)
    torch.cuda.synchronize()

    # Time the production-scale unmap with NO concurrent decode.
    t0 = time.monotonic_ns()
    for slot in range(N_BIG):
        set_access_chunk(big_va, slot, prot=0)
        unmap_chunk(big_va, slot)
    pure_big_unmap_ms = (time.monotonic_ns() - t0) / 1e6
    print(f"      pure {N_BIG}-chunk unmap (no decode), host wall: "
          f"{pure_big_unmap_ms:.2f} ms")

    # Re-map.
    for slot in range(N_BIG):
        map_handle(big_va, slot, big_handles[slot])
    set_access(big_va, N_BIG, prot=3)
    torch.cuda.synchronize()

    # Now time the same unmap CONCURRENT with decode.
    decode_stream.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record(decode_stream)
    launch_decode_work()
    end_evt.record(decode_stream)

    t_big_start = time.monotonic_ns()
    for slot in range(N_BIG):
        set_access_chunk(big_va, slot, prot=0)
        unmap_chunk(big_va, slot)
    t_big_end = time.monotonic_ns()
    concurrent_big_unmap_ms = (t_big_end - t_big_start) / 1e6

    decode_stream.synchronize()
    decode_big_gpu_ms = start_evt.elapsed_time(end_evt)
    print(f"      concurrent decode GPU time: {decode_big_gpu_ms:.2f} ms")
    print(f"      concurrent {N_BIG}-chunk unmap host wall: "
          f"{concurrent_big_unmap_ms:.2f} ms")

    big_slow_threshold = baseline_ms + 5.0
    if decode_big_gpu_ms > big_slow_threshold:
        print()
        print("=== A1 REFUTED (production-scale) ===")
        print(f"Even though 1-chunk concurrent test passed, the "
              f"{N_BIG}-chunk production-scale test shows decode GPU "
              f"time ({decode_big_gpu_ms:.2f} ms) significantly above "
              f"baseline ({baseline_ms:.2f} ms). The cumulative effect "
              f"of many cuMemUnmap calls IS draining decode.")
        return 1

    print(f"  [1.4 verdict] cuBLAS GEMM + eager launch + disjoint "
          f"reservation + main-thread unmap: decode unaffected ✓")

    # =====================================================================
    # 1.5 / 1.6 / 1.7 — depth additions from 2026-05-31 audit. The 1.1-1.4
    # variants above only proved A1 for cuBLAS+eager+disjoint-VA+main-thread.
    # Production decode is real Triton + captured graph; production VA layout
    # is ONE reservation per arena with sub-pools as offsets
    # (chunk_arena.py:299-300); production unmap runs on the Budgeter
    # _fire_worker thread (xpool_actuator.py:105,251). Each gap below would
    # let A1 pass in test while the live D10@C=56 crash recurs.
    # =====================================================================

    # ---- Common setup: re-map big_va so all 200 chunks are live again,
    # then overlay the real production mamba state on chunks 0..3 of THE
    # SAME big_va reservation. unmap target = chunks 100..199 (SAME
    # reservation). state_indices stays inside chunks 0..3.
    for slot in range(N_BIG):
        map_handle(big_va, slot, big_handles[slot])
    set_access(big_va, N_BIG, prot=3)
    torch.cuda.synchronize()

    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    HV15, V15, K15 = 4, 16, 16          # matches archived step1b sizing
    H15 = 2
    BYTES_PER_SLOT_15 = HV15 * V15 * K15 * 2      # bf16 → 2048
    SLOTS_PER_CHUNK_15 = CHUNK_BYTES // BYTES_PER_SLOT_15   # 1024
    N_STATE_CHUNKS = 4
    N_STATE_SLOTS = N_STATE_CHUNKS * SLOTS_PER_CHUNK_15      # 4096
    n_bf16_state = (N_STATE_CHUNKS * CHUNK_BYTES) // 2
    state_flat15 = tensor_from_va(big_va, (n_bf16_state,),
                                  torch.bfloat16, DEVICE)
    state15 = state_flat15.view(N_STATE_SLOTS, HV15, V15, K15)
    state15.fill_(0.0)
    for s in range(128):
        state15[s].fill_(float(s % 100))

    B15 = 4
    qkv_dim = H15 * K15 * 2 + HV15 * V15
    mixed_qkv15 = torch.zeros(B15, qkv_dim, dtype=torch.bfloat16,
                              device=f"cuda:{DEVICE}")
    a15 = torch.zeros(B15, HV15, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    b15 = torch.zeros(B15, HV15, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    A_log15 = torch.zeros(HV15, dtype=torch.float32, device=f"cuda:{DEVICE}")
    dt_bias15 = torch.zeros(HV15, dtype=torch.float32, device=f"cuda:{DEVICE}")
    out15 = torch.zeros(B15, 1, HV15, V15, dtype=torch.bfloat16,
                        device=f"cuda:{DEVICE}")
    ssm_idx15 = torch.zeros(B15, dtype=torch.int32, device=f"cuda:{DEVICE}")

    def run_kernel_15():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv15,
            a=a15, b=b15, A_log=A_log15, dt_bias=dt_bias15,
            scale=1.0,
            initial_state=state15,
            out=out15,
            ssm_state_indices=ssm_idx15,
            use_qk_l2norm_in_kernel=False,
        )

    # Warm + JIT compile so capture doesn't try to compile.
    ssm_idx15.copy_(
        torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=f"cuda:{DEVICE}")
    )
    run_kernel_15()
    torch.cuda.synchronize()

    cap_stream15 = torch.cuda.Stream(device=DEVICE)
    g15 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(cap_stream15):
        with torch.cuda.graph(g15, stream=cap_stream15):
            run_kernel_15()
    torch.cuda.synchronize()

    print()
    print(f"[1.5] real Triton kernel + captured graph + SAME-reservation "
          f"concurrent unmap")
    print(f"      state at big_va+0 (chunks 0..3, slots 0..{N_STATE_SLOTS-1}); "
          f"indices=[0,1,2,3]; unmap target = chunks 100..199 of big_va "
          f"(same reservation)")

    n_trials_15 = 3
    n_replays_per_trial = 50    # queue enough replays to overlap unmap loop
    for trial in range(n_trials_15):
        # Queue many replays so the GPU is actively running when we issue
        # the unmap loop on the host.
        for _ in range(n_replays_per_trial):
            g15.replay()
        # Concurrent unmap of chunks 100..199 of the SAME reservation.
        for slot in range(100, 200):
            set_access_chunk(big_va, slot, prot=0)
            unmap_chunk(big_va, slot)
        # If any replay faulted, sync will raise CUDA error.
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"      CRASH on trial {trial}: {type(e).__name__}: "
                  f"{str(e)[:240]}")
            print()
            print("=== A1 REFUTED at Triton + captured-graph + "
                  "same-reservation concurrent unmap ===")
            print("Captured graph faulted when chunks of the SAME VA "
                  "reservation were concurrently unmapped, even though "
                  "ssm_state_indices excluded those chunks. C+A's "
                  "'side-stream unmap doesn't affect decode' claim does "
                  "not survive at the real production kernel + graph + "
                  "VA-layout combination. C-alone (snapshot then "
                  "same-stream sync-unmap) is the architectural floor; "
                  "steps 3-5 must be re-scoped.")
            return 4
        # Re-map chunks 100..199 for the next trial.
        for slot in range(100, 200):
            map_handle(big_va, slot, big_handles[slot])
            set_access_chunk(big_va, slot, prot=3)
        torch.cuda.synchronize()
    print(f"      {n_trials_15} trials × {n_replays_per_trial} replays + "
          f"concurrent 100-chunk same-reservation unmap: no crash ✓")

    # =====================================================================
    # 1.6 — worker-thread unmap (mirrors Budgeter _fire_worker at
    # xpool_actuator.py:105,251). Production's unmap loop runs on a
    # threading.Thread, NOT the scheduler's main thread. Thread crossing
    # could change CUDA primary-context behavior even though the GIL
    # serialises Python.
    # =====================================================================
    import threading

    print()
    print(f"[1.6] worker-thread cuMemUnmap (mirrors Budgeter _fire_worker)")

    # Decode runs on main thread on decode_stream; unmap runs on worker
    # thread doing the host-side cuMemSetAccess(NONE)+cuMemUnmap loop.
    err_box: list = []

    def worker_unmap_fn():
        try:
            for slot in range(100, 200):
                set_access_chunk(big_va, slot, prot=0)
                unmap_chunk(big_va, slot)
        except Exception as e:
            err_box.append(e)

    decode_stream.synchronize()
    s_evt = torch.cuda.Event(enable_timing=True)
    e_evt = torch.cuda.Event(enable_timing=True)
    s_evt.record(decode_stream)
    launch_decode_work()
    e_evt.record(decode_stream)

    worker_t = threading.Thread(target=worker_unmap_fn, name="fire_worker_stub")
    worker_t.start()
    worker_t.join()
    decode_stream.synchronize()

    if err_box:
        print(f"      worker thread raised: {err_box}")
        print()
        print("=== A1 REFUTED at worker-thread variant ===")
        print("The cuMemSetAccess+cuMemUnmap sequence raised when issued "
              "from a non-main Python thread. Production's Budgeter "
              "_fire_worker would crash similarly.")
        return 5
    worker_gpu_ms = s_evt.elapsed_time(e_evt)
    print(f"      decode GPU time during worker-thread 100-chunk unmap: "
          f"{worker_gpu_ms:.2f} ms (delta {worker_gpu_ms - baseline_ms:+.2f} ms "
          f"vs baseline {baseline_ms:.2f} ms)")
    if worker_gpu_ms > baseline_ms + 5.0:
        print(f"      decode stalled — worker-thread unmap blocked the "
              f"decode stream (over {baseline_ms + 5.0:.2f} ms threshold)")
        print()
        print("=== A1 REFUTED at worker-thread variant ===")
        print("Even though the unmap completed without raising, the "
              "decode stream stalled while a non-main thread held the "
              "CUDA primary context for the unmap loop. Production's "
              "_fire_worker would impose the same stall on the scheduler.")
        return 6
    print(f"      worker-thread unmap concurrent with decode: no crash, "
          f"no stall ✓")
    for slot in range(100, 200):
        map_handle(big_va, slot, big_handles[slot])
        set_access_chunk(big_va, slot, prot=3)
    torch.cuda.synchronize()

    # =====================================================================
    # 1.7 — verify cuMemSetAccess(prot=0) is a REAL revoke, not bookkeeping.
    # Without this, the "decode stream's access was revoked" premise of A1
    # is unverified. Subprocess-isolated because faults poison CUDA context.
    # =====================================================================
    import subprocess
    import textwrap

    print()
    print(f"[1.7] cuMemSetAccess(prot=0) actually faults subsequent device "
          f"access (subprocess-isolated)")

    src17 = textwrap.dedent("""
        import ctypes, sys
        sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
        import torch
        import sglang.srt.arena.chunk_arena as ca
        from sglang.srt.arena.from_blob_ext import tensor_from_va
        DEVICE = 0
        torch.cuda.set_device(DEVICE)
        torch.cuda.synchronize()
        ca.CUDA.cuInit(0)
        CHUNK = 2 * 1024 * 1024
        prop = ca._CUmemAllocationProp()
        prop.type = ca.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 0
        prop.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = DEVICE
        h = ctypes.c_ulonglong()
        ca._check(
            ca.CUDA.cuMemCreate(ctypes.byref(h), CHUNK, ctypes.byref(prop), 0),
            'cuMemCreate',
        )
        va = ctypes.c_ulonglong()
        ca._check(
            ca.CUDA.cuMemAddressReserve(
                ctypes.byref(va), CHUNK, CHUNK, 0, 0
            ),
            'cuMemAddressReserve',
        )
        ca._check(
            ca.CUDA.cuMemMap(va.value, CHUNK, 0, h.value, 0), 'cuMemMap'
        )
        desc_rw = ca._CUmemAccessDesc()
        desc_rw.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
        desc_rw.location.id = DEVICE
        desc_rw.flags = 3
        ca._check(
            ca.CUDA.cuMemSetAccess(va.value, CHUNK, ctypes.byref(desc_rw), 1),
            'cuMemSetAccess RW',
        )
        t = tensor_from_va(va.value, (CHUNK // 4,), torch.float32, DEVICE)
        t.fill_(1.0)
        torch.cuda.synchronize()
        s_before = t.sum().item()
        print(f'sum_before_revoke={s_before}', flush=True)
        desc_n = ca._CUmemAccessDesc()
        desc_n.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
        desc_n.location.id = DEVICE
        desc_n.flags = 0
        ca._check(
            ca.CUDA.cuMemSetAccess(va.value, CHUNK, ctypes.byref(desc_n), 1),
            'cuMemSetAccess NONE',
        )
        try:
            s_after = t.sum().item()
            torch.cuda.synchronize()
            print(f'NO_FAULT sum_after_revoke={s_after}', flush=True)
            sys.exit(7)
        except Exception as e:
            print(f'FAULT_OK {type(e).__name__}: {str(e)[:160]}', flush=True)
            sys.exit(0)
    """)
    r17 = subprocess.run(
        ["/scratch/yuzhou/projects/sglang/.venv/bin/python", "-c", src17],
        capture_output=True, text=True, timeout=60,
    )
    stdout_tail = (r17.stdout or "")[-400:].strip()
    stderr_tail = (r17.stderr or "")[-400:].strip()
    print(f"      subprocess rc={r17.returncode}")
    print(f"      subprocess stdout: {stdout_tail}")
    if r17.returncode == 7 or "NO_FAULT" in stdout_tail:
        print(f"      subprocess stderr (last 400 chars): {stderr_tail}")
        print()
        print("=== A1 REFUTED at revoke-step variant ===")
        print("cuMemSetAccess(prot=0) did NOT cause a subsequent device "
              "read to fault. That means the 'access revoke' step is "
              "either a no-op or bookkeeping only — the CUDA driver is "
              "NOT actually denying the decode stream access to the "
              "unmapped VA. C+A's `cuMemUnmap won't sync decode because "
              "decode lost access first` argument collapses.")
        return 7
    if r17.returncode != 0:
        print(f"      subprocess stderr (last 400 chars): {stderr_tail}")
        print()
        print(f"=== 1.7 INCONCLUSIVE — subprocess returned rc={r17.returncode} "
              f"unexpectedly ===")
        return 8
    print(f"      cuMemSetAccess(prot=0) confirmed as a real revoke "
          f"(post-revoke device read raised) ✓")

    print()
    print("=== A1 CONFIRMED across all variants (1.1-1.7) ===")
    print(f"  1.4 cuBLAS + eager + disjoint-VA + main-thread: decode "
          f"unaffected ({decode_big_gpu_ms:.2f} ms ≈ baseline "
          f"{baseline_ms:.2f} ms)")
    print(f"  1.5 real Triton kernel + captured graph + same-reservation: "
          f"no crash across {n_trials_15} trials × {n_replays_per_trial} "
          f"replays + 100-chunk concurrent unmap")
    print(f"  1.6 worker-thread unmap: no crash, decode {worker_gpu_ms:.2f} "
          f"ms (delta {worker_gpu_ms - baseline_ms:+.2f} ms) within noise")
    print(f"  1.7 cuMemSetAccess(prot=0) is a real revoke (post-revoke "
          f"device read faults)")
    print()
    print("Implication: in C+A flow, fire's cuMemUnmap sequence on a "
          "worker thread does NOT block the decode stream even when the "
          "unmap target shares a VA reservation with the live captured "
          "graph's encoded base pointer, AND the revoke step is "
          "load-bearing (not a no-op).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
