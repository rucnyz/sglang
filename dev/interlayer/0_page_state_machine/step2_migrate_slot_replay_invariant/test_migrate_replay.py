"""Step 2 — migrate_slot + state_indices rewrite preserves captured
graph correctness.

Real CUDA + real production Triton kernel
(`fused_recurrent_gated_delta_rule_packed_decode`) + real VMM-backed
tensor. No mocks.

Falsifies/confirms A2 of the C+A combo: if the captured graph's
`ssm_state_indices` is rewritten between replays AND the dst slot
was prefilled with src's contents (via side-stream copy), does the
kernel produce the same output as if we'd indexed src directly?

Run:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python \\
    dev/interlayer/0_zero_blocking_fire/step2_migrate_slot_replay_invariant/test_migrate_replay.py

Exit 0 = A2 confirmed. Non-zero = A2 refuted.
"""
from __future__ import annotations

import ctypes
import sys
import time

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

import sglang.srt.arena.chunk_arena as ca
from sglang.srt.arena.from_blob_ext import tensor_from_va
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

DEVICE = 0
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()
ca.CUDA.cuInit(0)


# ---- CUDA primitives (same as step 1b) ----

CHUNK_BYTES = 2 * 1024 * 1024


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


def set_access(va_base: int, n_slots: int) -> None:
    desc = ca._CUmemAccessDesc()
    desc.location.type = ca.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = DEVICE
    desc.flags = 3
    ca._check(
        ca.CUDA.cuMemSetAccess(
            va_base, n_slots * CHUNK_BYTES, ctypes.byref(desc), 1
        ),
        "cuMemSetAccess",
    )


# ---- kernel sizing ----

H = 2
HV = 4
V = 16
K = 16
DTYPE = torch.bfloat16
DTYPE_BYTES = 2

BYTES_PER_SLOT = HV * V * K * DTYPE_BYTES   # 2048
SLOTS_PER_CHUNK = CHUNK_BYTES // BYTES_PER_SLOT  # 1024

N_CHUNKS = 4
N_SLOTS = N_CHUNKS * SLOTS_PER_CHUNK         # 4096


def main() -> int:
    print("=== Step 2: migrate_slot + index rewrite replay invariant ===\n")

    handles = alloc_handles(N_CHUNKS)
    va_base = reserve_va(N_CHUNKS * CHUNK_BYTES)
    for c in range(N_CHUNKS):
        map_handle(va_base, c, handles[c])
    set_access(va_base, N_CHUNKS)
    n_bf16 = (N_CHUNKS * CHUNK_BYTES) // DTYPE_BYTES
    flat = tensor_from_va(va_base, (n_bf16,), DTYPE, DEVICE)
    initial_state = flat.view(N_SLOTS, HV, V, K)

    # Fill slots with sentinels — first 256 slots have unique non-zero
    # patterns we can identify. Higher slots = 0 (they're the dst
    # holding area for migrate).
    initial_state.fill_(0.0)
    for s in range(256):
        # Slot s gets a tensor with values = (s + 1) * 0.01 — distinct
        # for each slot, bf16-representable.
        initial_state[s].fill_(float(s + 1) * 0.01)
    torch.cuda.synchronize()
    print(f"[setup] initial_state ({N_SLOTS} slots × {HV}×{V}×{K} bf16) "
          f"at 0x{initial_state.data_ptr():x}")

    # Build kernel inputs.
    B = 4
    qk_dim = H * K
    qkv_dim = qk_dim * 2 + HV * V
    # Use RANDOM input so the kernel's output is sensitive to all of
    # b_q, b_k, b_v, b_h (= initial_state[idx]). If we used zeros, the
    # output would be the same regardless of which slot we read.
    torch.manual_seed(42)
    mixed_qkv = torch.randn(B, qkv_dim, dtype=DTYPE, device=f"cuda:{DEVICE}")
    a = torch.randn(B, HV, dtype=DTYPE, device=f"cuda:{DEVICE}") * 0.1
    b_in = torch.randn(B, HV, dtype=DTYPE, device=f"cuda:{DEVICE}") * 0.1
    A_log = torch.randn(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    dt_bias = torch.randn(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    out = torch.zeros(B, 1, HV, V, dtype=DTYPE, device=f"cuda:{DEVICE}")

    ssm_state_indices = torch.zeros(B, dtype=torch.int32,
                                      device=f"cuda:{DEVICE}")

    def run_kernel():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a, b=b_in, A_log=A_log, dt_bias=dt_bias,
            scale=1.0,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=False,
        )

    # IMPORTANT: kernel writes back to initial_state (line 381-382 of
    # fused_recurrent.py: `h0=initial_state, ht=initial_state`). So
    # every replay MUTATES the slot we read from. We save a pristine
    # snapshot and restore it before each timed replay.
    pristine_state = initial_state.clone()
    print(f"[note] kernel writes back to initial_state; saving pristine "
          f"snapshot so each replay starts from same input")

    # Warmup so capture doesn't pick up JIT artifacts.
    ssm_state_indices.copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32,
                                          device=f"cuda:{DEVICE}"))
    run_kernel()
    torch.cuda.synchronize()
    initial_state.copy_(pristine_state)
    torch.cuda.synchronize()

    # Capture graph.
    decode_stream = torch.cuda.Stream(device=DEVICE)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(decode_stream):
        with torch.cuda.graph(g, stream=decode_stream):
            run_kernel()
    torch.cuda.synchronize()
    print(f"[capture] captured graph over fused_recurrent kernel")
    # Restore after the captured run.
    initial_state.copy_(pristine_state)
    torch.cuda.synchronize()

    # ---- 2.1 baseline ----
    src_indices = [0, 1, 2, 3]
    ssm_state_indices.copy_(torch.tensor(src_indices, dtype=torch.int32,
                                          device=f"cuda:{DEVICE}"))
    g.replay()
    torch.cuda.synchronize()
    baseline_outputs = out.clone()
    print(f"[2.1] baseline replay with indices={src_indices}: "
          f"out mean={out.float().mean().item():.4f} "
          f"max={out.float().abs().max().item():.4f}")
    # Restore pristine before next test.
    initial_state.copy_(pristine_state)
    torch.cuda.synchronize()

    # ---- 2.4 (negative control, BEFORE migrate) ----
    # dst slots still have their pristine 0s — replay should diverge
    # from baseline because slots 2000-2003 have all-zero state, not
    # the sentinels at 0-3.
    dst_indices = [2000, 2001, 2002, 2003]
    ssm_state_indices.copy_(torch.tensor(dst_indices, dtype=torch.int32,
                                          device=f"cuda:{DEVICE}"))
    g.replay()
    torch.cuda.synchronize()
    unmigrated_dst_outputs = out.clone()
    diff_unmigrated = (unmigrated_dst_outputs.float() -
                       baseline_outputs.float()).abs().max().item()
    print(f"[2.4 neg-ctrl] replay with un-migrated dst indices "
          f"{dst_indices}: max diff from baseline = {diff_unmigrated:.4f}")
    if diff_unmigrated < 1e-3:
        print(f"  ERROR: outputs are nearly equal — test cannot "
              f"discriminate. Sentinel values too small or kernel "
              f"insensitive. Need different test.")
        return 3
    initial_state.copy_(pristine_state)
    torch.cuda.synchronize()

    # ---- 2.2 migrate src→dst on SIDE stream ----
    side_stream = torch.cuda.Stream(device=DEVICE)
    decode_stream.synchronize()
    side_stream.synchronize()

    # Migrate happens on the PRISTINE state, mirroring what production
    # would do at the moment fire decides to move src out of the way.
    t_mig_start = time.monotonic_ns()
    with torch.cuda.stream(side_stream):
        for src_i, dst_i in zip(src_indices, dst_indices):
            initial_state[dst_i].copy_(initial_state[src_i])
    side_stream.synchronize()
    t_mig_end = time.monotonic_ns()
    mig_ms = (t_mig_end - t_mig_start) / 1e6
    print(f"[2.2] migrated {len(src_indices)} slot states "
          f"src->dst on side stream: {mig_ms:.3f} ms")

    # Verify migration was byte-correct by reading dst directly.
    for src_i, dst_i in zip(src_indices, dst_indices):
        if not torch.equal(initial_state[src_i], initial_state[dst_i]):
            print(f"  ERROR: slot {src_i} data not equal to slot {dst_i} "
                  f"after migrate. Side-stream copy failed.")
            return 4
    print(f"  byte-correct: initial_state[dst] == initial_state[src] ✓")

    # Update the snapshot so subsequent restores include the migrated
    # dst contents (otherwise restoring pristine zeros out the dst
    # slots we just populated).
    pristine_with_migration = initial_state.clone()
    # Optional: zero out src to simulate "src is now invalidated, only
    # dst holds the data". This is the production scenario where unmap
    # will follow.
    for src_i in src_indices:
        initial_state[src_i].zero_()
    torch.cuda.synchronize()
    print(f"  zeroed src slots {src_indices} (simulating post-unmap state)")
    pristine_post_invalidate = initial_state.clone()

    # ---- 2.3 replay with REWRITTEN indices ----
    initial_state.copy_(pristine_post_invalidate)
    ssm_state_indices.copy_(torch.tensor(dst_indices, dtype=torch.int32,
                                          device=f"cuda:{DEVICE}"))
    g.replay()
    torch.cuda.synchronize()
    migrated_dst_outputs = out.clone()
    diff_migrated = (migrated_dst_outputs.float() -
                     baseline_outputs.float()).abs().max().item()
    print(f"[2.3] replay with migrated dst indices {dst_indices} "
          f"(src zeroed, dst holds copied state): "
          f"max diff from baseline = {diff_migrated:.6f}")

    TOL = 1e-3  # bf16 precision tolerance
    if diff_migrated > TOL:
        print(f"\n=== A2 REFUTED ===")
        print(f"After migrating slot data from src to dst and rewriting "
              f"ssm_state_indices, replay output differs from baseline "
              f"by {diff_migrated:.6f} (> tolerance {TOL}). The captured "
              f"graph + index rewrite is NOT byte-correct. C+A's "
              f"snapshot half is broken.")
        return 1

    print(f"\n  byte-correct: replay(migrated dst) == replay(original src) ✓")

    # ---- 2.5 timing: side-stream migrate during decode ----
    print()
    print(f"[2.5] side-stream migrate during concurrent decode timing:")
    MAT_N = 4096
    A_m = torch.randn(MAT_N, MAT_N, dtype=torch.float32,
                       device=f"cuda:{DEVICE}")
    B_m = torch.randn(MAT_N, MAT_N, dtype=torch.float32,
                       device=f"cuda:{DEVICE}")
    C_m = torch.empty(MAT_N, MAT_N, dtype=torch.float32,
                       device=f"cuda:{DEVICE}")
    LOOPS = 20

    def launch_decode():
        with torch.cuda.stream(decode_stream):
            for _ in range(LOOPS):
                torch.matmul(A_m, B_m, out=C_m)

    # Warmup.
    for _ in range(3):
        launch_decode()
    decode_stream.synchronize()

    baseline_runs = []
    for _ in range(5):
        decode_stream.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(decode_stream)
        launch_decode()
        e1.record(decode_stream)
        decode_stream.synchronize()
        baseline_runs.append(e0.elapsed_time(e1))
    baseline_decode_ms = sorted(baseline_runs)[2]
    print(f"      baseline decode GPU time: {baseline_decode_ms:.2f} ms")

    # Now decode + concurrent migrate on side stream.
    decode_stream.synchronize()
    side_stream.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record(decode_stream)
    launch_decode()
    e1.record(decode_stream)

    with torch.cuda.stream(side_stream):
        # Migrate 50 slots' state, more substantial than 4-slot copy
        for src_i in range(50):
            initial_state[2100 + src_i].copy_(initial_state[src_i])
    side_stream.synchronize()
    decode_stream.synchronize()
    concurrent_decode_ms = e0.elapsed_time(e1)
    print(f"      concurrent decode GPU time: {concurrent_decode_ms:.2f} ms")
    delta = concurrent_decode_ms - baseline_decode_ms
    print(f"      delta: {delta:.2f} ms (threshold: +5 ms)")

    if delta > 5.0:
        print(f"\n  WARN: side-stream migrate slowed decode by "
              f"{delta:.2f} ms. C+A's overlap assumption is weaker than "
              f"step 1's pure-unmap case.")

    print()
    print("=== A2 CONFIRMED ===")
    print(f"migrate_slot data move + ssm_state_indices rewrite preserves "
          f"replay byte-correctness within bf16 tolerance "
          f"(max diff {diff_migrated:.6f}). Side-stream migrate of 50 "
          f"slots overlaps with decode at delta {delta:.2f} ms.")
    print()
    print("Implication: C half of C+A is sound. The pre-fire snapshot "
          "step can move state to safe slots; subsequent replay (with "
          "rewritten indices) reads migrated state byte-correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
