"""decode_wall — production unmap is safe under captured Triton graph (regression).

Production fire path uses **no defensive sync, no setAccess(NONE)
revoke** (post-#205). The safety contract is layer 0 (`fire_planner`
only picks free pages not referenced by any in-flight req's
`state_indices`). This regression test verifies that the layer-0
contract + cuMemUnmap's intrinsic atomicity is sufficient under a
captured Triton CUDA graph replaying concurrently with a worker-
thread unmap of a different VA range in the same reservation.

This is the production-code-bound counterpart to A1 step 1.5
(`0_page_state_machine/step1_stream_isolated_unmap/`): step 1.5
proved physics allows the pattern with raw cuMemUnmap (no sync, no
setAccess); this file proves sglang's production
`arena.shrink_explicit` (post-#205, also no sync, no setAccess)
matches.

Setup
-----
- `ChunkArena` with one pool, 200 chunks mapped.
- `state` tensor over chunks 0..3 (slots 0..4095).
- Captured CUDA graph runs `fused_recurrent_gated_delta_rule_packed_decode`
  with `ssm_state_indices = [0, 1, 2, 3]` (kernel only touches
  chunk 0).
- Thread A: replays the graph in a loop.
- Thread B (worker): calls `arena.shrink_explicit(pool, [100..199])`
  on chunks the kernel never reads.

Pass criterion: 3 trials × 200 replays + concurrent 100-chunk unmap,
no `cudaErrorIllegalAddress`. Equivalent to A1 step 1.5 result
(+0.11 ms decode wall, no crash) carried through production code.

If this regression fails, layer-0 invariant + cuMemUnmap atomicity
is insufficient on the test platform — the design's fail-fast
posture (design.md §"Transfer protocol" Stage 3) needs to be
revisited.

Run:
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python \\
    dev/interlayer/0_page_state_machine/decode_wall/test_no_crash.py
"""
from __future__ import annotations

import sys
import threading

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

import sglang.srt.arena.chunk_arena as ca
from sglang.srt.arena.chunk_arena import ChunkArena
from sglang.srt.arena.from_blob_ext import tensor_from_va

DEVICE = 0
torch.cuda.set_device(DEVICE)
torch.cuda.synchronize()
ca.CUDA.cuInit(0)

CHUNK_BYTES = 2 * 1024 * 1024
N_SLOTS = 200
POOL_NAME = "test_pool"


def main() -> int:
    print("=== decode_wall — production unmap safe under captured Triton graph ===")
    print()

    arena = ChunkArena(
        device_id=DEVICE,
        chunk_size=CHUNK_BYTES,
        n_handles=N_SLOTS,
        pool_capacities=[(POOL_NAME, N_SLOTS)],
    )
    n_mapped = len(arena.grow(POOL_NAME, N_SLOTS))  # #213: grow returns list[int]
    assert n_mapped == N_SLOTS
    pool = arena.pools[POOL_NAME]

    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    HV, V, K = 4, 16, 16
    H = 2
    BYTES_PER_SLOT = HV * V * K * 2  # bf16
    SLOTS_PER_CHUNK_K = CHUNK_BYTES // BYTES_PER_SLOT  # 1024
    N_STATE_CHUNKS = 4
    N_STATE_SLOTS = N_STATE_CHUNKS * SLOTS_PER_CHUNK_K  # 4096

    n_bf16_state = (N_STATE_CHUNKS * CHUNK_BYTES) // 2
    state_flat = tensor_from_va(
        pool.va_base, (n_bf16_state,), torch.bfloat16, DEVICE
    )
    state = state_flat.view(N_STATE_SLOTS, HV, V, K)
    state.fill_(0.0)
    for s in range(128):
        state[s].fill_(float(s % 100))

    B = 4
    qkv_dim = H * K * 2 + HV * V
    mixed_qkv = torch.zeros(B, qkv_dim, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    a = torch.zeros(B, HV, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    b_in = torch.zeros(B, HV, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    A_log = torch.zeros(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    dt_bias = torch.zeros(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    out = torch.zeros(B, 1, HV, V, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    ssm_idx = torch.zeros(B, dtype=torch.int32, device=f"cuda:{DEVICE}")
    ssm_idx.copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=f"cuda:{DEVICE}"))

    def run_kernel():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b_in,
            A_log=A_log, dt_bias=dt_bias,
            scale=1.0, initial_state=state, out=out,
            ssm_state_indices=ssm_idx,
            use_qk_l2norm_in_kernel=False,
        )

    run_kernel()
    torch.cuda.synchronize()
    print("[setup] kernel warmed up, JIT done")

    cap_stream = torch.cuda.Stream(device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(cap_stream):
        with torch.cuda.graph(graph, stream=cap_stream):
            run_kernel()
    torch.cuda.synchronize()
    print("[setup] captured CUDA graph over real Triton kernel")
    print()

    n_trials = 3
    n_replays_per_trial = 200
    unmap_slot_range = list(range(100, 200))  # 100 chunks; kernel reads chunks 0..3

    fault = []

    def worker_unmap():
        try:
            arena.shrink_explicit(POOL_NAME, unmap_slot_range)
        except Exception as e:
            fault.append(("worker", type(e).__name__, str(e)[:200]))

    for trial in range(n_trials):
        if trial > 0:
            n_remapped = len(arena.grow(POOL_NAME, len(unmap_slot_range)))  # #213
            assert n_remapped == len(unmap_slot_range)
            torch.cuda.synchronize()

        t = threading.Thread(target=worker_unmap, name="d2b_unmap_worker")
        t.start()

        try:
            for _ in range(n_replays_per_trial):
                graph.replay()
            torch.cuda.synchronize()
        except Exception as e:
            fault.append(("main", type(e).__name__, str(e)[:200]))

        t.join(timeout=30.0)
        if t.is_alive():
            fault.append(("worker", "TIMEOUT", "worker thread did not finish"))

        if fault:
            break

        print(f"[trial {trial}] {n_replays_per_trial} replays + "
              f"100-chunk worker unmap: no crash ✓")

    print()
    if fault:
        print(f"=== decode_wall: FAILED — {len(fault)} fault(s) reproduced ===")
        for src, typ, msg in fault:
            print(f"  [{src}] {typ}: {msg}")
        print()
        print("Layer-0 invariant (fire_planner picks free pages) + "
              "cuMemUnmap atomicity is NOT sufficient on this platform.")
        print("Action: revisit design.md §\"Transfer protocol\" Stage 3 — "
              "the fail-fast posture is wrong, defensive layer must be "
              "added back. See A1 step1.7 for setAccess(NONE) option.")
        return 1
    print("=== decode_wall: PASS — production code (no defense) safe under "
          "captured Triton graph + concurrent unmap ===")
    print(f"  {n_trials} trials × {n_replays_per_trial} replays + "
          f"100-chunk worker unmap, all clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
