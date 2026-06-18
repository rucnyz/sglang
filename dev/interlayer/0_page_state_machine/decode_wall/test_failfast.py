"""decode_wall — fail-fast on layer-0 violation (subprocess-isolated).

Pins design.md §"Threading model" + §"Transfer protocol" Stage 3
fail-fast claim: if the layer-0 invariant is violated (fire_planner
picks a page whose slot indices an in-flight req's state_indices
references), the next captured-graph replay touching that page must
fault with `cudaErrorIllegalAddress` — fast and visible, not silently
swallowed.

This is the "we don't defend, but if you misuse, you fail loud" half
of #205's posture. Without this test, the layer-0 contract is a paper
claim — we'd have no evidence that an actual misuse would surface.

Subprocess-isolated because CUDA illegal-access faults poison the
context (any subsequent CUDA op also faults). Mirrors step 1.7's
subprocess pattern.

Setup (inside the subprocess)
-----------------------------
- Same as test_no_crash.py: ChunkArena, 200 chunks, captured Triton
  graph over `fused_recurrent_gated_delta_rule_packed_decode`.
- BUT `ssm_state_indices = [150]` so the kernel reads slots backed by
  chunk 150 (which differs from the no-crash case where indices were
  [0,1,2,3] — disjoint from the unmap range).
- Worker thread calls `arena.shrink_explicit(pool, [100..199])`. The
  unmap range NOW includes chunk 150.
- Replay the captured graph after the unmap returns.

Pass criterion
--------------
- Replay raises `AcceleratorError` / `RuntimeError` whose message
  contains "illegal memory access" or `cudaErrorIllegalAddress`.
- Subprocess prints `FAULT_OK` and exits 0.
- Parent asserts rc == 0 and FAULT_OK marker in stdout. If subprocess
  exits 7 (`NO_FAULT` marker), the fail-fast claim is REFUTED — replay
  silently succeeded against unmapped VA, which means the design's
  fail-fast safety story is hollow.

Run
---
  CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python \\
    dev/interlayer/0_page_state_machine/decode_wall/test_failfast.py
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


SUBPROC_SRC = textwrap.dedent(r"""
    import sys
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
    OVERLAP_CHUNK = 150  # kernel reads slots in this chunk; we unmap it

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
    # state spans the full N_SLOTS chunks so OVERLAP_CHUNK is addressable.
    N_STATE_SLOTS = N_SLOTS * SLOTS_PER_CHUNK_K

    n_bf16_state = (N_SLOTS * CHUNK_BYTES) // 2
    state_flat = tensor_from_va(
        pool.va_base, (n_bf16_state,), torch.bfloat16, DEVICE
    )
    state = state_flat.view(N_STATE_SLOTS, HV, V, K)
    state.fill_(0.0)
    # Pre-write into the overlap chunk so the read isn't trivially
    # optimized away.
    slot_in_overlap = OVERLAP_CHUNK * SLOTS_PER_CHUNK_K
    state[slot_in_overlap].fill_(1.0)
    torch.cuda.synchronize()

    B = 1
    qkv_dim = H * K * 2 + HV * V
    mixed_qkv = torch.zeros(
        B, qkv_dim, dtype=torch.bfloat16, device=f"cuda:{DEVICE}"
    )
    a = torch.zeros(B, HV, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    b_in = torch.zeros(B, HV, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    A_log = torch.zeros(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    dt_bias = torch.zeros(HV, dtype=torch.float32, device=f"cuda:{DEVICE}")
    out = torch.zeros(B, 1, HV, V, dtype=torch.bfloat16, device=f"cuda:{DEVICE}")
    ssm_idx = torch.zeros(B, dtype=torch.int32, device=f"cuda:{DEVICE}")
    # KEY: the kernel will read from slot_in_overlap (chunk 150).
    ssm_idx.copy_(torch.tensor(
        [slot_in_overlap], dtype=torch.int32, device=f"cuda:{DEVICE}"
    ))

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

    cap_stream = torch.cuda.Stream(device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(cap_stream):
        with torch.cuda.graph(graph, stream=cap_stream):
            run_kernel()
    torch.cuda.synchronize()
    print(f"setup_ok chunk_overlap={OVERLAP_CHUNK} slot={slot_in_overlap}", flush=True)

    # ---- The layer-0 violation: unmap a chunk the captured graph reads.
    unmap_range = list(range(100, 200))  # includes OVERLAP_CHUNK=150
    n_freed = arena.shrink_explicit(POOL_NAME, unmap_range)
    print(f"unmap_done freed={n_freed}", flush=True)

    # ---- Replay against unmapped VA: expect cudaErrorIllegalAddress.
    try:
        graph.replay()
        torch.cuda.synchronize()
        # If we get here, no fault — design's fail-fast claim is REFUTED.
        print("NO_FAULT replay_completed_against_unmapped_VA", flush=True)
        sys.exit(7)
    except Exception as e:
        msg = str(e)[:240]
        # Look for the canonical signatures.
        ok = (
            "illegal memory access" in msg.lower()
            or "cudaErrorIllegalAddress" in msg
            or "an illegal memory access was encountered" in msg.lower()
        )
        if ok:
            print(f"FAULT_OK {type(e).__name__}: {msg}", flush=True)
            sys.exit(0)
        else:
            print(f"WRONG_FAULT {type(e).__name__}: {msg}", flush=True)
            sys.exit(8)
""")


def main() -> int:
    print("=== decode_wall — fail-fast on layer-0 violation (subprocess-isolated) ===")
    print()
    print("Driving layer-0 violation in subprocess: captured graph reads "
          "chunk 150, worker unmaps chunks 100..199.")
    print("Expect: subprocess replay raises cudaErrorIllegalAddress.")
    print()

    r = subprocess.run(
        ["/scratch/yuzhou/projects/sglang/.venv/bin/python", "-c", SUBPROC_SRC],
        capture_output=True, text=True, timeout=120,
    )
    stdout_tail = (r.stdout or "")[-1200:].strip()
    stderr_tail = (r.stderr or "")[-600:].strip()

    print(f"[subprocess] rc={r.returncode}")
    print("[subprocess stdout]")
    for line in stdout_tail.splitlines():
        print(f"  {line}")
    if r.returncode != 0 and stderr_tail:
        print("[subprocess stderr (tail)]")
        for line in stderr_tail.splitlines()[-12:]:
            print(f"  {line}")
    print()

    if r.returncode == 7 or "NO_FAULT" in stdout_tail:
        print("=== decode_wall fail-fast: REFUTED ===")
        print("Replay against unmapped VA did NOT fault. The design's "
              "fail-fast safety story (layer-0 violation → "
              "cudaErrorIllegalAddress) is hollow on this platform — a "
              "layer-0 bug in fire_planner would corrupt silently.")
        print("Action: design.md §\"Transfer protocol\" Stage 3 needs to "
              "re-add a defensive layer (sync OR cuMemSetAccess(NONE)). "
              "See A1 step 1.7 for the setAccess option.")
        return 1
    if r.returncode == 8 or "WRONG_FAULT" in stdout_tail:
        print("=== decode_wall fail-fast: INCONCLUSIVE ===")
        print("Replay raised an exception, but not the expected "
              "cudaErrorIllegalAddress signature. Investigate the actual "
              "exception above — the failure mode may have changed.")
        return 1
    if r.returncode != 0 or "FAULT_OK" not in stdout_tail:
        print("=== decode_wall fail-fast: INCONCLUSIVE ===")
        print(f"Subprocess exited rc={r.returncode} without the expected "
              "FAULT_OK marker. Check stderr above.")
        return 1

    print("=== decode_wall fail-fast: PASS ===")
    print("Layer-0 violation (kernel reads chunk 150, worker unmaps it) "
          "produced cudaErrorIllegalAddress on next replay, as design "
          "requires. Fail-fast safety story is real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
