# vmm_boot_smoke — VMM mechanism boots + transfer cycle correctness

What it tests: the low-level cuMem-based mechanism layer in
`python/sglang/srt/arena/chunk_arena.py` boots cleanly and executes
the transfer cycle (grow / shrink / cross-arena transfer) without
losing handles, corrupting bytes, or racing in-flight kernels.

15 sub-tests across 3 themes:
- handle-pool lifecycle (grow / shrink / cleanup, lazy growth, retry
  on stranded handles, owned vs external)
- VA semantics (disjoint reservations, over-provisioned VA, full-chunk
  byte integrity, owned-handle identity)
- transfer correctness (cross-arena handle identity, 10× ping-pong,
  re-grow no double-pop, tail-evict explicit)

The original test_16 (subprocess race detection with/without
`torch.cuda.synchronize()` before `cuMemUnmap`) was removed: per #205
production runs the no-defense leg (no sync, no `cuMemSetAccess(NONE)`
revoke), and the equivalent fail-fast safety story is now pinned by
[`../decode_wall/test_failfast.py`](../decode_wall/test_failfast.py)
against the actual production code path.

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python dev/interlayer/0_page_state_machine/vmm_boot_smoke/test_vmm_boot_smoke.py
```

No env-vars; takes ~30s; needs a GPU with CUDA VMM (any H100/H200).

## Result

15/15 PASS. Two test_13 / test_14 assertions were updated to the
`list[int]` return from `chunk_arena.grow()` introduced by #213
(was `int`).

Commit: `428b3c6b91` (introduced).
