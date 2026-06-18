# cuda_graph_safety — CUDA graph safety post-transfer

What it tests: a CUDA graph captured BEFORE a cross-arena transfer
still works AFTER the transfer remaps the physical handles underneath
its VA. The kernel binds to virtual address, not physical handle —
cuMemMap into the same VA window with a new handle should be
transparent to a captured + replayed graph.

3 sub-tests:
- test_1: intra-arena graph + transfer + page_table update + VA
  stability + full-chunk byte verify
- test_2: cross_arena_transfer production path graph safety
- test_3: shrink+regrow with VERIFIED-different physical handle
  (snapshot old handle_idx, force LIFO via `pool.grow(1)`, assert
  `new_handle_idx != old_handle_idx`) then graph reads the new value
  — proves the kernel binds to VA not physical handle (after test_3
  was rewritten following review — earlier version had a "reading
  same value twice" anti-pattern that didn't actually verify the
  new-handle claim)

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python dev/interlayer/0_page_state_machine/cuda_graph_safety/test_cuda_graph_safety.py
```

No env-vars; takes ~10s; needs a GPU.

## Result

3/3 PASS (v2 after strict review). Captures the load-bearing CUDA
guarantee: cuMemMap is a VA-level operation, captured graphs see the
new physical mapping transparently. If this ever regresses (e.g. PyTorch
changes graph internals), this test catches it.

**Scope**: correctness invariant only — tests move 4 chunks under a
captured graph. Large-n behavior (decode-stream wall + concurrent
unmap at ~100 chunks per sub-pool) is covered by
[`../decode_wall/`](../decode_wall/).

Commit: `428b3c6b91`
