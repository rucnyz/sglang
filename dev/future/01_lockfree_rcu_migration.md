# Lock-free RCU migration protocol

## Motivation
The current atomic-migration path for residual live pages (design.tex §3.2.3)
uses `cuStreamSynchronize` to enforce that all in-flight kernels finish reading
a block before its physical page is unmapped. This is a coarse barrier that
blocks the migrating thread for ~10 μs per fire on top of the ~1 ms D2D copy.

Linux memory hot-unplug solves the analogous CPU-side problem with
RCU (Read-Copy-Update): the migration thread copies, atomically swaps the page
table entry, then waits for a grace period during which old readers naturally
drain. New readers see the new mapping immediately; the migration thread never
blocks on synchronous fences.

## Approach
- Per-block reference counter exposed via shared device memory.
- Migration thread:
  1. D2D copy block bytes from old page to new page.
  2. Atomically swap allocator's per-block VA pointer (release semantics).
  3. Spin until ref-count of old page hits zero (RCU grace period).
  4. Unmap old page.
- Kernels touching a block do an acquire load of the block's VA pointer + a
  refcount increment on entry and decrement on exit.

## Trade-offs
- Saves ~10 μs/fire of stream-sync wall.
- Adds 2 atomic ops per block-access on the kernel hot path. On H200 with
  ~80 K prefill tokens/s this could be measurable; would need micro-bench.
- Protocol verification is non-trivial; getting the ABA-safety and
  release-acquire ordering right is real work.

## When to do it
At fire frequencies above ~1/s (heavy oscillating workload) where the
sync-barrier overhead becomes a measurable fraction of throughput. Below that
frequency the savings are sub-percent. Probably never worth it for current
agent-serving workloads where fires are minute-scale.
