# mamba_pool_perf — arena MambaPool hot-path overhead

Pins (and gates the fix of) the accidental per-step overhead the arena
`MambaPool` per-layer-list temporal layout adds over the non-budgeter stacked
single-tensor baseline.
The budgeter turns the temporal state per-layer (`SGLANG_MAMBA_PERLAYER`, auto-on with `SGLANG_ARENA_SHARED`) so cross-pool transfer can map physical bytes per sub-pool;
the baseline keeps one stacked `[num_layers, size+1, ...]` tensor.
Every per-request op (`alloc`/`free`/`copy_from`) then loops over `num_layers` tensors instead of touching one, paying extra kernel launches, and `free` paid a device sync on every call.
On a high-concurrency short swarm (thousands of small COW hits, often B=1) those launches dominated and showed up as the ~2-3.7% default-split decode regression in the agentreplay A/B.

This is the mamba-pool analog of the KV-allocator `free()` isin tax fixed in #322;
the cost is launch + sync overhead, not time/space complexity, so removing it where the pool has never been touched by a cross-fire is strictly free.

## What is pinned

| file | what it pins | needs GPU |
|---|---|---|
| `bench_mamba_pool_ops.py` | microbench: per-op us (`copy_from`/`alloc`/`free`/`live_size`) for stacked vs per-layer-list at production geometry (num_layers=24, temporal (32,128,128) fp32, conv (8192,3) bf16, size=443), `_capped_slots` empty. Sweeps B=1/4/147 so launch overhead (small B) is visible and the large-B byte cost is confirmed layout-independent. | yes |
| `test_mamba_free_fastpath.py` | the `_no_cross_fire` predicate gating `free`'s fast path: True only when `_capped_slots` empty AND `size == max_size`; flips False on shrink / populated `_capped_slots`, so a capped (unmapped) slot is never handed back (4 sub-tests). | no |
| `test_mamba_pool_invariants.py` | correctness + conservation on a real tiny CPU pool: alloc zero-init equivalence, `copy_from` byte fidelity, fast-path == slow-path free output, fast path provably skips `torch.isin`, and a 400-op alloc/free/shrink/grow/migrate/unmark churn keeps {free, capped, allocated} a partition of [1, max_size] after every op (8 sub-tests, both layouts). | no |
| `test_mamba_pool_perf_targets.py` | N=3 wall-time asserts of the fix's design targets: `free` ~0 (≤6us), `alloc` ≤100us (hoist realized), `live_size` ~0, `copy_from` ≤130us (flagged-inherent regression guard). | yes (idle) |

## The fix (in `MambaPool`, `python/sglang/srt/mem_cache/memory_pool.py`)

Three accidental costs, all ~0 in the stacked baseline, all exercised heavily by a high-concurrency swarm:

- `free` did `torch.isin(free_index, _capped_slots)` + a `free_index > self.size` mask whose `.any().item()` forces a device-to-host sync on EVERY free, even with no cross-fire ever active.
  Fix: the `_no_cross_fire` fast path (`_capped_slots` empty AND `size == max_size`) returns freed ids straight to `free_slots` via `torch.cat`, the baseline path, no isin / no mask / no sync.
  The instant any path (`migrate_slot`, `set_capacity_slots` shrink, the above-cap free branch) populates `_capped_slots` or lowers the cap, the predicate is False and the full capped-aware path runs verbatim, so a capped slot can never be returned (the #312/#329 unmapped-VA guard).
  Pre-fix +11us/call, post-fix ~0.

- `alloc` allocated a fresh `torch.zeros(1)` inside each per-layer iteration (24 tiny allocation kernels in production) for an identical zeroed result.
  Fix: hoist the scalar zero OUT of the loop, one per dtype, broadcast-expanded per layer.
  Pre-fix +168us, post-fix ~+55us (the residual is the inherent per-layer indexed-write launches).

- `live_size` already early-returns `self.size` when `_capped_slots` is empty (no `.item()` sync); unchanged, confirmed ~0.

## Flagged for human (NOT fixed)

The residual `copy_from` +92us and `alloc` +55us are the inherent 24-vs-1 per-layer indexed-write launches.
Cutting them needs an arena-layout change (a single stacked temporal backing under one VA range), which is unsound as a drop-in:
the arena lays each per-layer tensor over a SEPARATE `from_blob` VA range, so a stacked `as_strided` view is storage-out-of-bounds, would address each layer's unmapped headroom tail, and would have to be rebuilt on every cross-pool transfer, coupling the hot path to arena internals and breaking per-sub-pool independence.
CUDA-graphing the copy is not free either (indices and size vary per call).
Tracked as a deferred arena-layout decision (#332), not a bug; regression-guarded today by `test_mamba_pool_perf_targets.py` (`copy_from` ≤130us).
