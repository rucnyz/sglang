# 0_page_state_machine — A1 + A2 evidence

Empirical proofs for the two structural properties that the page
state machine in [`../design.md`](../design.md) §"Page ownership
state" + §"Threading model" depends on:

- **A1** (`step1_stream_isolated_unmap/`): raw worker-thread
  `cuMemUnmap` does not crash a captured CUDA graph and does not
  stall the decode stream, given the layer-0 invariant
  (`fire_planner` picks only pages no in-flight req's
  `state_indices` references). `cuMemSetAccess(prot=NONE)` revoke
  is an equivalent (also-safe) alternative defense per step 1.7,
  but production (post-#205) takes the no-defense leg.
- **A2** (`step2_migrate_slot_replay_invariant/`): side-stream
  `MambaPool.migrate_slot` + `ssm_state_indices` rewrite is
  byte-exact under captured-graph replay.
- **`alloc_lock/`**: pins **allocator-internal cross-thread
  mutation safety** — `_alloc_lock` race + leak reproducers +
  contention perf characterization. Covers #222
  (`MambaPool._alloc_lock`) and the KV-side parallel. Other
  Phase-0 sub-folders that spawn threads (`decode_wall/`,
  `step1_stream_isolated_unmap/`) model the worker thread against
  a synthetic decode workload to pin the cuMem syscall's stream-
  isolation properties; this folder is the only one that pins the
  allocator's mutating entry points under concurrent access.
- **`decode_wall/`**: production-bound regression for property A1 +
  §"Transfer protocol" Stage 3 + the decode-stream-wall half of
  §"fire_wall_curve + decode_wall — per-batch-size physical-floor
  budgets". Three tests: `test_no_crash` (production unmap safe
  under captured Triton graph), `test_failfast` (layer-0 violation
  surfaces as `cudaErrorIllegalAddress`), `test_decode_wall`
  (multi-sub-pool worker-thread unmap loop ≤ 0.10 ms wall budget —
  ~3-5× headroom over the stable 0.00–0.03 ms median).
- **`dtype_unit_sizes/`**: spec-level pin (no GPU) that sglang's
  per-slot byte arithmetic matches independently hand-derived
  constants. Covers design.md §dtype_unit_sizes.
- **`pristine_saturation/`**: engine-level pin that sglang's actual
  allocator output matches those same hand-derived constants under
  pristine boot. Covers design.md §pristine_saturation. Pairs with `dtype_unit_sizes/`
  (imports `HAND_VERIFIED_*` from there).
- **`vmm_boot_smoke/`**: chunk_arena boots cleanly and executes the
  transfer cycle (grow / shrink / cross-arena handle transfer) without
  losing handles, corrupting bytes, or racing in-flight kernels.
  15 sub-tests across handle-pool lifecycle, VA semantics, transfer
  correctness. Covers design.md §vmm_boot_smoke.
- **`va_reservation_hbm/`**: mechanism-level pin that
  `cuMemAddressReserve` of N GiB of VA costs zero HBM. After the
  arena wires in, total HBM = sglang baseline + actually-mapped
  chunks (no surprise overhead from the reservation). Covers design.md
  §va_reservation_hbm at the mechanism layer.
- **`fire_wall_curve/`**: per-n actuator-wall curve at
  n ∈ {4, 8, 16, 30} — p50 wall ≤ `n × 70 µs × 1.4` budget plus p99/p50
  < 5×. Pins the physical floor (~70 µs/chunk TLB invalidation) and
  catches super-linear regressions. Covers design.md §fire_wall+decode_wall.
- **`cuda_graph_safety/`**: a captured CUDA graph survives a
  cross-arena transfer that remaps the underlying physical handle —
  the kernel binds to VA, not handle. Correctness invariant only
  (n=4 chunks); large-n behavior is exercised by `decode_wall/`.
  Covers design.md §cuda_graph_safety.

## A1 — `step1_stream_isolated_unmap/`

Confirmed across all production-relevant axes (real Triton kernel,
captured CUDA graph, same-VA reservation, `threading.Thread`
worker, subprocess-isolated revoke-fault check):

```
[1.4] cuBLAS GEMM + eager + disjoint VA + main thread:  delta +0.02 ms
[1.5] real Triton kernel + captured graph + SAME-VA reservation +
      3 trials × 50 replays + concurrent 100-chunk unmap:  no crash
[1.6] worker-thread cuMemUnmap (mirrors Budgeter _fire_worker):
      decode delta +0.11 ms (within noise)
[1.7] cuMemSetAccess(prot=0) verified to fault subsequent device
      reads (real revoke, not bookkeeping)
```

The actuator's worker-thread Unmap step exploits this property:
after Drain or Migration has pre-conditioned a page to FREE state,
the unmap happens on a side thread without disturbing the captured-
graph replay on the scheduler thread. Fire wall on the decode stream
is ~0 ms. Production (post-#205) uses raw `cuMemUnmap` without
defensive sync or setAccess revoke — see
[`./decode_wall/`](./decode_wall/) for the
production-bound regression (`test_no_crash`, `test_failfast`,
`test_decode_wall`).

## A2 — `step2_migrate_slot_replay_invariant/`

`MambaPool.migrate_slot(src→dst)` on a side stream + runtime rewrite
of `ssm_state_indices` to point at dst preserves captured-graph
replay output byte-exactly. Side-stream copy of 50 slots' worth of
state (~8 MiB) overlaps a 53 ms decode kernel at zero measured cost
(delta -0.01 ms).

```
[2.1] baseline replay with indices=[0,1,2,3]:        out recorded
[2.4] neg-ctrl (uncopied dst): diff from baseline = 0.5317 (control discriminates)
[2.2] migrated 4 slot states src→dst on side stream: 0.091 ms
[2.3] replay with migrated dst indices:              diff = 0.000000 ✓
[2.5] side-stream migrate during 53 ms decode:       delta -0.01 ms
```

The Migration transition (paper appendix line 53-54; design.md
§"Page ownership state") uses this property: when the Admitter
picks `own_migrate` or `cross_migrate`, the actuator calls
`migrate_slot` to relocate the slot state to a different page,
then rewrites the in-flight req's `ssm_state_indices`. A2
guarantees the next captured-graph replay reads byte-identical
state.

## Implementation locations in sglang

The properties above guarantee that these production code paths are
safe:

| Property | Production site |
|---|---|
| A1 — worker-thread Unmap | `python/sglang/srt/arena/xpool_actuator.py:70` (`XPoolActuator` class), `:105` (`_fire_inflight` lock), `:251` (worker `with self._fire_inflight:`), `:254` (`_execute_async_locked`) — the worker thread that drives `cuMemUnmap` |
| A1 — cuMem syscall | `python/sglang/srt/arena/chunk_arena.py:388` `_unmap_slots_batched` (raw `cuMemUnmap`, no defensive sync, no setAccess revoke per #205). `cuMemSetAccess(prot=3)` is still called by `_map_slots_batched` to set RW on newly-mapped chunks — that's the normal grow-path, not the revoke pattern step 1.7 verified |
| A2 — migrate_slot primitive | `python/sglang/srt/mem_cache/memory_pool.py:756` (`MambaPool.migrate_slot`) |
| A2 — runtime index tensor | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:485` (`state_indices_list[bs-1][:len(mamba_indices)].copy_(mamba_indices)`) — the per-replay index rewrite |
| A2 — kernel that reads indexed state | `python/sglang/srt/layers/attention/fla/fused_recurrent.py:268` (`fused_recurrent_gated_delta_rule_packed_decode`) |

## §Transfer protocol coverage map

design.md §"Transfer protocol" lays out four stages. This folder
covers the mechanism-layer pins for Stages 2-3 + Stage 0 Migration;
Stage 0 Drain and Stage 1 Cap-barrier live one folder up in
`1_dyn_admission_cap/`.

| Stage | Where pinned |
|---|---|
| 0 — Drain (CACHED→FREE via sglang evict) | `1_dyn_admission_cap/test_mark_no_realloc.py::test_8` (evict returning capped slots) |
| 0 — Migration (LIVE→FREE via migrate_slot + index rewrite) | `step2_migrate_slot_replay_invariant/` (A2 byte-exact replay) |
| 1 — Cap-barrier (FREE→CAPPED, scheduler thread) | `1_dyn_admission_cap/test_mark_no_realloc.py` tests 1-3: `alloc` skips capped (test_1), no-realloc mark (test_2), mark-unmark idempotency (test_3) |
| 2 — Verify (`F.issubset(src.capped)`) | `1_dyn_admission_cap/test_mark_no_realloc.py::test_5` (verify subtracts `_capped_pages` from `free_pages` before the `isin` check — Phase 6 D6 first-attempt root cause) + `::test_31` (post-#215: verify moved to worker thread; cap_t leak abort path) |
| 3 — Unmap + Map (worker thread) | `decode_wall/` (no-crash + fail-fast + wall budget); `step1_stream_isolated_unmap/` (A1 raw cuMemUnmap stream isolation); `cuda_graph_safety/` (captured-graph survives remap) |

## Cross-references

- Main design: [`../design.md`](../design.md) §"Page ownership state",
  §"Threading model", §"Transfer protocol".
- Implementation plan: [`../PLAN.md`](../PLAN.md) Phase 3
  (`own_migrate` / `cross_migrate` Admitter integration depends on A2).
- Falsified design framings retained for traceability:
  [`../archive/c_plus_a_framing.md`](../archive/c_plus_a_framing.md),
  [`../archive/0_batch_boundary_fire/`](../archive/0_batch_boundary_fire/).
