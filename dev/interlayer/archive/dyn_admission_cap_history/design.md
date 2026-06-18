# Design: dynamic admission cap (option 1b, ideal arch)

**Status:** drafting (post-audit synthesis, pre-implementation).
**Inputs:** `audit_consumers.md`, `audit_other_arrays.md`,
`audit_cuda_graphs.md`, `audit_design_intent.md`.

## 1. Goal

When the cross-pool actuator grows `mamba_pool.size` from N → M,
sglang's per-request bookkeeping arrays grow correspondingly so that
admission can accept up to `floor(M / mamba_per_req_ratio)` concurrent
requests at runtime — capped by user-set `--max-running-requests` as
an upper ceiling.

The Budgeter must not be a no-op: D8 v1 measured Δ=-1.70% throughput
(target +10%) precisely because the actuator grew mamba but sglang's
admission stayed at the init-time cap of 33.

## 2. Architectural choice: VA-stable wrapping (audit_cuda_graphs Option B)

The cleanest mechanism is **VA-stable backing for the per-request
arrays**, matching the pattern already used by `MultiTensorArena` for
the KV and mamba pools:

1. Reserve VA up to `MAX_POSSIBLE = floor(arena_max_chunks / ratio)`
   at init time. VA reservation is free (D5 confirms).
2. Physically map only the rows the current admission cap requires.
3. Construct the tensor via `at::from_blob` over the VA base — the
   tensor's `data_ptr()` is the VA base address and **never moves**.
4. On grow, `cuMemMap` more pages within the same VA range. Tensor's
   pointer unchanged.
5. CUDA-captured Triton kernels (which bake in `req_to_token_ptr` per
   `audit_cuda_graphs.md` §2) remain valid — they read from the same
   VA, which is now physically backed for the new rows too.

Per `audit_cuda_graphs.md` §6 Option B + the project-level "ideal
architecture" guideline (memory: `feedback_ideal_architecture.md`),
this is the chosen mechanism. The alternative Option A (re-capture on
grow) trades correctness for a ~1-2 s latency spike per grow event —
acceptable but expedient.

## 3. Scope: which arrays change

Per `audit_other_arrays.md`, the arrays that MUST grow when admission
cap grows are tiered by criticality:

### Tier 1 — required for our test config (Qwen3.5-9B hybrid, no disagg)

| Array | File | Shape | Backing strategy |
|---|---|---|---|
| `ReqToTokenPool.req_to_token` | `mem_cache/memory_pool.py:154` | `(size, max_context_len)` int32 | VA-stable arena (new) |
| `ReqToTokenPool.free_slots` | `mem_cache/memory_pool.py:157` | Python list of `size` ints | extend in-place; not graph-captured |
| `HybridReqToTokenPool.req_index_to_mamba_index_mapping` | `mem_cache/memory_pool.py` (around 875) | `(size, ...)` int | VA-stable arena (new) |
| `FutureMap.token_ids_buf` | `overlap_utils.py:66-68` | `(5 × max_running + padding,)` int64 | VA-stable arena (new) |
| `FutureMap.{topk_p, topk_index, verified_id, hidden_states}_buf` | `overlap_utils.py:92-119` | `(future_buffer_len, ...)` | VA-stable arena (lazy-allocated) |

### Tier 2 — skip for now (gated by features we don't use)

- `MetadataBuffers` (disagg only) — disagg mode off in our test
- `HiSparseCoordinator` — NSA + HiSparse only
- `RoutedExpertsCapturer` — MoE only
- `NgramEmbedding` — n-gram models only

Each of these will need the same treatment when its feature is in
play, but they don't block D8.

### Tier 3 — already handled

- `MambaPool.{conv_state, temporal_state}` — actuator already grows
  these via `MambaArenaActuator.set_capacity_tokens()`. No new work.

## 4. Component-by-component changes

### 4.1. `ReqToTokenPool` — VA-stable backing

Today (`memory_pool.py:139-157`):

```python
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device, ...):
        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros((size, max_context_len), dtype=int32, device=...)
        self.free_slots = list(range(size))
```

After:

```python
class ReqToTokenPool:
    def __init__(self, init_size, max_possible_size, max_context_len, device, ...):
        self.size = init_size       # live admission cap
        self.max_size = max_possible_size  # VA reservation upper bound
        self.max_context_len = max_context_len

        # Reserve VA for max_possible_size rows. Physically map only
        # init_size rows. Tensor is constructed via at::from_blob over
        # the full VA range → shape (max_size, max_context_len), but
        # rows >= self.size are unmapped (would fault on read; admission
        # gate prevents this by not handing out slot ids >= self.size).
        self._va_arena = ReqTokenVAArena(
            n_rows_max=max_possible_size,
            row_bytes=max_context_len * 4,  # int32
            device=device,
        )
        self._va_arena.set_mapped_rows(init_size)
        self.req_to_token = self._va_arena.as_tensor(
            dtype=torch.int32,
            shape=(max_possible_size, max_context_len),
        )
        self.free_slots = list(range(init_size))

    def grow(self, new_size: int) -> None:
        """Map more physical pages so admission can hand out slots
        [self.size, new_size). Tensor data_ptr unchanged."""
        if new_size <= self.size:
            return
        assert new_size <= self.max_size, "exceeds VA reservation"
        self._va_arena.set_mapped_rows(new_size)
        self.free_slots.extend(range(self.size, new_size))
        self.size = new_size

    def shrink(self, new_size: int) -> None:
        """Unmap pages for slots [new_size, self.size). Slots must be
        FREE (not currently held by a running req). The Budgeter is
        responsible for not invoking shrink while slots are in use."""
        ...
```

Notes:
- `ReqTokenVAArena` is a small new helper (analogous to
  `MultiTensorArena` but specialized for the int32 row-table shape).
  Could be implemented by reusing `chunk_arena.SharedHandlePool` and
  `from_blob_ext.tensor_from_va` primitives — they're already in the
  arena module and known-good (D1/D4 PASS).
- Row size = 1 MiB on Qwen3.5-9B (`max_context_len=262144 × 4`). The
  arena's chunk granularity is 2 MiB → we map 2 rows per chunk. Grow
  granularity is 2 rows. Acceptable.
- `max_possible_size = floor(max_mamba_chunks / mamba_per_req_ratio)`.
  Computed at init from arena's pre-reserved max (default 80 GiB
  headroom → 40960 mamba chunks → 13687 req rows for ratio=3).

### 4.2. `HybridReqToTokenPool` (Mamba models)

Same treatment for `req_index_to_mamba_index_mapping` and
`mamba_spec_state` tensors (per `audit_consumers.md` §3): wrap in a
VA-stable arena, grow in lockstep with the base
`req_to_token`. Same `grow()` method called from the same trigger.

### 4.3. `FutureMap` (overlap mode)

Per `audit_other_arrays.md` Tier 2 row 3, FutureMap's `token_ids_buf`
is a circular buffer of `~5×max_running_requests` int64s. On overflow
it silently overwrites pending tokens — `audit_other_arrays.md`
flagged this as the worst overflow mode (silent corruption).

Same VA-stable backing pattern. The lazy spec-decode buffers
(`topk_p_buf` etc.) follow the same treatment when first allocated.

### 4.4. `DecodeReqToTokenPool` (disagg-only, skipped for D8)

Per `audit_consumers.md` §3, `DecodeReqToTokenPool` already
pre-allocates beyond `size`. For D8 (no disagg) this class isn't
instantiated. Out of scope for the first PR.

## 5. Resize trigger: who calls `pool.grow()` and when

### Trigger source

The Budgeter agent receives fire results in `_maybe_fire` (post-fire
callback already exists in `agent.py`). After each successful fire
that grew mamba, the agent:

1. Reads new `mamba_pool.size` (which the actuator already updated).
2. Computes `new_admission_cap = min(user_ceiling, mamba_pool.size // ratio)`.
3. Calls `scheduler.req_to_token_pool.grow(new_admission_cap)`. This
   triggers cascade to `HybridReqToTokenPool` and `FutureMap`
   (registered as listeners or invoked in sequence).
4. Updates `scheduler.max_running_requests` to `new_admission_cap`
   so any code paths reading the scalar see the new value.

### Concurrency model

`grow()` is called from the scheduler thread, INSIDE the Budgeter's
`tick()` call (which runs on the scheduler event loop). Per
`audit_consumers.md` §"Recommended Approach" + §"Implementation
Strategy", the safe window is **between batches, before any backend
fetches pool refs**. The Budgeter tick already runs at this point
(not during model forward).

Key guarantees:
- VA reservation is FREE (D5 confirms) — no GPU memory cost.
- `cuMemMap` is the only physical op — typically 50-100 µs per chunk
  (D7-measured). Bounded.
- No torch tensor reallocation; `data_ptr()` stable forever.
- No CUDA stream synchronize needed (cuMemMap is host-side syscall).
- `free_slots.extend(...)` is a single Python list op — atomic under
  GIL.

The shrink path (mamba pool shrunk by fire) is symmetric but
constrained: only unmap rows whose slot is FREE (not held). The
Budgeter must consult `free_slots` to compute the safe unmap range.

### Failure modes

- VA exhaustion (admission would exceed `max_possible_size`):
  `grow()` returns early; admission cap stays at current. Logged.
- cuMemMap failure (out of physical memory): rare; log + skip the
  resize. Subsequent admissions blocked but no crash.
- Shrink with slot still held: budgeter computes wrong range. Defended
  by `assert` in `shrink()` that the slot is in `free_slots`.

## 6. Implementation plan (phased)

### Phase 1: ReqTokenVAArena + ReqToTokenPool

- Add `ReqTokenVAArena` helper (reuses `chunk_arena.SharedHandlePool`
  + `from_blob_ext.tensor_from_va`).
- Refactor `ReqToTokenPool.__init__` to take `max_possible_size`,
  back with VA arena.
- Add `grow()` and `shrink()` methods. Make them no-op when
  shared_arena disabled (back-compat).
- Unit test: grow from N to 2N, verify data_ptr unchanged, verify
  rows [N, 2N) writable + readable.

### Phase 2: HybridReqToTokenPool

- Same treatment for `req_index_to_mamba_index_mapping` and friends.
- Ensure lockstep grow with base.

### Phase 3: FutureMap

- VA-stable backing for `token_ids_buf` + lazy spec buffers.
- Same `grow()` API.

### Phase 4: Budgeter wiring

- In `BudgetAgent._maybe_fire`, post-fire: compute new admission cap,
  call `pool.grow()` on each registered pool.
- Test with D8 — expect throughput Δ ≥ +10% over off baseline.

### Phase 5: Design.md updates

- Add §350-365 (admission cap coupling) per `audit_design_intent.md`
  suggested wording.
- Update §D8 conjecture per same.

### Phase 6: D8 re-validation + sweep

- Run D8 (saturated single-pool), assert PASS.
- Document outcomes in `dev/interlayer/verify/D8/README.md`.

## 7. Test coverage

- **Unit (Phase 1)**: D1-style smoke for `ReqTokenVAArena` grow path.
  Pure-Python, runs under D1's framework. ~5 sub-tests.
- **Unit (Phase 4)**: Budgeter post-fire grow callback. Mock pool +
  mock fire result, assert `pool.grow()` called with correct value.
- **Integration (Phase 6)**: D8 PASS — throughput Δ ≥ +10%.
- **Regression**: D7 still PASSes (no behavior change at small caps).
  D8b still PASSes (idle workload, no fires, no resize).

## 8. Out of scope / future work

- DecodeReqToTokenPool (disagg-only).
- MetadataBuffers, HiSparse, RoutedExperts, NgramEmbedding — gated by
  features we don't exercise. To be addressed when those workloads
  are validated.
- Long-term: consolidate `ReqTokenVAArena` into `MultiTensorArena` as
  another sub-pool type. Reduces code duplication.

## 9. Open questions (deferred to implementation)

1. **VA reservation cost on small GPUs**: 13.7 GiB VA reservation
   for max_possible. Should be free (D5), but if `cuMemAddressReserve`
   has soft limits, may need to clamp `max_possible_size` to e.g.
   `2 × init` × some_user-configurable safety factor.
2. **chunk_arena reuse semantics**: `SharedHandlePool` was designed
   for sharing across KV+mamba arenas. Should `ReqTokenVAArena` join
   the same shared pool (consuming handles) or use a separate
   instance? Separate is simpler; same shared pool fits the design
   philosophy better.
3. **Multi-DP / multi-TP**: each DP worker has its own
   `req_to_token_pool`; each TP worker shares it. Need to verify
   `grow()` is called once per DP and not once per TP.
4. **Eviction races**: if budgeter shrinks pool while a req is being
   admitted, the slot id returned by `alloc()` might be in the about-
   to-be-unmapped range. Mitigation: shrink only after admission has
   moved on (single-thread scheduler makes this trivial; just call
   shrink AFTER admission decisions for the current iteration).

These are tracked in `progress.md` for resolution during
implementation.
