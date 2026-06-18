# Implementation plan — phases + acceptance criteria

6 phases, each with concrete deliverables + a clear pass gate.
Designed to be checkpointable: any phase can stop and resume cleanly.

## Phase 1 — `ReqTokenVAArena` helper

**Goal:** stand up the VA-stable backing primitive for ReqToTokenPool
(and later FutureMap). Reuses existing `SharedHandlePool` and
`tensor_from_va` from the arena module — no new low-level mechanism.

**Files:**
- New: `python/sglang/srt/arena/req_token_arena.py` (~150 lines)
- New: `dev/interlayer/dyn_admission_cap/test_phase1.py` (~120 lines,
  ~6 sub-tests)

**API:**
```python
class ReqTokenVAArena:
    def __init__(self, n_rows_max: int, row_bytes: int, device_id: int,
                 *, shared_pool: Optional[SharedHandlePool] = None): ...
    @property
    def mapped_rows(self) -> int: ...
    def set_mapped_rows(self, n: int) -> None: ...
    def as_tensor(self, dtype: torch.dtype,
                  shape: Tuple[int, ...]) -> torch.Tensor: ...
    def data_ptr(self) -> int: ...
    def cleanup(self) -> None: ...
```

**Acceptance (test_phase1.py):**
- 1: construct with n_rows_max=8, row_bytes=2 MiB → tensor data_ptr ≠ 0,
  initial `mapped_rows == 0`
- 2: `set_mapped_rows(2)` → writes to rows [0:2] succeed, read-back exact
- 3: `set_mapped_rows(4)` → rows [0:2] data preserved, rows [2:4] writable
- 4: data_ptr unchanged across grow (this is the WHOLE point)
- 5: shrink: `set_mapped_rows(2)` after writing to [0:4] → rows [0:2]
  still read correct data
- 6: `cleanup()` unmaps + frees VA without faulting

Run: `.venv/bin/python dev/interlayer/dyn_admission_cap/test_phase1.py`

**Done when:** 6/6 sub-tests PASS. Commit with message
`dyn_admission_cap: Phase 1 — ReqTokenVAArena helper + 6/6 unit tests`.

---

## Phase 2 — `ReqToTokenPool` refactor

**Goal:** make the base `ReqToTokenPool` (used for non-Mamba models)
VA-stable + growable. Preserve back-compat for callers that just want
the small init-size pool.

**Files:**
- Modified: `python/sglang/srt/mem_cache/memory_pool.py`
  (lines 136-200, ReqToTokenPool class)
- Modified: `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
  (init site of the pool — needs new max_possible_size param)
- New: `dev/interlayer/dyn_admission_cap/test_phase2.py`

**API additions:**
- `ReqToTokenPool.__init__` gains `max_possible_size` param (default =
  `size`, i.e. no growth ⇒ back-compat).
- `ReqToTokenPool.grow(new_size: int) -> None`
- `ReqToTokenPool.shrink(new_size: int) -> None`
- `ReqToTokenPool.max_size: int` property
- Internal: `_va_arena` when `max_possible_size > size`, else falls
  back to current `torch.zeros` allocation (zero diff for callers
  that don't pass `max_possible_size`).

**Gate (back-compat first):**
- When `max_possible_size == size` (default), behavior is byte-identical
  to today's pool. No arena involved.
- New behavior only triggers when caller explicitly opts in by
  passing `max_possible_size > size`.

**Acceptance (test_phase2.py):**
- 1: back-compat — `ReqToTokenPool(size=16, ..., max_possible_size=16)`
  matches current behavior exactly (data_ptr stable across alloc/free,
  free_slots semantics unchanged, .write() works)
- 2: with `max_possible_size=64, size=16`: `grow(32)` extends free_slots,
  rows [16:32] writable, data_ptr unchanged
- 3: post-grow `alloc()` returns ids in [16:32]
- 4: `free()` of a grown slot returns it to free_slots
- 5: `shrink(8)` rejects when slot 10 is held (assert raises)
- 6: `shrink(8)` succeeds when slots [8:16] are all free; rows [8:16]
  unmapped (subsequent grow re-maps)

**Done when:** 6/6 PASS + existing D1, D4, D7, D8b tests still PASS
(no regression). Commit: `dyn_admission_cap: Phase 2 — ReqToTokenPool
grow/shrink (back-compat default)`.

---

## Phase 3 — `HybridReqToTokenPool` (Mamba models)

**Goal:** extend Phase 2 to `HybridReqToTokenPool` and any associated
`req_index_to_mamba_index_mapping` arrays. Grow in lockstep with the
base pool.

**Files:**
- Modified: `python/sglang/srt/mem_cache/memory_pool.py` (~line 844+,
  `HybridReqToTokenPool` and friends)
- New: `dev/interlayer/dyn_admission_cap/test_phase3.py`

**Acceptance:**
- 1: back-compat default (`max_possible_size == size`) — current
  behavior preserved
- 2: `grow(N)` extends both `req_to_token` AND
  `req_index_to_mamba_index_mapping`
- 3: post-grow Hybrid `alloc()` works for new slot id

**Done when:** 3/3 PASS + existing tests PASS. Commit:
`dyn_admission_cap: Phase 3 — HybridReqToTokenPool lockstep grow`.

---

## Phase 4 — `FutureMap` (overlap mode)

**Goal:** VA-stable backing for `FutureMap.token_ids_buf` and lazy
spec buffers. Grow when admission cap grows.

**Files:**
- Modified: `python/sglang/srt/managers/overlap_utils.py` (FutureMap
  class)
- New: `dev/interlayer/dyn_admission_cap/test_phase4.py`

**Acceptance:**
- 1: back-compat default — behavior preserved
- 2: grow extends token_ids_buf without changing data_ptr
- 3: circular-buffer wrap math correct after grow (newly-grown range
  isn't clobbered by stale wrap)
- 4: spec buffers lazy-allocate at correct size post-grow

**Done when:** 4/4 PASS. Commit:
`dyn_admission_cap: Phase 4 — FutureMap dynamic grow`.

---

## Phase 5 — Budgeter wiring (grow trigger)

**Goal:** Budgeter's post-fire callback computes new admission cap
from `mamba_pool.size` and triggers grow on registered pools. This is
the gluing piece.

**Files:**
- Modified: `python/sglang/srt/budgeter/agent.py` (`_maybe_fire`
  post-fire path)
- Modified: `python/sglang/srt/managers/scheduler.py` (register pools
  with budgeter; update `self.max_running_requests` on grow callback)
- New: `dev/interlayer/dyn_admission_cap/test_phase5.py` (unit test
  with mock pool + mock fire result)

**Mechanism:**
- On fire completion (k2m grew mamba), agent reads
  `mamba_pool.size`, computes
  `new_cap = min(user_ceiling, mamba_pool.size // ratio)`.
- Calls `req_to_token_pool.grow(new_cap)`.
- For Hybrid pools, that cascades to mamba index mapping.
- For FutureMap: `scheduler.future_map.grow(new_cap)` (if overlap on).
- Updates `scheduler.max_running_requests = new_cap` (the scalar field
  any code path reads).
- Shrink path: post-fire if direction is m2k, mamba shrunk; compute
  new (smaller) cap and call `shrink()`. shrink rejects held slots.

**Acceptance:**
- 1: mock fire result with `direction=k2m`, `granted_pages=N`,
  mamba_pool.size grew from 100 → 100+N → callback invokes
  `pool.grow(floor((100+N)/ratio))`
- 2: user ceiling respected — never grow above
  `--max-running-requests`
- 3: shrink path with held slots: callback observes held slots, calls
  shrink with the safe new cap (≥ held slots), not the naive one
- 4: integration: in BudgetAgent live tick, exercise post-fire with
  real (stub-scheduler) plumbing

**Done when:** 4/4 PASS. Commit:
`dyn_admission_cap: Phase 5 — Budgeter post-fire admission cap update`.

---

## Phase 6 — design.md update + D8 re-validation

**Goal:** document the now-implemented mechanism in design.md, then
re-run D8 to confirm throughput +10%.

**Files:**
- Modified: `dev/interlayer/design.md` (per `audit_design_intent.md`
  suggested wording — new §350-365 + update D8 conjecture)
- Modified: `dev/interlayer/verify/D8/D8_saturated.sh` — `--max-mamba-
  cache-size 100` stays (now actually limits init, not max — fires can
  grow past it)
- New: `dev/interlayer/verify/D8/D8_v2.sh` if config changes needed
- Updated: `dev/interlayer/verify/D8/README.md` with result

**Acceptance:**
- D8: throughput Δ ≥ +10% PASS
- D7: still PASS (28+ fires, all 4 checks)
- D8b: still PASS (0 fires on idle, no regression)
- D1, D4, D5, D5b, D5c: still PASS (mechanism unchanged)

**Done when:** D8 PASS + no regression. Commit:
`dyn_admission_cap: Phase 6 — D8 PASS, design.md updated`.

---

## Stop-condition policy

Each phase has its OWN test gate. If a phase fails:
- Phase 1-4: fail loudly, fix in place, retry. Don't proceed to next
  phase.
- Phase 5: integration is risky; if test fails, capture diagnostic
  data and add hypothesis to `progress.md` before fixing.
- Phase 6: D8 fail with mechanism PASS (fires happen but throughput
  doesn't grow) = workload tuning issue, not arch issue. Don't roll
  back the arch change.

## Subagent audits after Phase 5

Per user's "派subagent来audit" directive, dispatch 2 additional audits
once Phase 5 lands (before Phase 6 D8 run):

- **Code review audit**: review the actual diffs across phases 1-5 for
  concurrency / lifetime / cleanup bugs.
- **Test coverage audit**: review test_phase1-5 to find admission-time
  edge cases the tests miss (e.g. grow during alloc, shrink with
  pending fire, multi-DP).

These audits gate the Phase 6 D8 run.

---

## Phase 7 — MambaPool dynamic resize (added after D8 v2 FAILED)

**Why**: D8 v2 ran with all Phases 1-5 + the P0 CUDA-graph test
passing, but still throughput Δ = -1.73%. Diagnosis (see
`progress.md` 2026-05-26 later still): MambaPool itself is
init-bounded at THREE places — actuator's `max_slots = pool.size`,
`MambaPool.set_capacity_slots` clamps at `self.size`, and
`MambaPool.conv_state` is `torch.zeros((num_layers, size+1, ...))`
sized at init. Fires move bytes into mamba arena VA but MambaPool
can't address the new slots.

The audit (`audit_other_arrays.md`) had flagged MambaPool as Tier 1
critical; we misread it as "already handled by the arena" because
the temporal_state IS arena-backed. conv_state and the allocator's
self.size are NOT.

**Scope** (mirrors Phases 1-3 for MambaPool):

### Phase 7.1 — MambaPool.conv_state dynamic backing

- conv_state per-slot size on Qwen3.5-9B: ~48 KB × 24 layers ≈ 1.15 MB
- At max_size = 4× init = 400 slots → 460 MB pre-allocation
- Decision: pre-allocate at max_size (simpler than full VA-stable
  arena since the tensors are bounded and reasonably small at our
  test factor). Matches the Phase 3 approach for
  `HybridReqToTokenPool` tiny mapping tensors.

### Phase 7.2 — MambaPool.set_capacity_slots growth

- Current (line 780): `n_slots = max(1, min(n_slots, self.size))`
- New: `n_slots = max(1, min(n_slots, self.max_size))` and on growth,
  expose new free slot ids `[self.size, n_slots)` via free_slots
  extension.
- Mirrors `ReqToTokenPool.grow` from Phase 2.

### Phase 7.3 — MambaArenaActuator.max_slots from arena

- Current (line 33): `self.max_slots = pool.size` (init clamp)
- New: `self.max_slots = arena.max_chunks_per_pool × arena.tokens_per_chunk`
- Tiny change (1 line).

### Phase 7.4 — Wiring

- `model_runner_kv_cache_mixin.py` HybridReqToTokenPool init: also pass
  `mamba_max_size = max_size` to its `mamba_size` derivation, so the
  underlying MambaPool gets max_size > size.
- `_maybe_update_admission_cap` in agent.py: ensure mamba growth is
  driven by the same trigger that grows ReqToTokenPool.

### Phase 7.5 — D8 re-run

- Expected: throughput Δ ≥ +10% PASS.
- Also check: `[admission-cap] grew pool.size N → M` log AND
  `MambaPool.set_capacity_slots N → M` log present.

### Acceptance

- Unit test for MambaPool grow/shrink (mirror test_phase2)
- D7 + D8b still PASS (no regression)
- D8 PASS

### What we learned (record for future audits)

When auditing per-req arrays, **trace ALL levels of clamping**, not
just the obvious tensor allocation. MambaPool had three independent
caps that all needed lifting. The audit_other_arrays.md report
correctly listed MambaPool as Tier 1; the misread was thinking
arena-backed implies grow-capable. Verifying actual call paths
end-to-end (Actuator → Pool.set_capacity_slots → tensor index) is
necessary to confirm.
