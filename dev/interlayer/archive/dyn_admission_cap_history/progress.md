# Progress log

## 2026-05-26

- D8 v1 ran, failed: throughput Δ = -1.70% (target +10%).
  Diagnosed: sglang's `ReqToTokenPool.free_slots` array is init-sized
  for `max_running_requests=33`; fires grow mamba but admission can't
  exceed the array size.
- Considered 3 options (1a/1b/1c); user chose 1b (dynamic resize).
- Created this folder. Wrote `README.md`, `discussion.md`, this file.
- Dispatching 4 parallel audit subagents:
  - audit_consumers.md: who reads `ReqToTokenPool.size` and indexes
    into the array
  - audit_other_arrays.md: what OTHER init-sized arrays are bound to
    `max_running_requests`
  - audit_cuda_graphs.md: does CUDA graph capture embed array pointers
    that go stale on resize
  - audit_design_intent.md: does sglang's design.md or any doc
    already discuss dynamic admission cap

## Open questions (resolved via audits)

1. **Safe to resize tensor in use?** — No, BUT we don't need to:
   chosen approach is VA-stable wrapping (Option B from
   audit_cuda_graphs). `data_ptr()` never changes; `cuMemMap` adds
   physical pages within the same VA. Resize itself isn't needed.
2. **CUDA graph capture risk?** — `req_to_token` data_ptr IS baked
   into Triton kernels (`audit_cuda_graphs.md` §2). VA-stable wrapping
   avoids the issue; no re-capture needed.
3. **Other per-req arrays?** — `audit_other_arrays.md` Tier 1: also
   `HybridReqToTokenPool.req_index_to_mamba_index_mapping`,
   `FutureMap.token_ids_buf` + lazy spec buffers. Tier 2
   (disagg/HiSparse/MoE/ngram) skipped for D8.
4. **Hot-path reads?** — Backends cache tensor refs during forward.
   Resize must happen between batches. Budgeter tick already runs in
   that window. VA-stable wrapping makes this a non-issue anyway.
5. **Existing mechanism to reuse?** — Yes:
   `chunk_arena.SharedHandlePool` + `from_blob_ext.tensor_from_va`
   already used by MultiTensorArena (D1/D4 PASS). Build
   `ReqTokenVAArena` as a thin wrapper.

## 2026-05-26 (D8 v6: mechanism PASS, throughput regresses +4% TPOT — TODO)

After xpool_actuator multi-subpool fix (commit `818ecb06a6`), D8 v6:
- 44 fires, 0 aborted, no crash ✓
- admission cap grew 33→38+ over the run (log: `[admission-cap] grew
  pool.size 33 -> 34 ... 37 -> 38`) ✓
- MambaPool.set_capacity_slots cascaded correctly:
  `100 -> 101, 101 -> 103, 103 -> 105 ...` ✓
- Mechanism end-to-end CONFIRMED CORRECT.

**BUT** throughput regressed:

| | off | inter |
|---|---|---|
| completed | 5760 | 5760 |
| throughput | 8.00 req/s | 7.68 req/s (**−3.99%**) |
| TTFT | 266s | 281s (+5.6%) |
| TPOT | 7.98ms | 8.30ms (**+4.0%**) |

TPOT +4% is the kicker — that's per-token decode cost, applied to
ALL in-batch reqs. Suggests a systemic per-iteration cost added by
Phase 7's changes, not just fire overhead (44 fires × ~50ms = 2.2s ≈
0.3% of wall — too small to explain 4% TPOT).

**Suspect: MambaPool conv_state pre-allocated 4× larger** under
Phase 7 dynamic mode (with SGLANG_ADMISSION_MAX_FACTOR=4):
- Old: (24 layers, 101 slots, ...) = ~115 MB
- New: (24 layers, 397 slots, ...) = ~455 MB
- 340 MB of conv_state more → potential TLB pressure on
  attention/mamba kernel reads.

Other candidates:
- ReqToTokenPool's from_blob VA-backed tensor (Phase 1-2) — not
  measured separately vs torch.zeros baseline; possible overhead
- Fire CUDA resource contention (cuMemMap + memset) — small per-fire
  but maybe higher than estimated
- Budgeter tick + snapshot overhead (cheap)

**Bisect proposal**:
- D8 v6.5 with SGLANG_ADMISSION_MAX_FACTOR=2 (mamba pre-alloc only 2x)
  — if TPOT regression shrinks, MambaPool pre-alloc size IS the cost
- D8 v6.6 with factor=1 (back-compat) — should match off baseline
- D8 v7 open-loop (RPS=inf) — confirms whether admission-cap growth
  delivers throughput when bench is admission-bound

**Tracked as TODO** — to be solved AFTER user-requested regression
sweep across prior D-tests confirms no other tests are affected.

## 2026-05-26 (D8 v4 — Phase 7 + live_size fix, still crashes)

After Phase 7 (MambaPool dynamic resize) + Budgeter `live_size` fix:

D8 v4 log shows the dyn-admission chain is now correct:
- `[admission-cap] init: mamba_live=100 max_running=33 ratio=3 user_ceiling=256 pool_max=132 mamba_phys_max=396`
  (ratio = 3, was 12 in v3 — fix worked)
- `MambaPool.set_capacity_slots: 100 -> 103` (actuator grew mamba)
- `[admission-cap] grew pool.size 33 -> 34 (mamba 100 -> 103, ratio=3)`
  (Budgeter correctly cascaded the grow)
- Then: `CUDA error: an illegal memory access was encountered` on the
  next req admission's `req_index_to_mamba_index_mapping[select_index]`
  = mamba_index_tensor write.

The crash is async — could be anywhere in the prior operations.
Possible root causes:
- ReqToTokenPool grow path's tensor access at row 33 after partial
  re-map (chunk granularity rounding)
- MambaPool conv_state at slot 101+ where arena chunk mapping vs
  cap_slots is off-by-one
- FutureMap circular buffer wrap with grow on the same tick
- Captured CUDA graph holding pre-Phase-7 pointer to a tensor
  reallocated by Phase 7 (mamba conv_state newly sized at max_size)

Needs CUDA_LAUNCH_BLOCKING=1 rerun to isolate. Also Phase 7 lacks
the same rigor as Phases 1-5: no unit tests for MambaPool grow + no
P0 CUDA-graph test for the conv_state pre-allocation. Should add
those before another D8 attempt.

Status: dyn-admission orchestration logic CONFIRMED correct (logs
show the chain firing as designed). Bug is now in the lower-level
tensor access path under live workload. Parking pending CUDA_LAUNCH_
BLOCKING debug session + Phase 7 unit tests.

## 2026-05-26 (later still — D8 v2 FAILED + finding)

After Phases 1-5 + Phase 5b CUDA-graph validation (25 + 4 unit tests
all PASS, D1/D4 regression PASS), ran D8 v2 with
`SGLANG_ADMISSION_MAX_FACTOR=4`. Result: **FAIL** — throughput Δ =
-1.73% (target +10%), identical to D8 v1.

**Diagnosis**: dyn_admission_cap fix is correct for ReqToTokenPool
but a deeper layer also caps:

1. **MambaArenaActuator.max_slots** (`mamba_actuator.py:33`) is set
   to `pool.size` at __init__ (=100). Line 59 clamps every
   `set_capacity_tokens(n)` to `min(n, self.max_slots)`. So even
   when the actuator is asked to bump capacity to 164, it returns 100.

2. **MambaPool.set_capacity_slots** (`memory_pool.py:780`) ALSO clamps:
   `n_slots = max(1, min(n_slots, self.size))`. Even if (1) is fixed,
   MambaPool clamps at its own init size.

3. **MambaPool.conv_state** is `torch.zeros((num_layers, size+1, ...))`
   sized at init self.size. Cannot index beyond `self.size` (OOB).
   Temporal_state IS arena-backed (good — already at arena max
   shape under shared_arena mode), but conv_state isn't.

So the fires DO move bytes into mamba arena VA, but MambaPool can't
USE them — its allocator and tensors are init-bounded. Cross-pool
fires under D8 are pure overhead (-1.73%).

**Audits got this right** (audit_other_arrays.md listed MambaPool as
Tier 1 critical) but I missed it during scoping. The audit assumed
MambaPool's existing arena infrastructure handled growth; in fact
only temporal_state is grown-capable, not conv_state nor the
allocator's `self.size`.

**Phase 7+ scope** (new, not yet planned): extend dyn_admission_cap
to MambaPool itself —
  - conv_state: VA-stable backing (mirror ReqToTokenPool work)
  - set_capacity_slots: allow growth past init self.size
  - Actuator max_slots: derive from arena max not pool init

This is analogous in size to Phases 1-3 (the ReqToTokenPool work).

## 2026-05-26 (later)

Audits complete. Synthesized into `design.md`:
- VA-stable backing for `ReqToTokenPool.req_to_token` (Tier 1)
- Same for `HybridReqToTokenPool.req_index_to_mamba_index_mapping`
- Same for `FutureMap` (overlap mode default)
- DecodeReqToTokenPool / MetadataBuffers / HiSparse / MoE deferred
- Budgeter `_maybe_fire` post-fire callback as resize trigger
- 6 implementation phases (helper → ReqToTokenPool → Hybrid →
  FutureMap → Budgeter wiring → design.md → D8 validation)

Open implementation-time questions in `design.md` §9 (VA reservation
on small GPUs, SharedHandlePool reuse, multi-DP, eviction races).

Ready for Phase 1 implementation pending user review of design.

## 2026-05-27

D8 ideal-arch validation: Phase 1-7 all unit tests PASS (44/44), D7 PASS,
D8b idle PASS. D8 saturated still shows -4% throughput / +4.4% TPOT in
closed-loop bench despite mechanism working correctly (admission grew
33→48, 44 fires, no aborts).

User: accept ~4% as real regression; deep-dive root cause.

**Bisect** (see `d8_regression_bisect.md`) at 90 s workload, 4 phases:

| phase | TPOT | ΔTPOT |
|---|---|---|
| off | 7.849 ms | +0.00% |
| arena (`SGLANG_ARENA_SHARED=1`) | 7.903 ms | +0.69% |
| bg_killfire (`SGLANG_XPOOL_DISABLE_FIRES=1`) | 7.918 ms | +0.88% |
| bg_fire (21 fires) | 8.139 ms | +3.70% |
| bg_nofire (also 21 fires) | 8.209 ms | +4.59% |

**Root cause: cuMemMap/Unmap operations inside fires.**
- arena alone: +0.69%
- + budgeter (no fires): +0.19% (~noise)
- + 21 fires: +3-4%

**~10× impact amplification**: 21 × 60 ms = 1.2 s direct, ~14 s lost
throughput. Each fire stalls/slows decode for ~600 ms beyond its own
wall — strongly suggests GPU TLB shootdown / CUDA driver lock / implicit
device-wide sync, not the cuMem op time itself.

Added debug kill switch `SGLANG_XPOOL_DISABLE_FIRES=1` to
`xpool_planner.py:566` for bisection. Should remain (cheap, lets us A/B
fire cost in future workloads).

Next: identify exact GPU mechanism (TLB / driver / sync) via per-iter
timing or nsys profile around a fire.

## 2026-05-27 (later) — root cause + fix #1

Per-iter timing probe → per-tick probe → per-`_maybe_fire`-step probe.
Localized 200ms-per-fire cost to `XPoolFirePlanner.build()` →
`SchedulerOwnerProvider.build_kv_owner_map()` Python loop:
- `allocator.free_pages.cpu().tolist()` for ~2.4M slot ids
- `{int(x) for x in ...}` → set construction
- `for p in range(2398): if all(s in free_set for s in page_slots)` —
  2398 × 1024 = **2.5M Python set lookups per fire**

NOT cuMemMap / TLB / driver lock. The ~10× amplification I initially
suspected was a misread (worker's cuMemMap runs in parallel with
scheduler and is invisible to TPOT).

**Fix** (`scheduler_owner_provider.py`): vectorize on GPU —
bool-mask + reshape + `all(dim=1)` + `nonzero().cpu()`.

Unit tests in `test_owner_map_vectorized.py` (6/6 PASS):
- correctness vs ref Python on 5 small cases
- KV-scale: 180ms → 0.11ms = **1669× speedup**

D8 re-validation (single rep): inter +3.70% → +2.59% TPOT vs off.
Apparent ~1.1 pp recovered. tickprobe shows `build` cost gone; the 1
remaining slow tick (36ms) is `cap_barrier`'s `.item()` GPU sync.

**N=5 reps for trustworthy mean ± std** (user request):
- off:   **7.841 ± 0.008 ms** (per-rep: 7.831, 7.838, 7.837, 7.844, 7.853)
- inter: **8.077 ± 0.020 ms** (per-rep: 8.056, 8.084, 8.105, 8.060, 8.078)
- Δ TPOT = **+3.01% ± 0.30%** (24σ — clearly real, not noise)

**Honest correction**: single-rep claim of "1.1 pp recovered" was
inflated by single-rep variance. True recovery is **~0.7 pp**
(3.70% → 3.01%). Still real, but smaller.

Residual ~3% to investigate (Task #129):
- ~0.9% from per-iter Python+arena overhead (bg_killfire single-rep)
- ~2 pp from fires: 21 × ~36ms cap_barrier (0.2%) + ~6× amplified
  cuMemMap interference (~1.8%); ~300ms cost per fire vs 50ms wall.

### 2026-05-27 (Task #129 closed)

Full per-iter (all bs, all iters) profile + correlation with fires:
- iter time vs seconds-since-most-recent-fire: **flat** (no correlation)
- iter time BEFORE first fire: 6.89 ms (baseline)
- iter time AFTER first fire: 7.32 ms (permanent +170μs)

→ Cost is NOT per-fire spike. It's a **permanent state-change after
admission_cap grew beyond init**.

Direct GPU profile (`profile_arena_decode.py`): cuMem operations on
arena-backed memory cause **0% measurable slowdown** to subsequent
decode kernels reading that arena (Δ < 0.4% across 3, 48, 192 chunks).
GPU TLB / driver-lock hypothesis is **refuted**.

True residual mechanism is scheduler-side (Python) overhead engaged by
admission_cap growth. Likely scheduler iterates over larger
max_running_requests in some per-iter loop, or ReqToTokenPool
indexing pays more on bigger buffer. Untested in detail; accepted as
architectural cost of dynamic admission for now.

**Bottom line for D8:**
- Pre-fix bench: +3.70% TPOT (build_kv_owner_map Python loop dominated)
- Post-fix N=5: +3.01% ± 0.30% TPOT
- ~0.7 pp recovered by fix
- ~3 pp residual = architectural cost (not GPU, not Python loop)

### 2026-05-27 (later) — Cap-grow vs fires bisect (Task #130)

Added `SGLANG_DISABLE_ADMISSION_GROW=1` kill switch to
`agent._maybe_update_admission_cap`. N=3 sweep:

| phase | TPOT mean ± std | Δ vs off |
|---|---|---|
| off | 7.835 ± 0.007 | baseline |
| grow_off (fires, no admission resize) | 8.071 ± 0.018 | **+3.01%** |
| grow_on (fires + admission resize) | 8.077 ± 0.003 | +3.10% |

**Cap-grow contributes +0.09% — essentially zero.**

→ The **entire dynamic admission resize chain (Phase 1-5 + 7) is free**.
The ~3% cost was already there from "fires happening" before we did
any of this work.

The +2.1% from fires (above arena+budgeter baseline) is NOT GPU-side
(proven by direct profile) and NOT from admission resize. Most likely
in Python overhead of the fire path (cap_barrier sync, snapshot
reads, actuator state changes).

> Note: #129's earlier claim "permanent state-change after
> admission_cap grew beyond init" was REFUTED by #130's bisect — cap-grow
> contributes +0.09%. The "permanent state" must be in fire side effects
> not on the admission cap (e.g. mamba_pool's free_slots / capped_slots
> tensor state after `set_capacity_slots`).

### 2026-05-27 (later still) — Worker no-op bisect (Task #131)

To further isolate within "fires happening", added kill switch
`SGLANG_XPOOL_WORKER_NOOP=1` to `_fire_worker_loop`. When set, the
worker unmarks `cap_t` (rolls back cap_barrier) and emits a synthetic
aborted completion **without** any cuMemUnmap/cuMemMap, **without**
`torch.cuda.synchronize()`, **without** `set_capacity_slots`.

Bisect plan (N=3):
- off
- worker_active: full fire path (= current D8 inter)
- worker_noop: scheduler cap_barrier runs (.item() sync), worker is no-op

**Final result:**
- off: 7.843 ± 0.004 ms (n=3)
- worker_active: 8.081 ± 0.002 ms (n=2, +3.04%)
- worker_noop: 8.067 ± 0.012 ms (n=3, +2.86%)

→ Worker contributes **+0.18%** (essentially zero). **Entire ~2.9%
cost is on scheduler thread in `cap_barrier` path**, NOT in cuMem
ops, NOT in worker sync, NOT in pool side effects.

The cost lives in some combination of:
- `mark_pages_capped` allocator mutation
- `torch.isin().sum().item()` verify GPU sync
- `unmark_pages_capped` rollback (runs in both modes)

Direct cap_barrier wall is only ~36 ms × 21 fires = 0.76 s = 0.2%.
Observed cost is 2.86% = 10 s. So the ~10× gap suggests cap_barrier
triggers some persistent state effect on subsequent decode iters.

### 2026-05-27 (later) — cap_barrier drill (Tasks #132)

Added two kill switches: `SGLANG_XPOOL_SKIP_VERIFY=1` (skip
`.item()` GPU sync) and `SGLANG_XPOOL_SKIP_MARK=1` (skip
`mark_pages_capped`).

**Verify bisect (N=3):**
- off: 7.861 ± 0.022 ms
- noop_verify_on: 8.078 ± 0.019 ms (+2.76%)
- noop_verify_off: 8.108 ± 0.028 ms (+3.14% — WORSE)

→ verify (`.item()` sync) is NOT the cost; skipping it makes things
slightly worse (~+0.4%).

**Mark bisect (N=3):**
- off: 7.860 ± 0.019 ms
- bg_killfire: 7.917 ± 0.033 ms (+0.72%)
- nomark (cap_barrier no-op): 7.904 ± 0.024 ms (+0.55%)

→ **Skipping `mark_pages_capped` recovers the entire ~2.86% back to
baseline.** Same as bg_killfire (which never calls cap_barrier).

🎯 **ROOT CAUSE: `allocator.mark_pages_capped` reallocates the 2.4M-element
`free_pages` tensor on every fire** (`allocator.py:151-159`):

```python
mask = torch.isin(self.free_pages, target)      # 2.4M comparisons
held = self.free_pages[mask]                    # ~4k held
self.free_pages = self.free_pages[~mask]        # NEW 2.4M tensor (~19 MB)
self._capped_pages = torch.cat([existing, held])
```

Per fire ~57 MB tensor reallocation. 21 fires → ~1.2 GB tensor churn
through PyTorch's caching allocator. Subsequent operations pay diffuse
per-iter cost (memory layout / fragmentation / cache locality).

### 2026-05-27 (later still) — Fix #134 IMPLEMENTED & VALIDATED

`allocator.py` mark/unmark rewritten:
- mark only updates `_capped_pages` (small, fast); doesn't touch
  `free_pages` (no 19 MB realloc)
- unmark only removes from `_capped_pages`
- `alloc()` filters against `_capped_pages`: fast path when none,
  cheap-case when first need_size has no capped, slow-case scans
- `available_size()` and `free_page_mask()` subtract `_capped_pages`

Unit tests `test_mark_no_realloc.py` (4/4 PASS):
- alloc correctly skips capped slots (matches old semantics)
- mark+unmark peak GPU alloc: 24 KB (was 138 MB — **5750× reduction**)
- mark+unmark idempotent
- KV-scale mark wall: 0.006 ms (was 1.14 ms — **190× faster**)

D8 saturated N=3 re-validation:

| phase | TPOT (mean ± std) | Δ vs off |
|---|---|---|
| off | 7.927 ± 0.089 ms | baseline |
| inter (post #128 + #134) | 7.906 ± 0.015 ms | -0.26% (within noise) |

Initial N=3 looked great. But N=3 with high off variance (0.089 ms)
is noisy. User asked for N=5 — done next.

### 2026-05-27 (final) — D8 N=5 re-confirm (Task #135)

| phase | TPOT (N=5 mean ± std) | per-rep |
|---|---|---|
| off | 7.911 ± 0.048 ms | 7.907, 7.955, 7.957, 7.841, 7.895 |
| inter | **7.959 ± 0.061 ms** | 8.034, 8.018, 7.915, 7.922, 7.907 |

Δ TPOT: **+0.61% (1.4σ — borderline significant)**

**Correction to optimistic earlier claim:** the N=3 -0.26% was
session-specific noise; cross-session N=5 shows ~+0.6%. The N=5
number is the trustworthy one.

**Final timeline:**
- Pre-fix (original D8 inter): +3.70% TPOT regression
- Post-fix #128 (vectorized owner-map, N=5): +3.01% ± 0.30% (24σ)
- **Post-fix #128 + #134 (mark no-realloc, N=5): +0.61% (1.4σ)**

→ Recovered ~3.1 pp out of 3.7 pp (~84% of regression). Residual
+0.6% is within day-to-day system noise (off std was 0.048 ms ≈
0.6% itself). Accepted as fully resolved.

All existing Phase 1-7 + balanced atomic + owner-map + no-realloc
tests continue to pass (48/48). Fix is safe and correct.
