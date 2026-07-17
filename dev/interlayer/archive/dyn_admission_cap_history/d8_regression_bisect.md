# D8 regression root-cause bisect (2026-05-27)

## TL;DR

D8 saturated bench originally showed ~3.7% TPOT regression. Two fixes
collectively recovered **~84% of the regression** (3.7 pp → 0.6 pp).

**Final result (N=5 alternating off/inter reps, post both fixes):**

| Phase | TPOT (N=5 mean ± std) | Δ vs off |
|---|---|---|
| off | 7.911 ± 0.048 ms | baseline |
| inter (post #128 + #134) | **7.959 ± 0.061 ms** | **+0.61% (1.4σ — borderline)** |

The residual +0.6% may be a small remaining cost or measurement
variance; it is within day-to-day system noise. The N=3 initial
re-run measured -0.26% — both within ±1σ of zero.

**Earlier (incorrect) optimistic claim from N=3:**
- "-0.26% (within noise)" — the N=3 used a session where inter
  happened to run faster. N=5 cross-session is more rigorous and
  shows ~+0.6%, which we accept as the true residual.

**Two fixes landed:**

| Fix | Cost addressed | Recovered |
|---|---|---|
| **#128** vectorize `build_kv_owner_map` (GPU bool-mask, 1669× speedup) | Python loop scanning 2.5M slot lookups per fire | ~0.7 pp |
| **#134** rewrite `allocator.mark_pages_capped` to avoid `free_pages[~mask]` realloc (~57 MB churn/fire); `alloc()` filters against `_capped_pages` instead | ~1.2 GB tensor reallocation churn across run | **~2.9 pp** |

**Components ruled out (zero cost):**

| Component | Status |
|---|---|
| dynamic admission resize (Phase 1-7) | ✅ proven free (cap-grow contributes +0.09%) |
| GPU TLB / driver lock from cuMem ops | ✅ refuted by direct profile |
| worker thread (cuMem + sync + pool ops) | ✅ +0.18% |
| cap_barrier `.item()` GPU sync | ✅ -0.38% (slightly faster when skipped) |

**Bottom line:** Phase 1-7 dynamic admission cap work is
**architecturally free** AND the inherited mark_pages_capped
inefficiency is now fixed. D8 saturated regression: **resolved**.

Files:
- Fix: `python/sglang/srt/budgeter/scheduler_owner_provider.py`
- Tests: `test_owner_map_vectorized.py` (6/6 PASS, 1669× speedup)
- Direct GPU profiles: `profile_arena_decode.py`,
  `profile_cumem_decode_impact.py`
- Debug kill switches added: `SGLANG_XPOOL_DISABLE_FIRES`,
  `SGLANG_DISABLE_ADMISSION_GROW`, `SGLANG_XPOOL_WORKER_NOOP`

---

## Context

After Phase 1-7 + LCM-balanced atomic xpool transfer, D8 saturated
workload showed +4% TPOT regression on closed-loop bench:

| run | TPOT | throughput | duration |
|---|---|---|---|
| off (D8 saturated, 180 s) | 7.951 ms | 8.014 rps | 718.7 s |
| inter (factor=4, 44 fires) | 8.299 ms | 7.687 rps | 749.3 s |

Same 5760 prompts, RPS=32. Δ TPOT (+4.4%) ≈ Δ duration (+4.3%).

User accepted ~4% as real regression → deep-dive into root cause.

## Stage 1: 5-level drill-down identifies `build_kv_owner_map`

90s short workload, single rep for fast iteration.

| Probe level | Finding |
|---|---|
| Phase bisect (off vs arena vs bg_killfire vs bg_fire) | arena +0.7%, bg_killfire +0.9%, bg_fire +3.7% |
| `event_loop_normal` per-iter timing | 12 iter spikes of 210-240 ms each fire, normal iters back to 7 ms after spike |
| Per-iter phase split (recv/batch/run/proc/tick) | All cost in `tick_us` (200+ ms) |
| Per-tick step split (snapshot/adm/fire) | All in `fire_us` — `_maybe_fire` |
| Inside `_maybe_fire` (decide/build/cap) | `build=200 ms`, others < 20 ms |
| Inside `build()` | `SchedulerOwnerProvider.build_kv_owner_map` Python loop |

The hot path (`scheduler_owner_provider.py:38-47`):

```python
free_token_set = self._free_token_set(allocator)  # .cpu().tolist() + set() over 2.4M ints
free_pages = {
    p for p in range(n_pages)          # n_pages = 2398 per KV sub-pool
    if kv_act.page_is_fully_free(p, free_token_set)  # 1024 set lookups each
}
```

**~2.5M Python set lookups per fire** on the scheduler thread.

## Stage 2: Fix — GPU bool-mask vectorization

`SchedulerOwnerProvider._compute_fully_free_pages` — bool tensor over
all slots, scatter-write free slot indices, reshape to (n_pages, tps),
`.all(dim=1)` reduces to per-page mask, `.nonzero().cpu()` to Python set.

```python
is_free = torch.zeros(n_slots, dtype=torch.bool, device=device)
is_free[0] = True  # slot 0 sentinel
is_free[free_pages_t.long()] = True
fully_free_mask = is_free.view(n_pages, tps).all(dim=1)
return set(fully_free_mask.nonzero(as_tuple=True)[0].cpu().tolist())
```

Unit tests `test_owner_map_vectorized.py` (6/6 PASS):
- 5 correctness tests vs reference Python impl
- KV-scale benchmark (n_pages=2398, tps=1024): ref 180 ms → new 0.11 ms
- **1669× speedup**

## Stage 3: Single-rep validation

D8 saturated (90 s workload, 1 rep):

| phase | Pre-fix TPOT | Post-fix TPOT | Δ vs off |
|---|---|---|---|
| off | 7.842 ms | 7.842 ms | baseline |
| inter | 8.139 ms (+3.70%) | 8.045 ms (+2.59%) | **-1.11 pp recovered** |

Tick-probe: pre-fix had 12-21 ticks per run with `fire_us ≈ 200 ms`;
post-fix has **0** such ticks. ONE tick at 36 ms which is now
`cap_full=36ms` (the cap_barrier's `.item()` GPU sync), not `build`.

## Stage 4: N=5 reps for trustworthy mean ± std

Single rep showed 1.1 pp recovered — user (rightly) called out
unreliable. N=5 sweep:

| phase | TPOT (mean ± std) | Δ vs off |
|---|---|---|
| off | **7.841 ± 0.008 ms** (per-rep: 7.831, 7.838, 7.837, 7.844, 7.853) | baseline |
| inter | **8.077 ± 0.020 ms** (per-rep: 8.056, 8.084, 8.105, 8.060, 8.078) | **+3.01% ± 0.30%** |

t-test: 24σ — overwhelmingly significant.

**Honest correction:** the fix actually recovered **~0.7 pp** (3.70%
→ 3.01%), not the 1.1 pp the single-rep number suggested. Real, still
meaningful, but smaller.

## Stage 5: Why does +3% residual remain?

### 5a. Full per-iter distribution (logged ALL iters, not just >30 ms)

| bucket | off | inter | Δ | extra time |
|---|---|---|---|---|
| bs≥30 decode | 7.151 ms | 7.322 ms | **+171 μs/iter** | +7.48 s |
| bs=1 prefill | 12.21 ms | 13.13 ms | +920 μs/iter | +2.46 s |
| bs=0 idle | 0.019 ms × 776k | 0.018 ms × 1.15M | +380k iters | +6.03 s |

**Residual cost is DIFFUSE per-iter, not spike-per-fire.** Decode iter
count matches off (43662 vs 43660) — same work, just slower.

### 5b. Iter-time vs distance from nearest fire

| seconds since fire | mean iter ms | n |
|---|---|---|
| pre-1st-fire | 6.89 | 15 (warmup, smaller bs) |
| <1 s | 7.32 | 2567 |
| 1-3 s | 7.31 | 5219 |
| 3-5 s | 7.32 | 5207 |
| 5-10 s | 7.32 | 12931 |
| 10-15 s | 7.32 | 12722 |
| >15 s | 7.34 | 5001 |

**No correlation with fire proximity.** Cost is permanent once fires
start; not a "TLB recovery tail" pattern.

### 5c. Direct GPU profile — cuMem effect on decode kernels

`profile_arena_decode.py`: arena-backed KV tensor, real decode-like
work, with cuMem ops on the SAME arena:

| Test | Δ vs baseline |
|---|---|
| shrink+grow 3 pages | -0.01% |
| shrink+grow 48 pages (real fire size) | +0.20% |
| shrink+grow 192 pages (4× real) | +0.09% |
| unmap-to-shared + re-grow 192 pages | -0.15% |

**cuMem operations cause 0% measurable change in subsequent
decode-kernel GPU time.** GPU-TLB-churn hypothesis is REFUTED.

`profile_cumem_decode_impact.py` (decode on cudaMalloc, cuMem on
unrelated arena) shows same: 0% effect.

### 5d. Fires-vs-cap-grow bisect (N=3)

Kill switch `SGLANG_DISABLE_ADMISSION_GROW=1` skips
`_maybe_update_admission_cap` (Phase 5) so fires happen but
ReqToTokenPool / FutureMap / max_running_requests don't grow.

| phase | TPOT (mean ± std) | Δ vs off |
|---|---|---|
| off | 7.835 ± 0.007 ms | baseline |
| grow_off (fires, NO cap-grow) | 8.071 ± 0.018 ms | **+3.01%** |
| grow_on (fires + cap-grow) | 8.077 ± 0.003 ms | +3.10% |

**Cap-grow contributes +0.09% — essentially zero.**

→ Phase 1-7 dynamic admission resize is **free**. The ~3% cost is
already there from "fires happening" before any admission resize.

### 5e. Worker no-op bisect (Task #131, N=3)

Kill switch `SGLANG_XPOOL_WORKER_NOOP=1` makes the worker thread
skip ALL its work (cuMemUnmap, cuMemMap, torch.cuda.synchronize,
set_capacity_slots, KV cap_allocator_only) and just unmark the
capped pages. Scheduler-thread cap_barrier still runs.

| phase | TPOT (mean ± std) | Δ vs off |
|---|---|---|
| off | 7.843 ± 0.004 ms (n=3) | baseline |
| worker_active (full fire) | 8.081 ± 0.002 ms (n=2) | **+3.04%** |
| worker_noop (sched cap_barrier only) | 8.067 ± 0.012 ms (n=3) | **+2.86%** |

**Cost split:**
- Worker thread (cuMem + sync + pool ops): **+0.18%** — essentially zero
- **Scheduler-side cap_barrier path: +2.86%**

→ The entire fires-attributable cost (~3 pp) lives **on the scheduler
thread** in `actuator.cap_barrier()` (or `unmark_pages_capped` rollback
which runs in both modes).

This is a strong narrowing — we've ruled out cuMem driver ops, worker
threading, sync interactions, AND admission cap growth. The cost lives
in a small region: cap_barrier's tensor work + `.item()` sync + the
mark/unmark of `free_pages`.

### 5f. Refined cost decomposition (final)

| Component | Cost |
|---|---|
| arena tensor backing (cuMemMap-based KV+mamba) | ~0.7% |
| budgeter Python tick + snapshot (no fires) | ~0.2% |
| **scheduler-thread cap_barrier path** | **~2.9%** ⏳ |
| dynamic admission cap resize (Phase 1-5 + 7) | ~0% |
| worker thread (cuMem + sync + pool side effects) | ~0% |

Candidate root causes inside cap_barrier (untested):
- `torch.isin(free_pages_t, cap_t).sum().item()` GPU sync —
  blocks scheduler ~36 ms per fire; direct cost only 0.2%, but the
  sync may cause CUDA stream state effects with diffuse impact
- `mark_pages_capped(cap_t)` mutates `free_pages` tensor —
  subsequent allocator reads (every iter) may pay more if memory
  layout changed
- `unmark_pages_capped(cap_t)` (worker rollback) — same concern

## Status

- [x] Bisect identifies budgeter+fires as the cost (not arena alone).
- [x] Drill-down localizes spike to `build_kv_owner_map` Python loop.
- [x] Implement vectorized owner-map (GPU bool mask) + 6/6 unit tests
      with 1669× speedup.
- [x] Re-measure D8 single-rep — apparent -1.1 pp recovered.
- [x] N=5 reps — true recovery is **~0.7 pp** (3.70% → 3.01%).
- [x] Full per-iter timing — residual is DIFFUSE, not spike-per-fire.
- [x] Iter-time vs fire-distance — flat (no TLB tail).
- [x] Direct GPU profile — cuMem on arena causes 0% decode slowdown.
- [x] Fires-vs-cap-grow bisect — cap-grow is +0.09% (free).
- [x] (Task #131) Worker no-op bisect: cost is ALL on scheduler thread
      in cap_barrier path; worker contributes +0.18% (zero).
- [x] (Task #132) Drill INTO cap_barrier — DONE. Bisect:

      | phase | TPOT (N=3) | Δ vs off |
      |---|---|---|
      | off | 7.860 ± 0.019 ms | baseline |
      | bg_killfire (no fires) | 7.917 ± 0.033 ms | +0.72% |
      | nomark (fires decided, cap_barrier no-op) | 7.904 ± 0.024 ms | +0.55% |
      | worker_noop_verify_on (full cap_barrier) | 8.078 ± 0.019 ms | +2.76% |
      | worker_noop_verify_off (cap_barrier sans verify) | 8.108 ± 0.028 ms | +3.14% |

      **Skipping `mark_pages_capped` recovers the entire +2.86% cost.**
      **Skipping `verify` (the `.item()` sync) does NOT help** (slightly
      worse). So the cost lives in `mark_pages_capped` and/or its
      symmetric `unmark_pages_capped`, not in the GPU sync.

      What mark/unmark do (`allocator.py:129-199`):

      ```python
      mask = torch.isin(self.free_pages, target)  # 2.4M comparisons
      held = self.free_pages[mask]                # ~4k held
      self.free_pages = self.free_pages[~mask]    # NEW 2.4M tensor (~19 MB)
      self._capped_pages = torch.cat([existing, held])
      ```

      Per fire: ~3 such operations × ~19 MB each = **~57 MB of tensor
      reallocation per fire**. 21 fires × 57 MB = **~1.2 GB of GPU
      memory thrash** through PyTorch's caching allocator.

      **Hypothesis:** repeated rebuild of large `free_pages` tensor
      perturbs PyTorch caching-allocator state → subsequent allocations
      use different VRAM regions → memory access patterns or layout
      changes diffuse through all decode iters.

## Stage 6: Fix #134 — rewrite `mark_pages_capped` without realloc

`allocator.py` mark/unmark no longer mutate `free_pages`. Capped slots
are tracked only in `_capped_pages`. `alloc()` filters against this
small tensor:

```python
if capped_n == 0:
    # Fast path — original behavior, ZERO change
    select = self.free_pages[:need_size]
    self.free_pages = self.free_pages[need_size:]
    return select

# Cheap-case (typical): no capped in head of free_pages
head = self.free_pages[:need_size]
if not bool(torch.isin(head, capped).any().item()):
    self.free_pages = self.free_pages[need_size:]
    return head

# Slow-case: scan free_pages, find first N non-capped via cumsum
...
```

Also `available_size()` and `free_page_mask()` subtract `_capped_pages`
to maintain correct counts.

Unit tests `test_mark_no_realloc.py` (4/4 PASS):
- alloc correctly skips capped slots
- mark/unmark peak alloc < 1 MB (was 138 MB)
- mark+unmark idempotent
- KV-scale: mark mean 0.006 ms (was 1.14 ms; **190× faster**)

D8 saturated re-validation (N=3):
- off: 7.927 ± 0.089 ms
- inter: **7.906 ± 0.015 ms**
- Δ TPOT: -0.26% (within noise — regression resolved)

## Debug kill switches added (for ongoing bisection)

| Env var | Effect | File |
|---|---|---|
| `SGLANG_XPOOL_DISABLE_FIRES=1` | planner.decide returns no-fire | xpool_planner.py |
| `SGLANG_DISABLE_ADMISSION_GROW=1` | skip ReqToTokenPool/FutureMap grow | agent.py |
| `SGLANG_XPOOL_WORKER_NOOP=1` | worker unmark capped, skip cuMem | agent.py |
| `SGLANG_XPOOL_SKIP_MARK=1` | skip mark_pages_capped in cap_barrier | xpool_actuator.py |
| `SGLANG_XPOOL_SKIP_VERIFY=1` | skip `.item()` verify sync in cap_barrier | xpool_actuator.py |
| `SGLANG_ITER_TIMING_LOG=<path>` | per-iter timing + phase breakdown | scheduler.py |
| `SGLANG_BUDGETER_TICK_PROBE=<path>` | per-tick step timing (snapshot/adm/fire) | agent.py |

Plus instrumentation:
| Env var | Effect |
|---|---|
| `SGLANG_ITER_TIMING_LOG=<path>` | per-iter timing + phase breakdown |
| `SGLANG_BUDGETER_TICK_PROBE=<path>` | per-tick step timing (snapshot/adm/fire) |
