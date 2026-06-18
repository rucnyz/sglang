# alloc_lock — open follow-ups (lock perf)

The `_alloc_lock` added in commit `9e6e349f50` makes the allocator
thread-safe (proven by `race.py:test_1`) but **measurably adds
TTFT overhead** in idle_no_regression's 3-model perf sweep:

| Model | async no-lock TTFT Δ | async + lock TTFT Δ | Δ-from-lock |
|---|---|---|---|
| Qwen3.5-9B (TP=1) | -0.10% | +3.8% | +3.9 pp |
| Qwen3.5-35B-A3B (TP=1) | +1.00% | +2.15% | +1.15 pp |
| Qwen3.5-122B-A10B (TP=2) | -2.09% | +4.99% | +7.1 pp |

(Throughput Δ within 0.03% on all — only TTFT is affected.)

All cells still PASS idle_no_regression's bound (throughput ≤1%, TTFT ≤5% degradation
or any improvement). But the lock cost is non-trivial, especially on
122B where fires happen more often (22 fires vs 7 on 9B).

**BEFORE doing any of the TODOs below**, first do TODO 0 to confirm
the regression is real, not noise.

---

## TODO 0 — characterize noise floor (✅ DONE)

**Verdict: lock-induced regression is PURE NOISE.** TODOs 1 and 2 not
needed. Below is the data.

Ran idle_no_regression on Qwen3.5-9B (TP=1, GPU 3, sequential — no contamination)
3 reps with lock + 3 reps with lock reverted (`git checkout
9e6e349f50~1 -- allocator.py`). Compared via `noise_compare.py`.

```
                                  TTFT_off       TTFT_inter      Δ (inter vs off)
WITH lock     (3 reps mean ± σ)   26.00 ± 0.20   27.46 ± 0.60    +5.62 ± 2.47 %
WITHOUT lock  (3 reps mean ± σ)   26.54 ± 0.94   27.94 ± 0.55    +5.34 ± 3.04 %

Lock-induced Δ in regression:  +0.28 pp
Pooled standard error      :  ±2.26 pp
|Δ| / SE                   :   0.12   (need ≥ 2.0 to be significant)

→ Lock contributes essentially ZERO measurable TTFT cost.
```

Individual reps reveal the noise band:
```
WITH lock:    Δ = +5.14%, +8.30%, +3.44%
WITHOUT lock: Δ = +4.89%, +2.56%, +8.58%
```
Both groups span ~5pp run-to-run on the same code. The earlier
single-run apparent "3.9-7.1pp regression" was 100% within this band.

Throughput Δ in all 6 reps: ≤0.01% (lock doesn't touch the rate
allocator processes requests at; only the lock-acquire overhead
matters, and it's negligible).

**Why the lock cost vanishes**: lock acquire is ~100ns per call.
Even at 100K alloc/free per 180s run, total overhead is ~10ms —
0.005% of wall time. Worker holds lock only during
`set_capacity_pages` (microseconds per fire × 7 fires = also
negligible).

**What the persistent ~5% TTFT-inter > TTFT-off actually is**: not
lock cost; it's the async fire's cap_barrier work on the scheduler
thread (~1-2ms per fire) plus the queue-Full skip path. This was
already documented as the design's cost-of-flexibility; doesn't
warrant further optimization since at the paper's target model
(122B) the capacity benefit dominates.

## TODOs 1, 2, 3 — N/A (lock cost is noise; no optimization needed)

Skipped per TODO 0 verdict. Keep below for context if future work
wants to reduce the persistent ~5% TTFT cost (which is FROM THE ASYNC
FIRE WORK ITSELF, not the lock — they are different concerns).

### TODO 1 (skipped — TODO 0 closed) — finer-grained critical section

## TODO 1 — finer-grained critical section (if TODO 0 confirms regression)

Today the lock wraps the entire body of `mark_pages_capped`,
`unmark_pages_capped`, `set_capacity_pages`, `alloc`, `free`,
`merge_and_sort_free`. The body includes `torch.isin`, `torch.cat`
operations that take µs each — lock is held for full duration.

**Idea**: compute the new tensor OUTSIDE the lock, atomically swap
INSIDE:

```python
# Current
with self._alloc_lock:
    mask = torch.isin(self.free_pages, target)
    held = self.free_pages[mask]
    self.free_pages = self.free_pages[~mask]

# Proposed
loaded_free = self.free_pages    # snapshot read (no lock; relies on GIL atomicity of attr read)
mask = torch.isin(loaded_free, target)
held = loaded_free[mask]
new_free = loaded_free[~mask]
with self._alloc_lock:
    # Verify nothing else mutated since snapshot
    if self.free_pages is loaded_free:
        self.free_pages = new_free
    else:
        # Retry: another thread rebinded between snapshot and lock
        ... retry loop
```

**Test-first protocol** (critical):
1. Modify `race.py:test_1` to also assert post-state consistency:
   `free_pages ∪ release_pages ∪ _capped_pages = {1..size}` and
   `_capped_pages` contains exactly the `target` after mark.
2. Run modified test against current lock impl → expect PASS (baseline)
3. Apply finer-grained design
4. Run test → MUST still PASS
5. Run idle_no_regression perf → measure reduction in regression

**Risk**: Python attribute assignment is atomic for individual ops
but composite read-then-rebind ISN'T. The retry loop addresses that
but needs careful proof.

### TODO 2 (skipped — TODO 0 closed) — worker-active flag

Insight: worker thread is ACTIVE only during fire execution (a few
ms per fire, fires are seconds apart). 99% of `alloc()`/`free()`
calls happen while worker is idle → no contention possible → don't
need the lock.

**Design**:
```python
# BudgetAgent
self._worker_active = threading.Event()  # initially clear

# Worker, around the lock-relevant section:
self._worker_active.set()
try:
    dst_act.cap_allocator_only(...)  # this calls set_capacity_pages
finally:
    self._worker_active.clear()

# Allocator (modified alloc/free/mark):
def alloc(self, n):
    if not _get_worker_active().is_set():
        # Fast path: no concurrent mutation possible
        return self._alloc_unlocked(n)
    else:
        with self._alloc_lock:
            return self._alloc_unlocked(n)
```

**Correctness argument**:
- Worker SETs flag BEFORE entering critical section. set() includes a
  memory barrier (it's a threading.Event internal lock).
- alloc reads flag. If clear, no concurrent writer can be in CS
  (because worker would've set it first).
- If worker sets the flag DURING alloc's flag check, alloc might
  proceed without lock while worker enters CS — RACE.

**Need careful sync**:
- After worker.set(), check no alloc is in-flight (via a separate
  in-alloc counter)
- Or: worker uses a "soft cooldown" — flag goes set 1ms before actual
  modify, giving alloc time to see it. Hacky.

**Test-first protocol**:
1. Extend `race.py` to drive thousands of alloc/free WHILE
   worker repeatedly enters/exits critical section
2. Assert no double-alloc, no IndexError, no leaked pages
3. Run against current full-lock design → expect PASS
4. Apply worker-active-flag design
5. Run test → MUST still PASS (test should be sensitive to the
   transition window race)
6. If race detected in (5), retry with stricter sync; if all attempts
   fail, abandon this approach

**Risk**: getting the transition-window sync right is hard. Suggest
spawning a subagent code-review pass before declaring this done.

### TODO 3 (skipped — TODO 0 closed) — accept-and-document

---

## TODO 0.5 — active-only usage in persist consec (✅ DONE)

Followed up from TODO 0: even though lock cost is noise, the
underlying ~5% TTFT-inter-vs-off was hypothesized to be from
"phantom" fires triggered by `mamba_radix_cache` filling the pool
to 99% (planner saw saturated, but admission wasn't actually
stalled — cached snapshots are LRU-evictable for free).

design.md §"Budgeter — steady-state pressure rebalance" already says persist signal should be on "admission
ceiling" — live state only — not total occupancy. Code was reading
`pool_occupancy_mamba` (total). Aligned them.

**Changes**:
- `agent.py:_maybe_fire` populates `snapshot["usage_kv_active"]` and
  `snapshot["usage_mamba_active"]` = (used − evictable) / total
- `xpool_planner.py:_decide_inner` (nb_direction_aware branch) reads
  active fields when present, falls back to total when absent
- **Falsy-zero bug caught in first pass**: `snap.get(k, fb) or fb`
  treats 0.0 as falsy → wrong fallback. Fixed with explicit `in`
  check. Regression-guarded by `no_spike/D6c_unit_nb_multisource.py:test_F`.

**Outcome** (idle_no_regression N=3 sequential, see `../idle_no_regression/README.md`):
- 9B: pre-fix +5.62 ± 2.47 % → activev2 **+3.20 ± 3.99 %** (fires
  7 → **0**)
- 122B: pre-fix +4.99 % (N=1) → activev2 **+1.90 ± 1.71 %** (fires
  22 → **0**)

Fires correctly drop to 0 on idle workloads. Residual ~2-3% TTFT
mean is **independent of fires** (0 fires, still ~3% slower) — i.e.,
the active-fix did its job (eliminated phantom fires) but exposed
that the cost wasn't fires to begin with.

Next: see `TODO 1` (now repurposed) — track the residual ~3% to its
real source.

---

## TODO 1 (post-active-fix) — analyze residual TTFT overhead — ✅ CLOSED

**Root cause: pytorch issue 165419 (MemPool path).
Fix: arena defaults to from_blob; MemPool branch deleted entirely.
Commits: `cd3902bcc6` (verdict), `241463552d` (cleanup).**

**Updated 2026-05-26: BISECTION RESULT — arena alone owns the +5%;
budgeter contributes ~0 on top.**

### 3-phase bisection (bisect_3phase.sh, N=3 paired)

```
                 mean TTFT (ms)   N
  off          : 26.47 ± 1.03    3      (no arena, no budgeter)
  arena_only   : 27.12 ± 0.51    3      (SGLANG_ARENA_SHARED=1, no budgeter)
  inter        : 27.18 ± 0.69    3      (SGLANG_HIMA=1 → arena + budgeter)

  arena_only vs off      : per-rep Δ = [+2.93, -2.42, +7.23]  mean +2.58% SE 2.79
  inter      vs off      : per-rep Δ = [+3.82, +1.09, +3.21]  mean +2.71% SE 0.83
  inter      vs arena_only: per-rep Δ = [+0.87, +3.60, -3.74]  mean +0.24% SE 2.14
```

### Verdict

Adding budgeter to arena-only adds +0.24% (well inside noise). The
cost is in the arena tensor backing under real inference. The 4
exonerating micro-tests covered isolated code paths but missed the
arena tensor's interaction with the real inference path (CUDA graph
capture, attention kernel access pattern, or warmup priming).

### Combined N=10 (idle_no_regression r1-r6 + bisect 3 off/inter pairs + idle_no_regression r10)

```
  off    N=10: mean 26.41 ± 0.80 ms (SE 0.25)
  inter  N=10: mean 27.33 ± 0.71 ms (SE 0.22)
  per-rep Δ: +7.81, +0.71, +1.09, +3.86, +7.06, +9.89, +3.82, +1.09, +3.21, -2.74
  mean = +3.58% ± 3.81 pp (σ), SE = 1.21 pp
  |mean|/SE = 2.97  → significant
```

**Drift across N**: +1.92% (N=3) → +5.07% (N=6 inflated by r6 +9.89%
outlier) → +3.62% (N=9) → +3.58% (N=10). True mean settles ~+3.5%,
noise σ ~4pp. N=6 alone over-stated the magnitude. The user's
original "+3% regression" reading was correct in magnitude.

Closing out:
- Budgeter is design-correct AND perf-neutral once arena is in
- Arena tensor backing IS the +3.5% cost
- Drill into arena → see next section: MemPool path is the source

### Sub-bisection (bisect_arena_path.sh, N=3 paired, 2026-05-26)

Drilled INTO arena to find which tensor-backing path owns the cost.

```
                 mean TTFT (ms)    N
  off              26.84 ± 0.98    3       (no arena)
  arena_mempool    27.24 ± 0.59    3       (SGLANG_ARENA_SHARED=1, SGLANG_ARENA_FROM_BLOB=0)
  arena_fromblob   26.43 ± 0.11    3       (SGLANG_ARENA_SHARED=1, SGLANG_ARENA_FROM_BLOB=1)

  arena_mempool  vs off            : per-rep Δ = [-1.35, +2.18, +3.79]  mean +1.54% SE 1.52
  arena_fromblob vs off            : per-rep Δ = [-3.73, -3.55, +2.97]  mean -1.44% SE 2.20
  arena_fromblob vs arena_mempool  : per-rep Δ = [-2.41, -5.62, -0.79]  mean -2.94% SE 1.42 (|Δ|/SE=2.07 sig)
```

**ROOT CAUSE CONFIRMED**: PyTorch issue 165419 — `torch.cuda.MemPool`
silently disables `expandable_segments` for the whole process while
in scope, which costs ~3% TTFT under live attention + CUDA graph
capture. The from_blob path bypasses MemPool entirely (using
`at::from_blob` over cuMemMap-backed VA) and recovers the regression.

Note: `mempool_penalty_demo.py` (N=10 synthetic alloc kernels)
was NOT WRONG — at the demo level, MemPool penalty is invisible. The
penalty only shows up under real serving (attention kernels + graph
capture + concurrent intermediate allocs).

**Direction**: from_blob also has dramatically lower TTFT variance
(σ=0.11 vs 0.59 / 0.98) — MemPool adds noise too, not just mean cost.

**Fix candidate** (proposed; not yet applied):
- Default SGLANG_ARENA_FROM_BLOB to 1 (currently defaults to 0).
- Or: when SGLANG_ARENA_SHARED=1, auto-promote SGLANG_ARENA_FROM_BLOB=1
  (same pattern as SGLANG_HIMA → SGLANG_ARENA_SHARED).
- Cost: zero — from_blob is already implemented and used in tests.
- Risk: from_blob has a no-op deleter; lifetime of arena vs tensor
  must be carefully verified (probably already is, since this code
  path exists). Test under byte_transfer to confirm transfers still work.

N=6 sequential reps on 9B activev2 (HEAD = lock + active-fix v2):
  Δ per rep:  +7.79%, +0.69%, +1.08%, +3.83%, +7.06%, +9.91%
  mean = +5.07%, std = ±3.77 pp, SE = 1.54 pp
  |mean|/SE = 3.29 (significant, threshold = 2.0)

So the +5% TTFT IS a real overhead, not server-boot noise.

**Micro-tests that exonerated 4 candidates remain valid but
insufficient**:
- Lock acquire (TODO 0): 0.009%
- MemPool/expandable_segments (`mempool_penalty_demo` N=10): refuted on
  our PyTorch config
- Budgeter per-tick (`tick_cost`): 0.009% over 180s
- Arena tensor backing kernel-level (`arena_tensor_perf`): +0.20% ± 0.13 SE

**Candidates the micro-tests did NOT cover** (and thus where the
+5% might still hide):
1. **CUDA graph capture / replay with arena tensors**: arena tensors
   are valid CUDA pointers, but capture might handle them differently
   (e.g. additional state in captured graph metadata). Per-boot effect,
   not per-call — invisible at kernel level.
2. **Sglang scheduler dispatch path call rate**: `tick_cost` assumed
   1000Hz event loop. Actual sglang event loop may iterate faster
   (e.g. 10000Hz when batches small), pushing 75ns × N into the % range.
3. **Per-request KV-cache write/read patterns under real inference**:
   the `arena_tensor_perf` synthetic gather/scatter may not exercise
   the same access pattern as real attention; could miss a TLB/coalesce
   regression specific to attention kernel × arena layout.
4. **Allocator lock contention WITH workload-driven alloc rate**: prior
   test ran lock without contention; real bench has multiple
   intermediate allocations per forward.

The N=6 data is enough to declare the regression real but **not to
identify the source**. Need targeted empirical bisection:
- Run sglang with `no SGLANG_HIMA SGLANG_ARENA_SHARED=1` (arena
  only, no budgeter). N=3+. If TTFT shows the same +5% → arena
  tensor backing under real inference is the source.
- If arena-only is clean and `SGLANG_HIMA=1 SGLANG_ARENA_SHARED=0`
  also clean, the cost might be in the interaction (CUDA graphs
  capturing arena state PLUS budgeter dispatching during inference).

**No fix candidate yet — need more bisection data.**

---

### TODO 1 (skipped — closed) — original notes preserved for context:

After active-fix v2: 0 fires under idle, throughput identical, but
inter still ~3% slower than off. **Cost is independent of fires.**

**Strong hypothesis** (from `multi_tensor_arena.py:233` comment):
PyTorch silently disables `expandable_segments` when a user MemPool
is active (CUDACachingAllocator.cpp:1587-1591, pytorch issue 165419).
SGLANG_ARENA_SHARED=1 creates the user MemPool. Documented penalty:
+6-7% TTFT.

The alt path `SGLANG_ARENA_FROM_BLOB=1` bypasses the MemPool by
using `torch.from_blob` directly against the VA-mapped chunks.

**Protocol**:
1. Re-run idle_no_regression on 9B with `SGLANG_ARENA_FROM_BLOB=1` (force from_blob
   path), N=3 sequential
2. If TTFT Δ drops below ~1% → hypothesis confirmed, document
3. If still ~3% → look elsewhere (per-tick budgeter snapshot cost,
   ARENA boot overhead, etc.)

**Cost**: 3 × 8 min = 24 min wall on 9B alone. Don't sweep 122B
unless 9B confirms hypothesis (122B's pools are already tight,
expandable_segments matters less).

**Where it lives**: this folder. Test command added to
`../idle_no_regression/README.md` once verified.

Honestly: the lock cost is bounded (≤5% on idle_no_regression's 5% bound, ≤7pp on
worst-case 122B). Throughput is unaffected. Cross-pool transfer's
WHOLE POINT is per-fire bytes moved, not per-request TTFT.

If TODO 0 says "lock cost is mostly noise" or "real but small", the
right call is **document it as known cost** and move on. Add a note
in `alloc_lock/README.md` quantifying lock overhead and explaining the
trade-off (correctness > tiny TTFT cost). No code change.

---

## Cross-reference

Perf measurement source: `idle_no_regression/README.md` (sweep table includes pre-
async, async no-lock, and async + lock rows).

Race test that the lock enables: `race.py:test_1` (FAILs
without the lock — proven by revert-and-retry in commit `9e6e349f50`
commit message).
