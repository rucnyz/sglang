# Audit: sync fire path for Admitter

## 1. `XPoolActuator.execute(plan)` — synchronous, already exists

`python/sglang/srt/arena/xpool_actuator.py:331-346`:

```python
def execute(self, plan): 
    token = self.cap_barrier(plan)
    return self.execute_async(token)
```

Performs the **identical** physical cuMemUnmap + cuMemMap, torch.cuda
.synchronize, and dst `cap_allocator_only`. Same code path the
legacy sync branch in `agent.py:611-627` already uses.

## 2. Direct inline call from Admitter

**Yes, with caveats.** Both phases are designed thread-callable
(xpool_actuator.py:14-22 docstring).

**Lock map:**
- Source/dst allocator `_alloc_lock` (`allocator.py:70`):
  `mark_pages_capped`, `unmark_pages_capped`, `set_capacity_pages`
  (inside `cap_allocator_only`) acquire it. Scheduler's own
  `alloc()`/`free()` also use it.
- `SharedHandlePool` (`chunk_arena.py`) has **NO lock** —
  `_free_handles` is a Python list mutated by `shrink_explicit` /
  `grow`. Currently safe ONLY because the Budgeter's worker is
  single-threaded.

**Race risk:** if Admitter fires synchronously while the Budgeter's
worker is mid-fire on the same shared handle pool, the Python-list
ops on `_free_handles` race. GIL makes individual list ops atomic
but the sequence "pop handle / map it / record" is not — handle ids
could be lost or double-used.

## 3. Worker-thread serialization

Worker drains one fire at a time (`agent.py:763-809`). Today's
invariant holds because there is only one producer thread.

If Admitter fires synchronously while a Budgeter fire is in flight:
- Same `SharedHandlePool._free_handles` and `kv_arena._arena
  ._external_pool` are touched
- Src/dst PAGES do not overlap (each fire picks own free-page set
  via `XPoolFirePlanner.build`, fire_planner.py:103) — cuMemUnmap
  targets are disjoint
- BUT the shared FREE-HANDLE LIST is not protected

**Mitigation options:**
- (a) Admitter shares Budgeter's `_fire_queue` and blocks on completion
- (b) Add mutex around `SharedHandlePool` operations
- (c) Always run fires (Admitter + Budgeter) on the worker thread,
  with Admitter blocking on a per-fire future

## 4. `cap_barrier` cost

**36 ms outlier is bootstrap-only**, not steady state. From D8
bisect:
- `d8_regression_bisect.md:122` reports "ONE tick at 36 ms" — a
  single tick, not per-fire steady state
- Verify-skip bisect showed skipping verify does NOT help

The expensive part was `mark_pages_capped`'s tensor realloc, fixed
by #134.

**Steady-state cap_barrier**: dominated by
`expand_pages_to_token_slots` Python loop
(`xpool_actuator.py:148`, ~per-page) + `.item()` sync (~hundreds of µs).

For a 1-page Admitter fire: expect a few hundred µs — well under
1 ms.

`SGLANG_XPOOL_SKIP_VERIFY=1` (`xpool_actuator.py:176`) exists as
kill switch if needed.

## 5. `mark_pages_capped` is O(small) post-#134

Confirmed: `allocator.py:160-166` just does
`torch.cat([existing, target])` on small `_capped_pages`. No
`free_pages` realloc.

Unit test `test_mark_no_realloc.py`: mean 0.006 ms at KV scale.

**Safe for Admitter hot path.**

## 6. Smallest plan: ≥ LCM(n_src, n_dst) pages

`XPoolFirePlanner.build` (`fire_planner.py:80`) clamps `n = max(1,
n_pages_target)`, so 1 page is technically valid.

BUT actuator's atomic-cross-pool LCM logic
(`xpool_actuator.py:276-279`):

```python
total = (target_total // lcm(n_src, n_dst)) * lcm
```

If 1 page < LCM of subpool counts (e.g. n_kv_subpools=64,
n_mamba_subpools=48 ⇒ lcm hundreds), `total` rounds down to ZERO and
no transfer happens.

**The Admitter must size requests to ≥ lcm(n_src_subpools,
n_dst_subpools) pages**, not 1.

## 7. Reservation step

The Budgeter does not reserve granted bytes for any specific
arrival — it just bumps the dst cap (`xpool_actuator.py:316-317`).

For the Admitter, after `execute()` returns the new dst capacity is
visible to EVERY waiting arrival; a second arrival on the scheduler
thread could `alloc()` from the same fresh slots before the
triggering arrival completes admission.

**The Admitter needs an additional RESERVATION step under the dst
allocator's `_alloc_lock` between `execute` and resuming admission.**
`cap_allocator_only` alone does not reserve.

## Verdict

Existing `execute()` is **technically reusable for one-shot
synchronous Admitter fires**, but two new pieces are required for
production safety:

1. **Serialize against Budgeter's worker** — either share the
   worker's `_fire_queue` and block on completion, or add a mutex
   covering `SharedHandlePool` access.
2. **Reserve granted slots** before resuming the arrival, so a
   concurrent arrival can't race-allocate them.

Skipping verify and using #134's `mark_pages_capped` keeps the hot
path comfortably under 1 ms for a small (few-LCM-pages) plan.

## Key files

- `arena/xpool_actuator.py` (124, 211, 263-289, 316-317, 331)
- `arena/chunk_arena.py` (428, 446, 475, 478, 498)
- `arena/kv_actuator.py`
- `budgeter/agent.py` (89, 611-627, 745, 763-809, 824)
- `budgeter/fire_planner.py` (80, 103)
- `mem_cache/allocator.py` (70, 134-189)
- `dev/interlayer/dyn_admission_cap/d8_regression_bisect.md` (122,
  177-187, 269-276, 325-326)
