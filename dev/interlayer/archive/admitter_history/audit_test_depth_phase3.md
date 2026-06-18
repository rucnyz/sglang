# Test depth audit — Phase 3 (EvictCostIndex)

Subagent audit of `test_evict_cost_index.py` (10 tests initial → 12 after
gap fixes). Identified 3 gaps; landed fixes for 2 (Gap 1 + Gap 3); deferred
Gap 2 as not required for Phase 4.

## Gaps identified

### Gap 1: test_6c is serial, not actually concurrent — CRITICAL

`test_6c_concurrent_query_during_rebuild_safe` did serial rebuild→query→
rebuild→query. The implementation comment claims "Atomic swap — Python
attribute assignment is GIL-atomic" but that contract was never tested
under real concurrency.

**Risk:** Phase 5 calls `set_evict_index` at every Budgeter tick (1 Hz)
on a worker thread while the scheduler thread queries `c_evict_us` per
arrival (~1k qps). If list snapshotting isn't truly GIL-atomic, a query
could read `_prefix_tokens` from snapshot v2 but `_sorted_cpt` from v1,
with mismatched lengths → `IndexError`.

**Fix landed:** `test_6e_concurrent_thread_pool_race` —
ThreadPoolExecutor with 1 writer (rebuilds N=100↔50k every loop) + 8
readers (querying random X). Run 2s, assert no exceptions, all returns
are finite float or +inf (no NaN, no IndexError, no torn read).
Result: **1.18M reads under writer churn, 0 torn.**

### Gap 2: "almost-enough" boundary (defer)

`test_2_single_block` covers one specific overshoot. Test 3 covers
boundary `c_evict_us(61) == +inf`. But no test covers production-realistic
"N small blocks summing to just below X".

**Risk:** if a refactor drifts to "return cumulative-so-far when X >
total" (a "best-effort partial"), Admitter would silently pick own-evict
on infeasible inputs → allocator failure at fire time.

**Status:** deferred to a future phase. Phase 4 (sync fire path) doesn't
read partial-eviction output — it only consumes the scalar `c_evict_us`
in `decide()` arg-min. The risk is real but doesn't block Phase 4.

### Gap 3: hit_prob unit contract — CRITICAL

Test 3 names the third tuple element `hit_prob` and feeds raw
probabilities, but no test pins the units. Phase 5 will derive hit_prob
from `TreeNode.hits_in_window()` — natural temptation is to pass the raw
count (typically 0-1000s) instead of a normalised probability.

**Risk:** raw counts make `c_evict_us` ~1000× higher than `c_xfer_us`
absolute scale. The Admitter's `decide()` arg-min would then nearly
always pick cross-* even when own-evict is genuinely cheaper. D6n
(prefers own-evict when src cache hot) would fail silently.

**Fix landed:**
- `test_8_hit_prob_dimensional_contract`: builds two indexes with
  identical sizes but `hit_prob ∈ {0.5, 500}`; asserts the c_evict_us
  ratio is EXACTLY 1000× at all query points. Pins linearity.
- Docstring update on `EvictCostIndex.__init__` (cost_model.py:343-361):
  explicitly states "hit_prob MUST be dimensionless ∈ [0, 1]"; warns
  Phase 5 wirers against passing raw `hits_in_window()` counts.

## Genuinely deep tests (per subagent verdict)

1. `test_4_prefix_sum_matches_linear_scan` — 1000 random leaves × 200
   random X cross-checked vs from-scratch linear-scan reference. Defends
   against off-by-one in bisect/partial math.
2. `test_6b_rebuild_scales_below_n_log_n` — per-element ratio < 10×
   across N=100→100k. Locks down `sort(...)` against accidental O(N²)
   regression.
3. `test_7_cost_model_facade_routes_to_index` — covers the four-state
   matrix (no index, plugged-feasible, plugged-infeasible, second-pool
   unconfigured). Defends both facade contract and back-compat default.
4. `test_5_lock_ref_excludes_block` — locked block 10× the unlocked
   one's tokens. If filter regressed, `total_evictable_tokens` would be
   1100 not 100, easily caught.
5. **NEW** `test_6e_concurrent_thread_pool_race` — real writer/reader
   parallelism. Documents GIL-atomicity claim.
6. **NEW** `test_8_hit_prob_dimensional_contract` — pins units against
   silent unit-drift bugs in Phase 5 wiring.

## Subagent verdict

> **Soft go for Phase 4** after landing Gaps 1 + 3 (≤ 1 hr work).

## Action taken

- `test_owner_map_vectorized.py::test_6` rewritten from single-sample
  to **N=5 median**. New threshold: median ≥ 200×. Rationale: empirical
  reps show a wide 19×–1552× range under GPU contention; a single
  sample can't be trusted. Median is robust — even when one rep hits
  19×, the median sits at 540×+. 200× is well above worst-rep noise
  and well below the pristine ~1500× ceiling.
- Tests 6e (race) + 8 (unit contract) landed.
- Docstring update on `EvictCostIndex` pins `hit_prob ∈ [0, 1]`.

## Phase 3 final status

12/12 tests PASS. Perf budget summary:

| Metric | Result | Budget |
|---|---|---|
| Per-query P99 @ N=10k | 0.5 µs | < 50 µs |
| Per-query max | 3.8 µs | — |
| Rebuild @ N=10k | 3.3 ms | < 50 ms |
| Rebuild @ N=100k | 49 ms | < 200 ms |
| Per-element scaling 100k/1k | 2.23× | < 10× (O(log N) ~2×) |
| Throughput @ N=2k | 4.49M qps | ≥ 50k qps |
| Concurrency torture | 1.18M reads, 0 torn | exception-free |
| Unit contract | hit_prob × 1000 ⇒ c_evict × 1000 | linear |

→ Phase 4 unblocked.
