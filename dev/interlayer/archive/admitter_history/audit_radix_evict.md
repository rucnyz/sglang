# Audit: radix-cache eviction surface vs Admitter `c^evict_i(X)`

## 1. Leaf enumeration

**KV-only (`radix_cache.py`, lines 362, 614):**
- Maintained as Python `set` `self.evictable_leaves`
- Kept in sync via `_update_leaf_status()` (892-905)
- `evict()` does `list(...)` + `heapq.heapify` — **O(N log N) per call**
- No incremental priority queue

**HiMA mamba cache (`mamba_radix_cache.py`):**
- Dual doubly-linked LRU lists `full_lru_list` (KV) and
  `mamba_lru_list` (snapshot), wrapped by `LRUList` (273-525)
- Eviction-priority heap built lazily: `_lpb_build_eviction_heap`
  (978) and `_lpb_build_full_eviction_heap` (1233)
- **O(N) build, O(log N) pop**, rebuilt every `evict()` call

## 2. Per-leaf data available

`TreeNode` (`mamba_radix_cache.py:64-264`):
- `value` (int64 KV-page tensor — size = `value.numel()` tokens)
- `mamba_value` (slot-id tensor)
- `last_access_time`
- `hit_count` (cumulative; never reset)
- `_hit_times` deque (sliding window, `SGLANG_LPB_WINDOW_S=60s` default)
- `hits_in_window()` → recent hit-count
- `full_lock_ref` / `mamba_lock_ref` (>0 ⇒ evict-blocked)
- `priority`, `host_value`
- `eviction_priority()` (223) computes `hits_per_byte` with real
  `lpb_bytes_per_mamba_slot` ratio (initialised line 562) —
  **already exactly `hit_prob × s_b` numerator**, memoised on
  `_cached_priority`

## 3. KV vs mamba — separate or shared

Same `TreeNode` (one node carries both `value` and `mamba_value`)
BUT **separate LRU lists**, separate evictable counters:
- `full_evictable_size()` (line 1395) — KV
- `mamba_evictable_size()` (line 1398) — mamba

Budgeter's `agent.py:415-432` already calls both. Pure-KV models
use `RadixCache` with single `evictable_size()`.

## 4. Existing `evict()` API

`EvictParams(num_tokens, swa_num_tokens, mamba_num)` →
`EvictResult(num_tokens_evicted, ...)` (`base_prefix_cache.py:74-88`).

- Returns **count freed only — no cost estimate, no dry-run mode,
  no "evict at least X cheapest" simulator**
- `evict_full(N)` (1271) and `evict_mamba(N)` (1090) actually free
  pages; cannot be polled non-destructively
- `record_recovery_len_kv/_rec/_retract` (`common.py:316-347`) EWMA
  `\bar L_i` updated only as side-effect of real evictions

## 5. Per-block recompute cost

**No per-block prefill-latency telemetry.** Only proxy is `\bar L_i`
EWMA (`common.py:313`) — a single scalar (mean recovery length), not
per-block. The Admitter must interpolate `c_i(s_b)` from
`len(node.value)` and a global prefill-cost-per-token coefficient.

LPB's `eviction_priority` already uses `value.numel()` and mamba
`bytes_per_slot` as a byte-cost denominator (234-264) — same
building block, but **inverted (value-per-byte, not cost-per-block)**.

## 6. Hot-path cheap `c^evict_i(X)` — proposal

**No subtree-aggregated cost annotation exists** in the radix tree.
Only scalar root-level sums (`evictable_size_`, `protected_size_`).
Computing `Σ cheapest blocks: hit_prob_b × c_i(s_b)` requires
iterating leaves (O(N)).

For HiMA workloads N can be 10³-10⁴ live nodes — **100 µs budget
rules out full O(N) heapify per arrival**.

**Proposal — incrementally-maintained cost-sorted index:**
- Wrap `evictable_leaves` with `SortedKeyList[(cost_per_token, node)]`
- Keyed by `c_i(s_b) × hit_prob_b / s_b`
- Update sites already exist: `_update_leaf_status`, `record_hit`,
  `_delete_leaf`, `inc/dec_lock_ref` — these are the same places
  `_cached_priority` invalidates
- Push delta into sorted structure at each existing invalidation hook
- Maintain prefix-sum cache (Fenwick / BIT) on `(cost, cum_tokens)`
- `c^evict_i(X)` = `lower_bound(X tokens)` + partial last block →
  **O(log N) query**
- Re-priority on hit is O(log N) (remove+reinsert)
- Fits 100 µs budget for N ≤ 10⁴ even in Python

**Alternative (simpler):** if Admitter only queries `c^evict_i(X)`
for a small set of X values, maintain a sorted vector of
`(eviction_priority, len(node.key))`, refresh once per Budgeter tick
(~1 s), cache prefix-sums. Per-arrival cost = O(log N) bisect.

## 7. Empty-evictable case

YES — `evictable_leaves` can be empty when every leaf is locked by
a live req (`lock_ref > 0` ⇒ excluded by `_update_leaf_status`,
`radix_cache.py:893`).

Same for `full_lru_list.get_leaf_lru_no_lock()` returning `None`
(1292), `_lpb_build_full_eviction_heap` returning empty (1240).
`evictable_size()` returns 0 in that case. `evict_full(N)` silently
exits returning 0 (1284).

**Admitter must handle this explicitly:**
`c^evict_i(X) = +∞` whenever `evictable_size_i() < X`,
then fall through to cross-free / cross-evict / defer.

No exception raised — signal is purely the returned count.

## Key files

- `mem_cache/radix_cache.py` (211-271, 362, 608-639, 892-905)
- `mem_cache/mamba_radix_cache.py` (64-264, 273-525, 978-1026,
  1090-1189, 1191-1231, 1233-1268, 1271-1323, 1391-1411)
- `mem_cache/common.py` (313-337)
- `mem_cache/base_prefix_cache.py` (73-88)
- `budgeter/agent.py` (415-432)

**Hot-path verdict:** out-of-box cheapest cost-aware query is
`_lpb_build_eviction_heap` at O(N) — borderline for N≤500, blown
for N≥2000. To meet <100 µs per arrival robustly, need
incrementally-maintained sorted index (proposal above) hooked into
the four existing mutation sites.
