# Audit: scheduler arrival path (Admitter hook point)

## 1. Hook point for `Admitter.decide(req)`

**Primary hook:** `python/sglang/srt/managers/scheduler.py:2212` inside
`_add_request_to_queue` (def at line 2205), immediately before
`self.waiting_queue.append(req)`.

```python
2211    self._prefetch_kvcache(req)
2212    self.waiting_queue.append(req)   # ← Admitter.decide(req) goes BEFORE this
2213    req.time_stats.set_wait_queue_entry_time()
```

State available at this point:
- `self.token_to_kv_pool_allocator`
- `self.tree_cache`
- `self.running_batch`
- `self.waiting_queue` (for queue length `Q`)
- `req` itself

**Important caveat:** at `_add_request_to_queue` the req has ONLY
`origin_input_ids`. `req.extend_input_len`, `req.prefix_indices`,
`req.host_hit_length`, `req.mamba_pool_idx` are NOT set yet — they
are populated inside `PrefillAdder.add_one_req` after
`req.init_next_round_input(self.tree_cache)` at `scheduler.py:2706`.

So a "pure per-arrival" admitter at line 2212 must derive demand
from `len(req.origin_input_ids)`, OR the hook must be moved into the
prefill loop (line 2706 area) to see prefix-matched demand.

## 2. Demanded-size derivation

- KV-tokens demand X: after `init_next_round_input`, use
  `req.extend_input_len` (set at `schedule_batch.py:1059`).
  Pre-cache, use `len(req.origin_input_ids)`.
- Mamba-slots demand: **1 slot per req** (the mamba pool is per-req,
  not per-token).
- Max-new-tokens budget contribution:
  `req.sampling_params.max_new_tokens * new_token_ratio`.

## 3. Current "can-admit" check

`PrefillAdder.add_one_req` at `schedule_policy.py:767`. Criteria:
- `total_tokens = extend_input_len + max_new + page_size >= self.rem_total_tokens` → `NO_TOKEN`
- `swa_needed >= self.rem_swa_tokens` → `NO_TOKEN`
- `real_input_tokens >= self.rem_input_tokens` → `OTHER`

`rem_total_tokens = allocator.available_size() +
tree_cache.evictable_size() - rem_total_token_offset`. **This is
already "own-free + own-evictable space".**

Pool capacity query APIs:
- KV pool free: `token_to_kv_pool_allocator.available_size()`
- KV pool evictable: `tree_cache.evictable_size()` /
  `full_evictable_size()`
- Mamba pool free: `mamba_pool.available_size()` =
  `len(self.free_slots)` (`memory_pool.py:660`)
- Mamba pool live cap: `mamba_pool.live_size`

## 4. Existing defer points

Today, `_add_request_to_queue` ALWAYS appends to `waiting_queue` —
there is no per-arrival rejection by capacity. The actual deferral
is IMPLICIT: requests sit in `waiting_queue` until
`get_new_batch_prefill` picks them up, and `PrefillAdder.add_one_req`
returns `NO_TOKEN`/`OTHER` which breaks the loop.

→ Admitter's "defer" candidate maps cleanly onto **"leave req in
`waiting_queue`"; no new code path needed**. Only "own-evict /
cross-free / cross-evict" need new actuator triggers.

## 5. Reusability of `_fire_worker_loop` for synchronous fire

**Not directly usable.** The existing async flow:
1. `cap_barrier(plan)` synchronously on scheduler thread (returns FireToken)
2. `_fire_queue.put_nowait(token)` → worker thread → `execute_async`

For Admitter sync fire, pages must be migrated BEFORE the next
`alloc()` in `add_one_req`. Use `XPoolActuator.execute(plan)`
(`xpool_actuator.py:331`) — the legacy sync path that does cap_barrier
+ execute_async inline.

## 6. Thread-safety concerns

- `_alloc_lock` (`allocator.py:70`) is non-reentrant
- Admitter on scheduler thread; Budgeter worker on separate thread
- Both touch SharedHandlePool which is **lockless** today (safe only
  because Budgeter worker is single-producer)
- Concurrent Admitter + Budgeter fires would race on
  `_free_handles` (chunk_arena.py:446,475,498)
- Mitigation: either share Budgeter's `_fire_queue` and wait for
  completion, OR add mutex on SharedHandlePool

## Key files

- `scheduler.py` (2205-2212, 2566-2750, 2706, 2895)
- `schedule_policy.py` (767-899, esp 460, 487, 526)
- `schedule_batch.py` (1059, 1296)
- `mem_cache/allocator.py` (70, 75, 134-189, 226-269)
- `mem_cache/memory_pool.py` (316, 660-689)
- `budgeter/agent.py` (89, 268, 540-627, 726, 745)
- `arena/xpool_actuator.py` (124, 211, 331)
- `dev/interlayer/design.md` §355-410
