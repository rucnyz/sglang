# HiMA → v0.5.16 rebase notes (branch HiMA-v0516)

Goal: rebase HiMA (33 commits on 1981464ba4) onto upstream/main b78d3999b5
(~1680 upstream commits, v0.5.11→v0.5.16 era). Worktree: /data/yuzhou/projects/sglang-rebase.
The main checkout at /data/yuzhou/projects/sglang is untouched (user WIP in admitter.py).

## Doctrine
- Prefer upstream's evolved structures; layer HiMA deltas on top.
- For each conflicted file: consult OUR TRUE DELTA vs merge-base
  (`git diff $(git merge-base HiMA upstream/main) 0c9eeb02dd -- <file>`),
  never resolve from conflict markers alone.
- Functions that MOVED upstream get our hooks re-implanted at the new home.
- Scheduler/ModelRunner __init__ additions must be init_*/maybe_init_* helpers
  (skill: large-class-init-style). Env vars: skill env-var-conventions.
- After each file: ast.parse check, then `git add`.

## Mega-commit (0c9eeb02dd) resolutions so far
- `.gitignore`, `server_args.py`: union ("lpb" added to RADIX_EVICTION_POLICY_CHOICES;
  upstream now also has an .extend() hook at L406 — consider registering lpb there later).
- `mem_cache/common.py`: took upstream; re-added the three record_recovery_len_*
  EWMA recorders (+_RECOVERY_LEN_EWMA_ALPHA) above evict_from_tree_cache.
- `mem_cache/allocation.py` (NEW upstream home of alloc_token_slots): inserted the
  on-demand `_kv_grow_hook` retry block before the OOM raise.
- `mem_cache/swa_memory_pool.py`: upstream wholesale. Our loc-cache
  (+invalidate_loc_cache) was superseded by upstream's per-forward
  unwrap_write_loc pre-translation (fires happen between forwards → fire-safe);
  no external callers existed. Our in-file SWA allocator copy dropped
  (upstream keeps it in allocator/swa.py; base-class max_size is default-safe).
- `mem_cache/allocator/base.py`: hand-merged single class (our branch had the
  class DEFINED TWICE — latent bug, fixed here). Upstream skeleton + HiMA ctor
  extension (max_size param, _kv_grow_hook, _set_capped_pages/_capped_lo,
  _cap, _alloc_lock), live_size property, locked merge_and_sort_free with
  _merge_and_sort_free_unlocked. Dropped base-level backup/restore_state
  (upstream removed; our token.py/swa.py define their own). Kept upstream's
  resize(config) + free_segment(s).
- `mem_cache/radix_cache.py`: kept upstream evict loop (heap + free_segment,
  eviction_strategy.get_priority hook intact → lpb plugs in); added
  record_recovery_len_kv after each eviction. Kept `import collections`
  (LPB _hit_times deque), dropped unused hashlib.
- `mem_cache/hiradix_cache.py`: upstream pool_host imports; LPBStrategy import
  already survived at L61.
- `mem_cache/mamba_radix_cache.py`: EWMA/counter field inits + upstream's
  widened assert (UnifiedMambaTokenToKVPoolAllocator); None-guard before
  upstream's translate_mamba_indices; record_recovery_len_* before upstream's
  free_segment / _free_mamba_value; upstream's insert comment+assert.

## Upstream developments to reconcile at test time
- Upstream VIRTUALIZED mamba slots itself (translate_mamba_indices,
  _free_mamba_value, UnifiedMambaTokenToKVPoolAllocator) — overlaps HiMA's
  arena mamba actuator; check double-translation on the arena path.
- Upstream added RadixLinearAttention (KDA radix caching) and a
  mamba_radix_cache_strategy family incl. "HiMambaRadixCache" naming.
- allocator base got resize(config) — possible future hook for HiMA set_capacity.
- runtime_context / arg_groups are new config plumbing (skill: sglang-runtime-context).

## Remaining (mega-commit)
- memory_pool.py: 21 conflict hunks (our +1021/-101: ReqToTokenPool mamba
  translation, arena-backed MambaPool, HybridReqToTokenPool donation,
  MHATokenToKVPool arena hooks, HybridLinearKVPool). Upstream added ReplaySSM,
  speculative_eagle_topk, quantized-KV in the same regions.
- model_runner.py (UU) + model_runner_kv_cache_mixin.py (DU — deleted upstream;
  our mamba-cap derivation edits must be re-homed, likely mem_cache/kv_cache_builder.py).
- scheduler.py + scheduler_components/invariant_checker.py.

## After mega-commit
- `git rebase --continue` through the remaining 32 commits (mostly our-own files).
- Static checks (py_compile/import), unit tests, 9B smoke (idle GPU, not 3/5/7),
  small t6 A/B, push HiMA-v0516 to origin (rucnyz) only.

## memory_pool.py resolution (mega-commit, DONE — syntax OK)
- Imports: math+os+threading union.
- MambaPool.__init__: upstream params (eagle_topk, replayssm flags, envelope_layout)
  + our max_size; upstream field block + our _alloc_lock + alloc_size=max_size.
  Construction region REBUILT: attrs/decision block (perlayer/arena) hoisted before
  branches; envelope_layout branch (upstream) with max_slots=alloc_size+1 and an
  assert vs arena; else-branch conv at alloc_size+1 + upstream NPU/AMX + our
  arena|perlayer|stacked temporal 3-way; upstream ReplaySSM rings at alloc_size+1;
  per-slot cursors (write_pos/cache_base/is_flush) at alloc_size+1; our
  free_slots/_capped_slots init + upstream mem_usage accounting.
- clear_slots: upstream fused path with list-guard for temporal; our broadcast-zero
  CUDA body (list/stacked); upstream NPU else. (Bug fixed: stacked-else tail was in
  the old common region — re-added temporal_zero expand.)
- copy_from: our list-aware temporal + upstream replayssm cursor resets + fork_from.
- get_cpu_copy/load_cpu_copy: ours list-aware + upstream cursor round-trip.
- PD-transfer: upstream _NON_TRANSFER_STATE_FIELDS + _iter_transfer_state_tensors
  (patched: per-layer-split list yields ONCE as the list itself) + OUR
  state_views-based get_contiguous_buf_infos; get_state_dim_per_tensor upstream with
  list-guard; get_state_slice_outer_counts list-guard.
- KVCache base: upstream layer_shard/post_capture attrs + our _kv_arena decl +
  can_move_kv_cache.
- MHA _create_buffers: upstream dispatcher (quantized/post-capture/normal) with
  self._kv_arena=None at top; _create_buffers_normal carries our arena gate
  (use_arena now ALSO excludes use_hnd and vectorized_5d layouts) + arena build +
  early return; upstream plain body after. Old data_ptrs tail superseded by
  upstream _init_data_ptrs_and_strides.
- HybridLinearKVPool: upstream post_capture/finalize_backing + our _kv_arena
  property forward.
POST-SMOKE CHECKS: arena vs upstream resize()/finalize_backing interplay;
UnifiedMambaTokenToKVPoolAllocator path vs our arena (double-virtualization).

## Final mega-commit pieces (DONE)
- invariant_checker: upstream structure (dcp rounding, leak/msg call) + our
  live_size-aware `total`.
- scheduler.py: budget_agent.tick() kept before upstream's new
  is_disable_overlap_for_batch(batch, last_batch); _apply_forced_tokens moved
  INSIDE `if batch_result.has_sampled_token_ids:` as first line.
- model_runner.py: upstream init_cuda_graphs (capture_cuda_graphs refactor) +
  `self.maybe_warmup_arena_tlb()` appended; warmup gate wrapped as
  maybe_warmup_arena_tlb + _arena_tlb_warmup kept; ours-side old-upstream init
  tail and adjust_hybrid_swa_layers_for_pp dropped (upstream removed both).
- model_runner_kv_cache_mixin.py: DELETION ACCEPTED; all five hooks re-homed
  into mem_cache/kv_cache_configurator.py: (1) _resolve_max_admission_size
  helper, (2) HybridReqToTokenPool(max_size=...), (3) kv_live_migration_enabled
  OR'd into 4 enable_kv_cache_copy gates, (4) T2 placement-bias need_sort,
  (5) arena dynamic-cap max_size for TokenToKVPoolAllocator (page_size==1 site,
  uses sizes.max_total_num_tokens + token_to_kv_pool local).

## Rebase COMPLETE (31 commits on b78d3999b5; c^evict-cache pair skipped as net-zero)
Later-commit resolutions:
- 959461d2e8: common.py took upstream (file gutted upstream); factor logic in
  allocation.py unified onto HybridReqToTokenPool.mamba_slots_per_req (#338).
- 9e3c4fdddf: schedule_policy took HEAD — OUR #339-343 machinery was UPSTREAMED
  verbatim (rem_mamba_slots/_mamba_slots_needed) and evolved (mamba_gap_reserve);
  mamba_radix kept our LPB-loss cumulative fields + wide assert.
- 0c8ea1ccae: bench_one_batch is a deprecation shim now; SSU-init ported to
  python/sglang/benchmark/one_batch.py (import from
  sglang.kernels.ops.mamba.triton_ops.ssu_dispatch) between alloc_memory_pool
  and init_attention_backends.
- f130df4627 + cde49a0541: SKIPPED as an exact net-zero pair (verified).
- e99973a403: LPBStrategy import added alongside upstream's new imports.

## Static checks
- py_compile over every HiMA-touched python file: PASS.
- Core imports from worktree (PYTHONPATH override over the scratch venv):
  server_args, memory_pool, kv_cache_configurator, allocation, radix caches,
  budgeter agent/admitter, arena: PASS.

## Test methodology (in progress)
- dev/interlayer suite run on BOTH trees (rebase@GPU2, old-with-WIP@GPU4);
  acceptance = NO NEW failures vs the old tree (old tree already has stale
  tests, e.g. test_scheduler_hook 16F/9P on both — pre-existing contract drift
  from the no-backlog/fast-path Admitter evolution).
