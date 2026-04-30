# Eval blockers — append-only log

When a setting from `SETTINGS.md` can't run, document the blocker here and move on.

## Format

Each entry:
- **Setting:** which one
- **Blocker:** what's preventing it
- **Date observed:** when
- **Workaround:** if any
- **Resolved at:** if/when

---

## Path-axis dispatcher (Settings 5.A, 5.B, A6) — **BLOCKED**

- **Setting:** 5.A (`dev/eval/09_path_dense.sh` not written), 5.B, A6
- **Blocker:** path-axis dispatcher itself is not implemented. Paper §4.5 calls for a per-batch dispatcher that detects `K≤1` batches and routes them through a prefill-only fast path that bypasses paged-attention and DeltaNet slot pool. Estimated ~600 LoC in scheduler + model-runner. Substantial integration risk; needs user discussion before autonomous start.
- **Date observed:** 2026-04-30 night
- **Workaround:** none. Setting 5 cannot run until dispatcher exists.
- **Resolved at:** —

## Setting 2.2 (KV↔LoRA sweep) — **fixed and re-running 2026-04-30**

- **Setting:** 2.2 (`dev/eval/05_sweep_lora.sh`)
- **Original issue:** `--lora-name "lora_0,lora_1,..."` (comma-separated string) was sent to the server as ONE adapter name. The server logged `Got LoRA adapter that has never been loaded: lora_0,lora_1,lora_2,...,lora_31`. Bench reported `Mean TTFT: 0.00ms` because every request errored out fast.
- **Fix:** SGLang `bench_serving` `--lora-name` uses `nargs="*"` (space-separated, not comma). Changed the script to `--lora-name lora_0 lora_1 lora_2 ... lora_31`.
- **Status:** re-running on GPU 2.
- **Date observed / fixed:** 2026-04-30 night.
- **Resolved at:** 2026-04-30, after seeing server-side `Got LoRA adapter that has never been loaded` errors.

## Phase 3.d (heterogeneous granularity, K_BIG=8192) — **partially fixed 2026-04-30**

- **Setting:** Phase 3.d e2e and any Setting-1 cell with L1=1 (which sets SGLANG_K_BIG=8192).
- **Symptom 1 (crash):** Setting-1 v2 attempt: `cache_unfinished_req` asserts `new_prefix_len=512, len(new_indices)=0`. The K_BIG suppression created tombstone leaves at depth 512 (NOT past the 8192 chunked-prefill boundary) and `_match_prefix_helper` returns 0 indices for tombstone-only chains.
- **Symptom 2 (warning):** strict mem leak detector reports 7 slots / 1.26M (0.0006%) leaked at idle. tombstone-leaf nodes' KV cannot be evicted because `_evict_leaf_node` asserts `mamba_value is not None` — full LRU never reclaims them.
- **Root cause:** `MambaRadixCache.insert` set `mamba_value=None` on every non-aligned depth, including those *below* the first big-page boundary. With no depth-K_big ancestor in the chain, match_prefix returns 0; with the new node being a tombstone leaf, evict_full skips it.
- **Fix landed:** `mamba_radix_cache.py` line ~582 — only suppress when `insert_depth >= k_big AND insert_depth % k_big != 0`. Inserts shorter than K_big always carry their snapshot (no tombstone leaf created). Setting-1 v3 launched with the fix.
- **Outstanding:** even with the fix, deep tombstone leaves can still arise when an insert at depth 9000 (with k_big=8192) is the FIRST request to land beyond depth 8192 on this branch. Need to audit `_evict_leaf_node` to handle tombstone-leaf eviction OR audit `_iteratively_delete_tombstone_leaf` to walk down to handle tombstone-leaves with no children. Workaround: keep `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` for now.
- **Date observed:** 2026-04-30 night.
- **Resolved at:** partial fix landed 2026-04-30.

## Setting 1 (24-h phase-shift trace) — **partially blocked**

- **Setting:** 1 (`dev/eval/01_phase_shift_trace.sh` not written)
- **Blocker:** trace requires three datasets (Phase A: alpaca+lora, Phase B: rerank-style ShareGPT, Phase C: WildChat 8-turn). The pd_exp generator at `aproj/vllm/pd_exp/serve/generate_distribution_shift_dataset.py` produces (1) and approximates (2). WildChat for Phase C needs `pd_exp/multiturn/export_dataset.py` with `--dataset wildchat` — first time may need HF download.
- **Date observed:** 2026-04-30 night
- **Workaround:** start with Phase A + B only on the trace and document Phase C as "coming after WildChat export."
- **Resolved at:** —

