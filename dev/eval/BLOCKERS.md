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

## Phase 3.d (heterogeneous granularity, K_BIG) — **DISABLED in eval scripts as of 2026-04-30**

- **Setting:** Phase 3.d e2e (`dev/2e/34_phase3d_e2e.sh`), Setting 1 L1=1 cells, A2 K_big sweep, Setting 3.A.
- **Symptom A (insert assertion crash):** `cache_unfinished_req` line 800: `AssertionError: new_prefix_len=N, len(new_indices)=M` with `N > M`. Reproduces on **any** workload that crosses the 8192-token chunked-prefill boundary. Examples:
    - Setting-1 v2 (short prompts): `new_prefix_len=512, len(new_indices)=0` — fix landed (insert_depth >= k_big).
    - Setting-1 v5 multi-turn (long context): `new_prefix_len=12883, len(new_indices)=8192` — partial fix did NOT cover this.
    - A2 K_big sweep on GSP (12K system prompt): same crash signature at depth 12883.
- **Symptom B (idle leak):** even when bench survives, the `_evict_leaf_node` invariant (`mamba_value is not None`) means tombstone leaves' KV is never reclaimed. Idle-time leak detector reports 7-13K slots out of 1.26M.
- **Root cause:** `_match_prefix_helper` walks the tree and updates `best_value_len` ONLY when visiting nodes whose `mamba_value` is not None (line 1150). When the deepest matched ancestor is a snapshot at depth 8192 and the request's path beyond that goes through a tombstone-only chain (no further snapshots), `match_prefix` returns indices up to depth 8192 but `insert.prefix_len` returns the full traversal depth. The two-call invariant `match_prefix(after-insert) ≥ insert.prefix_len` breaks.
- **Real fix needed:** rework `_match_prefix_helper` to track the full traversal-end depth, not just the deepest-snapshot depth, OR make `insert.prefix_len` return the deepest-snapshot depth (consistent with match's view), OR redesign the tombstone-leaf semantics so they're never returned as the leaf of a match.
- **Workaround:** **K_BIG is disabled in all eval scripts.** `dev/eval/07_phase_shift_trace.sh` L1=1 cells now set only `SGLANG_HPB_LRU=1` (no `SGLANG_K_BIG`); `08_A2_kbig_sweep.sh` is an open question (the sweep script requires K_BIG to be functional, so the sweep is effectively shelved until the fix lands). Layer 1's HPB-LRU contribution is reproducible in isolation in Q3.D (Table~tab:hpb-gsp).
- **Impact on paper:** Layer 1's claimed *heterogeneous granularity* benefit cannot be reproduced in this eval cycle. The HPB LRU half of Layer 1 IS reproducible. Paper §6.2 should report only the HPB-driven Layer 1 contribution (Phase B P99 stable, Phase C P95 -16% via Layer 2) and acknowledge K_BIG as future work.
- **Date observed:** 2026-04-30.
- **Resolved at:** —

## Setting 1 (24-h phase-shift trace) — **partially blocked**

- **Setting:** 1 (`dev/eval/01_phase_shift_trace.sh` not written)
- **Blocker:** trace requires three datasets (Phase A: alpaca+lora, Phase B: rerank-style ShareGPT, Phase C: WildChat 8-turn). The pd_exp generator at `aproj/vllm/pd_exp/serve/generate_distribution_shift_dataset.py` produces (1) and approximates (2). WildChat for Phase C needs `pd_exp/multiturn/export_dataset.py` with `--dataset wildchat` — first time may need HF download.
- **Date observed:** 2026-04-30 night
- **Workaround:** start with Phase A + B only on the trace and document Phase C as "coming after WildChat export."
- **Resolved at:** —

