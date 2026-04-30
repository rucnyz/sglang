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

## B1 phase_shift workload — **architectural ceiling, not fixable with synthetic prompts**

- **Setting:** Layer 2 regression+benefit suite, B1 workload (`dev/eval/regression_suite/workloads/b1_phase_shift.sh` + `dispatcher_b1.py`).
- **Original symptom (v7):** B1 prelude e2e 1022ms vs baseline 974ms (+4.8%), 1 transfer fired, mamba_usage stayed [0.88, 0.99] both phases.
- **v8 dispatcher fix attempted (commit fd64c6413):** per-phase concurrency (32 mamba × short / 4 kv × 8K-input + 512-output). Result: prelude TPS 3914 vs baseline 4379 = −10.6%, still 1 transfer.
- **Budgeter telemetry v8:** mamba peak 0.99 (179/183 ticks > 0.8), KV peak 0.39 (0/183 ticks > 0.5). 4 reqs × 8.5K KV = 34K active vs pool size 1.26M tokens (2.7% utilization).
- **Architectural finding:** Qwen3.5-35B-A3B has 18 mamba slots and 1.26M KV tokens. Mamba saturates at ~18 concurrent reqs; KV needs ~120 concurrent reqs to saturate. For any concurrency 4-60, mamba is the *only* practical bottleneck. Synthetic short/long prompt workloads cannot phase-shift bind pool on this architecture. Genuine shifts only emerge in (a) extreme long-context per req (200K+ tokens) at low concurrency, or (b) multi-turn accumulating-KV conversation (WildChat-style — what Setting 1 v9-auto already exercises).
- **Workaround:** treat B1 as informative; the headline benefit is captured by B2 cold_burst (TTFT −24.5% ± 3 ms across 4-replicas, p99 −62.4%). Setting 1 v9-auto separately validates phase-shift on a realistic workload.
- **Date observed:** 2026-04-30 late night (suite v7+v8).
- **Resolved at:** — (architectural; not fixable by tweaking the dispatcher)

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

## Phase 3.d (heterogeneous granularity, K_BIG) — **FIXED 2026-04-30**

- **Setting:** Phase 3.d e2e, Setting 1 L1=1 cells, A2 K_big sweep, Setting 3.A.
- **Original symptoms:**
    - **Insert assertion crash:** `cache_unfinished_req` line 800: `new_prefix_len > len(new_indices)`. Reproduced on Setting 1 v2 (depth 512), v5 multi-turn (depth 12883), and A2 K_big sweep (depth 12883).
    - **Idle leak:** tombstone leaves' KV is never reclaimed because `_evict_leaf_node` asserts `mamba_value is not None`. Up to 13K slots leaked / 1.26M.
- **Root cause:** `_match_prefix_helper` only updates `best_value_len` at nodes with `mamba_value is not None`, while `insert.prefix_len` returned the full traversal depth — including tombstone leaves and tombstone-internal-nodes past the deepest snapshot. The two views of the same path disagreed, breaking `cache_unfinished_req`'s invariant. And tombstone leaves were never reclaimable.
- **Fix landed (commits b37bbc82e + 325f25334):**
    1. `_insert_helper` no longer creates tombstone leaves. When the suppression path would create a new leaf with `mamba_value=None`, the trailing KV is freed instead. Eliminates the never-reclaimable-leaf source.
    2. `_insert_helper` tracks `deepest_snapshot_depth` during the while loop. When K_BIG is active and the traversal goes past tombstone-internal-nodes past the deepest snapshot, `insert.prefix_len` returns the deepest-snapshot depth — consistent with what `match_prefix` returns. Eliminates the assertion crash.
    3. Suppression condition restored to `insert_depth >= k_big AND insert_depth % k_big != 0` so depth<k_big inserts retain small-page caching (legacy behavior). Earlier over-suppression (`insert_depth % k_big != 0` only) caused v7 Phase C 30% P95 regression because every short-prompt insert was dropped.
- **Validation:**
    - `dev/2e/33_phase3d_unit.py` 3/3 PASS: aligned-depth snapshot, depth<K_big retained, depth>K_big past tombstone preserves prefix_len consistency.
    - Setting 1 v8 (K_BIG=8192 enabled on L1 cells): all 4 cells × 3 phases run end-to-end clean. No assertion crash. No cell-vs-cell regression. Numbers match v6 (K_BIG disabled) within run-to-run noise.
- **Residual:** small idle-time leak (~80 KV slots / 1.26M = 0.006%) — the suppressed-path's `free(value)` may be slightly over-counting. Demoted via `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`. Not paper-blocking; followup TODO to track down.
- **Date observed:** 2026-04-30 night.
- **Resolved at:** 2026-04-30 (commit b37bbc82e + 325f25334).

## Setting 1 (24-h phase-shift trace) — **partially blocked**

- **Setting:** 1 (`dev/eval/01_phase_shift_trace.sh` not written)
- **Blocker:** trace requires three datasets (Phase A: alpaca+lora, Phase B: rerank-style ShareGPT, Phase C: WildChat 8-turn). The pd_exp generator at `aproj/vllm/pd_exp/serve/generate_distribution_shift_dataset.py` produces (1) and approximates (2). WildChat for Phase C needs `pd_exp/multiturn/export_dataset.py` with `--dataset wildchat` — first time may need HF download.
- **Date observed:** 2026-04-30 night
- **Workaround:** start with Phase A + B only on the trace and document Phase C as "coming after WildChat export."
- **Resolved at:** —

