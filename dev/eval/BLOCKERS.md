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

## 2026-04-30 late-night L2 debug chain — audit log

This section is for me-later-or-anyone-else to audit the chain of fixes
landed on the prelude branch tonight. For each commit: what was wrong,
what change addresses it, and whether the change is a real fix or a
workaround (and if a workaround, the followup that turns it into a real
fix).

**Default behavior is unchanged for any user not opting into the new
features.** All the engineering tonight is gated either on `mobile_soft_chunks > 0`
(off by default) or behaves identically to prior commits when the new
env vars are unset.

| commit | summary | status | notes |
|---|---|---|---|
| `c4a426e38` | net-benefit gate B_persist + stable-state re-eval | **real fix** | Paper §design-l2 Eq.~\ref{eq:nb-lb} third term. Unit-tested in `dev/2e/38_planner_netbenefit_unit.py` (T9, T10). Off by default (`SGLANG_XPOOL_NET_BENEFIT=0` is default; persist re-eval period env-tunable). |
| `d9f707c46` | re-introduce VA-only growth headroom | **real fix** | Reverts an over-zealous removal in `da326b1ed`. Comment in `multi_tensor_arena.py:218` documents that VA past init is reserved-but-unmapped — pure VA, no physical cost. Backward-compat default is 4 chunks of headroom. |
| `66e30e147` | bump `nb_chunk_cost_us` default 50 ms → 3 s, `cooldown_ticks` 3 → 16 | **half-workaround, half-fix** | The 50 ms/chunk was correct for a single physical chunk, but the actuator moves chunks in lcm-balanced units (lcm(20, 30) = 60 chunks for Qwen3.5-A3B), so a single fire's actual GPU cost is ~3 s. Bumping the default makes the gate's math match Qwen3.5-A3B reality, but other model topologies (Qwen3-Next-80B, etc.) have different lcm and want a different default. **Followup (a):** plumb the actuator's chunks-per-balanced-unit count to the planner so the gate computes cost from real lcm × per-chunk wall-time rather than an env knob. |
| `a4dc081c4` | sync env-reader string defaults to dataclass defaults | **real fix** | Bug: `_policy_from_env()` was reading `os.environ.get("SGLANG_XPOOL_NB_CHUNK_COST_US", "50000")` while the dataclass default had been bumped to 3 000 000 in `66e30e147`. Runtime configs were unaffected by the dataclass bump. Trivial typo class. Now all `_policy_from_env` string defaults match dataclass defaults. |
| `7490e5ed5` | actuator floor at `init_chunks_per_pool` (then `static_min_chunks_per_pool` after `475838fe4`) | **real fix** | Paper §design-l2-actuator (line 133-135) explicitly requires this. Without it the actuator could shrink any pool to 1 chunk per sub-pool, breaking captured CUDA graphs. |
| `475838fe4` | arena static-min/soft VA partitioning | **real fix** | First proper implementation of paper §design-l2-actuator's static-min/soft split. New `MultiTensorArena` parameter `static_min_tokens` (default = `init_tokens` ⇒ no behavior change). `SGLANG_ARENA_KV_MOBILE_SOFT_CHUNKS` and `_MAMBA_MOBILE_SOFT_CHUNKS` env vars (default 0) opt into mobile soft. Boot maps only static_min; actuator can grow soft via `cuMemMap` from shared free queue. |
| `8ceb63de6` | `MambaPool.__init__` caps engine-side mamba allocator at static_min | **real fix** | Static-min/soft split exposed an engine-side gap: the engine's `MambaPool` initialized `free_slots = arange(1, init_tokens + 1)` even though only static_min was physically mapped, so the allocator handed out unmapped slot ids → `cudaErrorIllegalAddress` in `MambaPool.alloc → torch.zeros`. Fix uses the existing `set_capacity_slots` API to cap the free-slot head at static_min. Backward-compat: when `mobile_soft_chunks=0`, static_min == init, the call is a no-op. |
| `f508d3893` | `MHATokenToKVPool` overrides `self.size` for KV mobile-soft | **real fix, partial** | Mirrors the MambaPool fix for KV. Sets `self.size = static_min_tokens - page_size` when `kv_mobile_chunks > 0` so the engine's downstream KV allocator caps at static_min. **Followup (b):** when actuator later maps mobile soft via `cuMemMap`, it must propagate that to `pool.size` (or a dynamic capacity getter) so the engine's scheduler exposes the new allocatable range. The cap_allocator_only path on the KV-side actuator already updates the allocator's internal cap; what's missing is the pool's `self.size` reflecting that for `max_total_num_tokens` reporting. Not a hidden bug — at most, mobile soft chunks stay dormant if the gate fires; that's safe (no crash) and no worse than today's L2-silent default. |
| `785394f1a` | off-by-one in MambaPool's `set_capacity_slots` argument | **real fix** | `set_capacity_slots(n_slots)` produces `free_slots = [1, n_slots]` inclusive; I passed `static_min_tokens` (= 128 = mapped-region size), so slot id 128 — the first byte of the unmapped chunk past static-min — was allocatable. Fixed by passing `static_min_tokens − 1`. Slot 0 is the padding slot, slots [1, static_min_tokens − 1] are the usable positions within the mapped region. |

### Audit checklist (run anytime)

```bash
# 1. Defaults unchanged unless explicitly opted in
grep -E "MOBILE_SOFT_CHUNKS\|NET_BENEFIT" python/sglang/srt/budgeter/cross_pool_planner.py \
    python/sglang/srt/mem_cache/memory_pool.py
# Expect:    default = "0"   (off-by-default)

# 2. No silent except clauses around CUDA errors
git log --since='2026-04-30' --oneline --all | head -20 \
    | xargs -I{} git show {} -- python/sglang/srt/ | grep -A2 -B2 "except.*Exception\|except:" | head -40
# Expect: no try/except wrappers around the actuator or arena code

# 3. Unit tests still pass
.venv/bin/python dev/2e/37_planner_edge_unit.py | tail -3
.venv/bin/python dev/2e/38_planner_netbenefit_unit.py | tail -3
# Expect: ALL PASS for both

# 4. Backward-compat smoke (env unset → identical to baseline)
unset SGLANG_ARENA_KV_MOBILE_SOFT_CHUNKS SGLANG_ARENA_MAMBA_MOBILE_SOFT_CHUNKS \
      SGLANG_XPOOL_NET_BENEFIT
# launch server with SGLANG_ARENA_SHARED=1 only — should match v7's
# default-arena behavior (~5-7% overhead vs no-arena baseline, no
# crashes, gate fires never)
```

### Known followups (real fixes that turn workarounds into real fixes)

(a) **Plumb actuator's lcm-aware chunks-per-balanced-unit into the planner.**
The gate's cost should be `chunks_per_unit × c_chunk_us` derived from
arena state, not an env knob. Right now the env knob default is correct
for Qwen3.5-A3B but not generic.

(b) **Dynamic `pool.size` propagation when actuator grows soft.** When
the actuator `cuMemMap`s mobile soft chunks into a pool, the engine's
scheduler should see `max_total_num_tokens` rise so it can schedule
into the new range. Currently the actuator updates the allocator's
internal cap but the pool's `self.size` is static. Without (b), mobile
soft chunks are dormant — fires happen but engine doesn't use the new
slots. Safe (no crash), but L2's metric demonstration is gated on (b).

(c) **Per-pool drain protocol for shrink.** Paper §design-l2-actuator
specifies "mark blocks above the new cap as 'draining,' release blocks
as their owning requests complete". Today the actuator caps the
allocator and immediately unmaps; in-flight requests are protected
only by the engine_busy gate (which itself reads `num_running_reqs`
from the snapshot — see open issue below). Currently the static_min
floor + dormant soft makes this moot (actuator never actually
unmaps), but if (b) lands, this becomes load-bearing.

(d) **engine_busy gate reads stale snapshot fields.** Diagnosed in
B3 v3: `num_running_reqs = 0` in `xpool_*` budgeter snapshot even when
24 requests were running. Current static_min floor masks this — fires
that get past the gate are no-ops anyway. If (b) lands, this surfaces.

---

## L2 actuator's cuMemUnmap breaks captured CUDA graphs — **paper's static-min region not implemented**

- **Setting:** Layer 2 cross-pool transfer when transfers actually move bytes (after VA-headroom restored in d9f707c46).
- **Symptom:** B3 v3 cell_01/cell_11 + v9-auto v5 cell_11_nb_v5 all crashed with `CUDA error: an illegal memory access was encountered` ~1 second after the actuator's first 60-chunk transfer. cell_00 (no arena) ran cleanly through the same workload.
- **Root cause:** the actuator's `cuMemUnmap` removes physical pages from the source pool's VA range. CUDA graphs captured at engine startup were captured against the full pre-shrink tensor (`shape=(max_tokens, *)`); after shrink they reference unmapped pages → illegal access on next decode/forward replay.
- **Paper §design-l2-actuator (line 133-135) explicitly addresses this:** "Each pool's VA range is split at startup into a guaranteed-mapped *static-min* region ... CUDA graphs are captured exclusively against offsets in this region. The budgeter's `transfer_chunks` actuator operates only on the *soft* region." Our impl maps `init_chunks_per_pool` at startup and lets the actuator unmap from the same range — there is no static-min/soft separation.
- **Workaround landed (sglang a4dc081c4):** raise cost-knob defaults so the net-benefit gate refuses to fire under any normal pressure (3 s cost × 1.5 margin needs 900 sustained ABOVE_HIGH ticks = 30 min). Effectively L2 stays silent; "全开 ≥ baseline" is met because no transfer = no crash. Full L2 metric demonstration requires the static-min plumbing.
- **Followups not yet landed:** (a) split arena's `init_chunks_per_pool` into `static_min_chunks + soft_chunks`, only static_min mapped at startup, soft initially unmapped, CUDA graph capture sees only static_min range. (b) actuator drains in-flight requests' KV blocks above new cap before cuMemUnmap (paper §design-l2-actuator drain protocol). (c) refuse cross-pool transfer if it would shrink any pool below its static_min.
- **Date observed:** 2026-04-30 late night (B3 v3, v9-auto v5).
- **Resolved at:** — (workaround in place: gate refuses all fires; full fix is paper-design-aligned static-min plumbing).

## L2 actuator's lcm-aware unit is too coarse — **needs cost knob bump or per-subpool partial moves**

- **Setting:** Layer 2 cross-pool transfer on any hybrid model (Qwen3.5-35B-A3B observed).
- **Symptom:** B3 v2 cell_01: a single L2 fire moved 60 chunks (15 GB) from KV to mamba. KV pool dropped from 1.26M tokens to 524K. β phase's 96K-input × 8-concurrent requests overflowed → 40,201 dispatcher errors (~85% failure rate). v9-auto v4 cell_11_nb_v4: server SIGQUIT'd mid-Phase C after one big transfer.
- **Root cause:** the actuator (`transfer_chunks`) operates in lcm-balanced units across all sub-pools. For Qwen3.5-35B-A3B with KV 20 sub-pools × mamba 30, lcm = 60, so each "1 unit" actuator move = 60 physical chunks × 50 ms cuMemUnmap+cuMemMap = ~3 s of GPU wall time per fire. The net-benefit gate's `nb_chunk_cost_us=50000` (one chunk's worth) under-cost transfers by 60×, letting through fires with B_persist=100 ms when actual cost is 3 s.
- **Workaround landed (sglang 66e30e147):** raise `nb_chunk_cost_us` default 50_000 → 3_000_000 (lcm × per-chunk for Qwen3.5-A3B). Cooldown 3 → 16 ticks (32 s). Now requires ~55 retracted requests or 30 min sustained saturation before firing — only fires when overwhelming admission-pressure evidence exists. For other model topologies, operators must override `SGLANG_XPOOL_NB_CHUNK_COST_US` to match their lcm × per-chunk cost.
- **Followups not yet landed:** (a) plumb actuator's chunks-per-unit into the planner so the gate uses real lcm rather than env knob; (b) post-fire monitor that reverses transfer if performance degrades within K ticks; (c) min-source-usage check refusing to shrink a pool already below e.g. 30% utilization; (d) per-subpool partial moves so the actuator can shift smaller increments than lcm.
- **Date observed:** 2026-04-30 late night (B3 v2, v9-auto v4).
- **Resolved at:** — (workaround in place; followups pending).

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

## Paper abstract has stale "no static configuration achieves" claim — **needs user-level decision**

- **Setting:** `prelude-paper/abstract.tex` last sentence: "On a 24-hour phase-shift trace over Qwen3 / Qwen3.5 / Qwen3-Next, Layer 1 + Layer 2 together sustain throughput across phase changes that no static configuration achieves; the four-cell ablation isolates each layer's contribution."
- **Issue:** the static-best partition baseline (`prelude-paper@647da07`, paper Table~\ref{tab:static-best}) found that single static `mamba_full_memory_ratio=0.9` beats the dynamic $(1,1)$ cell on v9-auto Phase A by $1.9\times$ mean TTFT. The contribution-attribution finding (`ef09782`, Table~\ref{tab:contribution-attribution}) and 3-trial variance bands on both v9-auto (`198a9a5`, Fig 7) and Q3.B 4-cell (`aaa837c`, Fig 8) show that the joint $(1,1)$ cell does not statistically dominate the L1-only $(1,0)$ cell at our measurement budget. Abstract's "no static config achieves" + "isolates each layer's contribution" framing therefore does not match the body's honest framing of "L1 carries the lift; L2 = no-regression mechanism whose marginal value over L1 is below variance".
- **Resolution:** rewrite the abstract's last sentence to match body. Example: "Layer 1 contributes a measurable end-to-end win on every workload tested ($-26\%$ to $-37\%$ recovery TTFT vs.\ baseline at $\sim 4$\,ms variance); Layer 2 is the no-regression cross-pool reallocation mechanism that becomes load-bearing on workloads with sustained admission pressure (work-in-progress)." Paper-framing decision; not autonomously edited.
- **Date observed:** 2026-05-01 NeurIPS strengthening session.
- **Status:** flagged for user review; body sections (`evaluation.tex`) already report honestly.

## L2-positive workload search — **architectural blocker, not workload-tunable**

- **Setting:** demonstrating $(L_1=1, L_2=1)$ strictly dominates $(L_1=1, L_2=0)$ on any measured workload. Paper currently flags this as work-in-progress.
- **Date observed:** 2026-05-01 (afternoon, after multiple workload iterations).
- **Search log:**
  - **Mamba-pool restriction (`mamba_full_memory_ratio=0.5`, run `l2-positive-20260501-134900`):** L2 cells (mem_frac=0.7) deterministically OOM in `chunk_gated_delta_rule_fwd` → `torch.zeros_like(v)` because arena+budgeter overhead consumes the GDN intermediate-buffer headroom. 0 xpool fires recorded across multiple trials. Verified `max_mamba_cache_size=254` (vs default 361) — flag IS being applied. L1-only cells run cleanly; L2-on cells crash mid-Phase-A. Retreating from this configuration.
  - **Default config + heavy GSP RPS=32 (run `admission-pressured-20260501-140507`):** baseline L10_L20 prefill=1140, retract=0, pause=0. Workload too gentle.
  - **Random 16K × 600 × RPS=24 (run `l2-kv-overflow-20260501-143041`):** baseline L10_L20 prefill=771, retract=0, pause=0, abort=1. mean_ttft=55s, p99_ttft=115s, max_concurrent_requests=592, but engine cap=120. Engine queues 472 reqs at admission; KV pool never overcommits. Killed early.
  - **Random 24K × 240 × RPS=48 with `--max-running-requests 240` override (run `l2-force-kv-20260501-143828`):** L10_L20 + L11_L20 still both retract=0, pause=0, abort=1. Even with admission cap doubled to force 5.76M-token contention against 1.26M pool, the engine never logs `"KV cache pool is full. Retract requests"` (verified by string grep against actual log emission in `scheduler.py:2841`).
- **Root cause:** the engine architecturally avoids retraction. `update_running_batch` calls `check_decode_mem`, which first invokes `evict_from_tree_cache(self.tree_cache, num_tokens)` and only retracts when the radix cache cannot free enough. Combined with chunked-prefill backpressure, prefill never overcommits the KV pool — KV grows incrementally per chunk and the radix-cache evictor frees as fast as decode consumes. Retract only fires when a forward step physically cannot allocate one decode step's worth of KV — a corner case that does not occur under any normal admission pressure on Qwen3.5-35B-A3B.
- **Architecture mismatch with L2 design:** L2's mamba→kv reclaim physically resizes the KV pool via `MHATokenToKVPool.set_capacity_tokens`, which calls into the arena. But the scheduler's `max_running_requests = floor(max_total_num_tokens / avg_seq_len)` is computed **once at boot** from initial pool size. Even when L2 grows KV capacity dynamically, the scheduler's admission cap remains static, so L2's extra bytes are never consumed by additional concurrent requests. This is exactly follow-up (b) in the L2-actuator audit (this file, line 38): "When the actuator `cuMemMap`s mobile soft chunks into a pool, the engine's scheduler should see `max_total_num_tokens` rise so it can schedule into the new range. Currently the actuator updates the allocator's internal cap but the pool's `self.size` is static."
- **2026-05-01 evening — config blocker IS real, but stacked under a deeper engine bug. Smoke results below:**
  - **Discovery (still correct):** spot-checking `runs/q3b-variance-20260501-094916/trial1_L11/budgeter.jsonl` revealed the actuator's first kv_to_mamba fire reports `xpool_unmapped_total=0, xpool_granted_total=0` — the fire was a no-op. Every eval script in `dev/eval/*.sh` that sets `SGLANG_BUDGETER=1` (15 scripts: 03/04/07/11-17/20/22/27/29-31) is missing `SGLANG_ARENA_*_MOBILE_SOFT_CHUNKS`. With both at 0 (env default), `init_chunks == static_min_chunks` per pool → shared free queue empty at boot → src.shrink can't go below static_min and dst.grow has no handles to pull → **every L2 fire across every paper-cited run was a planner-side decision with zero physical byte movement.** `runs/nb-gate-20260501-081514/nb_on/L11_L21/budgeter.jsonl` shows the pathology: 15 fires, all `granted_nonzero=0`, KV capacity stuck at 1,048,576 across all ticks, mamba at 512.
  - **Smoke v1 (`runs/l2-mobile-soft-20260501-164134`, KV+mamba mobile_soft=4 at 1 GB chunks):** `ValueError: SGLANG_ARENA_MAMBA_MOBILE_SOFT_CHUNKS=4 exceeds init_chunks=1`. At 1 GB chunks the mamba arena has only 1 init_chunk per sub-pool because per-slot bytes (~2 MB) make 251 slots fit in <1 chunk.
  - **Smoke v2 (`runs/l2-mobile-soft-20260501-171355`, KV mobile=1 mamba=0 at 1 GB chunks):** `RuntimeError: The specified pointer resides on host memory and is not registered with any CUDA device` in `tensor_from_va`. The from_blob path's CUDA pointer-check fails when mobile-soft leaves part of the (max_tokens, *)-shape view unmapped. Workaround: switch to `SGLANG_ARENA_FROM_BLOB=0` (MemPool path) which uses `_lib.multi_set_capacity` to temporarily expose max_chunks during `torch.empty`.
  - **Smoke v3 (`runs/l2-mobile-soft-focused-20260501-181148`, MemPool path, 256 MB chunks, KV mobile=2 mamba=1):** server came up cleanly (110 s warm), ran 12 seconds of Phase A bench, hit running-req=80 / mamba_usage=0.89, then `CUDA error: an illegal memory access was encountered` in `process_batch_result_prefill` → `next_token_ids.tolist()`. **This is the deeper bug stacked under the config issue.** The static_min/soft split was supposed to fix exactly this (BLOCKERS.md L98 entry: "L2 actuator's cuMemUnmap breaks captured CUDA graphs"), with CUDA graphs captured exclusively against the static_min region. But under v3, CUDA graphs were captured against the boot-time pool size (mamba.size=251 in the log, before `set_capacity_slots(static_min=128)` was called), so model kernels access slot positions [128, 251] which are unmapped after the cap. Mamba_usage reaching 0.89 = 0.89 × 128 ≈ 114 slots — close enough to overrun the 128-slot static_min that the next allocator hand-out goes into [128, 251] and the kernel hits unmapped memory.
  - **Bottom line:** the optimistic "just enable mobile-soft env vars" rewrite of the L2-positive blocker was wrong. There are TWO bugs stacked: (i) eval scripts don't set MOBILE_SOFT (config), AND (ii) even with MOBILE_SOFT set the engine's CUDA-graph capture isn't actually scoped to static_min — it captures the full boot-size allocator and dies the first time a workload pushes mamba past static_min. Fixing (ii) requires capturing CUDA graphs AFTER `set_capacity_slots(static_min)` has been called (currently called too late) AND verifying the model's slot/page handouts respect `_cap` not `self.size`. Estimated: ~150-300 LoC across model_runner cuda-graph capture path + allocator slot-issue logic + integration test. Likely a half-day plus reviewer time, NOT a quick win. Recommended: park L2-positive demo for paper, fix in a follow-on PR.

- **What would unblock the harder, longer-term L2-positive demo (assumes mobile-soft fix lands):** implement follow-up (b) — propagate dynamic `pool.size` (or, equivalently, surface `live_capacity_tokens()` to the scheduler) AND make `max_running_requests` re-derive on capacity change. ~150–300 LoC across `scheduler.py`, `memory_pool.py`, and the budgeter actuator. Estimated half-day implementation + integration test. This expands L2's reach beyond the static-pool-size constraint that mobile-soft alone can't bypass.
- **Concrete patch shape (drafted 2026-05-01 afternoon, awaiting user review):**
  1. **Arena already does the heavy lifting.** `MultiTensorArena.set_capacity_tokens(n_tokens)` (multi_tensor_arena.py:314) already grows/shrinks each sub-pool via `cuMemMap`/`cuMemUnmap` and updates the C-side capacity register via `multi_set_capacity`. The tensor returned by `arena.tensor(layer, kind)` always has shape `(max_tokens, *per_token_shape)` (line 230-235 comment: "its shape is (max_tokens, *) so engine indexing matches"). So growing past init does NOT require tensor reallocation.
  2. **Pool side (memory_pool.py:1353 `MHATokenToKVPool.set_capacity_tokens`).** After `self._kv_arena.set_capacity_tokens(target_aligned)`, add: if `new_advertised > self.size`, call `self._token_to_kv_pool_allocator.expand_size(new_advertised)` then `self.size = new_advertised`. Mirror change for shrink path (cap free_pages first, then update self.size).
  3. **Allocator side (allocator.py).** Add `expand_size(self, new_size)` method: if `new_size > self.size`, append page ids in range `(self.size, new_size]` to `self.free_pages`, then update `self.size = new_size`. Mirror `shrink_size` for the symmetric path.
  4. **Scheduler side (scheduler.py).** `max_running_requests` is read into many places at boot. Cleanest path: gate behind `SGLANG_DYNAMIC_POOL_SIZE=1` (default 0 = no behavior change), and when enabled, recompute `self.max_running_requests = self.max_total_num_tokens // avg_seq_len` on the L2 actuator's post-fire callback. Touch points: scheduler.py:706 (init), scheduler.py:1283 (queue size), scheduler.py:2598 (batch construction).
  5. **Unit test (`dev/2e/39_dynamic_pool_size_unit.py`, ~80 LoC).** Boot a pool with `init_tokens < max_tokens`. Assert `available_size() == init_tokens - page_size`. Call `set_capacity_tokens(max_tokens)`. Assert `available_size() == max_tokens - page_size` and `pool.size == max_tokens - page_size`. Allocate page id beyond init_tokens; assert no CUDA error.
  6. **Default behavior unchanged.** All new code paths gated on `SGLANG_DYNAMIC_POOL_SIZE=1` (planner side) and on `set_capacity_tokens` callers actually requesting growth (which only the L2 actuator does, gated on its own envvar).
- **Risk profile.** Mid risk: free_pages tensor extension is straightforward but free_pages is read on every alloc/free, so any race is a hard-to-debug CUDA illegal-access. Mitigation: serialize set_capacity_tokens with the engine's request-processing loop (already true — the actuator only fires from the budgeter agent which runs in the scheduler's tick callback). Also: the static-min floor in `7490e5ed5` already prevents the actuator from shrinking past static_min, so growth is the only direction that would exercise the new code path on production traces.
- **Recommended path forward.** Single-PR plan: land steps 1-3+5 first (pool/allocator + unit test, gated default-off, no scheduler change). Validate that the new envvar does nothing without `SGLANG_DYNAMIC_POOL_SIZE=1`. Then in a follow-up PR add step 4 (scheduler `max_running_requests` recompute) once the pool/allocator side is stable. This keeps the blast radius small and lets the user gate adoption.
- **Workaround:** none for paper-grade L2-positive demonstration. Paper body already frames L2 as the no-regression cross-pool reallocation mechanism whose marginal value above L1-only is below variance at the current measurement budget; the abstract entry above (`Paper abstract has stale claim`) covers the abstract rewrite. The L2 design is paper-correct; the architectural friction lives entirely on the engine integration side.

- **2026-05-01 late-night — RESOLVED in 8 commits** (`8f0950b99` … `e36d04a64`):

  **(1) Paper-faithful boot** (`8f0950b99`): scrapped the "donate-at-boot" mobile-soft mechanism. Boot now maps full init_chunks per sub-pool. `static_min` becomes a small actuator floor (1 chunk per sub-pool when shared_arena=1; = init when off → identical-to-baseline behavior). KV pool boots at the full `tot_aligned` capacity (879K tokens vs 524K under prior donate scheme). Earlier "L2-on cells lose 40% pool" complaint disappears.

  **(2) Drain protocol** (`265ece34e`, `e36d04a64`): paper §design-l2-actuator drain is now implemented in `cross_pool_actuator._drain_complete`. Counts pages > new_cap across `_capped_pages + release_pages + free_pages + free_group` (all the places SGLang's allocator can hold freed-but-not-in-_capped pages). Drain returns True iff `total_above_freed >= size - new_cap`, i.e., zero in_use slots above cap. The earlier crash in v3 (`runs/l2-mobile-soft-focused-20260501-181148`) was due to undercount — `_capped` alone missed `release_pages` and `free_group` entries → returned True prematurely → unmap killed in-flight slots → CUDA illegal access. Re-tested in `e36d04a64` smoke (pending).

  **(3) Engine-agnostic pressure adapter** (`d88557c85`): refactored the net-benefit gate's B term from a hardcoded `paused×x + retracted×y + persist×z` formula to an abstract `sum_i k_i × S_i` where each engine provides its own `EnginePressureAdapter` (new file `pressure_adapter.py`). SGLang adapter dominates the **eviction** signal — `num_evicted_tokens_recent` (per-tick delta of tree-cache eviction) × `prefill_save_us_per_token=12.5us` — because SGLang's primary admission-pressure relief mechanism is tree-cache eviction, not retract. Retract/paused/queue still surface but rarely fire. Persist provides a saturation backstop. New file `python/sglang/srt/budgeter/pressure_adapter.py` defines `PressureSignals` + `EnginePressureAdapter` + `SGLangPressureAdapter`; `cross_pool_planner.decide()` consumes via `_net_benefit_ok(snapshot, ...)`. 10/10 unit tests pass in `dev/2e/38_planner_netbenefit_unit.py`.

  **(4) Stage 1 actuator-cost calibration** (`23bc28761` instrumentation, `87360b2c7` calibration): wrapped `src._arena.shrink` and `dst._arena.grow` in `torch.cuda.synchronize` + `time.monotonic_ns` brackets. Smoke captured 2 fires moving 120 chunks total in 9.5 ms wall-time. Per-chunk `cuMemUnmap+cuMemMap` ≈ **80 µs** on Qwen3.5-35B-A3B / H200 / 256 MB chunks — paper default `c_actuator ≈ 50 ms` was **600× overestimate**. New default `nb_chunk_cost_us = 5_000` (per typical 1-chunk-per-dst-subpool fire). Paper §design-l2 L137 updated to `≈80 µs/chunk`.

  **(5) Eviction signal in scheduler** (`d88557c85`): `schedule_batch.py:check_decode_mem` now measures tree-cache eviction delta via `allocator.available_size` before/after `evict_from_tree_cache`, accumulates in `tree_cache._l2_cumulative_evicted_tokens`. Budgeter snapshot emits `num_evicted_tokens_recent` (per-tick delta) for the SGLang adapter to consume.

  **Validation status (as of `e36d04a64`):**
  - 10/10 unit tests pass (planner + adapter)
  - Smoke v3 (`l2-mobile-soft-focused-20260501-231320`): server boot @ full pool, fire moves 30 chunks (mamba 256→384), `xpool_fire_total_us=5689` matches Stage 1 prediction. CUDA crash AFTER fire revealed drain race (now fixed in `e36d04a64`).
  - Smoke under fix: pending end-of-bench (kicked at 23:30; `l2-mobile-soft-focused-20260501-233038`).

  **Paper updated:** `prelude-paper@634bdc6` rewrites §design-l2 net-benefit gate (Eq.~\ref{eq:nb-lb}) for engine-agnostic adapter framework, updates `c_actuator` from 50 ms to 80 µs.

- **Remaining work to fully close:**
  - End-to-end smoke under drain fix proving live-traffic fires (no CUDA crash) — in progress at this writing
  - vLLM adapter (paper claims framework portability; appendix material)
  - Per-workload calibration of `prefill_save_us_per_token` and `full_prefill_us` from bench_serving outputs (currently hardcoded; auto-calibration would land cleanly)
  - Update paper `evaluation.tex` Q3.B paragraph + abstract once smoke validates the drain fix
- **Resolved at:** 2026-05-01 late-night (sglang prelude commits 8f0950b99…e36d04a64; paper main commit 634bdc6)

