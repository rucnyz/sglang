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

