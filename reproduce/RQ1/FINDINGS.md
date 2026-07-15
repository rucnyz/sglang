# RQ1 Official Results

Date: 2026-06-24.
Model: Qwen/Qwen3.5-9B, GPU: H200 (GPU 7), sglang branch HiMA.
Harness: agentreplay token-exact replay, N=3 fresh boots per arm.
Base arm: LRU eviction, no HiMA.
Sys arm: LPB eviction (default 60s window) + HiMA (Budgeter + Admitter, async fire, cap floor).

## Current-build reproduction (2026-07-06)

Re-run of all three cases on the post-rebase build (upstream rebase #335, Budgeter
simplified to PaybackPlanner, swarm fixes #339-344) via `run_official_case123.sh`
(now Case3 = t6_v2 @ conc 128, matching FINDINGS; run_arm waits for the prior
server's GPU memory to free between reps). The win holds:

| Case | trace @ conc | base tps (N=3) | sys tps (N=3) | dTPS | dTTFT mean | err |
|------|--------------|----------------|---------------|------|------------|-----|
| Case1 | t6_v2 @ 64  | 444.3±0.9 [445.5,443.5,443.8] | 468.9±0.9 [467.6,469.3,469.7] | **+5.5%** | **−46.4%** | 0 |
| Case2 | t12   @ 64  | 282.4±7.8 [287.6,271.4,288.3] | 295.3±16.3 [306.7,306.9,272.3] | +4.6% | **−71.8%** | 0 |
| Case3 | t6_v2 @ 128 | 449.9±0.6 [450.3,449.0,450.4] | 475.7±1.1 [475.3,477.2,474.5] | **+5.7%** | **−59.4%** | 0 |

Case1 and Case3 reproduce tight (±<1.5) and match the 2026-06-24 numbers. Case2's
tps std is wide because the long t12 reps (~34 min at conc 64) each catch one
SW-power-cap dip rep (base rep2 271.4, sys rep3 272.3; the H200 is power-limited
at 700 W → SM 1515/1980 MHz, verified no external GPU contention). The dip is a
symmetric machine artifact: sys's clean reps [306.7, 306.9] still clear base's
[287.6, 288.3] by +6.5%, and Case2's TTFT win (−71.8%) is clean. Optionally
re-run Case2 at N=5 to dilute the dip.

## Summary (2026-06-24 original)

| Case | Trace | Conc | base tps | sys tps | tps delta | TTFT delta | TPOT delta |
|------|-------|------|----------|---------|-----------|------------|------------|
| Case1 | t6_v2 (1200 req) | 64 | 444.5±1.4 | 464.0±1.4 | **+4.4%** | **-44.6%** | **-6.7%** |
| Case2 | t12 (1795 req) | 64 | 287.4±0.4 | 303.3±0.6 | **+5.5%** | **-70.2%** | **-26.4%** |
| Case3 | t6_v2 (1200 req) | 128 | 449.4±0.2 | 481.1±1.9 | **+7.0%** | **-64.3%** | **-21.2%** |

## Case1: t6_v2 (long-horizon agent sessions, KV-bound)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 444.5±1.4 | 464.0±1.4 | **+4.4%** |
| cache_hit | 0.5814 | 0.6114 | +3.0pp |
| TTFT mean (ms) | 1437±20 | 796±8 | **-44.6%** |
| TTFT p50 (ms) | 533 | 418 | -21.6% |
| TTFT p99 (ms) | 11616 | 3940 | **-66.1%** |
| TPOT mean (ms) | 153.3±0.6 | 143.1±0.6 | **-6.7%** |
| E2E mean (ms) | 31160±135 | 29488±118 | **-5.4%** |
| wall_s | 741±3 | 710±2 | -4.2% |
| n_ok / n_error | 1200 / 0 | 1200 / 0 | |
| total_prompt | 37,015,694 | 37,015,694 | identical |
| total_out | 329,423 | 329,423 | identical |

Per-rep tps: base [443.8, 446.1, 443.5], sys [464.4, 465.1, 462.4].

## Case2: t12 (many-session agent swarm, high eviction volume)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 287.4±0.4 | 303.3±0.6 | **+5.5%** |
| cache_hit | 0.5452 | 0.5910 | +4.6pp |
| TTFT mean (ms) | 14892±60 | 4444±45 | **-70.2%** |
| TTFT p50 (ms) | 1075 | 596 | -44.6% |
| TTFT p99 (ms) | 67761 | 32976 | **-51.3%** |
| TPOT mean (ms) | 249.8±1.6 | 183.7±1.6 | **-26.4%** |
| E2E mean (ms) | 48024±84 | 43976±131 | **-8.4%** |
| wall_s | 2139±3 | 2027±4 | -5.2% |
| n_ok / n_error | 1795 / 0 | 1795 / 0 | |
| total_prompt | 80,355,225 | 80,355,225 | identical |
| total_out | 614,823 | 614,823 | identical |

Per-rep tps: base [287.6, 287.7, 286.9], sys [302.7, 303.9, 303.4].

## Case3: t6_v2 high-concurrency (concurrency scaling, conc=128)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 449.4±0.2 | 481.1±1.9 | **+7.0%** |
| cache_hit | 0.5840 | 0.6279 | +4.4pp |
| TTFT mean (ms) | 15817 | 5639 | **-64.3%** |
| TTFT p50 (ms) | 2642 | 1047 | -60.4% |
| TTFT p99 (ms) | 55672 | 25952 | **-53.4%** |
| TPOT mean (ms) | 364 | 287 | **-21.2%** |
| E2E mean (ms) | 55259 | 51385 | **-7.0%** |
| n_ok / n_error | 1200 / 0 | 1200 / 0 | |
| total_prompt | 37,015,694 | 37,015,694 | identical |
| total_out | 329,423 | 329,423 | identical |

Per-rep tps: base [449.3, 449.3, 449.7], sys [483.2, 479.6, 480.6].
Same trace as case1 but at 2x concurrency (128 vs 64): higher KV pressure
amplifies the m2k+LPB benefit from +4.4% to +7.0%.

## Config

run_arm.sh sys arm exports:
- SGLANG_HIMA=1, SGLANG_HIMA_TICK_S=1.0
- SGLANG_XPOOL_QUEUE_WAIT_US=100, SGLANG_XPOOL_COOLDOWN_S=1.0
- Calibrated cost model (SGLANG_CSIGMA_KV_*, c_M=0)
- LPB window: default 60s (SGLANG_LPB_WINDOW_S not set)
- Async fire (cap_barrier mark-only, cuMem* on worker thread)
- Admission-cap floor (max_running >= min(boot_cap, mamba_live//2))
- Eviction policy: --radix-eviction-policy lpb

Base arm: --radix-eviction-policy lru, no SGLANG_HIMA.

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang
bash reproduce/RQ1/run_official_case123.sh
```

## Qwen3.5-35B-A3B (2026-07, current build) — second measurable model

run_arm auto-selects the calibrated 35B csigma (`[[ MODEL == *35B* ]]` branch;
c_KV = 1.306635e-07·L² + 0.01602·L + 24.20 ms, c_M=0; from
`calibrate.sh Qwen/Qwen3.5-35B-A3B` + `--max-mamba-cache-size 16
--mem-fraction-static 0.7` on the bench, since 35B ssm_state is 51 GB at the
default cap and OOMs the profiler).

| cell | base (N=3) | sys (N=3) | dTPS | err |
|------|-----------|-----------|------|-----|
| Case1 t6_v2@64 (KV-bound, MAIN TABLE) | 475.1±0.2 | 481.1±0.9 | **+1.3%** (TTFT −38% P99) | 0 |
| vLLM t6_v2@64 | — | 184.1 (len_match 0.33) | — | 0 |
| static-best RATIO 0.8 | 477.0 | — | — | 0 |

35B is an MoE (~3B active), so decode is cheap and it is nearer compute-bound
than 9B -> smaller throughput gain (+1.3% vs 9B +5.5%); the latency win holds.

**35B Case2/Case3 (swarm/dynamic) \sys CRASH** (658-948 err/rep) even with the
35B c_KV: c_M=0 is wrong for 35B's mamba-bound regimes (open κ_M issue #276). So
only Case1 (the main-table cell) is measurable for 35B \sys; the RQ2-ablation
35B swarm/dynamic rows are not yet.

## Ling-2.6-flash (MLA) — \sys FUNDAMENTALLY unmeasurable (third-model blocker)

Resolves the "make Ling win or find the fundamental reason" investigation.
Ling-2.6-flash is `bailing_hybrid` (107B MoE, 28 linear-attn + 4 full-attn
layers) and the full-attn layers use **MLA** (kv_lora_rank 512, q_lora_rank
1536; KV cache 13.5M tokens / 57.8 GB = 4.3 KB/token compressed latent).

**Root cause (architectural, not tuning):** HiMA's arena-backing lives ONLY in
`MHATokenToKVPool` (`use_arena` gated on `head_dim==v_head_dim`, standard MHA
layout). MLA's KV cache is a different pool class (compressed latent), so the
arena code never runs → `kv_arena=None`. The MambaPool IS arena-backed, but with
no arena-backed KV pool the shared chain cannot build: the BudgetAgent reports
`chain_unavailable_reason="pools not arena-backed (kv_arena=None ...)"` at init.
The 57.8 GB of idle KV is locked in a non-arena allocation and cannot be moved to
the binding mamba pool. Cross-pool transfer (k2m AND m2k) is structurally
impossible. Confirmed: **0 cross-pool fires across 3 sys configs** (mamba cap
256, 160, and the swarm). Same class as the Kimi-48B blocker (ChunkCache).

**Also conceptually moot even if the MLA pool were arena-backed:** MLA exists to
make KV cheap, so the KV pool is over-abundant (13.5M tokens; realistic CC uses
~1M) and NEVER binds → m2k dead, LPB-on-KV dead. Mamba is the sole binding pool
in every regime → the optimal split never shifts → no dynamic win. And Ling's
compute scales sublinearly (base tps@peak-running: 4620@85, 6584@237, 7177@310;
marginal 12.9→8.1 tps/slot), so the compute knee is ~400-500 and a static mamba
pool at the largest that fits (~340; default mamba_cache 2601 OOMs, 384 OOMs)
already captures most of the throughput. The at-most win from borrowing idle KV
would be a few percent.

**Verified NOT the blocker (ruled out along the way):** the admission ceiling
follows the pool (`_maybe_update_admission_cap` raises `max_running_requests` +
`pp_max_micro_batch_size` on mamba grow); the PaybackPlanner already carries an
R_admission signal (`r_admit_m = W·1e6/N`); the mamba arena has VA headroom
(max_tokens 164100). All are inert because the physical grow is blocked at the
KV-pool-not-arena-backed step.

**Paper treatment:** Ling is a base + vLLM baseline + blocker-note row (mirror
the 48B row); \sys is not measurable. To make ANY MLA model win, one must first
implement arena-backing for the MLA KV pool (MultiTensorArena + SharedHandlePool
+ latent-layout chunk transfer + fa3-backend read) — a major extension.

## Nemotron-3-Super-120B-A12B (2026-07-15) — TP4, third measured model

GPU: 4×H200 (GPUs 3,4,5,7), TP=4, MEMFRAC=0.85, MAMBA_CAP=256,
MAMBA_STRAT=no_buffer, CUDA_GRAPH_DECODE=full, CUDA_GRAPH_PREFILL=disabled,
REASONING=none. Traces: corpus-built cc_nemotron_t6 (1199 req / 195 prog) and
cc_nemotron_t12 (1789 req / 146 prog). N=3 reps per arm, same-boot (not fresh-boot).

Two TP4-specific bugs fixed during this campaign:
1. `CappedFreeList.free()` missing `_norm(ids, self.device)` — ids from the
   scheduler (cuda:0) contaminated the free list on ranks 1-3 → device mismatch
   crash in `count_reachable`/`unmark`.
2. `BudgetAgent._fire_worker_loop` missing `torch.cuda.set_device(gpu_id)` — the
   worker thread defaulted bare `"cuda"` to cuda:0 regardless of rank, so `_norm`
   moved tensors to the wrong device.

### Summary

| Case | trace @ conc | base tps (N=3) | sys tps (N=3) | dTPS | dTTFT p99 | err |
|------|-------------|----------------|---------------|------|-----------|-----|
| Case1 | t6 @ 64    | 635.2±0.6 | 671.4±1.7 | **+5.7%** | **−22%** | 0 |
| Case2 | t12 @ 64   | 533.1±0.8 | 642.6±14.9 | **+20.5%** | **−22%** | 0 |
| Case3 | t6 @ 128   | 638.6±1.1 | 583.5±4.6 | −8.6% | **−91%** | 0 |

Case2 is the largest throughput win across all models (+20.5%). Case3 trades 8.6%
throughput for 95% TTFT improvement (20.5s→0.93s mean): at conc=128 the mamba pool
binds (4830 k2m fires, 0 m2k), HiMA correctly donates KV→mamba to admit more
requests, collapsing queue wait at the cost of higher per-token decode latency.

### Case1: cc_nemotron_t6 @ conc 64 (KV-bound, main-table cell)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 635.2±0.6 | 671.4±1.7 | **+5.7%** |
| cache_hit | 0.6196 | 0.7062 | +8.7pp |
| TTFT mean (ms) | 902 | 714 | **−20.8%** |
| TTFT p50 (ms) | 409 | 376 | −8.1% |
| TTFT p90 (ms) | 2412 | 1757 | **−27.2%** |
| TTFT p99 (ms) | 4721 | 3679 | **−22.1%** |
| TPOT mean (ms) | 108.8 | 97.3 | −10.6% |
| n_error | 0 | 0 | |

Per-rep tps: base [635.4, 634.5, 635.6], sys [672.2, 672.5, 669.4].

### Case2: cc_nemotron_t12 @ conc 64 (agent swarm, high eviction)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 533.1±0.8 | 642.6±14.9 | **+20.5%** |
| cache_hit | 0.5453 | 0.7301 | +18.5pp |
| TTFT mean (ms) | 1279 | 891 | **−30.3%** |
| TTFT p90 (ms) | 3205 | 2113 | **−34.1%** |
| TTFT p99 (ms) | 7583 | 5921 | **−21.9%** |
| TPOT mean (ms) | 123.4 | 92.8 | −24.8% |
| n_error | 0 | 0 | |

Per-rep tps: base [532.3, 533.1, 533.9], sys [632.6, 635.6, 659.7].

### Case3: cc_nemotron_t6 @ conc 128 (mamba-bound, QoE trade)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 638.6±1.1 | 583.5±4.6 | −8.6% |
| cache_hit | 0.6235 | 0.6989 | +7.5pp |
| TTFT mean (ms) | 20475 | 934 | **−95.4%** |
| TTFT p90 (ms) | 46768 | 2288 | **−95.1%** |
| TTFT p99 (ms) | 58392 | 5455 | **−90.7%** |
| TPOT p50 (ms) | 226.6 | 295.5 | +30.4% |
| n_error | 0 | 0 | |

Per-rep tps: base [639.3, 637.3, 639.2], sys [588.3, 583.0, 579.2].
Mamba-bound regime: 4830 k2m fires / 0 m2k. HiMA donates KV→mamba to increase
max_running, collapsing the 20s+ queue wait into sub-second TTFT.
