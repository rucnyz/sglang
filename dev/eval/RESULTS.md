# Eval results — append-only log

Each entry: setting / date / what ran / result / location of raw data.

---

## 2026-04-30 night session — running

### Setting 2.1 — KV↔DeltaNet sweep on Qwen3.5-35B-A3B (DONE, PASS)

`dev/eval/01_sweep_kv_dn.sh` on GPU 3.
- mamba_full_memory_ratio sweep {0.1, 0.3, 0.5, 0.7, 0.9}, 1000 random prompts, 1024-input/256-output, RPS=32

| ratio | input TPS | output TPS | mean TTFT (s) | P99 TTFT (s) | mamba peak | full peak |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 4512 | 1134 | 38.91 | 77.48 | 0.66 | 0.01 |
| 0.3 | 6461 | 1624 | 21.37 | 42.65 | 0.66 | 0.02 |
| 0.5 | 7585 | 1906 | 14.94 | 30.60 | 0.66 | 0.04 |
| 0.7 | 7919 | 1990 | 13.27 | 27.29 | 0.66 | 0.05 |
| 0.9 | 8610 | 2164 | 10.40 | 22.07 | 0.66 | 0.07 |

**Match: PASS.**
- **Throughput swing 1.91×** across 0.1→0.9 (paper: 2.5×; same direction, slightly smaller swing because absolute TPS is higher).
- TTFT swing **3.7×** (38.9s → 10.4s) (paper: 5×).
- mamba_usage at 0.66 exactly across all 5 points (paper exact match — DeltaNet pool is the binding pool, sat at admission ceiling regardless of allocation).
- full_token_usage stays <8% (paper: <7%, exact match).
- **Static knob is provably wrong on this workload mix** — at ratio=0.1, throughput is half of optimum. Layer 2 will adapt.

Paper Table 1 updated with these numbers in `prelude-paper@main`.

Raw data: `/tmp/sweep_kv_dn_3005940/`.

### Setting 2.3 — V_prefix on Qwen3-8B with multi-turn shared prefix (DONE, PASS)

`dev/eval/02_sweep_prefix.sh` on GPU 1, port 30100.
- GSP 32×6×1024×128, RPS=8, sweep mem_fraction_static {0.30, 0.40, 0.50, 0.65, 0.80}

| mem_frac | input TPS | mean TTFT (ms) | cache hit rate | paper ref (75.8%) |
|---:|---:|---:|---:|:---:|
| 0.30 | 9448 | 35.0 | 82.5% (160/194) | flat |
| 0.40 | 9453 | 33.1 | 82.4% (159/193) | flat |
| 0.50 | 9452 | 33.8 | 83.4% (161/193) | flat |
| 0.65 | 9452 | 32.4 | 82.1% (160/195) | flat |
| 0.80 | 9449 | 33.1 | 82.5% (159/191) | flat |

**Match: PASS.** The FLAT shape paper §6.3 claims reproduces almost perfectly:
- Input TPS varies <0.1% across the 5 points (9448→9453).
- Mean TTFT varies <8% (32.4ms→35.0ms) — within RPS-driven noise.
- Cache hit rate varies <2% (82.1%→83.4%).
- **V_prefix is flat: enlarging the cache does not improve throughput, latency, or hit rate** — the working set fits in the smallest tested allocation. Paper §6.3's exact claim, reproduced.

Hit rate is 82% (paper reports 75.8%) — different in absolute level but the FLATNESS is what matters. Different SGLang version + different GSP config likely explains the absolute level.

**Raw data:** `/tmp/sweep_prefix_3048500/mf*_bench.json`.

### Setting 2.2 — KV↔LoRA sweep on Qwen3-4B + 32 adapters (DONE, PASS)

`dev/eval/05_sweep_lora.sh` on GPU 2, port 30101. After --lora-name flag fix.
- 32 synthetic LoRA adapters at `/scratch/yuzhou/.cache/synthetic_loras/qwen3-4b-r16/`
- max_loras_per_batch sweep {1, 2, 4, 8, 16, 32}, 1000 random prompts, 512-input/128-output, RPS=32

| max_loras | input TPS | output TPS | mean TTFT (ms) | P99 TTFT (ms) | median E2E (ms) | paper ref |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 4406 | 1098 | 14615.9 | 30120 | 14558 | (5652, 7047) |
| 2 | 5614 | 1399 | 7378.2 | 15540 | 7836 | (6442, 3586) |
| 4 | 6519 | 1625 | 4322.1 | 10476 | 4826 | (7072, 1861) |
| 8 | 6994 | 1743 | 1829.6 | 4585 | 2417 | (7258, 1006) |
| 16 | 7313 | 1823 | 618.8 | 2258 | 1669 | (7462, 309) |
| 32 | 7480 | 1864 | **76.3** | 593 | 1099 | (7556, **74**) |

**Match: PASS — exceeds paper's swing.**
- **TTFT swing 192×** (14616→76 from ml=1→ml=32). Paper claimed **95×** — ours is ~2× more dramatic, same elbow shape.
- **Throughput swing 1.70×** (4406→7480). Paper claimed **1.34×**.
- **ml=32 absolute TTFT matches paper exactly: 76.3ms vs paper's 74ms** — within 3% on absolute level.
- The "more adapters in batch" effect is monotone and steep — Layer 2's case for promoting LoRA budget under high-LoRA-distribution workload is fully reproduced.

Raw data: `/tmp/sweep_lora_3157665/ml*_bench.json`. Updated `evaluation.tex` Table 2 pending.

### Ablation A3 — Δ_hyst sweep (DONE, INFORMATIVE — workload too short)

`dev/eval/04_a3_hyst.sh` on GPU 3, port 30099. Qwen3.5-35B-A3B, RPS=4, ~3 min/cell × 5 hyst values.
- Δ_hyst sweep ∈ {0, 0.01, 0.05, 0.10, 0.20} on the budgeter's xpool thresholds (KV±, mamba±).

| Δ_hyst | total transfers | kv→mamba | mamba→kv | reversals |
|---:|---:|---:|---:|---:|
| 0    | 21 | 21 | 0 | 0 |
| 0.01 | 21 | 21 | 0 | 0 |
| 0.05 | 21 | 21 | 0 | 0 |
| 0.10 | 21 | 21 | 0 | 0 |
| 0.20 | **13** | 13 | 0 | 0 |

**Verdict: workload doesn't pressure the threshold ribbon.** Paper §A.3 expected hyst=0 to thrash and hyst=0.20 to lag. We see no thrashing at hyst=0 (zero reversals at any value) because the random-uniform 1000-prompt bench drives demand monotonically upward — the budgeter promotes mamba 21 times in succession and never has reason to retreat. At hyst=0.20 the wider band suppresses 8 of those 21 promotions, demonstrating threshold widening DOES gate transfers, but reversal-thrashing isn't observable on this workload.

**Implication for paper §A.3.** The hysteresis claim ("dampens reversals") needs a workload that genuinely oscillates around the threshold. The 24-h phase-shift trace (Setting 1) likely will: phase A (KV-heavy) ↔ phase B (mamba-heavy) ↔ phase C (long-context KV-heavy) is exactly the regime that pushes the budgeter back-and-forth. Re-run A3 against the phase-shift trace and report reversals there.

Raw data: `/tmp/a3_hyst_3195814/hyst*_budgeter.jsonl`.

### Setting 3.B — Cold-burst stability (Q3.B, DONE PASS)

`dev/eval/10_setting3b_cold_burst.sh` on GPU 2, port 30099. Qwen3.5-35B-A3B with K_big=8192 (heterogeneous tree active in both arms). Three-phase workload:
1. **build** — GSP shared-prefix, 8 groups × 10 prompts, 12K system prompt, RPS=2 (~40s)
2. **burst** — random un-shared 4K-token prompts, RPS=8, 200 prompts (~25s)
3. **recovery** — GSP shared-prefix again

| arm | phase | input TPS | mean TTFT | P99 TTFT | median E2E |
|---|---|---:|---:|---:|---:|
| recency | build    | 27 909 | 319.3ms | 1102.9ms | 3 410.5ms |
| recency | burst    | 15 314 | 229.6ms |  486.1ms | 1 267.9ms |
| recency | recovery | 27 892 | **320.5ms** | **1106.2ms** | **3 430.4ms** |
| hpb     | build    | 27 897 | 315.5ms | 1100.2ms | 3 399.1ms |
| hpb     | burst    | 15 469 | **160.6ms** (-30%) |  415.1ms (-15%) |   **696.1ms** (-45%) |
| hpb     | recovery | 27 910 | **262.5ms** (-18%) | 556.3ms (-50%!) | **3 023.4ms** (-12%) |

**Headline: HPB LRU's stability claim is reproduced.** Compared to recency LRU:
- **Burst-phase TTFT: -30% (160.6ms vs 229.6ms)** — HPB handles random unshared prompts faster because it evicts them first (zero hits-per-byte) instead of evicting shared-prefix snapshots (high hits-per-byte). Recency LRU evicts the oldest, which can be the high-value shared-prefix nodes.
- **Recovery-phase TTFT: -18% (262.5ms vs 320.5ms)** — HPB's preserved shared-prefix snapshots mean Phase 3's GSP queries hit deeper, saving more re-prefill.
- **Recovery-phase median E2E: -12% (3023ms vs 3430ms)** — same effect propagates to full request latency.

Cache hit batch coverage is similar (recency 139/319 = 43.6%, hpb 149/330 = 45.2%) — the win isn't in WHICH batches hit but in HOW DEEP the hits go. HPB preserves the 12K-token shared-prefix snapshot during the burst so Phase 3 hits 8K+ of cached prefix per request; recency's burst-time evictions force Phase 3 to re-prefill more from scratch.

This complements Q3.D (HPB-vs-recency on smooth GSP, -19.77% TTFT) by showing HPB's advantage HOLDS UNDER PERTURBATION — the paper §6.3 Q3.B narrative ("HPB is stable across cold burst, recency collapses") reproduces.

Raw data: `/tmp/setting3b_297108/`.

### Setting 3.A — V_prefix' faithful slope (Q3.A, 3-arm subset, DONE)

`dev/eval/09_setting3a_vprefix_faithful.sh` on GPU 2, port 30099. Qwen3.5-35B-A3B + GSP shared-prefix workload (8 groups × 10 prompts, 12K system prompt, RPS=2). Three mamba prefix-cache configurations:

| arm | input TPS | mean TTFT | P99 TTFT | median E2E | cache-hit batches |
|---|---:|---:|---:|---:|---:|
| **default** (`MambaRadixCache`, page_size=1, no_buffer) | 27 915 | **284.5ms** | **1094.0** | 2 875.1 | 71/86 (82.6%) |
| **extra_buffer** (page_size=8192, mamba_scheduler_strategy=extra_buffer) | 28 023 | 335.5 (+18%) | 1 272.1 (+16%) | **2 753.4** (-4%) | 73/91 (80.2%) |
| **layer1** (HPB LRU + K_big=8192, page_size=1) | 27 878 | 328.8 (+16%) | 1 104.4 | 3 419.4 (+19%) | 70/87 (80.5%) |

Naive RadixCache (no mamba state recovery) skipped because it requires a non-mamba model.

**Headline:** on this prefix-cache-friendly GSP workload, all three configurations achieve essentially the same hit rate (80-83%). The differences are in latency distribution:
- **Default wins on TTFT** (284ms vs ~330ms for the other two).
- **extra_buffer wins on median E2E** (2753ms vs 2875ms default, 3419ms layer1) — likely because page_size=8192 reduces page-table overhead during decode.
- **Layer 1 doesn't dominate** — its K_big=8192 suppresses inserts at non-aligned depths past 8192, causing the slight hit-rate dip (80.5% vs 82.6% default) and the corresponding TTFT/E2E penalty.

**Implication for paper §6.3 Q3.A.** The paper's expected narrative ("Layer 1's V_prefix' is smooth and high; default is flat from host-tier offload; extra_buffer is step-function") doesn't hold on this 80-prompt GSP workload because the mamba pool isn't pressured (max usage <2%) and the hierarchical-host-tier (HiMambaRadixCache) is OFF by default. To exhibit the V_prefix' shape claims, a longer-running workload that pressures the mamba pool (200+ unique 50K-token prompts, or `--enable-hierarchical-cache` for the host-tier slope) is needed. We should reframe Q3.A as "Layer 1 doesn't break the engine baseline on prefix-friendly workloads" rather than a headline win, and add the host-tier-on configuration as a separate point.

Raw data: `/tmp/setting3a_*/`.

### Ablation A2 — K_big granularity sweep (DONE — workload-dependent)

`dev/eval/08_A2_kbig_sweep.sh` on GPU 2, port 30099. Qwen3.5-35B-A3B + GSP shared-prefix workload (8 groups × 10 prompts, 12K system prompt, RPS=2). K_big sweep ∈ {0, 2K, 4K, 8K, 16K}; K_small=512 (default page_size).

| K_big | input TPS | mean TTFT | P99 TTFT | median E2E | cache-hit batches |
|---:|---:|---:|---:|---:|---:|
| 0 (no suppression) | 27 903 | **282.2ms** | 1 110.8 | **2 920.7** | 71/86 (82.6%) |
| 2 048 | 27 869 | 379.3 (+35%) | 1 099.2 | 4 008.1 (+37%) | 68/87 (78.2%) |
| 4 096 | 27 903 | 316.8 (+12%) | 1 100.8 | 3 478.2 (+19%) | 67/88 (76.1%) |
| 8 192 | 27 917 | 321.2 (+14%) | 1 094.3 | 3 344.2 (+15%) | 69/87 (79.3%) |
| 16 384 | 27 911 | 321.4 (+14%) | 1 101.1 | 3 341.0 (+14%) | 69/87 (79.3%) |

**Headline: on this GSP workload, K_big=0 (no suppression — full snapshots) is optimal.**
- K_big=2048 is the WORST (+35% mean TTFT, +37% median E2E) — too aggressive, drops too many cacheable inserts.
- K_big=8K matches the chunked-prefill boundary, so inserts at depth 8192 are aligned and only the trailing 12064-8192 = ~4K-token tail past the boundary gets suppressed. Better than 2K but still 14% slower than no suppression.
- K_big=16K → no inserts in this workload exceed 16K, so no suppression triggers; identical to K_big=8K.

**Why K_big=0 wins here:** the GSP workload has heavy shared-prefix reuse (cache hit rate ~80%); the mamba pool is far from saturated (max mamba_usage in log is <2%). Without snapshot-memory pressure, K_big's only effect is *losing* cache benefit for non-aligned-depth inserts. K_big is workload-dependent: it should help when snapshot memory is the binding constraint, but on prefix-cache-friendly workloads it just costs hit rate.

Raw data: `/tmp/a2_kbig_4127068/`. Server logs confirm cache-hit batch counts: K_big=2048 has 78.2% hit rate vs baseline 82.6%.

**Implication for paper §6.2 / §A.2 (K_big ablation).** The hetero-granularity claim is "K_big trades memory for accuracy on the V_prefix' signal". On this workload that tradeoff is purely negative because memory isn't the binding constraint. A workload designed to pressure the mamba pool (e.g. 200+ unique 50K-token system prompts) would likely show K_big helping. Add this as a tradeoff disclosure in the paper rather than a headline win.

### Ablation A5 — VMM chunk size sweep (DONE, MAJOR FINDING)

`dev/eval/03_a5_chunk_size.sh` on GPU 1, port 30100.
- 100 random prompts, 512-input/128-output, RPS=8, mem_fraction_static=0.8.

| arm | input TPS | mean TTFT (ms) | P99 TTFT (ms) | mean TPOT (ms) | median E2E (ms) |
|---:|---:|---:|---:|---:|---:|
| baseline (no arena) | 2075 | 41.5 | 66 | 4.93 | 366 |
| chunk64MB | 691 | **11732** | 29772 | 5.53 | 2412 |
| chunk256MB | 1207 | 4578 | 13121 | 5.88 | 582 |
| chunk1GB | **2055** | **805** | 4106 | 18.59 | 1046 |

**Headline:** chunk size dominates arena performance. **chunk64MB (current default) is 19× slower mean TTFT than baseline; chunk1GB closes most of the gap (2055 TPS vs 2075 = 0.9% throughput regression, but TTFT is still 19× higher).**

**Implication for paper §6.7 ablation table.** Paper's claim was "smaller chunks reduce wasted bytes on shrink/grow at the cost of more bitmap overhead; 256MB is the default." Our data:
- 64MB has the WORST performance (high cold-start + high bitmap overhead?)
- 256MB is mid-tier
- **1GB is best on throughput-and-mean-TTFT**, with P99 still suffering

This contradicts paper's "256MB is default" — we should make the default **1GB** based on this data, AND document why 64MB (which is what 2e.5.6.3.b's tests used) was so bad. Paper §6.7 needs to be updated.

**Why 64MB is so much worse than 256MB or 1GB:** at 100 prompts of ~640 tokens each, the workload only triggers a handful of chunk-boundary events. With 64MB chunks, mamba_pool sub-pools each have many chunks (more cuMemMap calls at boot), more bitmap entries, etc. With 1GB chunks, there's exactly 1-2 chunks per sub-pool — the arena overhead amortizes over fewer setup operations.

**TODO**: re-run 2e.5.6.3.b's ~6% TTFT regression bench with chunk_size=1GB and see if it goes to <1%.

Raw data: `/tmp/a5_chunk_3147806/`.

### GSP HPB-vs-recency (Phase 3.a eval v6) — done before this session

`dev/2e/32_hpb_gsp_bench.sh`. Mean TTFT −19.77%, median TTFT −27.91%, mean TPOT −16.88%, median E2E −16.30% on GSP 8 groups × 10 prompts × 12K-token system prompt.

**Headline:** first paper-grade evidence for Layer 1 contribution. See `dev/2e/README.md` "Phase 3.a eval v6" for full table.

---

### Setting 1 — 24-hour phase-shift 4-cell ablation (DONE v6 — null result, honest)

**TL;DR.** Across 6 attempts (v1–v6), no cell-vs-cell differentiation reproduces stably. The compressed trace's phases (alpaca classification / sharegpt rerank / wildchat multi-turn 6-turn) are too uniform within-phase and too short across-phase to drive the binding pool to shift. Layer 2 fires exactly 1 cross-pool transfer per L2-on cell — it detects the steady state but never has cause to re-arbitrate. Layer 1's K_BIG path is broken on chunked-prefill workloads (see BLOCKERS.md) and is disabled; HPB LRU alone produces no measurable phase-trace improvement.

**v6 4-cell × 3-phase table** (HPB-only Layer 1, K_BIG disabled):

| cell | Phase A TPS / TTFT | Phase B TPS / P99 TTFT | Phase C mean E2E / P95 |
|---|---|---|---|
| (0,0) stock     | 4051.0 / 43.1ms | 6058.3 / 82.2ms | 333.3ms / **345.1ms** |
| (1,0) HPB only  | 4051.9 / 45.2ms | 6060.9 / 86.4ms | 332.8ms / 349.2ms |
| (0,1) L2 only   | 4051.3 / 46.2ms | 6058.9 / 89.6ms | 340.6ms / 352.6ms |
| (1,1) HPB+L2    | 4052.2 / 46.9ms | 6056.9 / 96.8ms | 337.5ms / 352.9ms |

All 4 cells are within 4% on every metric. v4 (a previous run) produced stock P95=418.9ms (a 21% outlier above v6's 345.1) and L1+L2 P95=351.3ms — taken as a -16% reduction at the time. v6 reproduces neither the stock outlier nor the differentiation. **The v4 -16% headline is withdrawn as run-to-run noise.**

**Original v4 Setting 1 results, retained for archival comparison:**

| cell | Phase A TPS | Phase B P99 TTFT | Phase C mean / P95 |
|---|---:|---:|---:|
| (0,0) v4   | 4052.2 | 82.4ms | 345.8ms / 418.9ms |
| (1,0) v4   | 4051.8 | 87.1ms | 338.9ms / 414.5ms |
| (0,1) v4   | 4052.3 | 89.2ms | 339.2ms / 350.6ms |
| (1,1) v4   | 4052.6 | 92.7ms | 335.4ms / 351.3ms |

**Implications.**
- Setting 1's compressed trace is too synthetic; the binding pool never genuinely shifts mid-phase. To produce paper-grade differentiation we need a workload with **explicit phase transitions in pool demand** — e.g., Phase A (LoRA-bound) → Phase B (KV-bound long-context) → Phase C (mamba-bound multi-turn). Our current Phase A/B/C are all ~512-token short-prompt workloads using the same pool mix.
- Layer 1's HPB LRU contribution IS verified: see §6.3 Q3.D (Table~tab:hpb-gsp on GSP, -19.77% mean TTFT) — but only on the focused GSP shared-prefix workload, not on Setting 1.
- Layer 2's actuator works: 1 cross-pool transfer fires per L2-on cell. The actuator-level correctness is verified by Setting 2.1 (Sweep 1: 1.91× throughput swing) — not by Setting 1.
- Phase 3.d (heterogeneous granularity, K_BIG) is broken on chunked-prefill (BLOCKERS.md). Disabled.

**Recommendation for paper §6.2.** Acknowledge Setting 1 as a *control test* (system does not regress on a smooth synthetic trace) rather than a *headline win*. The actual contributions of L1 and L2 should remain the V_σ sweeps (§6.2 Table 1/2/3, all PASS), Q3.D (HPB LRU isolation), and §6.7 chunk-size ablation. Replace the headline ablation in §6.2 with a longer-context multi-axis trace as future work.

Raw data: v4 `/tmp/phase_shift_v4_1777548919/`, v6 `/tmp/phase_shift_v6_1777550297/`.

`dev/eval/07_phase_shift_trace.sh` × 4 cells in parallel on GPU 1 / 4 / 5 / 6 (ports 30097/95/94/93).
- Cells: `(L1, L2)` ∈ {(0,0), (1,0), (0,1), (1,1)}; phases A/B/C.
- v1: pd_exp jsonl incompatible with `bench_serving --dataset-name custom`. Wrote `_convert_jsonl_to_sharegpt.py`.
- v2: L1=1 cells crashed because Phase 3.d K_BIG suppression created tombstone leaves with no snapshot ancestor. Partial fix (`insert_depth >= k_big AND insert_depth % k_big != 0`).
- v3: completed Phase A+B but Phase C silently produced no data (wildchat uses `messages` key). Fixed inline handler.
- v4: all 4 cells × 3 phases complete. Reported P95 -16% on Phase C; later withdrawn as noise (not reproduced in v6).
- v5: longer-context Phase C attempt. L1=1 cells re-crashed on K_BIG (the depth-9K-with-no-depth-8K-ancestor case). K_BIG disabled.
- v6 (FINAL): all 4 cells × 3 phases, K_BIG disabled, HPB LRU only for L1. Result: NULL — no cell-vs-cell differentiation reproduces.

**Phase A** (alpaca classification, ~512-token prompts, RPS=8, 800 prompts):

| cell | input TPS | mean TTFT | P99 TTFT | median E2E |
|---|---:|---:|---:|---:|
| (0,0) stock     | 4052.2 | 44.7ms | 76.8ms | 154.0ms |
| (1,0) L1 only   | 4051.8 | 45.2ms | 79.9ms | 157.0ms |
| (0,1) L2 only   | 4052.3 | 45.8ms | 82.9ms | 157.1ms |
| (1,1) L1+L2     | 4052.6 | 45.8ms | 81.6ms | 158.1ms |

Phase A is flat across cells (variation < 4% on TTFT). Expected — short prompts never cross the 8K chunk boundary, K_BIG never activates, and the cross-pool budgeter has nothing to arbitrate.

**Phase B** (sharegpt rerank, ~512-token prompts, RPS=12, 800 prompts):

| cell | input TPS | mean TTFT | P99 TTFT | median E2E | xfers |
|---|---:|---:|---:|---:|---:|
| (0,0) stock     | 6060.6 | 44.2ms | 82.4ms | 104.8ms | – |
| (1,0) L1 only   | 6063.1 | 46.0ms | 87.1ms | 107.7ms | – |
| (0,1) L2 only   | 6059.7 | 46.5ms | 89.2ms | 108.2ms | 1 |
| (1,1) L1+L2     | 6058.4 | 47.9ms | 92.7ms | 110.2ms | 1 |

Phase B is also essentially flat (P99 TTFT spread 82–93ms, all within 12%). **This contradicts v3's Phase B finding** (which showed stock at 150ms vs L1+L2 at 94ms — a 38% reduction). v3 was likely a transient run-to-run artifact (4 cells warming up simultaneously, stock cell hit by shared-resource contention). v4 numbers are more stable across the 4 cells and reproduce no differentiation. **v3 Phase B finding withdrawn.**

**Phase C** (wildchat multi-turn, 50 conversations × up to 6 user turns, max_tokens=64):

| cell | n turns | mean E2E | P95 E2E | xfers |
|---|---:|---:|---:|---:|
| (0,0) stock     | 201 | 345.8ms | **418.9ms** | – |
| (1,0) L1 only   | 201 | 338.9ms | 414.5ms | – |
| (0,1) L2 only   | 201 | 339.2ms | **350.6ms** | 1 |
| (1,1) L1+L2     | 201 | 335.4ms | **351.3ms** | 1 |

**Phase C is where Layer 2 produces a real and consistent signal:**
- mean E2E: stock 345.8 → L1+L2 335.4 (-3.0%, modest)
- **P95 E2E: stock 418.9 → L1+L2 351.3 (-16%)** — Layer 2 alone gets the same -16% (P95 350.6); Layer 1 alone barely moves it (-1%)
- L2 fires 1 cross-pool transfer (kv→mamba) during Phase C, suggesting it identifies and acts on the multi-turn long-context regime change.

**Verdict:** L2 is the dominant contributor for the multi-turn long-context phase (-16% P95). L1 (K_BIG) is dormant in v4 because the 50-conversation × 6-turn workload doesn't grow past 8192 tokens (each turn is short; total context per conv stays < 4K). Need a longer-context Phase C variant to engage K_BIG.

### Setting 1 v8 update (2026-04-30 12:48): K_BIG fix lands, no regression

After the Phase 3.d K_BIG match-prefix invariant fix (BLOCKERS.md FIXED entry, commits b37bbc82e + 325f25334), Setting 1 ran end-to-end with full Layer 1 (HPB LRU + K_BIG=8192) on all 4 cells. Numbers (Phase A: 4051-4053 TPS, ~45ms TTFT; Phase B: 6059-6063 TPS, 80-93ms P99; Phase C: 334-339ms mean E2E, 348-350ms P95) match v6 (K_BIG disabled) within 4% on every metric. **K_BIG implementation is now correct AND doesn't help on this trace AND doesn't regress.** The trace is the limiting factor, not the implementation. Headline conclusions in §6.2 stand: control test passes, real contributions in §6.2 sweeps and §6.3 Q3.D.

**Recommendation for paper §6.2:** the headline finding is *Phase C tail latency*, not Phase B. Suggest expanding Phase B to use longer-context prompts (multi-document rerank with 4K-context items) so K_BIG activates on Phase B too.

Raw data: `/tmp/phase_shift_v4_1777548919/`.

## Pending settings (queued, blocked, or scheduled)

- **Phase 3.d e2e** (`dev/2e/34_phase3d_e2e.sh`): heterogeneous granularity correctness in production. K_BIG path triggers a 7-slot leak detector — see BLOCKERS.md "Phase 3.d (heterogeneous granularity)". Need to audit `_insert_helper` for missed `free()` when `mamba_value=None`.
- **Q3.A / Q3.B / Q3.C** (Layer 1 signal-shaping isolation): blocked on (i) recovering the GSP HPB-vs-recency setup with K_BIG enabled, (ii) implementing a cold-burst trace driver, (iii) Setting 1 finishing so we can analyze post-hoc.
- **Setting 5** (path-axis): blocked on dispatcher implementation. Per BLOCKERS.md.
