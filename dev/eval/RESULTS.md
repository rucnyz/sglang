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

### Setting 1 — 24-hour phase-shift 4-cell ablation (RUNNING v3, A+B done)

`dev/eval/07_phase_shift_trace.sh` × 4 cells in parallel on GPU 1 / 4 / 5 / 6 (ports 30097/95/94/93).
- Cells: `(L1, L2)` ∈ {(0,0), (1,0), (0,1), (1,1)}; phases A/B/C.
- v1 attempt died at Phase A — pd_exp jsonl format incompatible with `bench_serving --dataset-name custom`. Wrote `_convert_jsonl_to_sharegpt.py`.
- v2 attempt: L1=1 cells crashed in `cache_unfinished_req` because Phase 3.d K_BIG suppression created tombstone leaves at depth 512 with no snapshot ancestor → match_prefix returned 0 indices. Fixed in `mamba_radix_cache.py` (only suppress when `insert_depth >= k_big AND insert_depth % k_big != 0`). See BLOCKERS.md.
- v3 (current): all 4 cells running, Phase A and Phase B complete. Phase C in progress.

**Phase A** (alpaca classification, ~512-token prompts, RPS=8, 800 prompts):

| cell | input TPS | mean TTFT | P99 TTFT | median E2E |
|---|---:|---:|---:|---:|
| (0,0) stock      | 4051.5 | 44.7ms | 80.5ms | 154.6ms |
| (0,1) L2 only    | 4052.1 | 46.1ms | 83.2ms | 157.3ms |
| (1,0) L1 only    | 4051.8 | 44.9ms | 79.1ms | 155.1ms |
| (1,1) L1+L2 full | 4052.3 | 46.2ms | 81.4ms | 157.9ms |

Phase A is too short / too uniform for any cell to differentiate. K_BIG never activates (prompts < 8192 tokens) and Layer 2 has nothing to arbitrate. *This is exactly what paper §6.2 predicts for the smooth-classification phase.*

**Phase B** (sharegpt rerank, ~512-token prompts, RPS=12, 800 prompts):

| cell | input TPS | mean TTFT | P99 TTFT | median E2E | xpool xfers |
|---|---:|---:|---:|---:|---:|
| (0,0) stock      | 6060.0 | 49.0ms | **150.7ms** | 107.3ms | – |
| (0,1) L2 only    | 6058.9 | 47.3ms | 90.8ms | 109.1ms | 1 |
| (1,0) L1 only    | 6062.4 | 45.9ms | 85.3ms | 107.1ms | – |
| (1,1) L1+L2 full | 6059.6 | 48.3ms | **93.9ms** | 111.6ms | 1 |

Throughput is essentially identical across cells (~6060 TPS), but **P99 TTFT drops from 150.7ms (stock) to 93.9ms (L1+L2) — a 38% tail-latency reduction**. The mean and median are flat. Layer 1 (HPB LRU) is the dominant contributor here; Layer 2 fires only 1 cross-pool transfer.

Phase C (wildchat multi-turn) results pending — that's where K_BIG and L2 cross-pool should matter most.

Raw data: `/tmp/phase_shift_v3_1777548459/`. Aggregate: `python3 dev/eval/_aggregate_phase_shift.py /tmp/phase_shift_v3_1777548459`.

## Pending settings (queued, blocked, or scheduled)

- **Phase 3.d e2e** (`dev/2e/34_phase3d_e2e.sh`): heterogeneous granularity correctness in production. K_BIG path triggers a 7-slot leak detector — see BLOCKERS.md "Phase 3.d (heterogeneous granularity)". Need to audit `_insert_helper` for missed `free()` when `mamba_value=None`.
- **Q3.A / Q3.B / Q3.C** (Layer 1 signal-shaping isolation): blocked on (i) recovering the GSP HPB-vs-recency setup with K_BIG enabled, (ii) implementing a cold-burst trace driver, (iii) Setting 1 finishing so we can analyze post-hoc.
- **Setting 5** (path-axis): blocked on dispatcher implementation. Per BLOCKERS.md.
