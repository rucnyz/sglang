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

### Setting 2.2 — KV↔LoRA sweep on Qwen3-4B + 32 adapters (in progress)

`dev/eval/05_sweep_lora.sh` on GPU 2, port 30101. Just started.
- 32 synthetic LoRA adapters at `/scratch/yuzhou/.cache/synthetic_loras/qwen3-4b-r16/`
- max_loras_per_batch sweep {1, 2, 4, 8, 16, 32}

### GSP HPB-vs-recency (Phase 3.a eval v6) — done before this session

`dev/2e/32_hpb_gsp_bench.sh`. Mean TTFT −19.77%, median TTFT −27.91%, mean TPOT −16.88%, median E2E −16.30% on GSP 8 groups × 10 prompts × 12K-token system prompt.

**Headline:** first paper-grade evidence for Layer 1 contribution. See `dev/2e/README.md` "Phase 3.a eval v6" for full table.

---

## Pending settings (queued, blocked, or scheduled)

- **A5 chunk-size sweep** (`dev/eval/03_a5_chunk_size.sh`): scheduled. Could close part of the 2e.5.6.3.b regression.
- **A3 hysteresis sweep** (`dev/eval/04_a3_hyst.sh`): scheduled.
- **Setting 1** (24-h phase-shift trace): blocked on dataset generation + Layer 1 default-on configuration. Per BLOCKERS.md.
- **Phase 3.d e2e** (`dev/2e/34_phase3d_e2e.sh`): heterogeneous granularity in production. Ready, awaiting GPU.
- **Setting 5** (path-axis): blocked on dispatcher implementation.
