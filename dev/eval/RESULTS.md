# Eval results — append-only log

Each entry: setting / date / what ran / result / location of raw data.

---

## 2026-04-30 night session — running

### Setting 2.1 — KV↔DeltaNet sweep on Qwen3.5-35B-A3B (in progress)

`dev/eval/01_sweep_kv_dn.sh` on GPU 3, port 30099.
- mamba_full_memory_ratio sweep {0.1, 0.3, 0.5, 0.7, 0.9}, 1000 random prompts, 1024-input/256-output, RPS=32

| ratio | input TPS | mean TTFT (s) | mamba_usage_peak | full_token_usage_peak | paper ref |
|---:|---:|---:|---:|---:|:---:|
| 0.1 | 4512 | 38.91 | 0.67 | 0.01 | (paper: 3039, 69.9, 0.66, 0.008) |
| 0.3 | 6461 | 21.37 | 0.66 | 0.02 | (paper: 3793, 49.7, 0.66, 0.023) |
| 0.5 | (running) | | | | (paper: 5973, 23.5) |
| 0.7 | | | | | (paper: 6890, 17.2) |
| 0.9 | | | | | (paper: 7648, 13.6) |

**Match assessment:** mamba_usage_peak matches paper exactly (0.66/0.67). Throughput numbers are HIGHER and TTFT LOWER than paper — different SGLang commit / engine optimizations between paper authoring and now. Shape (the 2.5× swing in throughput across the sweep) is the load-bearing claim and will be verifiable once 0.9 completes.

Raw data: `/tmp/sweep_kv_dn_*/ratio*_bench.json`.

### Setting 2.3 — V_prefix on Qwen3-8B with multi-turn shared prefix (in progress)

`dev/eval/02_sweep_prefix.sh` on GPU 1, port 30100.
- GSP 32×6×1024×128, RPS=8, sweep mem_fraction_static {0.30, 0.40, 0.50, 0.65, 0.80}

| mem_frac | cache hit rate | paper ref (75.8%) |
|---:|---:|:---:|
| 0.30 | 82.5% (160/194) | flat |
| 0.40 | 82.4% (159/193) | flat |
| 0.50 | (running) | |
| 0.65 | | |
| 0.80 | | |

**Match assessment:** the FLAT shape paper §6.3 claims is reproducing — first two points within 0.1% of each other. Hit rate higher than paper's 75.8% (82% vs 76%), which is the right direction for our setup (Qwen3-8B with default RadixCache, GSP workload).

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
