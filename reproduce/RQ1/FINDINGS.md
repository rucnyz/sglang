# RQ1 Official Results

Harness: agentreplay token-exact replay, N=3 per arm, sglang branch HiMA.
Base arm: LRU eviction, no HiMA.
Sys arm: LPB eviction + HiMA (Budgeter + Admitter, async fire, calibrated cost model c_M=0).

## Config

run_arm.sh sys arm exports:
- SGLANG_HIMA=1, SGLANG_HIMA_TICK_S=1.0
- SGLANG_XPOOL_QUEUE_WAIT_US=100, SGLANG_XPOOL_COOLDOWN_S=1.0
- Calibrated cost model (SGLANG_CSIGMA_KV_*, c_M=0)
- Async fire (cap_barrier mark-only, cuMem* on worker thread)
- Eviction policy: --radix-eviction-policy lpb

Base arm: --radix-eviction-policy lru, no SGLANG_HIMA.

## Qwen3.5-9B (2026-07-06) — 1×H200

GPU 7, TP=1. Traces: cc_qwen_t6_v2 (1200 req / 200 prog), cc_qwen_t12 (1795 req).
N=3 fresh boots per arm via `run_official_case123.sh`.

| Case | trace @ conc | base tps (N=3) | sys tps (N=3) | dTPS | dTTFT mean | err |
|------|--------------|----------------|---------------|------|------------|-----|
| Case1 | t6_v2 @ 64  | 444.3±0.9 | 468.9±0.9 | **+5.5%** | **−46.4%** | 0 |
| Case2 | t12   @ 64  | 282.4±7.8 | 295.3±16.3 | +4.6% | **−71.8%** | 0 |
| Case3 | t6_v2 @ 128 | 449.9±0.6 | 475.7±1.1 | **+5.7%** | **−59.4%** | 0 |

Case2 tps std is wide due to a symmetric SW-power-cap dip (base rep2 271.4,
sys rep3 272.3; H200 700 W throttle). Sys's clean reps [306.7, 306.9] clear
base's [287.6, 288.3] by +6.5%.

## Qwen3.5-35B-A3B (2026-07) — 1×H200

GPU 7, TP=1, MEMFRAC=0.85. Calibrated 35B csigma (`calibrate.sh`).

| cell | base (N=3) | sys (N=3) | dTPS | err |
|------|-----------|-----------|------|-----|
| Case1 t6_v2@64 (KV-bound, MAIN TABLE) | 475.1±0.2 | 481.1±0.9 | **+1.3%** (TTFT −38% P99) | 0 |
| vLLM t6_v2@64 | — | 184.1 (len_match 0.33) | — | 0 |
| static-best RATIO 0.8 | 477.0 | — | — | 0 |

35B is an MoE (~3B active), so decode is cheap and it is nearer compute-bound
than 9B -> smaller throughput gain (+1.3% vs 9B +5.5%); the latency win holds.

**35B Case2/Case3 \sys CRASH** (658-948 err/rep): c_M=0 is wrong for 35B's
mamba-bound regimes (open issue #276). Only Case1 is measurable.

## Nemotron-3-Super-120B-A12B (2026-07-15) — 4×H200, TP4

GPUs 3,4,5,7, TP=4, MEMFRAC=0.85, MAMBA_CAP=256, MAMBA_STRAT=no_buffer,
CUDA_GRAPH_DECODE=full, CUDA_GRAPH_PREFILL=disabled, REASONING=none.
Traces: corpus-built cc_nemotron_t6 (1199 req / 195 prog) and
cc_nemotron_t12 (1789 req / 146 prog). N=3 reps per arm, same-boot.

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
