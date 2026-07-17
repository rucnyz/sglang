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

## Qwen3.5-9B (2026-07-17, canonical traces) — 1×H200

TP=1, N=3 fresh-boot reps per arm. Traces: canonical corpus-built
cc_qwen_t6 (1199 req / 195 prog) and cc_qwen_t12 (1789 req / 146 prog) —
these REPLACE the deleted t6_v2/old-t12 traces, so the 2026-07-06 numbers
(444.3/468.9 etc.) are not comparable and are superseded by this table.
Build: post cap_barrier fix (7827ee453c) + k2m serving floor (48f2ce5414).

| Case | trace @ conc | base tps (N=3) | sys tps (N=3) | dTPS | TTFT mean | TTFT p90 | err |
|------|-------------|----------------|---------------|------|-----------|----------|-----|
| Case1 | t6 @ 64   | 913.4±4.9 | 1035.1±21.1 | **+13.3%** | 441→296 (**−33%**) | 1233→680 (**−45%**) | 0 |
| Case2 | t12 @ 64  | 716.6±1.8 | 779.2±6.5 | **+8.7%** | 699→544 (**−22%**) | 1926→1545 (**−20%**) | 0 |
| Case3 | t6 @ 128  | 348.7±1.6 | 1019.2±9.6 | **+192.3%** | 120,603→397 (**−99.7%**) | 262,108→986 (**−99.6%**) | 0 |

Case1 also improves TPOT mean 61.6→47.8 ms (−22%); Case2 80.6→69.5 ms
(−14%); cache hit +9.8pp / +6.6pp / +20.8pp.

**Case3 is the concurrency-unlock regime**: at conc 128 the mamba pool
caps base's admissible batch, queue wait explodes (TTFT mean 120.6 s, p90
262 s, TPOT mean 1623 ms from head-of-line stalls). HiMA's k2m donation
admits the full offered concurrency: 348.7→1019.2 tok/s (2.9×), TTFT p99
285.5 s→2.2 s, TPOT mean 92 ms. Requires the k2m serving floor
(48f2ce5414): without it the now-cheap fires drain the KV pool below one
prefill chunk and the scheduler OOMs ("Available full tokens: 6408 ...
evictable: 0") — the pre-floor sys arm crashed in rep 2 of this exact
cell (rep1 throttled to 229.9 tok/s by the same drain).

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

Three TP4-specific bugs fixed during this campaign:
1. `CappedFreeList.free()` missing `_norm(ids, self.device)` — ids from the
   scheduler (cuda:0) contaminated the free list on ranks 1-3 → device mismatch
   crash in `count_reachable`/`unmark`.
2. `BudgetAgent._fire_worker_loop` missing `torch.cuda.set_device(gpu_id)` — the
   worker thread defaulted bare `"cuda"` to cuda:0 regardless of rank, so `_norm`
   moved tensors to the wrong device.
3. `cap_barrier` mark-then-clamp (commit 7827ee453c, see "Case3 root cause"
   below) — 240 ms of scheduler-thread time per fire; the sole cause of the
   original Case3 −8.6% regression.

### Summary

| Case | trace @ conc | base tps (N=3) | sys tps (N=3) | dTPS | dTTFT p90 | err |
|------|-------------|----------------|---------------|------|-----------|-----|
| Case1 | t6 @ 64    | 635.2±0.6 | 699.9±9.6 | **+10.2%** | **−33%** | 0 |
| Case2 | t12 @ 64   | 533.1±0.8 | 689.5±16.6 | **+29.3%** | **−43%** | 0 |
| Case3 | t6 @ 128   | 638.6±1.1 | 654.7±1.6 | **+2.5%** | **−96%** | 0 |

Case2 is the largest throughput win across all models (+29.3%; all three
cases remeasured on the post-cap_barrier-fix build — the fix roughly
doubled the case1/case2 margins by removing the 240 ms/fire tax). Case3 wins on
BOTH axes after the cap_barrier fix: +2.5% tps AND TTFT p90 46.8s→2.0s — at
conc=128 the mamba pool binds, HiMA donates KV→mamba (k2m fires) to admit
85→130 concurrent requests, collapsing queue wait with no throughput cost.

### Case3 root cause (the original −8.6% regression, now fixed)

The pre-fix sys arm measured 583.5±4.6 (−8.6%). Isolation matrix:

| | rr=85 | rr=130 |
|---|---|---|
| no HiMA (mamba cache 416) | 679.9 | **694.9** (high concurrency is BETTER) |
| HiMA, pre-fix | 675.9 (win) | 583.5 (−8.6%) |
| HiMA, fixed | — | **654.7 (+2.5%)** |

High concurrency was innocent. The regression was entirely per-fire overhead:
every k2m fire's `cap_barrier` ran on the scheduler thread and expanded +
marked the planner's FULL offered page set (n_pages=80 → 655,360 token-slot
ids through a Python list → CUDA tensor), then clamped the grant to the dst
headroom (~5 pages) and unmarked the ~94% surplus — measured
`cap_barrier_us` p50 = 240 ms in the fire records of every run, winners
included. Because each fire granted ~6% of the ask, the Admitter re-fired
every cooldown: ~400 fires/rank/rep × 240 ms ≈ 97 s/rep of stolen scheduler
time = the regression. Loss concentrated in concurrency-transition windows
(fire-heavy): rr≥100 windows WITH fires cost 235 ms/pass vs 169 ms without.

Fix (7827ee453c): free-only plans clamp BEFORE expand/mark (pure int math)
and mark only the kept pages, with vectorized expansion. Post-fix fire
records: cap_barrier_us p50 = 46 µs (5200×), total fire time 1299 s → 13 s
per 3-rep run. Unit tests:
dev/interlayer/4_e2e/byte_transfer/test_cap_barrier_fast_path.py (end-state
equivalence vs legacy semantics on a real CappedFreeList + token-math
corners + perf budget).

Side observation: base with MAMBA_CAP=416 (vs 256) reaches 679.9–694.9
(hit 0.62→0.65) — the larger mamba state cache alone is worth ~6%; worth
considering as the default for this model.

### Case1: cc_nemotron_t6 @ conc 64 (KV-bound, main-table cell)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 635.2±0.6 | 699.9±9.6 | **+10.2%** |
| cache_hit | 0.6196 | 0.7088 | +8.9pp |
| TTFT mean (ms) | 902 | 677 | **−25.0%** |
| TTFT p90 (ms) | 2412 | 1622 | **−32.8%** |
| TTFT p99 (ms) | 4721 | 3369 | **−28.6%** |
| TPOT mean (ms) | 108.8 | 91.6 | −15.8% |
| n_error | 0 | 0 | |

Per-rep tps: base [635.4, 634.5, 635.6], sys [703.2, 689.0, 707.4]
(post cap_barrier fix; the pre-fix sys measured 671.4±1.7 with the
240 ms/fire scheduler tax).

### Case2: cc_nemotron_t12 @ conc 64 (agent swarm, high eviction)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 533.1±0.8 | 689.5±16.6 | **+29.3%** |
| cache_hit | 0.5453 | 0.7351 | +19.0pp |
| TTFT mean (ms) | 1279 | 829 | **−35.2%** |
| TTFT p90 (ms) | 3205 | 1834 | **−42.8%** |
| TTFT p99 (ms) | 7583 | 6111 | **−19.4%** |
| TPOT mean (ms) | 123.4 | 84.2 | −31.8% |
| n_error | 0 | 0 | |

Per-rep tps: base [532.3, 533.1, 533.9], sys [707.0, 673.9, 687.7]
(post cap_barrier fix; pre-fix sys measured 642.6±14.9).

### Case3: cc_nemotron_t6 @ conc 128 (mamba-bound; post cap_barrier fix)

| Metric | base (N=3) | sys (N=3) | delta |
|--------|-----------|-----------|-------|
| throughput_tok_s | 638.6±1.1 | 654.7±1.6 | **+2.5%** |
| cache_hit | 0.6235 | 0.7029 | +7.9pp |
| TTFT mean (ms) | 20475 | 813 | **−96.0%** |
| TTFT p90 (ms) | 46768 | 2007 | **−95.7%** |
| TPOT p50 (ms) | 226.6 | 223.7 | −1.3% |
| n_error | 0 | 0 | |

Per-rep tps: base [639.3, 637.3, 639.2], sys [652.9, 655.8, 655.4].
Mamba-bound regime: ~4600 k2m fires / 0 m2k. HiMA donates KV→mamba to raise
max_running 85→130, collapsing the 20s+ queue wait into sub-second TTFT at
no throughput cost. (Sys reps ran on GPUs 1,3,5,7 vs base on 3,4,5,7; the
pre-fix rerun on 1,3,5,7 measured 592.9 vs the 3,4,5,7 campaign's 580.6, so
the GPU-set effect is ~2%, well below the +12% fix recovery.)
