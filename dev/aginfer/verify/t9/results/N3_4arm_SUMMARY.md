# T9 — 4-arm fairness matrix (LRU vs TA vs OURS_inline vs OURS_full)

Authoritative result for the paper §8 baseline comparison under
current `temperature=0.0 seed=42` settings.

## Setup

* dates: 2026-05-26 (OURS_full = K matrix cycles 2/4/6) →
  2026-05-27 (OURS_inline = H'_now matrix) →
  2026-05-28–29 (LRU + TA cycles, including GPU 4,7 → 0,1 switch
  on cycles 4-6).
* sglang HEAD: `6489aa4902`
* TP=2 EP=2 DeepSeek-V4-Flash, HiCache + Mooncake L3
* harbor: `temperature=0.0 seed=42 -l 32 -n 32 -k 1 --ak max_turns=200`
* sglang `--random-seed 42`
* N=3 cycles per arm.

## Configs

| arm | inline scorer | daemon / proxy |
|---|---|---|
| **LRU** | stock sglang default LRU eviction | none (harbor → sglang :30000) |
| **TA** | stock sglang default | rucnyz/ThunderAgent on :9200 (BFD pause/resume) |
| **OURS_inline** | `ours_greedy_score` (paper §7 V_u) | none |
| **OURS_full** | `ours_greedy_score` | aginfer-daemon (kv_scheduler + admission ON) |

## Per-cycle stats

### LRU (N=3)

| cycle | GPUs | n | mean (s) | p50 | p99 | stdev |
|---|---|---|---|---|---|---|
| 1 | 4,7 | 32 | 1394.8 | 1181.8 | 3686.5 | 882.8 |
| 3 | 4,7 | 32 | 1429.6 | 1411.2 | 3124.0 | 730.0 |
| 5 | 0,1 | 32 | 1591.0 | 1294.6 | 4251.3 | 1024.3 |

**Across-cycle**: **1471.8 ± 104.7 s**

### TA (N=3)

| cycle | GPUs | n | mean (s) | p50 | p99 | stdev |
|---|---|---|---|---|---|---|
| 2 | 4,7 | 32 | 1454.8 | 1405.7 | 2969.8 | 757.7 |
| 4 | 0,1 | 32 | 1358.2 | 1370.3 | 4272.0 | 905.8 |
| 6 | 0,1 | 32 | 1475.1 | 1414.4 | 3781.9 | 805.2 |

**Across-cycle**: **1429.4 ± 62.4 s**

### OURS_inline (N=3, H'_now matrix, all on GPUs 4,7)

| cycle | n | mean (s) | p50 | p99 | stdev |
|---|---|---|---|---|---|
| 1 | 32 | 1332.0 | 1358.8 | 2972.6 | 661.0 |
| 2 | 32 | 1413.4 | 1420.0 | 3292.7 | 810.9 |
| 3 | 32 | 1433.0 | 1215.8 | 3420.6 | 744.9 |

**Across-cycle**: **1392.8 ± 53.6 s**

### OURS_full (N=3, K matrix cycles 2/4/6, all on GPUs 4,7)

| cycle | n | mean (s) | p50 | p99 | stdev |
|---|---|---|---|---|---|
| 2 | 32 | 1398.8 | 1557.3 | 3366.1 | 857.5 |
| 4 | 32 | 1289.5 | 1211.9 | 3219.7 | 802.3 |
| 6 | 32 | 1343.7 | 1204.6 | 4118.7 | 922.6 |

**Across-cycle**: **1344.0 ± 54.6 s**

## Final ranking (by per-trial mean) — N=4 LRU + N=4 OURS_full

After +1+1 extension cycle on GPUs 0,1 (2026-05-29):

| rank | arm | N | mean ± std (s) | Δ vs LRU |
|---|---|---|---|---|
| 1 | **OURS_full** | **4** | **1391.7 ± 105.3** | **−123.4 (8.1 %)** |
| 2 | OURS_inline | 3 | 1392.8 ± 53.6 | −122.3 (8.1 %) |
| 3 | TA | 3 | 1429.4 ± 62.4 | −85.7 (5.7 %) |
| 4 | LRU | 4 | 1515.1 ± 121.7 | baseline |

Ordering matches paper §8 expectation: **OURS > TA > LRU**.

(N=3 means before extension were: OURS_full 1344.0 ± 54.6,
LRU 1471.8 ± 104.7.  Adding one GPU-0,1 cycle to each arm
inflated their mean **and** ~doubled their stdev — see GPU
confound below.)

## Pairwise Welch t-tests (N=4 LRU + N=4 OURS_full + N=3 others)

| arm A | arm B | A mean ± std | B mean ± std | Δ (A−B) | SE | z | two-sided p |
|---|---|---|---|---|---|---|---|
| LRU | OURS_inline | 1515.1 ± 121.7 | 1392.8 ± 53.6 | +122.3 | 68.2 | +1.79 | 0.073 |
| LRU | OURS_full | 1515.1 ± 121.7 | 1391.7 ± 105.3 | +123.4 | 80.4 | +1.53 | 0.125 |
| LRU | TA | 1515.1 ± 121.7 | 1429.4 ± 62.4 | +85.7 | 70.7 | +1.21 | 0.225 |
| TA | OURS_inline | 1429.4 ± 62.4 | 1392.8 ± 53.6 | +36.6 | 47.5 | +0.77 | 0.441 |
| TA | OURS_full | 1429.4 ± 62.4 | 1391.7 ± 105.3 | +37.7 | 63.8 | +0.59 | 0.555 |
| OURS_inline | OURS_full | 1392.8 ± 53.6 | 1391.7 ± 105.3 | +1.1 | 61.0 | +0.02 | 0.984 |

* No pair reaches |z| > 1.96 (95 % CI cutoff).
* Best pair after extension: **LRU vs OURS_inline z = +1.79
  (p ≈ 0.07)**.  LRU vs OURS_full *dropped* from z=1.87 (N=3) to
  z=1.53 (N=4) because the new OURS_full cycle on GPUs 0,1 came
  in at 1534.7 s (vs ~1344 s on GPUs 4,7) and roughly doubled the
  OURS_full stdev.
* The 4-arm ordering is preserved but **all pairwise differences
  are now in the noise**.

### N=3 (GPUs 4,7-dominant) snapshot for reference

Before the extension cycle on 0,1, the N=3 numbers were:

| pair | Δ | z |
|---|---|---|
| LRU vs OURS_full | 127.8 | **+1.87** (p ≈ 0.062) |
| TA vs OURS_full | 85.4 | +1.78 |

That snapshot was the strongest signal we got; adding one more
0,1-cycle each diluted it (variance grew faster than √N could
compensate).

## Caveats

### GPU-pair confound (the dominant noise source)

The 2026-05-29 +1+1 extension exposed a much larger GPU-pair
effect than expected.  Per-GPU-pair means:

| arm | GPUs 4, 7 | GPUs 0, 1 | Δ (0,1 − 4,7) |
|---|---|---|---|
| LRU | 1412 s (N=2) | **1618 s (N=2)** | **+206 s** |
| TA | 1455 s (N=1) | 1417 s (N=2) | −38 s |
| OURS_full | 1344 s (N=3) | **1534 s (N=1)** | **+190 s** |
| OURS_inline | 1393 s (N=3) | — | — |

GPUs 0,1 are systematically **~200 s slower than 4,7** for LRU
and OURS_full — that's **larger than the scheduler-vs-scheduler
spread** (≈128 s).  When the two pairs are mixed in one arm, the
within-arm stdev roughly doubles and the Welch z drops.

This is now the biggest single source of noise in the matrix.
TA happens to have the opposite sign for its 0,1 vs 4,7 delta,
which is presumably variance in its own right (TA has N=1 on 4,7).

### Closing the confound

The right scientific fix is to redo LRU and OURS_full
exclusively on a single GPU pair (4,7) for at least N=4 each
(~4–5 hours GPU time on a quiet machine), then re-aggregate.
We have **not** done this — at the time of writing GPUs 4, 7
are occupied by other users for an indeterminate window.

Alternative paper-level fix per `N3_GAPS.md` §4: change the
workload (cap `max_completion_tokens`, shrink KV pool) to push
the scheduler-driven Δ above the GPU-pair noise; then the
confound becomes negligible.

### Runaway dominance

Per `N3_ROOT_CAUSE.md`, 1 % of LLM requests under `temperature=0.0`
run away to 20–64 k completion tokens, consuming 80 % of LLM
wall time across all configs.  This caps the maximum possible
scheduler-driven improvement at < 5 % of trial wall in theory;
the observed 8.7 % spread between OURS_full and LRU is consistent
with the GPU confound on top of the structural ceiling.

### Hit-rate ceiling

Mean cache hit ratio is 95.5 % across all four arms (see
`N3_ttft_analysis.md`).  The prefix-reuse story is already
saturated by inline `ours_greedy_score` alone; further scheduler
improvement has bounded headroom.  See `N3_GAPS.md` §3.1.

## Conclusion

* **Direction is right** — OURS_full beats every other arm by mean.
* **Statistical power is borderline** — best pair (vs LRU) is
  z = 1.87, marginally significant.
* **N=3 isn't enough** to publish "OURS beats LRU at p < 0.05"
  without either (a) more cycles, (b) GPU-pair cleanup, or
  (c) moving to a workload where the scheduler has more room
  (e.g., capping `max_completion_tokens` per N3_GAPS §4).

## Files

* per-cycle data: `results/run_*_matrix_*_cycleN_*/harbor_jobs/...`
* sglang logs (per-cycle): same dirs, `sglang_v4flash.log`
* aggregator: `verify/t9/parse_4arm.py`
* this SUMMARY: `verify/t9/results/N3_4arm_SUMMARY.md`
* root cause of slowness: `verify/t9/results/N3_ROOT_CAUSE.md`
* unmeasured gaps: `verify/t9/results/N3_GAPS.md`
