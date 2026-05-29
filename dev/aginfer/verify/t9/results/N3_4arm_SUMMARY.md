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

## Final ranking (by per-trial mean)

| rank | arm | mean ± std (s) | Δ vs LRU |
|---|---|---|---|
| 1 | **OURS_full** | **1344.0 ± 54.6** | **−127.8 (8.7 % faster)** |
| 2 | OURS_inline | 1392.8 ± 53.6 | −79.0 (5.4 %) |
| 3 | TA | 1429.4 ± 62.4 | −42.4 (2.9 %) |
| 4 | LRU | 1471.8 ± 104.7 | baseline |

Ordering matches paper §8 expectation: **OURS > TA > LRU**.

## Pairwise Welch t-tests

| arm A | arm B | A mean ± std | B mean ± std | Δ (A−B) | SE | z |
|---|---|---|---|---|---|---|
| LRU | OURS_full | 1471.8 ± 104.7 | 1344.0 ± 54.6 | **+127.8** | 68.2 | **+1.87** |
| TA | OURS_full | 1429.4 ± 62.4 | 1344.0 ± 54.6 | +85.4 | 47.9 | +1.78 |
| LRU | OURS_inline | 1471.8 ± 104.7 | 1392.8 ± 53.6 | +79.0 | 67.9 | +1.16 |
| OURS_inline | OURS_full | 1392.8 ± 53.6 | 1344.0 ± 54.6 | +48.8 | 44.2 | +1.10 |
| TA | OURS_inline | 1429.4 ± 62.4 | 1392.8 ± 53.6 | +36.6 | 47.5 | +0.77 |
| LRU | TA | 1471.8 ± 104.7 | 1429.4 ± 62.4 | +42.4 | 70.4 | +0.60 |

* No pair reaches |z| > 2 (95 % CI cutoff).
* **OURS_full vs LRU** is the closest at **z = +1.87**
  (one-sided p ≈ 0.031, two-sided p ≈ 0.062 — *marginally
  significant*).
* The 9 % spread in mean across the 4 arms is consistent with the
  paper §8 ordering but N=3 doesn't have enough power to call
  pairwise wins above noise.

## Caveats

### GPU-pair confound

Cycles 1, 3 (LRU) and 2 (TA) ran on **GPUs 4, 7**.
Cycles 4, 6 (TA) and 5 (LRU) ran on **GPUs 0, 1** (after a forced
switch when other users took 4, 7).  OURS_inline and OURS_full
ran entirely on GPUs 4, 7.

If GPUs 0, 1 are systematically slower (different NVLink lane,
PCIe topology, or memory bandwidth) than 4, 7, this would
inflate LRU's and TA's measured mean — making OURS look better
than it is.

Within-GPU partial comparison (GPUs 4, 7 only):
* LRU (N=2 of 3 cycles): 1412 s
* TA (N=1 of 3 cycles): 1455 s
* OURS_inline (N=3): 1393 s
* OURS_full (N=3): 1344 s

Ordering still preserved (OURS_full < OURS_inline < LRU), but TA
flips to ≈ LRU at N=1 on the cleaner pair.  Not enough data to
conclude.

Closing this cleanly would require redoing the LRU and TA arms
fully on GPUs 4, 7 — another ~6 cycles, ~6 h GPU time.

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
