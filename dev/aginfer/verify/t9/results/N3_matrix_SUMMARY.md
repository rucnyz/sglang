# T9 N=3 matrix — daemon-side T11a vs inline-only baseline

## Setup

* date: 2026-05-26 23:46 → 2026-05-27 06:13 (6 h 27 min wall)
* sglang HEAD: `41e06f73c5` (post-daemon-T11a, post-cleanup-fix)
* GPUs: 4, 7 (B300 SXM6)
* TP=2, EP=2, HiCache + Mooncake L3
* harbor: 32 trials × max_turns 200, swebenchpro / terminus-2,
  temperature=0.0 seed=42
* sglang `--random-seed 42`

Cycle alternation B-O-B-O-B-O.  Resume after pkill-self-bug fix
left cycle 1 baseline data intact; cycles 2–6 ran fresh.

## Configs

| config | daemon flags | what scores units |
|---|---|---|
| **baseline** | `kv_scheduler=disabled admission=disabled` | inline scorer (hits/age) only |
| **ours** | `kv_scheduler=enabled admission=enabled` | inline (hits/age) + daemon V_u (program-alive rule, T11a) |

Inline scorer identical in both → any latency delta is from
daemon-side T11a + admission.

## Per-cycle stats

| cycle | config | mean (s) | p50 (s) | p99 (s) | stdev (s) | new_tok | cached_tok | hit rate |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 1344.3 | 1164.3 | 3160.6 | 838.7 | 1.72 M | 37.6 M | 0.9562 |
| 2 | ours     | 1398.8 | 1557.3 | 3366.1 | 857.5 | 1.67 M | 34.2 M | 0.9536 |
| 3 | baseline | 1419.6 | 1292.3 | 3295.6 | 864.9 | 1.69 M | 39.1 M | 0.9586 |
| 4 | ours     | 1289.5 | 1211.9 | 3219.7 | 802.3 | 1.72 M | 37.7 M | 0.9563 |
| 5 | baseline | 1404.0 | 1267.8 | 3245.0 | 819.7 | 1.67 M | 39.5 M | 0.9595 |
| 6 | ours     | 1343.7 | 1204.6 | 4118.7 | 922.6 | 1.66 M | 34.6 M | 0.9542 |

## Aggregate

| config | N | per-trial mean (s) | cache hit rate |
|---|---|---|---|
| baseline | 3 | **1389.3 ± 39.7** | 0.9581 ± 0.0017 |
| ours     | 3 | **1344.0 ± 54.6** | 0.9547 ± 0.0014 |

## Statistical test

* Δ mean = −45.3 s (ours faster)
* SE (Welch) ≈ √(39.7²/3 + 54.6²/3) ≈ 39.0 s
* **z ≈ −1.16** (1-sided, p ≈ 0.12)
* Stopping rule: |Δ| < (σ_b + σ_o)/2 = 47.2 s → **no signal**

## Conclusion

**Daemon-side T11a alone does not produce a statistically
meaningful latency improvement** at N=3.  Best-case interpretation:
~3 % faster, within noise.  Cache hit rate essentially identical
across configs (94.5–95.6 %).

## Implication for T11 plan

Per methodology stopping conditions: if 6 cycles show no signal,
**investigate the H' ↔ kv_off 500 s gap before doing inline-side
T11a**.

The single-shot diagnostic numbers (1559 / 1549 / 1457) were noisier
than they looked: the proper N=3 baseline is 1389 s.  H' baseline
was 885 s.  The gap is **~500 s wall-time per trial** that exists
**without** any aginfer scheduling — just from running through
the daemon proxy.  No amount of daemon-side scheduling improvement
can close this gap if its root cause is upstream (HTTP routing
overhead, event bus load, proxy buffering).

Suggested next probes (all faster than another full matrix):
1. **Parse sglang's per-request JSON logs** from these 6 cycles.
   We turned on `--log-requests --log-requests-format json`; each
   request has prompt/output/cached_tokens/latency.  Per-turn TTFT
   distributions should be identical across configs if the issue
   is per-request proxy overhead; if TTFT differs, scheduling
   matters and we missed an effect at the wall-time level.
2. **One-trial direct-sglang comparison** (no daemon, no harbor
   proxy hop) on a single fixed instance.  If direct ≈ 200 s/
   trial-turn and daemon-proxy is ≈ 350 s, we've found the gap.
3. **Daemon profiling** under load — what's the event bus / proxy
   doing per turn?  Add per-event histogram of process time.

## Files

* harbor results: `results/run_K_*_matrix_20260526_234639_cycleN_*/`
* sglang_v4flash.log per cycle: same dirs
* aggregate (this file): `verify/t9/results/N3_matrix_SUMMARY.md`
* methodology: `verify/t9/methodology.md`
