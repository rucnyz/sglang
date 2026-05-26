# Run K — kv_off variant (T9 diagnostic)

## Setup
* date: 2026-05-26 (sglang head 888ea822, daemon-side T11a NOT yet
  active — kv_off ran against pre-T11a daemon module)
* GPUs: 4,7 (B300 SXM6)
* daemon flags: `--kv-scheduler=disabled --admission-controller=disabled`
* sglang: TP=2 EP=2, HiCache ON, Mooncake L3 ON, `ours_greedy_score`
  inline scorer
* harbor: 32 trials, swebenchpro / terminus-2, `--n 32 --ak max_turns=200`
* total wall: 48m 5s

## Per-trial stats (32 trials, 0 exceptions in result.json sense;
   30 reward-bearing + 2 RuntimeError per harbor table)

| metric | value |
|---|---|
| mean | **1457.5 s** |
| p50 | 1479.7 s |
| p99 | 2882.6 s |
| min | 180.5 s |
| max | 2882.6 s |
| stdev | 642.4 s |

## Comparison

| run | mean | vs H' (885) | scope of `hits/age` proxy |
|---|---|---|---|
| H' (inline only, no daemon) | 885 s | 1.00× | inline scorer only |
| **kv_off (inline only, daemon present)** | **1457 s** | **1.65×** | inline scorer only |
| K-a (kv_scheduler on, admission OFF) | 1549 s | 1.75× | inline + daemon |
| K full (both ON) | 1559 s | 1.76× | inline + daemon |
| target | < 716 s | < 0.81× | — |

## Diagnosis

Per the kv_off discriminator (designed in T11 README pre-run):
* "kv_off ≈ 885 s → daemon-side V_u is the culprit, fix daemon
  only"
* "kv_off ≈ 1550 s → inline scorer is *also* at fault"
* **Observed: 1457 s** — between the two pre-registered values.
  Conclusion: BOTH sides have the bug; daemon contributes ~100 s
  (1559 → 1457), inline contributes ~570 s (1457 → 885).

So T11a must touch BOTH:
* daemon-side: `kv_scheduler.py:build_paper_state` (DONE,
  commit 888ea822, not yet re-tested)
* inline-side: `baselines/sglang_adapter.py:_node_to_unit`
  (PENDING; needs daemon → sglang liveness push)

## Open puzzle

H' 885 s ↔ kv_off 1457 s, gap ≈ 570 s.  Both runs use the same
inline scorer and identical sglang/HiCache/Mooncake configs.  The
only known difference is the daemon proxy (kv_off routes harbor
through aginfer-daemon proxy; H' goes direct).

Hypotheses (untested):
1. Proxy HTTP routing overhead — 200 turns × ~3 ms RTT × 32
   concurrent trials ≈ 600 s wall, ballpark fits.
2. Event bus enqueue/dequeue load — `EventKind.LLM_PREFILL` fires
   on every turn even with handlers disabled; bus drains them
   but CPU contention on the sglang side could stall.
3. /aginfer/state polling — handlers disabled, scheduler.fetch
   shouldn't run; need to confirm by daemon log.

Out-of-scope for T11 — file as separate diagnostic if T11a fixes
the inline gap and the proxy overhead surfaces as the residual.

## Files

* harbor results:
  `dev/aginfer/results/run_K_kv_off/harbor_jobs/2026-05-26__17-50-07/`
* daemon log: `results/run_K_kv_off/daemon.log`
* sglang log: `results/run_K_kv_off/sglang.log`
