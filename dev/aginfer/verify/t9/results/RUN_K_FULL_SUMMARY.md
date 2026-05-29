# Run K (full) — 2026-05-26 14:16-15:08

> ⚠️ **SUPERSEDED — N=1 single-shot, claims invalidated.**
>
> Authoritative result is `N3_matrix_SUMMARY.md`
> (N=3 cycles, alternating B-O-B-O-B-O):
> * baseline (kv_off): **1389.3 ± 39.7 s**
> * ours (full): **1344.0 ± 54.6 s**
> * **Δ = −45.3 s, z = −1.16 → not statistically significant.**
>
> The "1.76× Run H' 885 s" framing below is wrong on two counts:
> 1. Single-shot 1559 s was inside ~150 s SEM; not a real signal vs N=3 mean 1344 s.
> 2. Run H' 885 s was measured under a DIFFERENT setting
>    (no `temperature=0.0 seed=42`).  Per `N3_ROOT_CAUSE.md`, this
>    matrix's runaway-generation outliers dominate trial time —
>    not anything T11a daemon scheduling does.  See H'_now N=3
>    matrix (1392.8 ± 53.6 s, no daemon, current settings) — the
>    daemon proxy adds ~4 s/trial, not 500 s.
>
> Keeping this doc as a historical artifact; targets/verdicts below
> are not authoritative.

**Variant:** kv_scheduler=enabled + admission_controller=enabled + HiCache ON

**Acceptance:**
| metric | result | T9 target | status |
|---|---|---|---|
| successful trials | 30 / 32 | ≥ 28 | ✓ |
| per-trial mean | 1559 s | < 716 s | ✗ (1.76× Run H' 885 s) |
| per-trial std | 678 s | — | — |
| per-trial p50 | 1500 s | — | — |
| per-trial p90 | 2511 s | — | — |
| per-trial p99 | 3120 s | < 1336 s | ✗ |
| per-trial max | 3120 s | — | — |
| sum | 49 900 s (13.9 h) | — | — |
| sglang crashes | 0 | 0 | ✓ |
| swebenchpro pass | 0/30 | informational | (V4-Flash + terminus-2 baseline; not our target) |

**Stack:**
* sglang TP=2 + EP=2 on GPUs 4,7, DSV4-Flash, HiCache + Mooncake L3
* kv_policy_loaded=baselines.sglang_adapter:ours_greedy_score (inline scorer)
* daemon: kv_scheduler=enabled, admission_controller=enabled, theta_hi=0.85, theta_lo=0.7
* harbor: terminus-2 / swebenchpro / 32 trials concurrent (-n 32)

**Observations:**
* 5177 `POST /v1/chat/completions` reached daemon (real agent traffic)
* 11748 `/aginfer/state` fetches (each kv_scheduler / admission event)
* No CUDA OOM, no sglang scheduler exits
* Reward mean 0.0 (= no agent fix succeeded; this is V4-Flash + terminus-2 capability ceiling, not our concern)

**Verdict:** K (full) FAILS the < 716 s mean target — we added overhead, didn't reduce.

**Cause-narrowing experiments needed (per T9 README §"WORST CASE"):**
1. **Run K-a** (admission OFF): isolates admission's contribution. If K-a ≈ Run H' (885 s), admission added ~675 s/trial. If K-a still bad, kv_scheduler is the culprit.
2. **Run J** (HiCache OFF): tests whether mooncake L3 is fighting our migrate decisions.

**Theta-mismatch suspicion:**
* sglang's webhook firer classify(occ, theta_hi=0.7, theta_crit=0.9) — fires memory_pressure when occ >= 0.7
* daemon's admission_controller theta_hi=0.85 — pauses only when occ >= 0.85
* Means daemon sees memory_pressure events for occ in [0.7, 0.85) but does NOTHING → wasted state fetches.
* Probably worth aligning both to the same theta (try daemon=0.7 to match sglang) before K-a / J.
